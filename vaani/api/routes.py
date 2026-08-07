"""Control-plane REST API.

This is what the Angular admin portal talks to: agent profiles, knowledge
ingestion, live call monitoring, analytics. The media plane is the WebSocket in
ws_voice.py; nothing here touches audio.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Body, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field

from vaani.agent.prompt import AgentProfile, render_system_prompt
from vaani.core.logging import get_logger

log = get_logger(__name__)
router = APIRouter()

_STARTED = time.time()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@router.get("/health", tags=["ops"])
async def health(request: Request) -> dict[str, Any]:
    services = request.app.state.services
    return {
        "status": "ok",
        "uptime_s": round(time.time() - _STARTED, 1),
        "providers": {
            "stt": services.stt.name,
            "llm": services.llm.name,
            "tts": services.tts.name,
        },
        "calls": {
            "live": request.app.state.calls.live_count,
            "capacity": services.settings.max_concurrent_calls,
        },
    }


@router.get("/ready", tags=["ops"])
async def ready(request: Request) -> dict[str, Any]:
    """Kubernetes readiness: refuse traffic while at capacity so the load
    balancer routes new calls to a pod that can actually take them."""
    manager = request.app.state.calls
    if manager.at_capacity:
        raise HTTPException(status_code=503, detail="at capacity")
    return {"status": "ready"}


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


class AgentProfileIn(BaseModel):
    key: str = Field(min_length=1, max_length=64)
    name: str = "Assistant"
    organisation: str = "the organisation"
    role: str = "Answer caller questions accurately and help them complete tasks."
    languages: list[str] = Field(default_factory=lambda: ["English"])
    tone: str = "warm, patient, professional"
    greeting: str = "Namaste. How may I help you today?"
    closing: str = "Thank you for calling. Have a good day."
    policies: list[str] = Field(default_factory=list)
    forbidden_topics: list[str] = Field(default_factory=list)
    escalation_rules: list[str] | None = None
    voice: str | None = None
    tools: list[str] = Field(default_factory=lambda: ["search_knowledge", "transfer_to_human"])
    max_tool_iterations: int = Field(default=4, ge=1, le=8)


@router.get("/agents", tags=["agents"])
async def list_agents(request: Request) -> list[dict[str, Any]]:
    profiles = request.app.state.services.profiles
    return [
        {
            "key": p.key,
            "name": p.name,
            "organisation": p.organisation,
            "languages": p.languages,
            "tools": p.tools,
        }
        for p in profiles.values()
    ]


@router.get("/agents/{key}", tags=["agents"])
async def get_agent(key: str, request: Request) -> dict[str, Any]:
    profile = request.app.state.services.profiles.get(key)
    if profile is None:
        raise HTTPException(404, f"No agent profile {key!r}")
    return {**profile.__dict__, "system_prompt": render_system_prompt(profile)}


@router.put("/agents/{key}", tags=["agents"])
async def upsert_agent(key: str, body: AgentProfileIn, request: Request) -> dict[str, Any]:
    services = request.app.state.services
    known = set(services.tools.names())
    unknown = [t for t in body.tools if t not in known]
    if unknown:
        raise HTTPException(400, f"Unknown tools: {unknown}. Available: {sorted(known)}")

    fields = body.model_dump(exclude_none=True)
    fields["key"] = key
    profile = AgentProfile(**fields)
    services.profiles[key] = profile
    log.info("agent profile saved", extra={"agent": key})
    return {"key": key, "system_prompt": render_system_prompt(profile)}


@router.delete("/agents/{key}", tags=["agents"])
async def delete_agent(key: str, request: Request) -> dict[str, str]:
    if key == "default":
        raise HTTPException(400, "The default profile cannot be deleted")
    request.app.state.services.profiles.pop(key, None)
    return {"status": "deleted"}


@router.get("/tools", tags=["agents"])
async def list_tools(request: Request) -> list[dict[str, Any]]:
    registry = request.app.state.services.tools
    return [registry.get(n).to_wire()["function"] for n in registry.names()]


# ---------------------------------------------------------------------------
# Knowledge
# ---------------------------------------------------------------------------


class TextIngest(BaseModel):
    text: str = Field(min_length=1)
    source: str = "manual-entry"
    agent_key: str = "default"
    metadata: dict[str, Any] = Field(default_factory=dict)


@router.post("/knowledge/text", tags=["knowledge"])
async def ingest_text(body: TextIngest, request: Request) -> dict[str, Any]:
    retriever = request.app.state.services.retriever
    count = await retriever.index_text(
        body.text, source=body.source, agent_key=body.agent_key, metadata=body.metadata
    )
    return {"indexed_chunks": count, "source": body.source, "agent_key": body.agent_key}


@router.post("/knowledge/upload", tags=["knowledge"])
async def ingest_file(
    request: Request,
    file: UploadFile = File(...),
    agent_key: str = Query("default"),
) -> dict[str, Any]:
    import tempfile
    from pathlib import Path

    suffix = Path(file.filename or "upload.txt").suffix or ".txt"
    raw = await file.read()
    if len(raw) > 64 * 1024 * 1024:
        raise HTTPException(413, "File exceeds the 64 MB limit")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)
    try:
        from vaani.rag.chunking import chunk_text, load_file

        text = load_file(tmp_path)
        chunks = chunk_text(text, source=file.filename or tmp_path.name)
        count = await request.app.state.services.retriever.index_chunks(
            chunks, agent_key=agent_key
        )
    except ValueError as exc:
        raise HTTPException(415, str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return {"indexed_chunks": count, "source": file.filename, "agent_key": agent_key}


@router.get("/knowledge/search", tags=["knowledge"])
async def search_knowledge(
    request: Request,
    q: str = Query(min_length=1),
    agent_key: str = "default",
    top_k: int = Query(5, ge=1, le=20),
) -> dict[str, Any]:
    """Exposed so operators can test retrieval quality without placing a call —
    the fastest way to diagnose 'the agent gave a wrong answer'."""
    hits = await request.app.state.services.retriever.search(
        q, agent_key=agent_key, top_k=top_k
    )
    return {
        "query": q,
        "hits": [
            {"text": h.text, "source": h.source, "score": h.score, "metadata": h.metadata}
            for h in hits
        ],
    }


@router.delete("/knowledge/source/{source}", tags=["knowledge"])
async def delete_source(
    source: str, request: Request, agent_key: str = "default"
) -> dict[str, Any]:
    removed = await request.app.state.services.retriever.delete_source(
        source, agent_key=agent_key
    )
    return {"source": source, "removed_chunks": removed}


@router.get("/knowledge/sources", tags=["knowledge"])
async def knowledge_sources(request: Request, agent_key: str = "default") -> list[dict[str, Any]]:
    """What is actually indexed, so an operator is not uploading blind."""
    pairs = await request.app.state.services.retriever.sources(agent_key=agent_key)
    return [{"source": source, "chunks": chunks} for source, chunks in pairs]


@router.get("/knowledge/stats", tags=["knowledge"])
async def knowledge_stats(request: Request, agent_key: str | None = None) -> dict[str, Any]:
    count = await request.app.state.services.retriever.count(agent_key)
    return {"agent_key": agent_key, "chunks": count}


# ---------------------------------------------------------------------------
# Calls and analytics
# ---------------------------------------------------------------------------


@router.get("/calls/live", tags=["calls"])
async def live_calls(request: Request) -> list[dict[str, Any]]:
    return request.app.state.calls.live()


@router.get("/calls", tags=["calls"])
async def call_history(
    request: Request, limit: int = Query(50, ge=1, le=500)
) -> list[dict[str, Any]]:
    """Served from the database when one is configured: the in-memory manager
    only knows about calls this process handled, which is not an audit trail."""
    repository = request.app.state.services.calls
    if repository is None:
        return request.app.state.calls.history(limit)
    return await repository.recent(limit)


@router.get("/calls/{call_id}", tags=["calls"])
async def get_call(call_id: str, request: Request) -> dict[str, Any]:
    manager = request.app.state.calls
    session = manager.get(call_id)
    if session is not None:
        return session.record.to_dict()
    for record in manager.history(500):
        if record["call_id"] == call_id:
            return record
    raise HTTPException(404, f"No call {call_id!r}")


@router.post("/calls/{call_id}/hangup", tags=["calls"])
async def hangup_call(call_id: str, request: Request) -> dict[str, str]:
    if not await request.app.state.calls.hangup(call_id):
        raise HTTPException(404, f"No live call {call_id!r}")
    return {"status": "ended"}


@router.get("/analytics/summary", tags=["analytics"])
async def analytics(request: Request) -> dict[str, Any]:
    manager = request.app.state.calls
    stats = manager.stats()

    # Knowledge-gap mining: the questions that produced no useful retrieval are
    # the highest-value input to the next round of content authoring.
    languages: dict[str, int] = {}
    unresolved: list[str] = []
    for record in manager.history(200):
        for lang in record.get("languages", []):
            languages[lang] = languages.get(lang, 0) + 1
        for turn in record.get("turns", []):
            if "search_knowledge" in turn.get("tools", []) and _looks_unresolved(turn):
                unresolved.append(turn["caller"])

    return {
        **stats,
        "languages": languages,
        "knowledge_gaps": unresolved[:25],
    }


def _looks_unresolved(turn: dict[str, Any]) -> bool:
    reply = (turn.get("agent") or "").lower()
    return any(
        phrase in reply
        for phrase in ("could not find", "do not have that", "don't have that", "not sure")
    )


# ---------------------------------------------------------------------------
# Text-only conversation, for testing an agent without audio
# ---------------------------------------------------------------------------


@router.post("/simulate/turn", tags=["testing"])
async def simulate_turn(
    request: Request,
    text: str = Body(embed=True),
    agent_key: str = Body("default", embed=True),
    history: list[dict[str, str]] = Body(default_factory=list, embed=True),
) -> dict[str, Any]:
    """Run one agent turn over text.

    Invaluable for regression testing prompts and knowledge: you can assert on
    the agent's answers in CI without synthesising a single second of audio.
    """
    from vaani.agent.runtime import ConversationAgent
    from vaani.agent.tools.base import ToolContext

    services = request.app.state.services
    ctx = ToolContext(
        call_id="simulated", agent_key=agent_key, services=services.as_tool_services()
    )
    agent = ConversationAgent(
        profile=services.profile(agent_key), llm=services.llm, tools=services.tools, ctx=ctx
    )
    for message in history:
        if message.get("role") == "user":
            agent.note_user(message.get("content", ""))
        elif message.get("role") == "assistant":
            agent.note_assistant(message.get("content", ""))

    turn = await agent.respond(text)
    return {
        "text": turn.text,
        "control": turn.control,
        "tool_calls": turn.tool_calls,
        "latency_ms": turn.latency_ms,
    }
