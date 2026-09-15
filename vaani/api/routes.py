"""Control-plane REST API.

This is what the Angular admin portal talks to: agent profiles, knowledge
ingestion, live call monitoring, analytics. The media plane is the WebSocket in
ws_voice.py; nothing here touches audio.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Body, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field

from vaani.agent.prompt import AgentProfile, profile_to_dict, render_system_prompt
from vaani.api.scope import ensure_visible, is_unrestricted, visible_organisation
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


# Defaults come from the dataclass so the two cannot disagree, and a test pins
# this field list to it: a field this model forgets is one every save resets.
_DEFAULTS = AgentProfile(key="_")


class AgentProfileIn(BaseModel):
    key: str = Field(min_length=1, max_length=64)
    name: str = _DEFAULTS.name
    organisation: str = _DEFAULTS.organisation
    role: str = _DEFAULTS.role
    languages: list[str] = Field(default_factory=lambda: list(_DEFAULTS.languages))
    tone: str = _DEFAULTS.tone
    greeting: str = _DEFAULTS.greeting
    closing: str = _DEFAULTS.closing
    policies: list[str] = Field(default_factory=list)
    knowledge_guidelines: str = _DEFAULTS.knowledge_guidelines
    forbidden_topics: list[str] = Field(default_factory=list)
    escalation_rules: list[str] | None = None
    voice: str | None = None
    voices: dict[str, str] = Field(default_factory=dict)
    objective: str = _DEFAULTS.objective
    extra_dispositions: list[str] = Field(default_factory=list)
    stall_after: int = Field(default=_DEFAULTS.stall_after, ge=1, le=10)
    ask_language: bool = _DEFAULTS.ask_language
    language_prompt: str = _DEFAULTS.language_prompt
    tools: list[str] = Field(default_factory=lambda: list(_DEFAULTS.tools))
    max_tool_iterations: int = Field(default=_DEFAULTS.max_tool_iterations, ge=1, le=8)


def _owned_agent(request: Request, key: str) -> AgentProfile:
    """The profile, or 404 if it belongs to another organisation.

    404 rather than 403 throughout: telling somebody that an agent exists but is
    forbidden discloses that the organisation is a customer and what its agents
    are called.
    """
    profile = request.app.state.services.profiles.get(key)
    if profile is None:
        raise HTTPException(404, f"No agent profile {key!r}")
    ensure_visible(request, getattr(profile, "organisation_id", None), "agent profile")
    return profile


@router.get("/agents", tags=["agents"])
async def list_agents(request: Request) -> list[dict[str, Any]]:
    profiles = request.app.state.services.profiles
    mine = visible_organisation(request)
    return [
        {
            "key": p.key,
            "name": p.name,
            "organisation": p.organisation,
            "languages": p.languages,
            "tools": p.tools,
        }
        for p in profiles.values()
        if mine is None or getattr(p, "organisation_id", None) == mine
    ]


@router.get("/agents/{key}", tags=["agents"])
async def get_agent(key: str, request: Request) -> dict[str, Any]:
    profile = _owned_agent(request, key)
    return {**profile_to_dict(profile), "system_prompt": render_system_prompt(profile)}


@router.put("/agents/{key}", tags=["agents"])
async def upsert_agent(key: str, body: AgentProfileIn, request: Request) -> dict[str, Any]:
    services = request.app.state.services
    existing = services.profiles.get(key)
    if existing is not None:
        # Editing somebody else's agent rewrites what their bot says out loud.
        _owned_agent(request, key)
    owner = (
        getattr(existing, "organisation_id", None)
        if existing is not None
        else visible_organisation(request)
    )
    known = set(services.tools.names())
    unknown = [t for t in body.tools if t not in known]
    if unknown:
        raise HTTPException(400, f"Unknown tools: {unknown}. Available: {sorted(known)}")

    fields = body.model_dump(exclude_none=True)
    fields["key"] = key
    # Ownership is never taken from the request body: an org admin creating an
    # agent gets their own organisation, and an existing one keeps the owner it
    # already had.
    fields["organisation_id"] = owner
    profile = AgentProfile(**fields)
    # Persist before applying, so a database failure surfaces as a failed save
    # rather than a profile that works until the next restart.
    if services.profile_store is not None:
        await services.profile_store.save(profile)
    services.profiles[key] = profile
    log.info("agent profile saved", extra={"agent": key})
    return {"key": key, "system_prompt": render_system_prompt(profile)}


@router.delete("/agents/{key}", tags=["agents"])
async def delete_agent(key: str, request: Request) -> dict[str, str]:
    if key == "default":
        raise HTTPException(400, "The default profile cannot be deleted")
    _owned_agent(request, key)
    services = request.app.state.services
    if services.profile_store is not None:
        await services.profile_store.delete(key)
    services.profiles.pop(key, None)
    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# Organisations and the numbers that reach them
# ---------------------------------------------------------------------------


class OrganisationIn(BaseModel):
    slug: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=160)


class OrganisationPatch(BaseModel):
    name: str | None = Field(default=None, max_length=160)
    active: bool | None = None


class DidIn(BaseModel):
    organisation_id: int
    agent_key: str = Field(min_length=1, max_length=64)
    label: str | None = Field(default=None, max_length=120)
    active: bool = True


def _tenancy(request: Request) -> Any:
    store = getattr(request.app.state.services, "tenancy", None)
    if store is None:
        # A bare install with no database. Every call is unattributed, and there
        # is nowhere to put an organisation, so this is a configuration answer
        # rather than a missing page.
        raise HTTPException(503, "No database is configured, so organisations cannot be stored")
    return store


@router.get("/organisations", tags=["tenancy"])
async def list_organisations(request: Request) -> list[dict[str, Any]]:
    mine = visible_organisation(request)
    return [
        asdict(o) for o in await _tenancy(request).organisations() if mine is None or o.id == mine
    ]


@router.post("/organisations", tags=["tenancy"], status_code=201)
async def create_organisation(body: OrganisationIn, request: Request) -> dict[str, Any]:
    try:
        org = await _tenancy(request).create_organisation(slug=body.slug, name=body.name)
    except ValueError as exc:
        # 409 rather than 400: the request is well formed, the slug is taken.
        raise HTTPException(409, str(exc)) from exc
    log.info("organisation created", extra={"slug": org.slug})
    return asdict(org)


@router.patch("/organisations/{organisation_id}", tags=["tenancy"])
async def update_organisation(
    organisation_id: int, body: OrganisationPatch, request: Request
) -> dict[str, Any]:
    store = _tenancy(request)
    if await store.organisation(organisation_id) is None:
        raise HTTPException(404, f"No organisation {organisation_id}")
    ensure_visible(request, organisation_id, "organisation")
    if body.name is not None:
        await store.rename_organisation(organisation_id, body.name)
    if body.active is not None:
        # Suspending an organisation takes its numbers out of service without
        # deleting anything: the calls already recorded against them still have
        # to be attributable.
        await store.set_organisation_active(organisation_id, body.active)
        log.info(
            "organisation availability changed",
            extra={"organisation_id": organisation_id, "active": body.active},
        )
    return asdict(await store.organisation(organisation_id))


@router.get("/dids", tags=["tenancy"])
async def list_dids(
    request: Request, organisation_id: int | None = Query(default=None)
) -> list[dict[str, Any]]:
    return await _tenancy(request).dids_with_organisation(
        visible_organisation(request, organisation_id)
    )


@router.put("/dids/{number}", tags=["tenancy"])
async def upsert_did(number: str, body: DidIn, request: Request) -> dict[str, Any]:
    store = _tenancy(request)
    if await store.organisation(body.organisation_id) is None:
        # Otherwise the number resolves to an organisation nobody can administer.
        raise HTTPException(404, f"No organisation {body.organisation_id}")
    try:
        did = await store.upsert_did(
            number=number,
            organisation_id=body.organisation_id,
            agent_key=body.agent_key,
            label=body.label,
            active=body.active,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    log.info(
        "did mapped",
        extra={"did": did.number, "organisation_id": did.organisation_id, "agent": did.agent_key},
    )
    return asdict(did)


@router.delete("/dids/{number}", tags=["tenancy"])
async def delete_did(number: str, request: Request) -> dict[str, str]:
    """Releases the number. The calls it already took keep their attribution,
    which is why they store the organisation rather than joining through here."""
    await _tenancy(request).delete_did(number)
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
    # Writing into another organisation's corpus is putting words in their
    # bot's mouth, so the agent is checked before a single chunk is embedded.
    _owned_agent(request, body.agent_key)
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

    _owned_agent(request, agent_key)
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
        count = await request.app.state.services.retriever.index_chunks(chunks, agent_key=agent_key)
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
    _owned_agent(request, agent_key)
    hits = await request.app.state.services.retriever.search(q, agent_key=agent_key, top_k=top_k)
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
    _owned_agent(request, agent_key)
    removed = await request.app.state.services.retriever.delete_source(source, agent_key=agent_key)
    return {"source": source, "removed_chunks": removed}


@router.get("/knowledge/sources", tags=["knowledge"])
async def knowledge_sources(request: Request, agent_key: str = "default") -> list[dict[str, Any]]:
    """What is actually indexed, so an operator is not uploading blind."""
    _owned_agent(request, agent_key)
    pairs = await request.app.state.services.retriever.sources(agent_key=agent_key)
    return [{"source": source, "chunks": chunks} for source, chunks in pairs]


@router.get("/knowledge/stats", tags=["knowledge"])
async def knowledge_stats(request: Request, agent_key: str | None = None) -> dict[str, Any]:
    if agent_key is not None:
        _owned_agent(request, agent_key)
    elif not is_unrestricted(request):
        # A total across every organisation is a platform figure, not a
        # customer's. Narrow it to their own rather than refusing outright.
        return await _own_knowledge_total(request)
    count = await request.app.state.services.retriever.count(agent_key)
    return {"agent_key": agent_key, "chunks": count}


async def _own_knowledge_total(request: Request) -> dict[str, Any]:
    retriever = request.app.state.services.retriever
    mine = visible_organisation(request)
    total = 0
    for profile in request.app.state.services.profiles.values():
        if getattr(profile, "organisation_id", None) == mine:
            total += await retriever.count(profile.key)
    return {"agent_key": None, "chunks": total}


# ---------------------------------------------------------------------------
# Calls and analytics
# ---------------------------------------------------------------------------


@router.get("/calls/live", tags=["calls"])
async def live_calls(request: Request) -> list[dict[str, Any]]:
    return request.app.state.calls.live(visible_organisation(request))


@router.get("/calls", tags=["calls"])
async def call_history(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
    organisation_id: int | None = Query(default=None),
) -> list[dict[str, Any]]:
    """Served from the database when one is configured: the in-memory manager
    only knows about calls this process handled, which is not an audit trail.

    `organisation_id` narrows the view for a platform administrator. For anyone
    else it is ignored — scope comes from who you are, not what you asked for.
    """
    mine = visible_organisation(request, organisation_id)
    repository = request.app.state.services.calls
    if repository is None:
        rows = request.app.state.calls.history(limit)
        return [r for r in rows if mine is None or r.get("organisation_id") == mine]
    return await repository.recent(limit, mine)


@router.get("/calls/{call_id}", tags=["calls"])
async def get_call(call_id: str, request: Request) -> dict[str, Any]:
    manager = request.app.state.calls
    session = manager.get(call_id)
    if session is not None:
        ensure_visible(request, session.record.organisation_id, "call")
        return session.record.to_dict()
    for record in manager.history(500):
        if record["call_id"] == call_id:
            ensure_visible(request, record.get("organisation_id"), "call")
            return record
    # Memory holds only what this process handled, which is why the history
    # endpoint above reads the database. This one must too, or every call that
    # endpoint lists answers 404 after a restart — and the per-turn metrics,
    # which is where the latency figures live, become unreachable entirely.
    repository = request.app.state.services.calls
    if repository is not None:
        stored = await repository.get_call(call_id)
        if stored is not None:
            ensure_visible(request, stored.get("organisation_id"), "call")
            stored["turns"] = await repository.turns_for(call_id)
            return stored
    raise HTTPException(404, f"No call {call_id!r}")


@router.post("/calls/{call_id}/hangup", tags=["calls"])
async def hangup_call(call_id: str, request: Request) -> dict[str, str]:
    manager = request.app.state.calls
    session = manager.get(call_id)
    if session is None:
        raise HTTPException(404, f"No live call {call_id!r}")
    # Checked before the hang-up, not after: ending a stranger's call and then
    # reporting it forbidden would be the worst of both.
    ensure_visible(request, session.record.organisation_id, "live call")
    if not await manager.hangup(call_id):
        raise HTTPException(404, f"No live call {call_id!r}")
    return {"status": "ended"}


@router.get("/analytics/summary", tags=["analytics"])
async def analytics(request: Request) -> dict[str, Any]:
    manager = request.app.state.calls
    mine = visible_organisation(request)
    stats = manager.stats()
    # Deployment-wide figures belong to the platform. A customer sees how many
    # of the lines are theirs, not how busy the box is.
    if mine is not None:
        own_live = manager.live(mine)
        stats = {**stats, "live": len(own_live)}
        stats.pop("capacity", None)
        stats.pop("total_calls", None)

    # Knowledge-gap mining: the questions that produced no useful retrieval are
    # the highest-value input to the next round of content authoring.
    languages: dict[str, int] = {}
    unresolved: list[str] = []
    for record in manager.history(200):
        if mine is not None and record.get("organisation_id") != mine:
            continue
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
