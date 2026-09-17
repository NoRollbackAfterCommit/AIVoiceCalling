"""Control-plane REST API.

This is what the Angular admin portal talks to: agent profiles, knowledge
ingestion, live call monitoring, analytics. The media plane is the WebSocket in
ws_voice.py; nothing here touches audio.
"""

from __future__ import annotations

import csv
import io
import time
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import APIRouter, Body, File, HTTPException, Query, Request, Response, UploadFile
from pydantic import BaseModel, Field

from vaani.agent.prompt import AgentProfile, profile_to_dict, render_system_prompt
from vaani.api.scope import ensure_visible, is_unrestricted, visible_organisation
from vaani.core.logging import get_logger
from vaani.db.analytics import default_range
from vaani.telephony.numbers import to_e164

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


# The shape the console has always told operators to use, now enforced. It also
# keeps an agent key from colliding with the `org:<id>` namespace the shared
# knowledge set lives in — an agent allowed to be called `org:1` would write
# straight into a customer's shared documents.
AGENT_KEY = r"^[a-z0-9][a-z0-9_-]*$"


class AgentProfileIn(BaseModel):
    key: str = Field(min_length=1, max_length=64, pattern=AGENT_KEY)
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


@router.get("/agent-defaults", tags=["agents"])
async def agent_defaults() -> dict[str, Any]:
    """What a new agent starts as, before anybody edits it.

    The console draws a form for a new agent and then posts it, so it needs the
    starting values to draw. Without them every field the operator leaves alone
    is posted empty and switches its own default off — which is how an agent
    with `tools: []` gets made: one that cannot search its documents, cannot
    transfer, and cannot hang up.

    Outside the /api/agents prefix on purpose. Under it, this would have to be
    registered ahead of /agents/{key} or be swallowed as a key, and nothing
    would catch that the day somebody reorders this file.
    """
    return {**profile_to_dict(_DEFAULTS), "key": ""}


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
async def upsert_agent(
    key: str,
    body: AgentProfileIn,
    request: Request,
    organisation_id: int | None = Query(
        default=None,
        description="Which organisation owns this agent. Honoured only for a "
        "platform administrator; for anyone else their own organisation is used "
        "whatever they ask for.",
    ),
) -> dict[str, Any]:
    services = request.app.state.services
    existing = services.profiles.get(key)
    if existing is not None:
        # Editing somebody else's agent rewrites what their bot says out loud.
        _owned_agent(request, key)

    # Ownership decides which organisation's shared training set this line
    # reads, so somebody has to be able to set it: until they could, every
    # agent a platform administrator created belonged to nobody and could never
    # use a customer's shared documents at all.
    #
    # A query parameter rather than a body field, and read through
    # visible_organisation, which honours it only for an unrestricted caller.
    # An org admin naming somebody else's organisation still gets their own —
    # scope comes from who you are, never from what you asked for.
    if existing is None:
        owner = visible_organisation(request, organisation_id)
    elif organisation_id is not None and is_unrestricted(request):
        owner = organisation_id
    else:
        # An ordinary edit must not quietly re-home the agent.
        owner = getattr(existing, "organisation_id", None)
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
    organisation = await store.organisation(body.organisation_id)
    if organisation is None:
        # Otherwise the number resolves to an organisation nobody can administer.
        raise HTTPException(404, f"No organisation {body.organisation_id}")

    profile = request.app.state.services.profiles.get(body.agent_key)
    if profile is None:
        # An unknown key falls through to the default agent at call time, so the
        # number answers — in the wrong voice, from the wrong documents, with
        # nothing anywhere to say the mapping never took.
        raise HTTPException(404, f"No agent {body.agent_key!r}")

    owner = getattr(profile, "organisation_id", None)
    if owner is not None and owner != body.organisation_id:
        # Found in production, where it had gone unnoticed. The number belonged
        # to one organisation and the agent answering it to another, which
        # breaks the shared training set silently: an upload at organisation
        # scope goes to the agent owner's set, while a call reads the set of the
        # organisation the number belongs to. The document lands where no caller
        # on that number can reach it and the bot answers exactly as before.
        theirs = await store.organisation(owner)
        raise HTTPException(
            400,
            f"Agent {body.agent_key!r} belongs to "
            f"{theirs.name if theirs else f'organisation {owner}'}, but this number "
            f"belongs to {organisation.name}. A number must be answered by its own "
            "organisation's agent, or they cannot share a training set.",
        )
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
    # "agent" puts the document on this line alone; "organisation" puts it in
    # the set every line of that agent's organisation reads.
    scope: Literal["agent", "organisation"] = "agent"
    metadata: dict[str, Any] = Field(default_factory=dict)


def _scope_of(request: Request, agent_key: str, scope: str) -> int | None:
    """The organisation to file a document under, or None for the line itself.

    Ownership is read off the agent, never taken from the request: the caller
    names a line they are allowed to touch, and which organisation that line
    belongs to is not theirs to assert. Writing into another organisation's
    corpus is putting words in their bot's mouth.
    """
    profile = _owned_agent(request, agent_key)
    if scope != "organisation":
        return None
    owner = getattr(profile, "organisation_id", None)
    if owner is None:
        raise HTTPException(
            400,
            f"Agent {agent_key!r} belongs to no organisation, so it has no shared set. "
            "Map it to one first, or file this against the agent itself.",
        )
    return owner


@router.post("/knowledge/text", tags=["knowledge"])
async def ingest_text(body: TextIngest, request: Request) -> dict[str, Any]:
    organisation_id = _scope_of(request, body.agent_key, body.scope)
    retriever = request.app.state.services.retriever
    count = await retriever.index_text(
        body.text,
        source=body.source,
        agent_key=body.agent_key,
        organisation_id=organisation_id,
        metadata=body.metadata,
    )
    return {
        "indexed_chunks": count,
        "source": body.source,
        "agent_key": body.agent_key,
        "scope": body.scope,
    }


@router.post("/knowledge/upload", tags=["knowledge"])
async def ingest_file(
    request: Request,
    file: UploadFile = File(...),
    agent_key: str = Query("default"),
    scope: Literal["agent", "organisation"] = Query("agent"),
) -> dict[str, Any]:
    import tempfile
    from pathlib import Path

    organisation_id = _scope_of(request, agent_key, scope)
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
            chunks, agent_key=agent_key, organisation_id=organisation_id
        )
    except ValueError as exc:
        raise HTTPException(415, str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "indexed_chunks": count,
        "source": file.filename,
        "agent_key": agent_key,
        "scope": scope,
    }


@router.get("/knowledge/search", tags=["knowledge"])
async def search_knowledge(
    request: Request,
    q: str = Query(min_length=1),
    agent_key: str = "default",
    top_k: int = Query(5, ge=1, le=20),
) -> dict[str, Any]:
    """Exposed so operators can test retrieval quality without placing a call —
    the fastest way to diagnose 'the agent gave a wrong answer'."""
    profile = _owned_agent(request, agent_key)
    # Both scopes, because that is what a real call reads; a preview that
    # searched only the line would not show what the caller would be told.
    hits = await request.app.state.services.retriever.search(
        q,
        agent_key=agent_key,
        organisation_id=getattr(profile, "organisation_id", None),
        top_k=top_k,
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
    source: str,
    request: Request,
    agent_key: str = "default",
    scope: Literal["agent", "organisation"] = "agent",
) -> dict[str, Any]:
    organisation_id = _scope_of(request, agent_key, scope)
    removed = await request.app.state.services.retriever.delete_source(
        source, agent_key=agent_key, organisation_id=organisation_id
    )
    return {"source": source, "removed_chunks": removed, "scope": scope}


@router.get("/knowledge/sources", tags=["knowledge"])
async def knowledge_sources(
    request: Request,
    agent_key: str = "default",
    scope: Literal["agent", "organisation", "both"] = "both",
) -> list[dict[str, Any]]:
    """What is actually indexed, so an operator is not uploading blind.

    Both scopes by default, each row saying which it came from: the question an
    operator is really asking is "what can this line answer from", and half an
    answer to that is how a document gets uploaded twice.
    """
    profile = _owned_agent(request, agent_key)
    owner = getattr(profile, "organisation_id", None)
    retriever = request.app.state.services.retriever

    rows: list[dict[str, Any]] = []
    if scope in ("agent", "both"):
        rows += [
            {"source": source, "chunks": chunks, "scope": "agent"}
            for source, chunks in await retriever.sources(agent_key=agent_key)
        ]
    if scope in ("organisation", "both") and owner is not None:
        rows += [
            {"source": source, "chunks": chunks, "scope": "organisation"}
            for source, chunks in await retriever.sources(organisation_id=owner)
        ]
    return rows


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
async def analytics(
    request: Request,
    organisation_id: int | None = Query(default=None),
    did: str | None = Query(default=None),
    days: int = Query(default=30, ge=1, le=366),
) -> dict[str, Any]:
    """Figures over the stored calls, for one organisation or the whole platform.

    Read from the database rather than from the in-memory ring of the last two
    hundred calls, which is lost whenever the container is recreated — and a
    deploy recreates it every time.
    """
    mis = _analytics(request)
    scope_id = visible_organisation(request, organisation_id)
    since, until = default_range(days)

    summary = await mis.summary(organisation_id=scope_id, did=did, since=since, until=until)
    summary["knowledge_gaps"] = await mis.knowledge_gaps(
        organisation_id=scope_id, did=did, since=since, until=until
    )
    summary["window_days"] = days
    summary["organisation_id"] = scope_id
    summary["live"] = len(request.app.state.calls.live(scope_id))
    if is_unrestricted(request):
        # Capacity is a property of the box, not of a customer's call centre.
        summary["capacity"] = request.app.state.services.settings.max_concurrent_calls
    return summary


@router.get("/analytics/export", tags=["analytics"])
async def export_calls(
    request: Request,
    organisation_id: int | None = Query(default=None),
    did: str | None = Query(default=None),
    days: int = Query(default=30, ge=1, le=366),
    limit: int = Query(default=5000, ge=1, le=50000),
) -> Response:
    """The call list as CSV, honouring exactly the scope the API does.

    An export that ignored scoping would be the easiest way to walk out with
    another organisation's calls.
    """
    repository = request.app.state.services.calls
    if repository is None:
        raise HTTPException(503, "No database is configured, so there is nothing to export")

    scope_id = visible_organisation(request, organisation_id)
    since, _until = default_range(days)
    rows = await repository.recent(limit, scope_id)
    wanted = to_e164(did) or did if did else None

    columns = [
        "call_id",
        "started_at",
        "organisation_id",
        "did",
        "caller_number",
        "agent_key",
        "direction",
        "duration_s",
        "outcome",
        "disposition",
        "reference",
        "language",
    ]
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(columns)
    for row in rows:
        if row.get("started_at", 0) < since:
            continue
        if wanted and row.get("did") != wanted:
            continue
        writer.writerow([_csv_safe(row.get(c)) for c in columns])

    stamp = datetime.now(tz=UTC).strftime("%Y%m%d")
    name = f"calls-{stamp}.csv"
    return Response(
        content=buffer.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


# A leading one of these makes a spreadsheet treat the cell as a formula.
_FORMULA_STARTERS = ("=", "+", "-", "@", chr(9), chr(13))


def _csv_safe(value: Any) -> Any:
    """Stop a spreadsheet treating a field as a formula.

    A caller number beginning with + is the obvious one, and a transcript
    fragment starting with = would be executed on open in Excel.
    """
    if isinstance(value, str) and value[:1] in _FORMULA_STARTERS:
        return "'" + value
    return value


def _analytics(request: Request) -> Any:
    mis = getattr(request.app.state.services, "analytics", None)
    if mis is None:
        raise HTTPException(503, "No database is configured, so there are no figures to report")
    return mis


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
