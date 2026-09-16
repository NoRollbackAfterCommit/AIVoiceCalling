"""A completed call must be readable from the database afterwards."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from vaani.api.routes import get_call
from vaani.config import Settings
from vaani.core.registry import build_services
from vaani.db.repository import CallRepository
from vaani.pipeline.session import CallSession, CallState

from .test_pipeline import FakeTransport, settle, silence, tone


def _settings() -> Settings:
    return Settings(
        stt_provider="mock",
        llm_provider="mock",
        tts_provider="mock",
        vector_store="memory",
        embedding_provider="hash",
        record_calls=False,
        end_of_turn_silence_ms=200,
        idle_prompt_after_s=120,
        idle_hangup_after_s=600,
    )


@pytest.fixture
async def services_with_db(tmp_path):
    svc = build_services(_settings())
    # See tests/test_pipeline.py: language selection has its own tests.
    svc.profiles["default"] = replace(svc.profiles["default"], ask_language=False)
    svc.calls = CallRepository(f"sqlite+aiosqlite:///{tmp_path / 'calls.db'}")
    await svc.calls.start()
    await svc.start()
    yield svc
    await svc.close()
    await svc.calls.close()


def _replied(session: CallSession) -> bool:
    """The agent has produced a reply for the turn just pushed.

    Either the turn is already on the record, or the reply is being played and
    the hangup below will finalise it. Waiting for the record alone would mean
    waiting out the whole reply at real time — about five seconds of mock audio
    — in every test that drives a turn.
    """
    return bool(session.record.turns) or session.state == CallState.SPEAKING


async def _run_one_turn(session: CallSession) -> None:
    """Drive one complete caller turn, then hang up.

    Waits on the session reaching each step rather than on a duration. The
    fixed sleeps this replaces were 0.1 s for the greeting and 0.8 s for the
    reply; on a machine under load — which is any machine running the whole
    suite — the agent was still thinking when the hangup landed, so the call
    was stored with no turns and whichever test read them first failed.
    """
    task = asyncio.create_task(session.run())
    try:
        reached = await settle(lambda: session.state == CallState.LISTENING)
        assert reached, f"greeting never finished; state is {session.state}"

        await session.push_audio(tone(700))
        await session.push_audio(silence(500))
        assert await settle(lambda: _replied(session)), (
            f"the utterance produced no reply; state is {session.state}"
        )
    except BaseException:
        # Without this the session task outlives a failed assertion and the
        # fixture's close() waits on it forever, so pytest hangs before it can
        # print why the assertion failed.
        task.cancel()
        raise

    await session.hangup("completed")
    await asyncio.wait_for(task, timeout=10)


async def test_a_call_is_persisted_with_its_turns(services_with_db):
    svc = services_with_db
    session = CallSession(
        transport=FakeTransport(),
        services=svc,
        agent_key="default",
        caller_number="+919876543210",
    )
    await _run_one_turn(session)

    row = await svc.calls.get_call(session.call_id)
    assert row is not None
    assert row["caller_number"] == "+919876543210"
    assert row["outcome"] == "completed"
    assert row["ended_at"] is not None

    turns = await svc.calls.turns_for(session.call_id)
    assert turns, "expected at least one persisted turn"
    assert turns[0]["seq"] == 0
    assert {t["role"] for t in turns} == {"caller", "agent"}


async def test_the_record_survives_a_restart(tmp_path, services_with_db):
    svc = services_with_db
    session = CallSession(transport=FakeTransport(), services=svc, agent_key="default")
    await _run_one_turn(session)
    call_id = session.call_id
    await svc.calls.close()

    reopened = CallRepository(f"sqlite+aiosqlite:///{tmp_path / 'calls.db'}")
    await reopened.start()
    try:
        assert (await reopened.get_call(call_id))["outcome"] == "completed"
        assert await reopened.turns_for(call_id)
    finally:
        await reopened.close()
    svc.calls = reopened  # so the fixture teardown closes something valid


class _ForgetfulManager:
    """A call manager with no memory of the call, as after any restart."""

    def get(self, call_id: str) -> None:
        return None

    def history(self, limit: int) -> list:
        return []


def _request(manager, repository) -> SimpleNamespace:
    state = SimpleNamespace(calls=manager, services=SimpleNamespace(calls=repository))
    return SimpleNamespace(app=SimpleNamespace(state=state))


async def test_a_stored_call_is_readable_once_the_process_has_forgotten_it(services_with_db):
    """`GET /api/calls/{id}` must fall through to the database.

    The history endpoint above it already reads the repository, for the reason
    written in its own docstring: the in-memory manager only knows the calls
    this process handled, which is not an audit trail. The detail endpoint read
    memory alone, so every call the list showed came back 404 after a restart —
    and per-turn latency metrics, which live on the turns, were unreachable.
    """
    svc = services_with_db
    session = CallSession(transport=FakeTransport(), services=svc, agent_key="default")
    await _run_one_turn(session)

    record = await get_call(session.call_id, _request(_ForgetfulManager(), svc.calls))

    assert record["call_id"] == session.call_id
    assert record["outcome"] == "completed"
    assert record["turns"], "the stored turns must come back with the call"
    assert {t["role"] for t in record["turns"]} == {"caller", "agent"}
    # The timings are the point: latency work cannot read what the API drops.
    # They are flat columns on the turn, not a nested metrics dict.
    first = record["turns"][0]
    assert {"stt_ms", "agent_ms", "tts_first_chunk_ms", "total_ms"} <= set(first)
    assert isinstance(first["total_ms"], int)


async def test_a_call_nobody_has_ever_seen_is_still_a_404(services_with_db):
    with pytest.raises(HTTPException) as caught:
        await get_call("no-such-call", _request(_ForgetfulManager(), services_with_db.calls))
    assert caught.value.status_code == 404


async def test_sessions_run_fine_without_a_repository():
    """Persistence is optional: a bare install must still place calls."""
    svc = build_services(_settings())
    svc.profiles["default"] = replace(svc.profiles["default"], ask_language=False)
    await svc.start()
    assert svc.calls is None

    session = CallSession(transport=FakeTransport(), services=svc)
    task = asyncio.create_task(session.run())
    await asyncio.sleep(0.1)
    await session.hangup("completed")
    await asyncio.wait_for(task, timeout=10)
    await svc.close()
