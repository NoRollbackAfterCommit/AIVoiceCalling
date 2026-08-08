"""A completed call must be readable from the database afterwards."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from vaani.config import Settings
from vaani.core.registry import build_services
from vaani.db.repository import CallRepository
from vaani.pipeline.session import CallSession

from .test_pipeline import FakeTransport, silence, tone


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


async def _run_one_turn(session: CallSession) -> None:
    task = asyncio.create_task(session.run())
    await asyncio.sleep(0.1)
    await session.push_audio(tone(700))
    await session.push_audio(silence(500))
    await asyncio.sleep(0.8)
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
