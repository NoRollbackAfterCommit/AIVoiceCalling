"""Which organisation's agent answers, decided by the number dialled.

This is the whole point of the tenancy model from a caller's side. Three numbers
reach a health university's agent, two reach a state portal's, and the call
record says which — because reporting is per organisation and per DID, and a
call attributed to the wrong one is worse than a call attributed to nobody.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from vaani.config import Settings
from vaani.core.registry import build_services
from vaani.db.repository import CallRepository
from vaani.db.tenancy import TenancyRepository
from vaani.pipeline.session import CallSession

from .test_pipeline import FakeTransport


def _settings() -> Settings:
    return Settings(
        stt_provider="mock",
        llm_provider="mock",
        tts_provider="mock",
        vector_store="memory",
        embedding_provider="hash",
        record_calls=False,
        smartflo_agent="fallback-agent",
    )


@pytest.fixture
async def world(tmp_path):
    """Two organisations, five numbers, laid out as the worked example."""
    svc = build_services(_settings())
    svc.calls = CallRepository(f"sqlite+aiosqlite:///{tmp_path / 'routing.db'}")
    await svc.calls.start()
    svc.tenancy = TenancyRepository(svc.calls.sessions)
    await svc.start()

    base = svc.profiles["default"]
    for key in ("health-admissions", "health-exams", "wb-general", "fallback-agent"):
        svc.profiles[key] = replace(base, key=key, ask_language=False)

    health = await svc.tenancy.create_organisation(slug="health", name="Health University")
    wb = await svc.tenancy.create_organisation(slug="wb", name="WB Centralised Portal")
    for number, agent in [
        ("+918065605871", "health-admissions"),
        ("+918065605872", "health-admissions"),
        ("+918065605873", "health-exams"),
    ]:
        await svc.tenancy.upsert_did(number=number, organisation_id=health.id, agent_key=agent)
    for number in ("+918065605874", "+918065605875"):
        await svc.tenancy.upsert_did(number=number, organisation_id=wb.id, agent_key="wb-general")

    yield svc, health, wb
    await svc.close()
    await svc.calls.close()


async def _route(svc, called: str):
    """What the telephony route does before building a session."""
    from vaani.telephony.routing import route_call

    return await route_call(svc, called)


async def test_each_number_reaches_its_own_organisations_agent(world):
    svc, health, wb = world

    assert await _route(svc, "+918065605871") == (health.id, "health-admissions")
    assert await _route(svc, "+918065605872") == (health.id, "health-admissions")
    assert await _route(svc, "+918065605873") == (health.id, "health-exams")
    assert await _route(svc, "+918065605874") == (wb.id, "wb-general")
    assert await _route(svc, "+918065605875") == (wb.id, "wb-general")


async def test_the_carriers_own_spelling_of_the_number_routes(world):
    """Tata sends the dialled number with a leading space and no plus."""
    svc, health, _ = world
    assert await _route(svc, " 918065605873") == (health.id, "health-exams")


async def test_an_unmapped_number_is_answered_not_dropped(world):
    """A configuration gap must never cost a real caller their call. They get
    the configured fallback agent, and the call is recorded unattributed so it
    shows up in reporting as needing a mapping."""
    svc, _, _ = world
    assert await _route(svc, "+919999999999") == (None, "fallback-agent")


async def test_a_call_with_no_dialled_number_still_gets_an_agent(world):
    svc, _, _ = world
    assert await _route(svc, None) == (None, "fallback-agent")


async def test_routing_survives_the_tenancy_store_being_absent():
    """A bare install has no database at all and must still answer."""
    svc = build_services(_settings())
    await svc.start()
    try:
        from vaani.telephony.routing import route_call

        assert await route_call(svc, "+918065605873") == (None, "fallback-agent")
    finally:
        await svc.close()


async def test_the_call_record_carries_the_organisation_and_the_number(world):
    """Reporting is per organisation and per DID, so both are stamped on the row
    when the call starts — not when it ends, or a crash mid-call loses the
    attribution along with everything else."""
    svc, health, _ = world
    session = CallSession(
        transport=FakeTransport(),
        services=svc,
        agent_key="health-exams",
        caller_number="+919876543210",
        organisation_id=health.id,
        did="+918065605873",
    )
    assert session.record.organisation_id == health.id
    assert session.record.did == "+918065605873"

    await svc.calls.create_call(session.record)
    stored = await svc.calls.get_call(session.call_id)
    assert stored["organisation_id"] == health.id
    assert stored["did"] == "+918065605873"


async def test_an_unattributed_call_is_stored_as_such(world):
    svc, _, _ = world
    session = CallSession(transport=FakeTransport(), services=svc, agent_key="fallback-agent")
    await svc.calls.create_call(session.record)
    stored = await svc.calls.get_call(session.call_id)
    assert stored["organisation_id"] is None
