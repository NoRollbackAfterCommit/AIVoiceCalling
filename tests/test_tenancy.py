"""Organisations, their DIDs, and which agent answers a given number.

One deployment runs several organisations' call centres. Which one a caller
reaches is decided entirely by the number they dialled, so this lookup is the
hinge the whole tenancy model turns on: get it wrong and a citizen asking a
health university about admissions is answered by a different government
portal's agent, trained on the wrong corpus.
"""

from __future__ import annotations

import pytest

from vaani.db.repository import CallRepository
from vaani.db.tenancy import TenancyRepository


@pytest.fixture
async def repos(tmp_path):
    calls = CallRepository(f"sqlite+aiosqlite:///{tmp_path / 'tenancy.db'}")
    await calls.start()
    yield calls, TenancyRepository(calls.sessions)
    await calls.close()


@pytest.fixture
async def tenancy(repos):
    return repos[1]


async def test_a_did_resolves_to_its_organisation_and_agent(tenancy):
    health = await tenancy.create_organisation(slug="health-university", name="Health University")
    await tenancy.upsert_did(
        number="+918065605873", organisation_id=health.id, agent_key="health-admissions"
    )

    found = await tenancy.resolve_did("+918065605873")
    assert found == (health.id, "health-admissions")


async def test_the_number_is_matched_however_the_carrier_wrote_it(tenancy):
    """Tata's live handshake sent a leading space and no plus."""
    org = await tenancy.create_organisation(slug="wb-portal", name="WB Centralised Portal")
    await tenancy.upsert_did(number="+918065605873", organisation_id=org.id, agent_key="wb-general")

    for spelling in ("918065605873", " 918065605873", "+91 80 6560 5873", "08065605873"):
        assert await tenancy.resolve_did(spelling) == (org.id, "wb-general"), spelling


async def test_a_number_stored_untidily_is_still_found(tenancy):
    """An operator pastes whatever their carrier portal shows them."""
    org = await tenancy.create_organisation(slug="o", name="O")
    await tenancy.upsert_did(number=" 91 80-6560 5873 ", organisation_id=org.id, agent_key="a")
    assert await tenancy.resolve_did("+918065605873") == (org.id, "a")


async def test_an_unknown_number_resolves_to_nothing_rather_than_a_guess(tenancy):
    org = await tenancy.create_organisation(slug="o", name="O")
    await tenancy.upsert_did(number="+918065605873", organisation_id=org.id, agent_key="a")
    assert await tenancy.resolve_did("+919999999999") is None
    assert await tenancy.resolve_did("") is None
    assert await tenancy.resolve_did(None) is None


async def test_a_deactivated_did_stops_answering(tenancy):
    """Taking a number out of service must not need the row deleted: the calls
    already recorded against it still have to be attributable."""
    org = await tenancy.create_organisation(slug="o", name="O")
    await tenancy.upsert_did(number="+918065605873", organisation_id=org.id, agent_key="a")
    await tenancy.upsert_did(
        number="+918065605873", organisation_id=org.id, agent_key="a", active=False
    )
    assert await tenancy.resolve_did("+918065605873") is None


async def test_a_deactivated_organisation_takes_its_numbers_with_it(tenancy):
    org = await tenancy.create_organisation(slug="o", name="O")
    await tenancy.upsert_did(number="+918065605873", organisation_id=org.id, agent_key="a")
    await tenancy.set_organisation_active(org.id, False)
    assert await tenancy.resolve_did("+918065605873") is None


async def test_moving_a_did_to_another_organisation_replaces_the_mapping(tenancy):
    """A number belongs to one organisation at a time. Two live mappings would
    make which agent answers depend on row order."""
    a = await tenancy.create_organisation(slug="a", name="A")
    b = await tenancy.create_organisation(slug="b", name="B")
    await tenancy.upsert_did(number="+918065605873", organisation_id=a.id, agent_key="a-agent")
    await tenancy.upsert_did(number="+918065605873", organisation_id=b.id, agent_key="b-agent")

    assert await tenancy.resolve_did("+918065605873") == (b.id, "b-agent")
    assert len(await tenancy.dids_for(a.id)) == 0
    assert len(await tenancy.dids_for(b.id)) == 1


async def test_one_organisation_can_run_several_lines(tenancy):
    """The worked example: three numbers for a health university, two for a
    state portal, and the university's two lines answered by different agents."""
    health = await tenancy.create_organisation(slug="health", name="Health University")
    wb = await tenancy.create_organisation(slug="wb", name="WB Portal")
    for number, agent in [
        ("+918065605871", "health-admissions"),
        ("+918065605872", "health-admissions"),
        ("+918065605873", "health-exams"),
    ]:
        await tenancy.upsert_did(number=number, organisation_id=health.id, agent_key=agent)
    for number in ("+918065605874", "+918065605875"):
        await tenancy.upsert_did(number=number, organisation_id=wb.id, agent_key="wb-general")

    assert await tenancy.resolve_did("+918065605872") == (health.id, "health-admissions")
    assert await tenancy.resolve_did("+918065605873") == (health.id, "health-exams")
    assert await tenancy.resolve_did("+918065605874") == (wb.id, "wb-general")
    assert len(await tenancy.dids_for(health.id)) == 3
    assert len(await tenancy.dids_for(wb.id)) == 2


async def test_a_slug_cannot_be_reused(tenancy):
    """The slug is how an organisation is named in URLs and exports."""
    await tenancy.create_organisation(slug="health", name="Health University")
    with pytest.raises(ValueError, match="health"):
        await tenancy.create_organisation(slug="health", name="Something Else")


async def test_organisations_survive_a_restart(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'restart.db'}"
    first = CallRepository(url)
    await first.start()
    org = await TenancyRepository(first.sessions).create_organisation(slug="health", name="H")
    await TenancyRepository(first.sessions).upsert_did(
        number="+918065605873", organisation_id=org.id, agent_key="a"
    )
    await first.close()

    second = CallRepository(url)
    await second.start()
    try:
        assert await TenancyRepository(second.sessions).resolve_did("+918065605873") == (
            org.id,
            "a",
        )
    finally:
        await second.close()
