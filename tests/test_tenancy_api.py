"""Configuring organisations and their numbers from the portal.

The point of these endpoints is that an operator can stand up a second call
centre without shell access: create the organisation, attach its numbers, point
each at an agent. Everything here is administration, not the call path.
"""

from __future__ import annotations

import httpx
import pytest

from vaani.db.repository import CallRepository
from vaani.db.tenancy import TenancyRepository
from vaani.main import create_app
from vaani.pipeline.manager import CallManager
from vaani.telephony.announce import CallAnnouncements


@pytest.fixture
async def client(services, settings, tmp_path):
    app = create_app(settings.model_copy(update={"api_token": "", "env": "dev"}))
    repository = CallRepository(f"sqlite+aiosqlite:///{tmp_path / 'api.db'}")
    await repository.start()
    services.calls = repository
    services.tenancy = TenancyRepository(repository.sessions)
    app.state.services = services
    app.state.calls = CallManager(max_concurrent=5)
    app.state.announcements = CallAnnouncements()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    await repository.close()


async def test_an_operator_can_stand_up_a_second_call_centre(client):
    """The worked example, done entirely through the API."""
    health = (
        await client.post(
            "/api/organisations", json={"slug": "health", "name": "Health University"}
        )
    ).json()
    wb = (await client.post("/api/organisations", json={"slug": "wb", "name": "WB Portal"})).json()

    for number, agent in [
        ("+918065605871", "health-admissions"),
        ("+918065605872", "health-admissions"),
        ("+918065605873", "health-exams"),
    ]:
        r = await client.put(
            f"/api/dids/{number}",
            json={"organisation_id": health["id"], "agent_key": agent},
        )
        assert r.status_code == 200, r.text
    for number in ("+918065605874", "+918065605875"):
        await client.put(
            f"/api/dids/{number}",
            json={"organisation_id": wb["id"], "agent_key": "wb-general"},
        )

    # "default" is seeded by the migration so an upgraded deployment's existing
    # agents and calls have an owner; the two new ones sit alongside it.
    listed = (await client.get("/api/organisations")).json()
    assert {"health", "wb"} <= {o["slug"] for o in listed}

    health_numbers = (await client.get(f"/api/dids?organisation_id={health['id']}")).json()
    assert len(health_numbers) == 3
    assert {d["agent_key"] for d in health_numbers} == {"health-admissions", "health-exams"}
    assert len((await client.get(f"/api/dids?organisation_id={wb['id']}")).json()) == 2


async def test_a_number_is_stored_canonically_however_it_was_typed(client):
    """An operator pastes whatever their carrier portal shows them, and the
    carrier sends something different again. Both must be the same row."""
    org = (await client.post("/api/organisations", json={"slug": "o", "name": "O"})).json()
    r = await client.put(
        "/api/dids/08065605873", json={"organisation_id": org["id"], "agent_key": "a"}
    )
    assert r.status_code == 200
    assert r.json()["number"] == "+918065605873"


async def test_an_unusable_number_is_refused_with_a_reason(client):
    org = (await client.post("/api/organisations", json={"slug": "o", "name": "O"})).json()
    r = await client.put(
        "/api/dids/not-a-number", json={"organisation_id": org["id"], "agent_key": "a"}
    )
    assert r.status_code == 400
    assert "number" in r.json()["detail"].lower()


async def test_a_number_cannot_be_attached_to_an_organisation_that_does_not_exist(client):
    """Otherwise the DID resolves to an organisation nobody can administer."""
    r = await client.put("/api/dids/+918065605873", json={"organisation_id": 999, "agent_key": "a"})
    assert r.status_code == 404


async def test_a_duplicate_slug_is_refused_not_silently_merged(client):
    await client.post("/api/organisations", json={"slug": "health", "name": "Health"})
    r = await client.post("/api/organisations", json={"slug": "health", "name": "Other"})
    assert r.status_code == 409


async def test_an_organisation_can_be_suspended_and_restored(client):
    org = (await client.post("/api/organisations", json={"slug": "o", "name": "O"})).json()
    await client.put(
        "/api/dids/+918065605873", json={"organisation_id": org["id"], "agent_key": "a"}
    )

    async def active_flag() -> bool:
        rows = (await client.get("/api/organisations")).json()
        return next(r["active"] for r in rows if r["id"] == org["id"])

    await client.patch(f"/api/organisations/{org['id']}", json={"active": False})
    assert await active_flag() is False

    await client.patch(f"/api/organisations/{org['id']}", json={"active": True})
    assert await active_flag() is True


async def test_releasing_a_number_leaves_the_calls_it_took(client):
    """Deleting the mapping must not be how call history is destroyed."""
    org = (await client.post("/api/organisations", json={"slug": "o", "name": "O"})).json()
    await client.put(
        "/api/dids/+918065605873", json={"organisation_id": org["id"], "agent_key": "a"}
    )
    assert (await client.delete("/api/dids/+918065605873")).status_code == 200
    assert (await client.get(f"/api/dids?organisation_id={org['id']}")).json() == []


async def test_the_did_list_says_which_organisation_each_number_belongs_to(client):
    """The page shows every number across every organisation, so each row has to
    name its owner without a second request per row."""
    a = (await client.post("/api/organisations", json={"slug": "a", "name": "Alpha"})).json()
    b = (await client.post("/api/organisations", json={"slug": "b", "name": "Beta"})).json()
    await client.put("/api/dids/+918065605871", json={"organisation_id": a["id"], "agent_key": "x"})
    await client.put("/api/dids/+918065605872", json={"organisation_id": b["id"], "agent_key": "y"})

    rows = (await client.get("/api/dids")).json()
    assert {r["organisation_name"] for r in rows} == {"Alpha", "Beta"}
