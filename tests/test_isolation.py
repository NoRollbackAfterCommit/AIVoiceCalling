"""One organisation must not be able to reach another's anything.

This is the security surface of the whole tenancy model, so the tests are
written as an attacker rather than as a user: every one of them is somebody
signed in to Alpha trying to reach Beta's data, by direct id, by query
parameter, and by crafted filter. A supervisor being *shown* only their own
calls is not the property under test — the property is that asking for
somebody else's does not work.
"""

from __future__ import annotations

from dataclasses import replace

import httpx
import pytest

from vaani.db.accounts import AccountRepository, Role
from vaani.db.analytics import AnalyticsRepository
from vaani.db.repository import CallRepository
from vaani.db.tenancy import TenancyRepository
from vaani.main import create_app
from vaani.pipeline.manager import CallManager
from vaani.pipeline.session import CallSession
from vaani.settings_store import SettingsStore
from vaani.telephony.announce import CallAnnouncements

from .test_pipeline import FakeTransport

TOKEN = "s3cret-token"
PASSWORD = "a-long-enough-password"


@pytest.fixture
async def world(services, settings, tmp_path):
    """Two organisations, a call and an agent each, and a supervisor in Alpha."""
    app = create_app(settings.model_copy(update={"api_token": TOKEN, "env": "dev"}))
    repository = CallRepository(f"sqlite+aiosqlite:///{tmp_path / 'isolation.db'}")
    await repository.start()
    services.calls = repository
    services.tenancy = TenancyRepository(repository.sessions)
    services.analytics = AnalyticsRepository(repository.sessions)
    services.accounts = AccountRepository(repository.sessions)
    app.state.services = services
    app.state.calls = CallManager(max_concurrent=10)
    app.state.announcements = CallAnnouncements()
    store = SettingsStore(path=tmp_path / "settings.json")
    store.update({"api_token": TOKEN})
    app.state.settings_store = store

    alpha = await services.tenancy.create_organisation(slug="alpha", name="Alpha")
    beta = await services.tenancy.create_organisation(slug="beta", name="Beta")

    base = services.profiles["default"]
    services.profiles["alpha-agent"] = replace(
        base, key="alpha-agent", organisation_id=alpha.id, ask_language=False
    )
    services.profiles["beta-agent"] = replace(
        base, key="beta-agent", organisation_id=beta.id, ask_language=False
    )

    calls = {}
    for org, agent in [(alpha, "alpha-agent"), (beta, "beta-agent")]:
        session = CallSession(
            transport=FakeTransport(),
            services=services,
            agent_key=agent,
            caller_number=f"+9198765{org.id:05d}",
            organisation_id=org.id,
            did=f"+9180656058{70 + org.id}",
        )
        await repository.create_call(session.record)
        calls[org.slug] = session.call_id

    await services.accounts.create_user(
        email="sup@alpha.example",
        password=PASSWORD,
        role=Role.SUPERVISOR,
        organisation_id=alpha.id,
    )
    await services.accounts.create_user(
        email="admin@alpha.example",
        password=PASSWORD,
        role=Role.ORG_ADMIN,
        organisation_id=alpha.id,
    )
    await services.accounts.create_user(
        email="root@euphoria.example",
        password=PASSWORD,
        role=Role.PLATFORM_ADMIN,
        organisation_id=None,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client, alpha, beta, calls
    await repository.close()


async def _as(client, email: str) -> None:
    r = await client.post("/api/auth/login", json={"email": email, "password": PASSWORD})
    assert r.status_code == 200, r.text


# -- calls -----------------------------------------------------------------


async def test_a_supervisor_sees_only_their_own_organisations_calls(world):
    client, _alpha, _beta, calls = world
    await _as(client, "sup@alpha.example")

    listed = (await client.get("/api/calls")).json()
    ids = {row["call_id"] for row in listed}
    assert calls["alpha"] in ids
    assert calls["beta"] not in ids, "another organisation's call appeared in the list"


async def test_asking_for_another_organisations_call_by_id_is_refused(world):
    """The list is filtered, so the obvious attack is to skip the list."""
    client, _alpha, _beta, calls = world
    await _as(client, "sup@alpha.example")

    r = await client.get(f"/api/calls/{calls['beta']}")
    assert r.status_code == 404, "a direct id reached another organisation's call"


async def test_passing_someone_elses_organisation_as_a_parameter_is_ignored(world):
    """Scope comes from who you are, never from what you asked for."""
    client, _alpha, beta, calls = world
    await _as(client, "sup@alpha.example")

    listed = (await client.get(f"/api/calls?organisation_id={beta.id}")).json()
    assert calls["beta"] not in {row["call_id"] for row in listed}


async def test_hanging_up_another_organisations_live_call_is_refused(world):
    """Ending a stranger's call is worse than reading about it."""
    client, _alpha, beta, _calls = world
    manager = client._transport.app.state.calls
    services = client._transport.app.state.services
    victim = CallSession(
        transport=FakeTransport(),
        services=services,
        agent_key="beta-agent",
        organisation_id=beta.id,
        did="+918065605872",
    )
    await manager.register(victim)

    await _as(client, "sup@alpha.example")
    r = await client.post(f"/api/calls/{victim.call_id}/hangup")
    assert r.status_code == 404
    assert manager.get(victim.call_id) is not None, "the call was ended anyway"


async def test_the_live_roster_only_shows_your_own_organisation(world):
    client, alpha, beta, _calls = world
    manager = client._transport.app.state.calls
    services = client._transport.app.state.services
    for org, agent in [(alpha, "alpha-agent"), (beta, "beta-agent")]:
        await manager.register(
            CallSession(
                transport=FakeTransport(),
                services=services,
                agent_key=agent,
                organisation_id=org.id,
                did=f"+91806560587{org.id}",
            )
        )

    await _as(client, "sup@alpha.example")
    rows = (await client.get("/api/calls/live")).json()
    assert rows, "the supervisor's own live call vanished too"
    assert {r["organisation_id"] for r in rows} == {alpha.id}


# -- agents and knowledge --------------------------------------------------


async def test_another_organisations_agent_is_neither_listed_nor_readable(world):
    client, _alpha, _beta, _calls = world
    await _as(client, "admin@alpha.example")

    keys = {a["key"] for a in (await client.get("/api/agents")).json()}
    assert "alpha-agent" in keys
    assert "beta-agent" not in keys

    assert (await client.get("/api/agents/beta-agent")).status_code == 404


async def test_another_organisations_agent_cannot_be_edited_or_deleted(world):
    """Overwriting a competitor's prompt is the most damaging thing here."""
    client, _alpha, _beta, _calls = world
    await _as(client, "admin@alpha.example")

    r = await client.put("/api/agents/beta-agent", json={"key": "beta-agent", "name": "Hijacked"})
    assert r.status_code == 404
    assert (await client.delete("/api/agents/beta-agent")).status_code == 404

    services = client._transport.app.state.services
    assert services.profiles["beta-agent"].name != "Hijacked"


async def test_knowledge_cannot_be_written_into_another_organisations_agent(world):
    """The corpus is what the bot says out loud. Writing into someone else's is
    putting words in their mouth."""
    client, _alpha, _beta, _calls = world
    await _as(client, "admin@alpha.example")

    r = await client.post(
        "/api/knowledge/text",
        json={"text": "Beta's fees are zero.", "source": "forged", "agent_key": "beta-agent"},
    )
    assert r.status_code == 404


async def test_knowledge_cannot_be_searched_in_another_organisations_agent(world):
    client, _alpha, _beta, _calls = world
    await _as(client, "admin@alpha.example")
    r = await client.get("/api/knowledge/search?q=fees&agent_key=beta-agent")
    assert r.status_code == 404


# -- tenancy administration ------------------------------------------------


async def test_an_org_admin_sees_only_their_own_organisation_and_numbers(world):
    client, alpha, _beta, _calls = world
    await _as(client, "admin@alpha.example")

    orgs = (await client.get("/api/organisations")).json()
    assert {o["id"] for o in orgs} == {alpha.id}


# -- the platform administrator --------------------------------------------


async def test_a_platform_admin_still_sees_everything(world):
    """The scoping must not be so enthusiastic that nobody can run the platform."""
    client, _alpha, _beta, calls = world
    await _as(client, "root@euphoria.example")

    ids = {row["call_id"] for row in (await client.get("/api/calls")).json()}
    assert {calls["alpha"], calls["beta"]} <= ids
    assert (await client.get(f"/api/calls/{calls['beta']}")).status_code == 200
    assert len((await client.get("/api/organisations")).json()) >= 2


async def test_the_shared_token_still_reaches_everything(world):
    """The deploy script and the carrier harness have no organisation."""
    client, _alpha, _beta, calls = world
    r = await client.get(
        f"/api/calls/{calls['beta']}", headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert r.status_code == 200


# -- reporting -------------------------------------------------------------


async def test_the_summary_counts_only_your_own_organisations_calls(world):
    client, _alpha, _beta, _calls = world
    await _as(client, "sup@alpha.example")

    mine = (await client.get("/api/analytics/summary")).json()
    assert mine["calls"] == 1, "another organisation's calls were counted"

    await client.post("/api/auth/logout")
    await _as(client, "root@euphoria.example")
    assert (await client.get("/api/analytics/summary")).json()["calls"] == 2


async def test_naming_another_organisation_in_the_summary_is_ignored(world):
    client, _alpha, beta, _calls = world
    await _as(client, "sup@alpha.example")
    r = await client.get(f"/api/analytics/summary?organisation_id={beta.id}")
    assert r.json()["calls"] == 1


async def test_capacity_is_a_platform_figure_not_a_customers(world):
    """How busy the box is says nothing useful to one customer, and says
    something about the others."""
    client, _alpha, _beta, _calls = world
    await _as(client, "sup@alpha.example")
    assert "capacity" not in (await client.get("/api/analytics/summary")).json()

    await client.post("/api/auth/logout")
    await _as(client, "root@euphoria.example")
    assert "capacity" in (await client.get("/api/analytics/summary")).json()


async def test_the_csv_export_cannot_walk_out_with_another_organisations_calls(world):
    client, _alpha, beta, calls = world
    await _as(client, "sup@alpha.example")

    r = await client.get(f"/api/analytics/export?organisation_id={beta.id}")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert calls["alpha"] in r.text
    assert calls["beta"] not in r.text, "an export leaked another organisation's calls"


async def test_the_export_neutralises_spreadsheet_formulas(world):
    """A caller number starting with + is read as a formula by Excel."""
    from vaani.api.routes import _csv_safe

    assert _csv_safe("+918065605873").startswith("'")
    assert _csv_safe("=cmd|'/c calc'!A0").startswith("'")
    assert _csv_safe("ordinary") == "ordinary"
    assert _csv_safe(42) == 42


# -- who may put an agent into an organisation -------------------------------


async def test_a_platform_admin_can_put_a_new_agent_into_an_organisation(world):
    """Somebody has to. A platform administrator sets a customer up, and until
    they could name the organisation, every agent they made belonged to none —
    which meant it could never read that customer's shared training set."""
    client, alpha, _beta, _calls = world
    await _as(client, "root@euphoria.example")

    created = await client.put(
        "/api/agents/new-line",
        params={"organisation_id": alpha.id},
        json={"key": "new-line"},
    )
    assert created.status_code == 200, created.text
    assert (await client.get("/api/agents/new-line")).json()["organisation_id"] == alpha.id


async def test_a_platform_admin_can_move_an_agent_that_belongs_to_nobody(world):
    """The agent a deployment starts with has no organisation. Without a way to
    adopt it, the first customer on an existing box could never use a shared
    set for the line they were already running."""
    client, alpha, _beta, _calls = world
    await _as(client, "root@euphoria.example")

    assert (await client.get("/api/agents/default")).json()["organisation_id"] is None
    moved = await client.put(
        "/api/agents/default", params={"organisation_id": alpha.id}, json={"key": "default"}
    )
    assert moved.status_code == 200, moved.text
    assert (await client.get("/api/agents/default")).json()["organisation_id"] == alpha.id


async def test_an_org_admin_naming_another_organisation_is_ignored(world):
    """Scope comes from who you are, never from what you asked for. The
    parameter is how a platform admin narrows; it must not let a customer
    hand their agent to somebody else, or take one."""
    client, alpha, beta, _calls = world
    await _as(client, "admin@alpha.example")

    created = await client.put(
        "/api/agents/alpha-new", params={"organisation_id": beta.id}, json={"key": "alpha-new"}
    )
    assert created.status_code == 200, created.text
    assert (await client.get("/api/agents/alpha-new")).json()["organisation_id"] == alpha.id


async def test_an_org_admin_cannot_move_their_agent_out_of_their_organisation(world):
    client, alpha, beta, _calls = world
    await _as(client, "admin@alpha.example")

    await client.put(
        "/api/agents/alpha-agent", params={"organisation_id": beta.id}, json={"key": "alpha-agent"}
    )
    assert (await client.get("/api/agents/alpha-agent")).json()["organisation_id"] == alpha.id


async def test_an_agent_keeps_its_organisation_when_none_is_named(world):
    """An ordinary edit — changing a greeting — must not quietly re-home it."""
    client, alpha, _beta, _calls = world
    await _as(client, "root@euphoria.example")

    await client.put("/api/agents/alpha-agent", json={"key": "alpha-agent", "greeting": "Hello."})
    assert (await client.get("/api/agents/alpha-agent")).json()["organisation_id"] == alpha.id
