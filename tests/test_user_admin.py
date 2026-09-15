"""Managing accounts from the portal.

The danger here is not reading somebody else's user list. It is that an
organisation administrator, who is trusted inside their own organisation, could
use these endpoints to award themselves a role or an organisation they were
never given — so most of this file is about privilege escalation rather than
about creating users.
"""

from __future__ import annotations

import httpx
import pytest

from vaani.db.accounts import AccountRepository, Role
from vaani.db.analytics import AnalyticsRepository
from vaani.db.repository import CallRepository
from vaani.db.tenancy import TenancyRepository
from vaani.main import create_app
from vaani.pipeline.manager import CallManager
from vaani.settings_store import SettingsStore
from vaani.telephony.announce import CallAnnouncements

TOKEN = "s3cret-token"
PASSWORD = "a-long-enough-password"


@pytest.fixture
async def world(services, settings, tmp_path):
    app = create_app(settings.model_copy(update={"api_token": TOKEN, "env": "dev"}))
    repository = CallRepository(f"sqlite+aiosqlite:///{tmp_path / 'users.db'}")
    await repository.start()
    services.calls = repository
    services.tenancy = TenancyRepository(repository.sessions)
    services.analytics = AnalyticsRepository(repository.sessions)
    services.accounts = AccountRepository(repository.sessions)
    app.state.services = services
    app.state.calls = CallManager(max_concurrent=5)
    app.state.announcements = CallAnnouncements()
    store = SettingsStore(path=tmp_path / "settings.json")
    store.update({"api_token": TOKEN})
    app.state.settings_store = store

    alpha = await services.tenancy.create_organisation(slug="alpha", name="Alpha")
    beta = await services.tenancy.create_organisation(slug="beta", name="Beta")
    await services.accounts.create_user(
        email="root@euphoria.example",
        password=PASSWORD,
        role=Role.PLATFORM_ADMIN,
        organisation_id=None,
    )
    await services.accounts.create_user(
        email="admin@alpha.example",
        password=PASSWORD,
        role=Role.ORG_ADMIN,
        organisation_id=alpha.id,
    )
    await services.accounts.create_user(
        email="admin@beta.example",
        password=PASSWORD,
        role=Role.ORG_ADMIN,
        organisation_id=beta.id,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client, alpha, beta
    await repository.close()


async def _as(client, email: str) -> None:
    r = await client.post("/api/auth/login", json={"email": email, "password": PASSWORD})
    assert r.status_code == 200, r.text


# -- the ordinary job ------------------------------------------------------


async def test_an_org_admin_adds_a_supervisor_to_their_own_organisation(world):
    client, alpha, _beta = world
    await _as(client, "admin@alpha.example")

    r = await client.post(
        "/api/users",
        json={
            "email": "new@alpha.example",
            "password": PASSWORD,
            "name": "New Supervisor",
            "role": Role.SUPERVISOR,
        },
    )
    assert r.status_code == 201, r.text
    assert r.json()["organisation_id"] == alpha.id
    assert "password" not in r.text and "hash" not in r.text


async def test_an_org_admin_sees_only_their_own_organisations_users(world):
    client, alpha, _beta = world
    await _as(client, "admin@alpha.example")

    listed = (await client.get("/api/users")).json()
    assert {u["email"] for u in listed} == {"admin@alpha.example"}
    assert all(u["organisation_id"] == alpha.id for u in listed)


async def test_a_platform_admin_sees_everybody(world):
    client, _alpha, _beta = world
    await _as(client, "root@euphoria.example")
    emails = {u["email"] for u in (await client.get("/api/users")).json()}
    assert {"root@euphoria.example", "admin@alpha.example", "admin@beta.example"} <= emails


# -- privilege escalation --------------------------------------------------


async def test_an_org_admin_cannot_mint_a_platform_administrator(world):
    """The whole deployment, awarded by one line of JSON."""
    client, _alpha, _beta = world
    await _as(client, "admin@alpha.example")

    r = await client.post(
        "/api/users",
        json={"email": "sneaky@alpha.example", "password": PASSWORD, "role": Role.PLATFORM_ADMIN},
    )
    assert r.status_code == 403


async def test_an_org_admin_cannot_put_a_user_in_another_organisation(world):
    """Scope comes from who you are: the body may name an organisation, and for
    an org admin it is ignored rather than honoured."""
    client, alpha, beta = world
    await _as(client, "admin@alpha.example")

    r = await client.post(
        "/api/users",
        json={
            "email": "planted@beta.example",
            "password": PASSWORD,
            "role": Role.SUPERVISOR,
            "organisation_id": beta.id,
        },
    )
    assert r.status_code == 201
    assert r.json()["organisation_id"] == alpha.id, "a user was planted in another organisation"


async def test_an_org_admin_cannot_suspend_another_organisations_user(world):
    client, _alpha, _beta = world
    await _as(client, "root@euphoria.example")
    victim = (await client.get("/api/users")).json()
    victim_id = next(u["id"] for u in victim if u["email"] == "admin@beta.example")

    await client.post("/api/auth/logout")
    await _as(client, "admin@alpha.example")
    r = await client.patch(f"/api/users/{victim_id}", json={"active": False})
    assert r.status_code == 404


async def test_an_org_admin_cannot_reset_another_organisations_password(world):
    """Resetting a password is taking the account."""
    client, _alpha, _beta = world
    await _as(client, "root@euphoria.example")
    users = (await client.get("/api/users")).json()
    victim_id = next(u["id"] for u in users if u["email"] == "admin@beta.example")

    await client.post("/api/auth/logout")
    await _as(client, "admin@alpha.example")
    r = await client.patch(f"/api/users/{victim_id}", json={"password": "a-brand-new-password"})
    assert r.status_code == 404

    # And the original password still works.
    await client.post("/api/auth/logout")
    await _as(client, "admin@beta.example")


async def test_an_org_admin_cannot_promote_themselves(world):
    client, _alpha, _beta = world
    await _as(client, "admin@alpha.example")
    me = (await client.get("/api/auth/me")).json()

    r = await client.patch(f"/api/users/{me['id']}", json={"role": Role.PLATFORM_ADMIN})
    assert r.status_code == 403


async def test_a_supervisor_cannot_reach_user_administration_at_all(world):
    client, alpha, _beta = world
    await _as(client, "root@euphoria.example")
    await client.post(
        "/api/users",
        json={
            "email": "sup@alpha.example",
            "password": PASSWORD,
            "role": Role.SUPERVISOR,
            "organisation_id": alpha.id,
        },
    )
    await client.post("/api/auth/logout")

    await _as(client, "sup@alpha.example")
    assert (await client.get("/api/users")).status_code == 403
    assert (
        await client.post(
            "/api/users",
            json={"email": "x@alpha.example", "password": PASSWORD, "role": Role.VIEWER},
        )
    ).status_code == 403


# -- not locking everyone out ----------------------------------------------


async def test_the_last_platform_administrator_cannot_suspend_themselves(world):
    """There would then be no way back in without shell access to the box."""
    client, _alpha, _beta = world
    await _as(client, "root@euphoria.example")
    me = (await client.get("/api/auth/me")).json()

    r = await client.patch(f"/api/users/{me['id']}", json={"active": False})
    assert r.status_code == 400
    assert "last" in r.json()["detail"].lower()


async def test_a_platform_admin_may_suspend_themselves_once_another_exists(world):
    client, _alpha, _beta = world
    await _as(client, "root@euphoria.example")
    await client.post(
        "/api/users",
        json={
            "email": "second@euphoria.example",
            "password": PASSWORD,
            "role": Role.PLATFORM_ADMIN,
        },
    )
    me = (await client.get("/api/auth/me")).json()
    assert (await client.patch(f"/api/users/{me['id']}", json={"active": False})).status_code == 200


async def test_changing_your_own_password_keeps_you_signed_in_with_the_new_one(world):
    client, _alpha, _beta = world
    await _as(client, "admin@alpha.example")
    me = (await client.get("/api/auth/me")).json()

    r = await client.patch(f"/api/users/{me['id']}", json={"password": "my-brand-new-password"})
    assert r.status_code == 200

    await client.post("/api/auth/logout")
    again = await client.post(
        "/api/auth/login",
        json={"email": "admin@alpha.example", "password": "my-brand-new-password"},
    )
    assert again.status_code == 200
