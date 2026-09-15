"""Signing in, staying signed in, and what the session is allowed to do.

The shared bearer token does not go away — it is how the deploy script, the
health probe and the carrier harness reach the API. What changes is that a
person gets an account instead of the master key.
"""

from __future__ import annotations

import time

import httpx
import pytest

from vaani.db.accounts import AccountRepository, Role
from vaani.db.repository import CallRepository
from vaani.db.tenancy import TenancyRepository
from vaani.main import create_app
from vaani.pipeline.manager import CallManager
from vaani.security.sessions import SessionSigner
from vaani.settings_store import SettingsStore
from vaani.telephony.announce import CallAnnouncements

TOKEN = "s3cret-token"
PASSWORD = "a-long-enough-password"


# -- the signed session ----------------------------------------------------


def test_a_session_round_trips():
    signer = SessionSigner("a-secret", ttl_s=3600)
    assert signer.verify(signer.issue(7)) == 7


def test_a_session_signed_with_another_key_is_refused():
    """Otherwise anyone who can set a cookie can become any user."""
    forged = SessionSigner("attackers-secret", ttl_s=3600).issue(1)
    assert SessionSigner("a-secret", ttl_s=3600).verify(forged) is None


def test_a_tampered_session_is_refused():
    signer = SessionSigner("a-secret", ttl_s=3600)
    token = signer.issue(7)
    _user_id, issued, sig = token.split(".")
    assert signer.verify(f"9.{issued}.{sig}") is None


def test_an_expired_session_is_refused():
    signer = SessionSigner("a-secret", ttl_s=60)
    assert signer.verify(signer.issue(7, issued_at=time.time() - 3600)) is None


@pytest.mark.parametrize("junk", ["", "nonsense", "1.2", "a.b.c", "1.notanumber.sig"])
def test_rubbish_is_refused_rather_than_raising(junk):
    assert SessionSigner("a-secret", ttl_s=60).verify(junk) is None


# -- logging in ------------------------------------------------------------


@pytest.fixture
async def world(services, settings, tmp_path):
    app = create_app(settings.model_copy(update={"api_token": TOKEN, "env": "dev"}))
    repository = CallRepository(f"sqlite+aiosqlite:///{tmp_path / 'login.db'}")
    await repository.start()
    services.calls = repository
    services.tenancy = TenancyRepository(repository.sessions)
    services.accounts = AccountRepository(repository.sessions)
    app.state.services = services
    app.state.calls = CallManager(max_concurrent=5)
    app.state.announcements = CallAnnouncements()
    # The settings endpoint saves through the store; without one it cannot tell
    # a role refusal from a missing dependency. The store is what the guard
    # reads, so the token has to live there too — otherwise the deployment reads
    # as open and every request is waved through.
    store = SettingsStore(path=tmp_path / "settings.json")
    store.update({"api_token": TOKEN})
    app.state.settings_store = store

    org = await services.tenancy.create_organisation(slug="health", name="Health University")
    await services.accounts.create_user(
        email="root@euphoria.example",
        password=PASSWORD,
        role=Role.PLATFORM_ADMIN,
        organisation_id=None,
    )
    await services.accounts.create_user(
        email="sup@health.example",
        password=PASSWORD,
        role=Role.SUPERVISOR,
        organisation_id=org.id,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c, org
    await repository.close()


async def _login(client, email: str) -> httpx.Response:
    return await client.post("/api/auth/login", json={"email": email, "password": PASSWORD})


async def test_the_login_page_is_reachable_without_credentials(world):
    """An operator with no session must land somewhere they can type one."""
    client, _ = world
    r = await client.get("/login")
    assert r.status_code == 200
    assert "password" in r.text.lower()


async def test_signing_in_sets_a_session_and_identifies_the_user(world):
    client, _ = world
    r = await _login(client, "root@euphoria.example")
    assert r.status_code == 200
    assert r.json()["role"] == Role.PLATFORM_ADMIN

    me = await client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json()["email"] == "root@euphoria.example"


async def test_the_session_cookie_is_not_readable_by_scripts(world):
    """An XSS in the console must not be able to lift the session."""
    client, _ = world
    r = await _login(client, "root@euphoria.example")
    raw = r.headers["set-cookie"].lower()
    assert "httponly" in raw
    assert "samesite=lax" in raw or "samesite=strict" in raw


async def test_a_wrong_password_is_refused_without_saying_which_part_was_wrong(world):
    client, _ = world
    r = await client.post(
        "/api/auth/login", json={"email": "root@euphoria.example", "password": "wrong-password"}
    )
    assert r.status_code == 401
    unknown = await client.post(
        "/api/auth/login", json={"email": "nobody@nowhere.example", "password": "wrong-password"}
    )
    assert unknown.status_code == 401
    # Identical wording: a different message tells an attacker which addresses exist.
    assert r.json()["detail"] == unknown.json()["detail"]


async def test_signing_out_ends_the_session(world):
    client, _ = world
    await _login(client, "root@euphoria.example")
    assert (await client.get("/api/auth/me")).status_code == 200
    await client.post("/api/auth/logout")
    assert (await client.get("/api/auth/me")).status_code == 401


async def test_a_session_still_works_when_the_shared_token_is_not_presented(world):
    """The whole point: a person signs in instead of holding the master key."""
    client, _ = world
    await _login(client, "sup@health.example")
    assert (await client.get("/api/calls")).status_code == 200


async def test_without_a_session_or_a_token_the_api_is_refused(world):
    client, _ = world
    assert (await client.get("/api/calls")).status_code == 401


async def test_the_shared_token_still_works_for_machines(world):
    """The deploy script and the health probe have no browser to log in with."""
    client, _ = world
    r = await client.get("/api/calls", headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200


async def test_a_suspended_user_loses_access_on_their_next_request(world):
    """Revoking access must not wait for a cookie to expire."""
    client, _ = world
    await _login(client, "sup@health.example")
    assert (await client.get("/api/auth/me")).status_code == 200

    accounts = None
    for user in await _accounts(client).users():
        if user.email == "sup@health.example":
            accounts = user
    await _accounts(client).set_active(accounts.id, False)

    assert (await client.get("/api/auth/me")).status_code == 401


def _accounts(client) -> AccountRepository:
    return client._transport.app.state.services.accounts


# -- what each role may do -------------------------------------------------


async def test_a_supervisor_cannot_change_deployment_settings(world):
    """Settings carry provider keys and the concurrency cap; they belong to the
    platform administrator, not to a customer's supervisor."""
    client, _ = world
    await _login(client, "sup@health.example")
    r = await client.put("/api/settings", json={"barge_in_ms": 500})
    assert r.status_code == 403


async def test_a_platform_admin_may_change_deployment_settings(world):
    client, _ = world
    await _login(client, "root@euphoria.example")
    r = await client.put("/api/settings", json={"barge_in_ms": 500})
    assert r.status_code == 200


async def test_a_supervisor_cannot_create_organisations(world):
    client, _ = world
    await _login(client, "sup@health.example")
    r = await client.post("/api/organisations", json={"slug": "sneaky", "name": "Sneaky"})
    assert r.status_code == 403


async def test_a_supervisor_may_still_watch_and_end_calls(world):
    """The job description: see the lines, hang up one that has gone wrong."""
    client, _ = world
    await _login(client, "sup@health.example")
    assert (await client.get("/api/calls/live")).status_code == 200
    # No such call, but the point is that it is 404 rather than 403.
    assert (await client.post("/api/calls/nope/hangup")).status_code == 404
