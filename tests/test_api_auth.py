"""One shared token in front of the control plane and the media plane.

Empty token: everything open, as the laptop demo needs. Token set: every /api
and /ws request presents it, except the three paths a load balancer or the
Asterisk dialplan hits without credentials, and the console pages, which are
harmless without data. Production never runs open: it refuses to boot without
a token, refuses a save that would clear it, and fails closed if it is cleared
anyway.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import uuid
from typing import Any

import httpx
import pytest

from vaani.api.auth import check_api_token
from vaani.config import Settings
from vaani.core.logging import configure_logging
from vaani.pipeline.manager import CallManager
from vaani.settings_store import SettingsStore
from vaani.telephony.announce import CallAnnouncements

TOKEN = "s3cret-token"


def _app(services, settings, token, env="dev"):
    from vaani.main import create_app

    app = create_app(settings.model_copy(update={"api_token": token, "env": env}))
    app.state.services = services
    app.state.calls = CallManager(max_concurrent=2)
    app.state.announcements = CallAnnouncements()
    return app


def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _scope(kind: str, path: str, query: str = "", root_path: str = "") -> dict[str, Any]:
    return {
        "type": kind,
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "ws" if kind == "websocket" else "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": root_path,
        "query_string": query.encode(),
        "headers": [],
        "server": ("test", 80),
        "client": ("test", 1),
        "subprotocols": [],
    }


async def _handshake(app, path: str, query: str = "", root_path: str = "") -> list[str]:
    """Drive a WebSocket handshake through the ASGI interface and return the
    message types the app sent. A refused handshake closes without accepting."""
    inbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    await inbox.put({"type": "websocket.connect"})
    sent: list[str] = []

    async def receive() -> dict[str, Any]:
        return await inbox.get()

    async def send(message: dict[str, Any]) -> None:
        sent.append(message["type"])

    await asyncio.wait_for(app(_scope("websocket", path, query, root_path), receive, send), 5)
    return sent


async def _status(app, path: str, root_path: str = "") -> int:
    """An HTTP request through the raw ASGI interface, so the scope can carry
    the root_path a reverse proxy or `--root-path` would set."""
    status = 0

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        nonlocal status
        if message["type"] == "http.response.start":
            status = message["status"]

    await asyncio.wait_for(app(_scope("http", path, root_path=root_path), receive, send), 5)
    return status


# -- the setting ---------------------------------------------------------------


def test_the_token_is_a_secret_that_needs_no_restart():
    meta = Settings.model_fields["api_token"].json_schema_extra
    assert meta["secret"] is True, "the settings API must never echo it back"
    assert meta["restart"] is False, "changing it must not rebuild providers"
    assert Settings().api_token is None


def test_prod_without_a_token_refuses_to_boot():
    with pytest.raises(RuntimeError, match="VAANI_API_TOKEN"):
        check_api_token(Settings(env="prod", api_token=None))


def test_prod_with_a_token_boots_quietly():
    assert check_api_token(Settings(env="prod", api_token=TOKEN)) is None


def test_dev_without_a_token_warns_but_boots():
    warning = check_api_token(Settings(env="dev", api_token=None))
    assert warning is not None and "open" in warning


# -- HTTP ----------------------------------------------------------------------


async def test_open_when_no_token_is_configured(services, settings):
    async with _client(_app(services, settings, None)) as c:
        assert (await c.get("/api/agents")).status_code == 200


async def test_a_missing_token_is_refused(services, settings):
    async with _client(_app(services, settings, TOKEN)) as c:
        r = await c.get("/api/agents")
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"


async def test_a_wrong_token_is_refused(services, settings):
    async with _client(_app(services, settings, TOKEN)) as c:
        r = await c.get("/api/agents", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


async def test_the_bearer_token_admits(services, settings):
    async with _client(_app(services, settings, TOKEN)) as c:
        r = await c.get("/api/agents", headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200


async def test_a_query_token_is_not_accepted_on_http(services, settings):
    """Tokens in URLs land in access logs and browser history. Only the
    WebSocket handshake, which cannot carry a header, gets that allowance."""
    async with _client(_app(services, settings, TOKEN)) as c:
        r = await c.get(f"/api/agents?token={TOKEN}")
    assert r.status_code == 401


async def test_health_ready_and_the_announce_stay_open(services, settings):
    async with _client(_app(services, settings, TOKEN)) as c:
        assert (await c.get("/api/health")).status_code == 200
        assert (await c.get("/api/ready")).status_code == 200
        r = await c.post("/api/telephony/announce", data={"uuid": str(uuid.uuid4())})
        assert r.status_code == 200


async def test_the_console_pages_stay_open(services, settings):
    async with _client(_app(services, settings, TOKEN)) as c:
        assert (await c.get("/")).status_code == 200
        assert (await c.get("/static/index.html")).status_code == 200


async def test_a_root_path_prefix_does_not_bypass_the_guard(services, settings):
    """Behind a proxy or `--root-path /vaani`, the scope path is /vaani/api/...;
    matching the raw path would wave every request through."""
    app = _app(services, settings, TOKEN)
    assert await _status(app, "/vaani/api/agents", root_path="/vaani") == 401
    # The older convention, where path excludes the prefix, must match too.
    assert await _status(app, "/api/agents", root_path="/vaani") == 401
    assert await _handshake(app, "/vaani/ws/monitor", root_path="/vaani") == ["websocket.close"]


# -- production never runs open -----------------------------------------------


async def test_prod_with_an_emptied_token_fails_closed(services, settings):
    """Boot refuses this state, so the only way in is a runtime clear. The
    guard then serves nothing rather than everything; health stays up so the
    operator can see the box is alive."""
    async with _client(_app(services, settings, None, env="prod")) as c:
        assert (await c.get("/api/agents")).status_code == 503
        assert (await c.get("/api/health")).status_code == 200
    assert await _handshake(_app(services, settings, None, env="prod"), "/ws/monitor") == [
        "websocket.close"
    ]


def _prod_store(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"env": "prod", "api_token": TOKEN}), encoding="utf-8")
    return SettingsStore(path)


async def test_the_settings_api_refuses_to_clear_the_token_in_prod(services, settings, tmp_path):
    from vaani.main import create_app

    store = _prod_store(tmp_path)
    app = create_app(store.settings)
    app.state.settings_store = store
    app.state.services = services
    app.state.calls = CallManager(max_concurrent=2)
    auth = {"Authorization": f"Bearer {TOKEN}"}

    async with _client(app) as c:
        r = await c.put("/api/settings", json={"api_token": ""}, headers=auth)
        assert r.status_code == 422, r.text
        r = await c.post("/api/settings/reset", json={"keys": ["api_token"]}, headers=auth)
        assert r.status_code == 422, r.text
        # And the token the operator had is still the one that works.
        assert (await c.get("/api/agents", headers=auth)).status_code == 200
    assert store.settings.api_token == TOKEN


# -- WebSocket -----------------------------------------------------------------


async def test_the_websocket_handshake_needs_the_token(services, settings):
    app = _app(services, settings, TOKEN)

    refused = await _handshake(app, "/ws/monitor")
    assert refused == ["websocket.close"], "must close without ever accepting"

    admitted = await _handshake(app, "/ws/monitor", query=f"token={TOKEN}")
    assert admitted[0] == "websocket.accept"


async def test_a_wrong_websocket_token_is_refused(services, settings):
    app = _app(services, settings, TOKEN)
    assert await _handshake(app, "/ws/monitor", query="token=nope") == ["websocket.close"]


async def test_the_websocket_is_open_without_a_configured_token(services, settings):
    app = _app(services, settings, None)
    assert (await _handshake(app, "/ws/monitor"))[0] == "websocket.accept"


def test_the_log_stream_redacts_the_websocket_token():
    """uvicorn writes the handshake line, query string included, on
    uvicorn.error rather than uvicorn.access. Whichever logger carries it, the
    token must not reach the stream."""
    configure_logging("INFO")
    stream = io.StringIO()
    logging.getLogger().handlers[0].setStream(stream)

    logging.getLogger("uvicorn.error").info(
        '%s - "WebSocket %s" [accepted]', "10.0.0.1:1", f"/ws/call?agent=default&token={TOKEN}"
    )
    logging.getLogger("uvicorn.access").info(
        '%s - "%s %s HTTP/%s" %d', "10.0.0.1:1", "GET", f"/api/x?api_token={TOKEN}", "1.1", 401
    )

    out = stream.getvalue()
    assert TOKEN not in out
    assert "token=***" in out, "the rest of the line survives"


def test_the_log_stream_redacts_a_secret_named_anything_a_carrier_might_use():
    """A credential is redacted by what it is, not by the name it arrives under.

    Smartflo's handshake is documented as `?key=`, but the name is set in their
    dashboard and a mistyped or differently-named parameter still carries the
    real webhook secret — which is the HMAC key that mints per-call tokens. One
    such request wrote it to the container log in full.
    """
    configure_logging("INFO")
    stream = io.StringIO()
    logging.getLogger().handlers[0].setStream(stream)

    for name in ("secret", "webhook_secret", "apikey", "x-api-key", "password", "sig"):
        logging.getLogger("uvicorn.access").info(
            '%s - "%s %s HTTP/%s" %d', "10.0.0.1:1", "POST", f"/api/x?{name}={TOKEN}", "1.1", 401
        )

    out = stream.getvalue()
    assert TOKEN not in out, "a credential reached the log under one of these names"
    assert out.count("***") == 6, "each line keeps its shape, with the value replaced"
