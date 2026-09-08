"""One shared token in front of /api and /ws.

Per-user accounts and roles are a phase-6 concern. What a pilot needs today is
that a VM with a public IP for its SIP trunk does not also expose a control
plane that can change providers, upload knowledge, edit profiles and hang up
calls. A single bearer token closes that, and an empty one keeps the laptop
demo zero-config.

HTTP presents the token as `Authorization: Bearer`. A browser cannot put a
header on a WebSocket handshake, so that one path may carry it as `?token=`;
HTTP may not, because a token in a URL lands in access logs and history (the
access log redacts the handshake's copy, see vaani.core.logging).

Three paths stay open by design: /api/health and /api/ready, which load
balancers and the Asterisk readiness GotoIf poll without credentials, and
/api/telephony/announce, which is LAN-only like AudioSocket itself — a token
there would only end up in the dialplan file in cleartext.

Production never runs open. Boot refuses without a token, the settings API
refuses a save that would clear it, and if the token is cleared anyway the
guard serves nothing rather than everything.
"""

from __future__ import annotations

import hmac
from typing import Any

from starlette.datastructures import Headers, QueryParams
from starlette.responses import JSONResponse
from starlette.routing import get_route_path

from vaani.config import Settings
from vaani.core.logging import get_logger

log = get_logger(__name__)

OPEN_PATHS = frozenset({"/api/health", "/api/ready", "/api/telephony/announce"})
GUARDED_PREFIXES = ("/api/", "/ws/")


def check_api_token(settings: Settings) -> str | None:
    """Boot-time policy. Raises in production without a token; otherwise returns
    the warning to log, or None when there is nothing to say."""
    if settings.api_token:
        return None
    if settings.env == "prod":
        raise RuntimeError(
            "VAANI_ENV=prod but VAANI_API_TOKEN is empty: a production deployment does "
            "not boot with an open control plane. Set a token, or run as staging."
        )
    return (
        "API is open: VAANI_API_TOKEN is empty, so anyone who can reach port "
        f"{settings.port} can change settings and hang up calls. Fine on a laptop; set "
        "a token before this host has a public address."
    )


class TokenGuard:
    """Pure ASGI so it sees WebSocket handshakes as well as HTTP requests.

    The expected token is read per request from the settings store, so one saved
    on the admin page takes effect on the next request without a restart.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        # Not scope["path"]: behind a proxy or `--root-path`, that carries the
        # mount prefix, and "/vaani/api/agents" would never match "/api/".
        path = get_route_path(scope)
        if path in OPEN_PATHS or not path.startswith(GUARDED_PREFIXES):
            await self.app(scope, receive, send)
            return

        settings = _settings(scope["app"])
        expected = getattr(settings, "api_token", None) or None
        if not expected:
            if getattr(settings, "env", "dev") == "prod":
                log.error("API token is empty in production; serving nothing")
                await _refuse(scope, receive, send, 503, "API token is not configured")
                return
            await self.app(scope, receive, send)
            return

        presented = _bearer(Headers(scope=scope).get("authorization"))
        if presented is None and scope["type"] == "websocket":
            presented = QueryParams(scope.get("query_string", b"")).get("token")

        if presented is not None and hmac.compare_digest(presented.encode(), expected.encode()):
            await self.app(scope, receive, send)
            return

        log.warning("refused request without a valid API token", extra={"path": path})
        await _refuse(scope, receive, send, 401, "API token required")


async def _refuse(scope: dict[str, Any], receive: Any, send: Any, status: int, why: str) -> None:
    if scope["type"] == "http":
        headers = {"WWW-Authenticate": "Bearer"} if status == 401 else None
        await JSONResponse({"detail": why}, status_code=status, headers=headers)(
            scope, receive, send
        )
    else:
        # Closing before accept rejects the handshake; the browser sees a
        # socket that never opened.
        await send({"type": "websocket.close", "code": 4401 if status == 401 else 1013})


def _settings(app: Any) -> Any:
    state = app.state
    store = getattr(state, "settings_store", None)
    return store.settings if store is not None else getattr(state, "settings", None)


def _bearer(header: str | None) -> str | None:
    if not header:
        return None
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return None
    return value.strip()
