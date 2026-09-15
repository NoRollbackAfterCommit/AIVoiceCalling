"""One shared token in front of /api and /ws.

Two credentials reach this guard, and they are for different kinds of caller.
A **person** signs in and carries a session cookie, which names their role and
their organisation. A **machine** — the deploy script, the health probe, the
carrier harness — presents the shared bearer token, has no browser to sign in
with, and is treated as a platform administrator. An empty token keeps the
laptop demo zero-config.

Roles are enforced here rather than per handler. A permission model spread
across fifty call sites is one nobody can audit, and the question an auditor
asks is "what can a supervisor do", not "what does this endpoint allow" —
see `required_roles` in api/accounts.py.

HTTP presents the token as `Authorization: Bearer`. A browser cannot put a
header on a WebSocket handshake, so that one path may carry it as `?token=`;
HTTP may not, because a token in a URL lands in access logs and history (the
access log redacts the handshake's copy, see vaani.core.logging).

Three paths stay open by design: /api/health and /api/ready, which load
balancers and the Asterisk readiness GotoIf poll without credentials, and
/api/telephony/announce, which is LAN-only like AudioSocket itself — a token
there would only end up in the dialplan file in cleartext.

Two more are exempt from this guard but not open: /api/telephony/smartflo/handshake
and /ws/smartflo. Tata's servers reach them over the public internet and send no
Authorization header, and the handshake's reply carries a WebSocket URL with a
token in it, so leaving them merely open would disclose a credential. Each
authenticates itself instead — a dedicated webhook secret on the handshake and
a two-minute per-call token on the socket — neither of which is the admin token
or unlocks anything else. See vaani/api/smartflo.py.

Production never runs open. Boot refuses without a token, the settings API
refuses a save that would clear it, and if the token is cleared anyway the
guard serves nothing rather than everything.
"""

from __future__ import annotations

import hmac
from http.cookies import SimpleCookie
from typing import Any

from starlette.datastructures import Headers, QueryParams
from starlette.responses import JSONResponse
from starlette.routing import get_route_path

from vaani.api.accounts import COOKIE, required_roles
from vaani.config import Settings
from vaani.core.logging import get_logger

log = get_logger(__name__)

OPEN_PATHS = frozenset(
    {
        "/api/health",
        "/api/ready",
        "/api/telephony/announce",
        # Both Smartflo paths carry their own, stronger authentication — a
        # dedicated webhook secret and a two-minute per-call token — because the
        # shared bearer token cannot travel on either. See vaani/api/smartflo.py.
        "/api/telephony/smartflo/handshake",
        "/ws/smartflo",
        # Sign-in and sign-out: guarding the way in would leave nobody a way in.
        # /api/auth/me is NOT here — it answers who you are, so it needs to know.
        "/api/auth/login",
        "/api/auth/logout",
    }
)
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

        settings = live_settings(scope["app"])
        expected = getattr(settings, "api_token", None) or None

        # A signed-in person is resolved first, and on every path — including the
        # open one. Skipping it when no token is configured left the laptop demo
        # unable to answer "who am I", because nothing populated the request.
        user = await _session_user(scope)
        if user is not None:
            allowed = required_roles(scope.get("method", "GET"), path)
            if user.role not in allowed:
                log.warning(
                    "refused an action outside the caller's role",
                    extra={"path": path, "role": user.role},
                )
                await _refuse(scope, receive, send, 403, "Your role does not allow this")
                return
            scope.setdefault("state", {})["user"] = user
            await self.app(scope, receive, send)
            return

        if not expected:
            if getattr(settings, "env", "dev") == "prod":
                log.error("API token is empty in production; serving nothing")
                await _refuse(scope, receive, send, 503, "API token is not configured")
                return
            # Open deployment, nobody signed in: the zero-config laptop demo.
            scope.setdefault("state", {})["user"] = None
            await self.app(scope, receive, send)
            return

        presented = _bearer(Headers(scope=scope).get("authorization"))
        if presented is None and scope["type"] == "websocket":
            presented = QueryParams(scope.get("query_string", b"")).get("token")

        if presented is not None and hmac.compare_digest(presented.encode(), expected.encode()):
            # A machine holding the shared token acts as a platform
            # administrator: it has no account to carry a narrower role.
            scope.setdefault("state", {})["user"] = None
            await self.app(scope, receive, send)
            return

        log.warning("refused request without a valid API token", extra={"path": path})
        await _refuse(scope, receive, send, 401, "API token required")


async def _session_user(scope: dict[str, Any]) -> Any:
    """The signed-in person, or None.

    The account is reloaded on every request rather than trusted from the
    cookie. Sessions are stateless, so this is what makes suspending a user take
    effect on their next click instead of whenever their cookie happens to
    expire.
    """
    signer = getattr(scope["app"].state, "session_signer", None)
    accounts = getattr(getattr(scope["app"].state, "services", None), "accounts", None)
    if signer is None or accounts is None:
        return None

    raw = Headers(scope=scope).get("cookie") or ""
    token = SimpleCookie(raw).get(COOKIE).value if COOKIE in SimpleCookie(raw) else None
    user_id = signer.verify(token)
    if user_id is None:
        return None
    try:
        user = await accounts.get(user_id)
    except Exception:
        log.exception("could not load the signed-in account")
        return None
    return user if user is not None and user.active else None


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


def live_settings(app: Any) -> Any:
    """What the guard and the Smartflo endpoints both read: the settings as
    saved on the admin page, falling back to what the app booted with."""
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
