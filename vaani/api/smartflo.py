"""Smartflo's dynamic endpoint, and the WebSocket it points at.

`vaani/api/auth.py` refuses `?token=` on HTTP on purpose: a token in a URL lands
in access logs and browser history. Smartflo's dynamic endpoint is plain HTTP
called by Tata's servers, so it cannot use the WebSocket exemption, and its
*response* hands out a URL containing a token — left open it would be a
token-disclosure hole rather than merely an unauthenticated endpoint.

So both paths here are exempt from the shared bearer guard and authenticate
themselves more strictly than it would. The handshake requires
`smartflo_webhook_secret`, which is not the admin token and grants nothing but
the ability to ask for a wss_url. The WebSocket requires a per-call token signed
with that secret, valid for two minutes, which grants exactly one thing: opening
one call.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import time
from typing import Any

from fastapi import APIRouter, Request
from starlette.responses import JSONResponse

from vaani.api.auth import live_settings
from vaani.core.logging import get_logger

log = get_logger(__name__)
router = APIRouter()

# Smartflo connects immediately after the handshake, so the window only has to
# cover a network hop. Anything longer is replay surface for no benefit.
CALL_TOKEN_TTL_S = 120


def _sign(secret: str, payload: str) -> str:
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def _encode(call_id: str) -> str:
    return base64.urlsafe_b64encode(call_id.encode()).decode("ascii").rstrip("=")


def _decode(encoded: str) -> str | None:
    padded = encoded + "=" * (-len(encoded) % 4)
    try:
        return base64.urlsafe_b64decode(padded.encode()).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None


def mint_call_token(secret: str, call_id: str, *, now: float | None = None) -> str:
    exp = int((time.time() if now is None else now) + CALL_TOKEN_TTL_S)
    encoded = _encode(call_id)
    return f"{encoded}.{exp}.{_sign(secret, f'{encoded}:{exp}')}"


def verify_call_token(secret: str, token: str, *, now: float | None = None) -> str | None:
    """The call id when the token is genuine and unexpired, otherwise None."""
    if not secret:
        # An unconfigured deployment refuses rather than accepting everything.
        return None
    parts = token.split(".")
    if len(parts) != 3:
        return None
    encoded, raw_exp, presented = parts
    try:
        exp = int(raw_exp)
    except ValueError:
        return None
    if not hmac.compare_digest(_sign(secret, f"{encoded}:{exp}"), presented):
        return None
    if (time.time() if now is None else now) > exp:
        return None
    return _decode(encoded)


async def _fields(request: Request) -> dict[str, Any]:
    """Smartflo may GET with query parameters or POST JSON or a form."""
    merged: dict[str, Any] = dict(request.query_params)
    if request.method == "POST":
        if "application/json" in request.headers.get("content-type", ""):
            try:
                body = await request.json()
            except Exception:
                body = None
            if isinstance(body, dict):
                merged.update(body)
        else:
            merged.update(dict(await request.form()))
    return merged


@router.api_route("/telephony/smartflo/handshake", methods=["GET", "POST"], tags=["telephony"])
async def smartflo_handshake(request: Request) -> JSONResponse:
    """Answer Smartflo's per-call lookup with the socket to stream to.

    Two thousand milliseconds is a hangup, not a retry, so this does no database
    work, touches no provider, and builds a string.
    """
    settings = live_settings(request.app)
    if not getattr(settings, "smartflo_enabled", False):
        return JSONResponse({"detail": "Smartflo is not enabled"}, status_code=404)

    secret = getattr(settings, "smartflo_webhook_secret", "") or ""
    fields = await _fields(request)
    presented = str(fields.get("key") or "")
    if not secret or not hmac.compare_digest(presented.encode(), secret.encode()):
        log.warning("refused a Smartflo handshake with a bad secret")
        return JSONResponse({"detail": "invalid webhook secret"}, status_code=401)

    host = getattr(settings, "smartflo_public_host", "") or ""
    if not host:
        log.error("smartflo_public_host is unset; cannot build a wss_url")
        return JSONResponse({"detail": "smartflo_public_host is not configured"}, status_code=503)

    call_id = str(fields.get("callId") or fields.get("callid") or "").strip() or "unknown"
    log.info(
        "smartflo handshake",
        extra={"call_id": call_id, "to": str(fields.get("toNumber") or "")[:32]},
    )
    token = mint_call_token(secret, call_id)
    # Exactly these two keys. Smartflo refuses anything else and drops the call.
    return JSONResponse({"success": True, "wss_url": f"wss://{host}/ws/smartflo?token={token}"})
