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
