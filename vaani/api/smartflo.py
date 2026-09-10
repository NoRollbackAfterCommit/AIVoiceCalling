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

import asyncio
import base64
import binascii
import contextlib
import hashlib
import hmac
import secrets
import time
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from starlette.responses import JSONResponse

from vaani.api.auth import live_settings
from vaani.core.logging import get_logger
from vaani.pipeline.manager import CallCapacityError
from vaani.pipeline.session import CallSession
from vaani.telephony.smartflo import SmartfloTransport
from vaani.telephony.smartflo_frames import SmartfloEvent, parse_frame

log = get_logger(__name__)
router = APIRouter()
# A router's prefix applies to every route on it, and this one is mounted under
# /api. The socket lives at /ws, so it gets a router of its own.
ws_router = APIRouter()

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


def _utf8(text: str) -> bytes | None:
    """`str.encode` raises on a lone surrogate, which a JSON body can carry as
    "\\ud800". Every string on this path comes from a stranger, so that has to
    be a refusal, never a traceback on a public endpoint."""
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError:
        return None


def mint_call_token(secret: str, call_id: str, *, now: float | None = None) -> str:
    exp = int((time.time() if now is None else now) + CALL_TOKEN_TTL_S)
    encoded = _encode(call_id)
    return f"{encoded}.{exp}.{_sign(secret, f'{encoded}:{exp}')}"


def verify_call_token(secret: str, token: str, *, now: float | None = None) -> str | None:
    """The call id when the token is genuine and unexpired, otherwise None."""
    if not secret or _utf8(secret) is None:
        # Unconfigured, or saved as text that cannot be encoded: refuse rather
        # than accept everything, or raise from `_sign` inside a public route.
        return None
    parts = token.split(".")
    if len(parts) != 3:
        return None
    encoded, raw_exp, presented = parts
    try:
        exp = int(raw_exp)
    except ValueError:
        return None
    presented_bytes = _utf8(presented)
    if presented_bytes is None or _utf8(encoded) is None:
        return None
    # Bytes, not str: compare_digest raises on non-ASCII text instead of
    # returning False, and either part may hold anything a stranger typed.
    if not hmac.compare_digest(_sign(secret, f"{encoded}:{exp}").encode(), presented_bytes):
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
    expected = _utf8(secret)
    presented = _utf8(str(fields.get("key") or ""))
    if not expected or presented is None or not hmac.compare_digest(presented, expected):
        log.warning("refused a Smartflo handshake with a bad secret")
        return JSONResponse({"detail": "invalid webhook secret"}, status_code=401)

    host = getattr(settings, "smartflo_public_host", "") or ""
    if not host:
        log.error("smartflo_public_host is unset; cannot build a wss_url")
        return JSONResponse({"detail": "smartflo_public_host is not configured"}, status_code=503)

    call_id = str(fields.get("callId") or fields.get("callid") or "").strip()
    if not call_id:
        # Not a fixed placeholder: a token for "unknown" would be one credential
        # shared by every callId-less caller for two minutes. Random, so each
        # such handshake still opens exactly one call; a warning, because a real
        # callback without a callId means the field is not named what we expect
        # and an operator should hear about it. Never a refusal: no live
        # Smartflo callback has been seen yet, and a real call must not be
        # dropped over a guess about its shape.
        call_id = f"unknown-{secrets.token_urlsafe(12)}"
        log.warning("smartflo handshake carried no callId; minted a random call id")
    if _utf8(call_id) is None:
        log.warning("refused a Smartflo handshake whose callId is not valid text")
        return JSONResponse({"detail": "callId is not valid text"}, status_code=400)
    log.info(
        "smartflo handshake",
        extra={"call_id": call_id, "to": str(fields.get("toNumber") or "")[:32]},
    )
    token = mint_call_token(secret, call_id)
    # Exactly these two keys. Smartflo refuses anything else and drops the call.
    return JSONResponse({"success": True, "wss_url": f"wss://{host}/ws/smartflo?token={token}"})


@ws_router.websocket("/ws/smartflo")
async def smartflo_stream(ws: WebSocket) -> None:
    """One call, from Smartflo's `start` frame to its `stop`.

    Shaped like /ws/call: the session and the socket reader are two tasks, and
    whichever ends first winds the other down. The reader has to keep draining
    while the agent thinks, or the carrier's audio arrives in late bursts.
    """
    settings = live_settings(ws.app)
    enabled = bool(getattr(settings, "smartflo_enabled", False))
    secret = getattr(settings, "smartflo_webhook_secret", "") or ""
    call_ref = verify_call_token(secret, ws.query_params.get("token", "")) if enabled else None
    if call_ref is None:
        # The token stays out of the log: forged, it is noise; genuine, it is a
        # credential for another two minutes.
        log.warning(
            "refused a Smartflo stream", extra={"reason": "bad token" if enabled else "disabled"}
        )
        # Closing before accept refuses the handshake outright, the way the
        # bearer guard does: Smartflo sees a socket that never opened.
        await ws.close(code=1008)
        return

    await ws.accept()
    transport = SmartfloTransport(ws.send_text)

    # Nothing to build until `start`: the caller's number and the stream id
    # arrive there, and no audio comes before it.
    start = await _await_start(ws)
    if start is None:
        await _close(ws)
        return
    transport.stream_sid = start.stream_sid or ""

    services = ws.app.state.services
    manager = ws.app.state.calls
    session = CallSession(
        transport=transport,
        services=services,
        agent_key=getattr(settings, "smartflo_agent", "default") or "default",
        caller_number=start.caller,
        direction=start.direction or "inbound",
    )
    try:
        await manager.register(session)
    except CallCapacityError as exc:
        # Smartflo's protocol has no "rejected" frame. Closing is the only
        # refusal it understands; the caller hears the line drop.
        log.warning("smartflo call refused", extra={"reason": str(exc)})
        await _close(ws)
        return
    log.info(
        "smartflo call started",
        extra={"call_id": session.call_id, "call_sid": start.call_sid, "token_call_id": call_ref},
    )

    session_task = asyncio.create_task(session.run(), name=f"session:{session.call_id}")
    reader_task = asyncio.create_task(
        _read(ws, transport, session), name=f"reader:{session.call_id}"
    )
    try:
        done, pending = await asyncio.wait(
            {session_task, reader_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if reader_task in done and not session_task.done():
            # Bounded, unlike /ws/call: with the peer wedged and the consumer
            # stuck under the transport lock, hangup's inbox sentinel waits on
            # a full queue. Cutting it short loses nothing that cancelling the
            # session task below does not already cover.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(session.hangup("caller_disconnected"), timeout=5)
            with contextlib.suppress(asyncio.CancelledError, TimeoutError):
                await asyncio.wait_for(session_task, timeout=10)
        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
    finally:
        await manager.unregister(session)
        await transport.close()
        # The transport only holds a sender and leaves the socket open on
        # purpose. Ending the call on the wire is this route's job; it is what
        # makes Smartflo hang up when the agent ends the call first.
        await _close(ws)


async def _await_start(ws: WebSocket) -> SmartfloEvent | None:
    """The `start` frame, or None if the socket ends before one arrives."""
    while (raw := await _next_frame(ws)) is not None:
        event = parse_frame(raw)
        if event.kind == "start":
            return event
        if event.kind == "stop":
            return None
    return None


async def _read(ws: WebSocket, transport: SmartfloTransport, session: CallSession) -> None:
    """Feeds the session until Smartflo sends `stop` or drops the socket."""
    while (raw := await _next_frame(ws)) is not None:
        event = parse_frame(raw)
        if event.kind == "media" and event.pcm:
            await session.push_audio(event.pcm)
        elif event.kind == "dtmf" and event.digit:
            await session.push_dtmf(event.digit)
        elif event.kind == "mark":
            transport.note_mark(event.mark_name)
        elif event.kind == "stop":
            return
        # `connected`, a repeated `start`, `invalid`: nothing a live call acts on.


async def _next_frame(ws: WebSocket) -> str | bytes | None:
    """The next frame, or None once the peer is gone.

    Smartflo promises text. A stray binary frame is still not a reason to end a
    call, so it goes to `parse_frame` like the rest and comes back `invalid`.
    """
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                return None
            if (text := message.get("text")) is not None:
                return text
            if (data := message.get("bytes")) is not None:
                return data
    except (WebSocketDisconnect, RuntimeError):
        # RuntimeError is receive() after the socket has closed, as in /ws/call.
        return None


async def _close(ws: WebSocket) -> None:
    with contextlib.suppress(Exception):
        await ws.close()
