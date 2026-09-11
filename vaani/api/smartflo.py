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
with that secret, which names one call and stops verifying two minutes after it
is minted. Nothing marks a token as used, so a replay inside that window opens a
second call — an accepted pilot risk (see docs/telephony.md), since the window
only has to cover the hop between the handshake and Smartflo's connect, and
neither credential can reach the settings API, the knowledge base, or a live
call.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import hashlib
import hmac
import json
import logging
import secrets
import time
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from starlette.responses import JSONResponse

from vaani.api.auth import live_settings
from vaani.api.limits import enforce_body_limit
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

# How long an accepted socket may stay silent before `start` arrives. Smartflo
# sends `connected` then `start` back to back, so the real gap is one network
# hop; ten seconds is three orders of magnitude of headroom for a carrier-side
# stall, and still under uvicorn's 20 s WebSocket ping interval, so an abandoned
# socket is gone before the keepalive would even go looking for it. Not a
# setting: an operator has nothing to weigh here, and a value low enough to drop
# real calls or high enough to matter would both be mistakes.
START_DEADLINE_S = 10.0


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


def _is_the_secret(presented: str, expected: bytes | None) -> bool:
    """Constant-time, and never a raise: both sides may hold anything a stranger
    typed, and an unconfigured secret must refuse rather than accept everything."""
    candidate = _utf8(presented)
    if not expected or candidate is None:
        return False
    return hmac.compare_digest(candidate, expected)


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
    """Smartflo may GET with query parameters or POST JSON or a form.

    Caller beware: this reads the body, so `enforce_body_limit` comes first.
    """
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
            try:
                form = dict(await request.form())
            except Exception:
                # python-multipart raises on a body that is not the form it
                # claims to be. Unguarded, that escaped as a 500 and a logged
                # traceback on a path any stranger can reach; a body we cannot
                # read simply carries no fields, exactly as on the JSON branch.
                log.warning("a Smartflo handshake body could not be parsed as a form")
                form = {}
            merged.update(form)
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
    expected = _utf8(secret)

    # The documented endpoint URL carries the secret in the query string, so in
    # the normal case authentication is settled here, before a byte of the body
    # is touched. A body only gets parsed for someone who already proved they
    # know the secret, or who left the query out entirely.
    in_query = request.query_params.get("key")
    if in_query is not None and not _is_the_secret(in_query, expected):
        log.warning("refused a Smartflo handshake with a bad secret")
        return JSONResponse({"detail": "invalid webhook secret"}, status_code=401)

    enforce_body_limit(request)
    fields = await _fields(request)
    if in_query is None and not _is_the_secret(str(fields.get("key") or ""), expected):
        log.warning("refused a Smartflo handshake with a bad secret")
        return JSONResponse({"detail": "invalid webhook secret"}, status_code=401)

    host = getattr(settings, "smartflo_public_host", "") or ""
    if not host:
        log.error("smartflo_public_host is unset; cannot build a wss_url")
        return JSONResponse({"detail": "smartflo_public_host is not configured"}, status_code=503)

    call_id = str(fields.get("callId") or fields.get("callid") or "").strip()
    if not call_id:
        # Not a fixed placeholder: a token is not single-use, so one minted for
        # "unknown" would be a credential every callId-less caller could replay
        # against every other's call for two minutes. Random, so a replay stays
        # confined to the handshake it came from; a warning, because a real
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
    tally = _FrameTally()
    try:
        start = await asyncio.wait_for(_await_start(ws, tally), timeout=START_DEADLINE_S)
    except TimeoutError:
        # Nothing is registered yet, so a socket parked here is invisible to the
        # concurrency cap, to `live()` and to `drain()` — a shutdown would not
        # even wait for it. A valid token must not buy one of those indefinitely.
        log.warning(
            "smartflo stream sent no start frame; closing",
            extra={
                "token_call_id": call_ref,
                "deadline_s": START_DEADLINE_S,
                # What did arrive, in case `start` is here under another name.
                "frame_counts": tally.snapshot(),
            },
        )
        start = None
    if start is None:
        await _close(ws)
        return
    if not start.stream_sid:
        # Every outbound frame carries the stream id, and `send_audio` drops the
        # lot without one: the call would run to its idle timeout as a registered,
        # connected line with dead air on it. Refusing costs the caller a redial
        # and gives the operator the reason; keeping it would cost them the call
        # anyway, plus a line out of the capacity budget and a call record that
        # looks like it worked.
        log.error(
            "smartflo start carried no streamSid; refusing the call, since the "
            "line would be silent for its whole duration",
            extra={"call_sid": start.call_sid or "", "token_call_id": call_ref},
        )
        await _close(ws)
        return
    transport.stream_sid = start.stream_sid

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
        _read(ws, transport, session, tally), name=f"reader:{session.call_id}"
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


class _FrameTally:
    """Counts the frames a call took no action on, and describes the first one.

    Discarding them in silence hides the likeliest first-live-call failure there
    is: if Tata's media arrives in a shape `parse_frame` does not match, every
    frame comes back `invalid`, the caller talks into silence, the idle hangup
    ends the call ninety seconds later, and the log for it holds the greeting and
    nothing else. A line per frame is no better — ten a second is nine hundred
    lines for that one call — so the first is described and the rest are counted.
    """

    __slots__ = ("_described", "counts")

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self._described = False

    def note(self, kind: str, raw: str | bytes) -> None:
        self.counts[kind] = self.counts.get(kind, 0) + 1
        if kind != "invalid" or self._described:
            return
        self._described = True
        # Structure, not content: the keys are what diagnoses a shape mismatch,
        # and unlike the body they carry no audio and nothing the caller said.
        log.info(
            "smartflo sent a frame the parser does not recognise",
            extra={"frame_shape": _frame_shape(raw), "frame_bytes": len(raw)},
        )
        # The body itself only at debug, where an operator has asked for it.
        log.debug("unrecognised smartflo frame", extra={"frame_body": _preview(raw)})

    def snapshot(self) -> dict[str, int]:
        return dict(self.counts)

    def report(self) -> None:
        if not self.counts:
            return
        # A warning only when something was unreadable: `connected` and a
        # repeated `start` are ordinary traffic and would be noise on every call.
        log.log(
            logging.WARNING if self.counts.get("invalid") else logging.DEBUG,
            "smartflo frames the call did not act on",
            extra={"frame_counts": self.snapshot()},
        )


def _frame_shape(raw: str | bytes) -> str:
    """What an unrecognised frame looked like, with none of what it held."""
    if isinstance(raw, bytes):
        return "binary"
    try:
        parsed = json.loads(raw)
    except Exception:
        # Including RecursionError on a deeply nested payload, as parse_frame does.
        return "not json"
    if not isinstance(parsed, dict):
        return f"json {type(parsed).__name__}"
    return ",".join(sorted(str(key)[:24] for key in parsed)[:8]) or "empty object"


def _preview(raw: str | bytes) -> str:
    text = raw if isinstance(raw, str) else raw.decode("utf-8", "replace")
    return text[:200]


async def _await_start(ws: WebSocket, tally: _FrameTally) -> SmartfloEvent | None:
    """The `start` frame, or None if the socket ends before one arrives."""
    while (raw := await _next_frame(ws)) is not None:
        event = parse_frame(raw)
        if event.kind == "start":
            return event
        if event.kind == "stop":
            return None
        # Nothing else advances the wait. Counted anyway: if a live `start` ever
        # arrives in a shape the parser misses, what did arrive is the only
        # diagnosis the operator gets, and the deadline's warning carries it.
        tally.note(event.kind, raw)
    return None


async def _read(
    ws: WebSocket, transport: SmartfloTransport, session: CallSession, tally: _FrameTally
) -> None:
    """Feeds the session until Smartflo sends `stop` or drops the socket."""
    try:
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
            else:
                # `connected`, a repeated `start`, `invalid`: nothing a live call
                # acts on, but a stream of them is why a call was silent.
                tally.note(event.kind, raw)
    finally:
        # In a finally so the count survives the cancellation that ends this task
        # when the agent, not the caller, hangs up first.
        tally.report()


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
