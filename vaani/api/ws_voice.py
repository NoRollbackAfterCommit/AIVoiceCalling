"""WebSocket media transport.

The wire protocol, deliberately minimal so a browser, a mobile app and the
Asterisk bridge can all speak it:

  client → server   binary : raw PCM16 mono 16 kHz caller audio
                    text   : JSON control — {"type":"start"|"dtmf"|"hangup", ...}

  server → client   binary : raw PCM16 mono 16 kHz agent audio
                    text   : JSON events — state, transcript, metrics,
                             barge_in, transfer, call_end

No codec negotiation and no jitter buffer: this is the transport for browsers and
for the SIP bridge, both of which sit on a reliable local link. Public WebRTC
should terminate at LiveKit and hand PCM to this same session object.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from vaani.core.logging import get_logger
from vaani.pipeline.manager import CallCapacityError
from vaani.pipeline.session import CallSession

log = get_logger(__name__)
router = APIRouter()


class WebSocketTransport:
    def __init__(self, ws: WebSocket) -> None:
        self._ws = ws
        self._open = True
        # Serialise sends: FastAPI's WebSocket is not safe for concurrent writes,
        # and audio streaming races event emission constantly.
        self._lock = asyncio.Lock()

    async def send_audio(self, pcm: bytes) -> None:
        if not self._open:
            return
        async with self._lock:
            try:
                await self._ws.send_bytes(pcm)
            except Exception:
                self._open = False

    async def send_event(self, event: dict[str, Any]) -> None:
        if not self._open:
            return
        async with self._lock:
            try:
                await self._ws.send_text(json.dumps(event, default=str))
            except Exception:
                self._open = False

    async def close(self) -> None:
        self._open = False
        with contextlib.suppress(Exception):
            await self._ws.close()


@router.websocket("/ws/call")
async def voice_call(
    ws: WebSocket,
    agent: str = Query("default", description="Agent profile key"),
    caller: str | None = Query(None, description="Caller number, if known"),
    direction: str = Query("inbound"),
) -> None:
    services = ws.app.state.services
    manager = ws.app.state.calls

    await ws.accept()
    transport = WebSocketTransport(ws)
    session = CallSession(
        transport=transport,
        services=services,
        agent_key=agent,
        caller_number=caller,
        direction=direction,
    )

    try:
        await manager.register(session)
    except CallCapacityError as exc:
        await transport.send_event({"type": "rejected", "reason": str(exc)})
        await transport.close()
        return

    await transport.send_event({"type": "call_start", "call_id": session.call_id})

    # The session loop and the socket reader run concurrently: the reader must
    # keep draining the socket even while the agent is thinking, or the client's
    # send buffer backs up and audio arrives late in bursts.
    session_task = asyncio.create_task(session.run(), name=f"session:{session.call_id}")
    reader_task = asyncio.create_task(_read(ws, session), name=f"reader:{session.call_id}")

    try:
        done, pending = await asyncio.wait(
            {session_task, reader_task}, return_when=asyncio.FIRST_COMPLETED
        )
        # Whichever finished first, wind the other down cleanly.
        if reader_task in done and not session_task.done():
            await session.hangup("caller_disconnected")
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(session_task, timeout=10)
        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
    finally:
        await manager.unregister(session)
        await transport.close()


async def _read(ws: WebSocket, session: CallSession) -> None:
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                return
            if (data := message.get("bytes")) is not None:
                await session.push_audio(data)
            elif (text := message.get("text")) is not None:
                await _control(text, session)
    except WebSocketDisconnect:
        return
    except RuntimeError:
        # Raised when receive() is called after the socket has closed.
        return


async def _control(raw: str, session: CallSession) -> None:
    try:
        message = json.loads(raw)
    except json.JSONDecodeError:
        return
    kind = message.get("type")
    if kind == "dtmf" and message.get("digit"):
        await session.push_dtmf(str(message["digit"])[:1])
    elif kind == "hangup":
        await session.hangup("caller_ended")
    elif kind == "start":
        log.info("client ready", extra={"client": message.get("client")})
