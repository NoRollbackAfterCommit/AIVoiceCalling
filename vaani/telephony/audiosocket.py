"""Asterisk AudioSocket bridge — real telephony, incoming and outgoing.

AudioSocket is the least ceremonious way to get PCM out of Asterisk: a plain TCP
socket carrying length-prefixed frames, no RTP stack, no SDP negotiation, no
media server in the path. That makes it the right first telephony integration,
and it is production-viable for on-premise deployments where Asterisk and this
service sit on the same LAN. (For carrier-grade NAT traversal and scale, put
LiveKit in front and hand its PCM to the same CallSession — nothing below the
transport changes.)

Wire format, per message:

    byte 0      type
    bytes 1-2   payload length, big endian
    bytes 3..n  payload

    0x00  hang up
    0x01  call UUID (16 bytes), the first message on every connection
    0x10  audio — signed linear 16-bit, mono, 8 kHz, 20 ms (320 bytes)
    0xff  error

Dialplan:

    exten => 1912,1,Answer()
     same  =>       ,AudioSocket(${UUID},vaani-host:9092)
     same  =>       ,Hangup()

Asterisk speaks 8 kHz; the pipeline speaks 16 kHz. Conversion happens here and
nowhere else.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import struct
import uuid
from typing import Any

from vaani.audio.resample import resample_pcm16
from vaani.config import SAMPLE_RATE, Settings
from vaani.core.logging import get_logger
from vaani.core.registry import Services
from vaani.pipeline.manager import CallCapacityError, CallManager
from vaani.pipeline.session import CallSession

log = get_logger(__name__)

TYPE_HANGUP = 0x00
TYPE_UUID = 0x01
TYPE_AUDIO = 0x10
TYPE_ERROR = 0xFF

ASTERISK_RATE = 8000
# 20 ms of 8 kHz PCM16. Asterisk expects frames at exactly this cadence.
ASTERISK_FRAME_BYTES = ASTERISK_RATE * 2 * 20 // 1000  # 320


class AudioSocketTransport:
    """Adapts a CallSession to an Asterisk TCP connection."""

    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self._writer = writer
        self._open = True
        self._lock = asyncio.Lock()
        # Agent audio arrives in 200 ms lumps; Asterisk wants a steady 20 ms
        # drip. This holds the remainder between writes.
        self._pending = bytearray()
        self.events: list[dict[str, Any]] = []

    async def send_audio(self, pcm: bytes) -> None:
        if not self._open:
            return
        downsampled = resample_pcm16(pcm, SAMPLE_RATE, ASTERISK_RATE)
        async with self._lock:
            self._pending.extend(downsampled)
            while len(self._pending) >= ASTERISK_FRAME_BYTES:
                frame = bytes(self._pending[:ASTERISK_FRAME_BYTES])
                del self._pending[:ASTERISK_FRAME_BYTES]
                try:
                    self._writer.write(
                        struct.pack(">BH", TYPE_AUDIO, len(frame)) + frame
                    )
                except Exception:
                    self._open = False
                    return
            with contextlib.suppress(Exception):
                await self._writer.drain()

    async def send_event(self, event: dict[str, Any]) -> None:
        """Asterisk has no channel for these, so they go to the log and to any
        websocket supervisor attached to this call."""
        self.events.append(event)
        kind = event.get("type")
        if kind in ("transcript", "transfer", "call_end", "barge_in"):
            log.info("call event", extra={"event_type": kind, "payload": json.dumps(
                {k: v for k, v in event.items() if k != "record"}, default=str)[:500]})

    def drop_pending(self) -> None:
        """Barge-in: discard audio not yet handed to Asterisk."""
        self._pending.clear()

    async def close(self) -> None:
        if not self._open:
            return
        self._open = False
        with contextlib.suppress(Exception):
            self._writer.write(struct.pack(">BH", TYPE_HANGUP, 0))
            await self._writer.drain()
            self._writer.close()
            await self._writer.wait_closed()


async def _read_message(reader: asyncio.StreamReader) -> tuple[int, bytes] | None:
    header = await reader.readexactly(3)
    kind, length = struct.unpack(">BH", header)
    payload = await reader.readexactly(length) if length else b""
    return kind, payload


class AudioSocketServer:
    def __init__(
        self,
        services: Services,
        manager: CallManager,
        *,
        host: str = "0.0.0.0",
        port: int = 9092,
        agent_key: str = "default",
        settings: Settings | None = None,
    ) -> None:
        self._services = services
        self._manager = manager
        self._host = host
        self._port = port
        self._agent_key = agent_key
        self._settings = settings or services.settings
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle, host=self._host, port=self._port
        )
        log.info("audiosocket listening", extra={"host": self._host, "port": self._port})

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername")
        transport = AudioSocketTransport(writer)
        call_uuid: str | None = None
        session: CallSession | None = None
        session_task: asyncio.Task | None = None

        try:
            while True:
                try:
                    message = await _read_message(reader)
                except (asyncio.IncompleteReadError, ConnectionError):
                    break
                if message is None:
                    break
                kind, payload = message

                if kind == TYPE_UUID:
                    call_uuid = str(uuid.UUID(bytes=payload)) if len(payload) == 16 else None
                    log.info("inbound call", extra={"peer": str(peer), "uuid": call_uuid})
                    session = CallSession(
                        transport=transport,
                        services=self._services,
                        agent_key=self._agent_key,
                        call_id=(call_uuid or uuid.uuid4().hex)[:16],
                        direction="inbound",
                        settings=self._settings,
                    )
                    try:
                        await self._manager.register(session)
                    except CallCapacityError as exc:
                        log.warning("rejecting call", extra={"reason": str(exc)})
                        break
                    session_task = asyncio.create_task(session.run())

                elif kind == TYPE_AUDIO:
                    if session is None or not payload:
                        continue
                    await session.push_audio(
                        resample_pcm16(payload, ASTERISK_RATE, SAMPLE_RATE)
                    )

                elif kind == TYPE_HANGUP:
                    break

                elif kind == TYPE_ERROR:
                    log.warning("asterisk error frame", extra={"payload": payload.hex()})

        except Exception:
            log.exception("audiosocket connection failed")
        finally:
            if session is not None:
                await session.hangup("caller_hung_up")
                if session_task is not None:
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(session_task, timeout=10)
                await self._manager.unregister(session)
            await transport.close()
