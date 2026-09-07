"""The Asterisk bridge actually answers: TCP in, greeting audio out, hangup.

This exercises the real server over a real socket — the mock providers make it
cheap, and the wire format is exactly what Asterisk sends, so a framing mistake
fails here rather than on a live trunk.
"""

from __future__ import annotations

import asyncio
import struct
import uuid

from vaani.config import Settings
from vaani.pipeline.manager import CallManager
from vaani.telephony.audiosocket import (
    ASTERISK_FRAME_BYTES,
    TYPE_AUDIO,
    TYPE_HANGUP,
    TYPE_UUID,
    AudioSocketServer,
)


def test_settings_expose_the_telephony_knobs():
    s = Settings()
    assert s.audiosocket_enabled is False, "telephony must be off until an operator turns it on"
    assert s.audiosocket_port == 9092
    assert s.audiosocket_host == "0.0.0.0"


async def _read_frame(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    header = await reader.readexactly(3)
    kind, length = struct.unpack(">BH", header)
    payload = await reader.readexactly(length) if length else b""
    return kind, payload


async def test_server_answers_greets_and_releases_the_line(services, settings):
    manager = CallManager(max_concurrent=2)
    server = AudioSocketServer(services, manager, host="127.0.0.1", port=0, settings=settings)
    await server.start()
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        writer.write(struct.pack(">BH", TYPE_UUID, 16) + uuid.uuid4().bytes)
        await writer.drain()

        for _ in range(200):
            await asyncio.sleep(0.02)
            if manager.live_count:
                break
        assert manager.live_count == 1, "the UUID frame must open a session"

        kind, payload = await asyncio.wait_for(_read_frame(reader), timeout=5)
        while kind != TYPE_AUDIO:
            kind, payload = await asyncio.wait_for(_read_frame(reader), timeout=5)
        assert len(payload) == ASTERISK_FRAME_BYTES, "greeting must arrive as 20 ms 8 kHz frames"

        writer.write(struct.pack(">BH", TYPE_HANGUP, 0))
        await writer.drain()
        for _ in range(400):
            await asyncio.sleep(0.02)
            if manager.live_count == 0:
                break
        assert manager.live_count == 0, "hangup must release the line"
        writer.close()
    finally:
        await server.stop()


async def test_announced_call_carries_caller_and_agent(services, settings):
    """The dialplan posts caller and agent under the UUID it hands to AudioSocket();
    the session opened for that UUID must carry both."""
    from vaani.telephony.announce import CallAnnouncements

    announcements = CallAnnouncements()
    manager = CallManager(max_concurrent=2)
    server = AudioSocketServer(
        services,
        manager,
        host="127.0.0.1",
        port=0,
        settings=settings,
        announcements=announcements,
    )
    await server.start()
    try:
        call_uuid = uuid.uuid4()
        announcements.announce(str(call_uuid), caller_number="+919876543210", agent_key="pension")

        _reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        writer.write(struct.pack(">BH", TYPE_UUID, 16) + call_uuid.bytes)
        await writer.drain()

        for _ in range(200):
            await asyncio.sleep(0.02)
            if manager.live_count:
                break
        (live,) = manager.live()
        assert live["caller_number"] == "+919876543210"
        assert live["agent_key"] == "pension"

        writer.write(struct.pack(">BH", TYPE_HANGUP, 0))
        await writer.drain()
        writer.close()
    finally:
        await server.stop()
