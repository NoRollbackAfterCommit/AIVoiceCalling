"""Tata Smartflo streaming wire format.

Smartflo has cloned Twilio Media Streams, so Twilio's examples transfer directly.
Audio is G.711 mu-law at 8 kHz, base64 inside JSON text frames — never binary —
which is where it differs from Exotel's signed-linear PCM.

This module is pure: JSON in, dataclass out, and JSON strings for the three
frames the bot sends. It owns the wire format and nothing else. Rate conversion
and the multiple-of-160 rule belong to the transport, next to the buffer that
enforces them.
"""

from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass, field
from typing import Any

from vaani.audio.resample import ulaw_to_pcm16


@dataclass(slots=True)
class SmartfloEvent:
    kind: str
    stream_sid: str | None = None
    call_sid: str | None = None
    pcm: bytes | None = None
    caller: str | None = None
    called: str | None = None
    direction: str | None = None
    custom: dict[str, str] = field(default_factory=dict)
    digit: str | None = None
    mark_name: str | None = None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def parse_frame(raw: str | bytes) -> SmartfloEvent:
    """Never raises. A carrier that sends one bad frame must not end a call."""
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return SmartfloEvent(kind="invalid")
    try:
        message = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError, RecursionError):
        # A deeply nested payload (e.g. "[" * 100_000) blows the interpreter's call
        # stack before the parser ever gets to raise JSONDecodeError. RecursionError
        # is a RuntimeError, not a ValueError, so it needs its own name here.
        return SmartfloEvent(kind="invalid")
    if not isinstance(message, dict):
        return SmartfloEvent(kind="invalid")

    kind = _text(message.get("event"))
    stream_sid = _text(message.get("streamSid"))
    if kind is None:
        return SmartfloEvent(kind="invalid")

    if kind == "start":
        start = message.get("start")
        start = start if isinstance(start, dict) else {}
        custom = start.get("customParameters")
        return SmartfloEvent(
            kind="start",
            stream_sid=stream_sid or _text(start.get("streamSid")),
            call_sid=_text(start.get("callSid")),
            caller=_text(start.get("from")),
            called=_text(start.get("to")),
            direction=_text(start.get("direction")),
            custom={str(k): str(v) for k, v in custom.items()} if isinstance(custom, dict) else {},
        )

    if kind == "media":
        media = message.get("media")
        payload = _text(media.get("payload")) if isinstance(media, dict) else None
        if payload is None:
            return SmartfloEvent(kind="invalid")
        try:
            ulaw = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            return SmartfloEvent(kind="invalid")
        return SmartfloEvent(kind="media", stream_sid=stream_sid, pcm=ulaw_to_pcm16(ulaw))

    if kind == "dtmf":
        dtmf = message.get("dtmf")
        digit = _text(dtmf.get("digit")) if isinstance(dtmf, dict) else None
        return SmartfloEvent(kind="dtmf", stream_sid=stream_sid, digit=digit[:1] if digit else None)

    if kind == "mark":
        mark = message.get("mark")
        return SmartfloEvent(
            kind="mark",
            stream_sid=stream_sid,
            mark_name=_text(mark.get("name")) if isinstance(mark, dict) else None,
        )

    if kind in ("connected", "stop"):
        return SmartfloEvent(kind=kind, stream_sid=stream_sid)

    return SmartfloEvent(kind="invalid")


# 160 bytes of mu-law at 8 kHz is exactly 20 ms. Smartflo refuses bot media that
# is under 160 bytes or not a multiple of it.
ULAW_FRAME_BYTES = 160


def media_frame(stream_sid: str, ulaw: bytes, seq: int) -> str:
    return json.dumps(
        {
            "event": "media",
            "streamSid": stream_sid,
            "sequenceNumber": str(seq),
            "media": {"payload": base64.b64encode(ulaw).decode("ascii")},
        }
    )


def mark_frame(stream_sid: str, name: str, seq: int) -> str:
    return json.dumps(
        {
            "event": "mark",
            "streamSid": stream_sid,
            "sequenceNumber": str(seq),
            "mark": {"name": name},
        }
    )


def clear_frame(stream_sid: str, seq: int) -> str:
    """Barge-in. Empties whatever the carrier still has buffered for the caller."""
    return json.dumps({"event": "clear", "streamSid": stream_sid, "sequenceNumber": str(seq)})
