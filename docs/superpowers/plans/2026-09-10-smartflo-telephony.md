# Tata Smartflo Telephony Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accept inbound PSTN calls streamed by Tata Smartflo's voice-bot channel, without changing anything in the call pipeline.

**Architecture:** A frame codec module (pure functions, no I/O), a `SmartfloTransport` implementing the existing three-method `Transport` Protocol, and two API endpoints — a dynamic-endpoint handshake that mints a short-lived per-call token, and a WebSocket that pumps frames into a `CallSession`. `session.py`, `vad.py` and the `Transport` Protocol are untouched: the transport reacts to the `barge_in` and `speech_end` events the session already emits.

**Tech Stack:** Python 3.11+, FastAPI, `audioop` (already a dependency via `vaani/audio/resample.py`), `base64`/`json`/`hmac`/`hashlib` (stdlib). No new packages.

**Spec:** `docs/superpowers/specs/2026-09-10-smartflo-telephony-design.md`

## Global Constraints

- ruff, line length 100, target py311. `from __future__ import annotations` at the top of every module.
- Tests run against mock providers only — no GPU, no network, whole suite about a second. Run with `.venv/Scripts/python.exe -m pytest`.
- Inside the pipeline audio is **PCM16 little-endian, mono, 16 kHz**. mu-law/8 kHz conversion happens only inside `vaani/telephony/`.
- Smartflo media the bot sends must be **at least 160 bytes and a multiple of 160** (160 bytes of mu-law at 8 kHz = 20 ms).
- Every message to Smartflo carries `streamSid` and an incrementing `sequenceNumber`.
- The handshake must return **exactly** `{"success": true, "wss_url": "..."}` — any extra key, bad JSON or non-200 makes Smartflo drop the call — within **2000 ms**.
- New settings are declared once in `vaani/config.py` with `cfg()` metadata. Never hand-write markup in `vaani/web/static/settings.html`.
- `secret=True` fields are never returned in cleartext by the settings API.
- Comments explain *why*, not what. Match the register of the existing module docstrings.
- Lint before each commit: `.venv/Scripts/python.exe -m ruff check --fix . && .venv/Scripts/python.exe -m ruff format .`

---

### Task 1: Smartflo settings

**Files:**
- Modify: `vaani/config.py` (Telephony group, after the `audiosocket_*` fields, around line 519-545)
- Test: `tests/test_smartflo.py` (create)

**Interfaces:**
- Consumes: `cfg()` from `vaani/config.py`.
- Produces: `Settings.smartflo_enabled: bool`, `Settings.smartflo_webhook_secret: str`, `Settings.smartflo_public_host: str`, `Settings.smartflo_agent: str`.

**Note on `depends_on`:** every existing `depends_on` in `config.py` keys off a string-valued provider `Literal`; no boolean is used as a gate anywhere. Do **not** invent `depends_on={"smartflo_enabled": ["true"]}` — the settings page has never rendered that shape and it is unverified. The fields simply always show.

- [ ] **Step 1: Write the failing test**

Create `tests/test_smartflo.py`:

```python
"""Tata Smartflo voice-bot streaming: codec, transport and endpoints.

Everything here runs against mock providers and a fake Smartflo client, so the
wire format is exercised without a Tata account, a phone line or a network.
"""

from __future__ import annotations

from vaani.config import Settings


def test_smartflo_settings_carry_ui_metadata_and_hide_the_secret():
    fields = Settings.model_fields

    assert fields["smartflo_enabled"].default is False
    assert fields["smartflo_agent"].default == "default"

    for name in (
        "smartflo_enabled",
        "smartflo_webhook_secret",
        "smartflo_public_host",
        "smartflo_agent",
    ):
        extra = fields[name].json_schema_extra
        assert extra["group"] == "Telephony", f"{name} must render under Telephony"
        assert extra["label"], f"{name} needs a label for the admin page"
        # These take effect per call; none of them rebuilds a provider.
        assert extra["restart"] is False, f"{name} must not demand a restart"

    secret = fields["smartflo_webhook_secret"].json_schema_extra
    assert secret["secret"] is True, "the webhook secret must never be echoed back"
    assert fields["smartflo_public_host"].json_schema_extra["secret"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_smartflo.py -v`
Expected: FAIL with `KeyError: 'smartflo_enabled'`

- [ ] **Step 3: Write minimal implementation**

In `vaani/config.py`, immediately after the `audiosocket_port` field:

```python
    smartflo_enabled: bool = cfg(
        False,
        group="Telephony",
        label="Tata Smartflo voice bot",
        restart=False,
        help="Accept calls streamed by Tata Smartflo's voice-bot channel. Tata must "
        "enable Channels Hub on the account first; setup is in docs/telephony.md.",
    )
    smartflo_webhook_secret: str = cfg(
        "",
        group="Telephony",
        label="Smartflo webhook secret",
        secret=True,
        restart=False,
        help="Shared secret Smartflo puts in the dynamic-endpoint URL, and the key that "
        "signs the per-call token in the wss_url handed back. Deliberately not the API "
        "token: it travels in a URL that lands in the carrier's logs, so a leak must not "
        "reach the control plane.",
    )
    smartflo_public_host: str = cfg(
        "",
        group="Telephony",
        label="Smartflo public host",
        restart=False,
        help="Host used to build the wss_url, e.g. voice.example.in. Not taken from "
        "request headers: behind a proxy those are attacker-influenced, and Smartflo "
        "refuses anything but an exact URL.",
    )
    smartflo_agent: str = cfg(
        "default",
        group="Telephony",
        label="Smartflo agent profile",
        restart=False,
        help="Agent profile that answers Smartflo calls.",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_smartflo.py -v`
Expected: PASS

Then confirm nothing else broke: `.venv/Scripts/python.exe -m pytest -q`

- [ ] **Step 5: Commit**

```bash
.venv/Scripts/python.exe -m ruff check --fix . && .venv/Scripts/python.exe -m ruff format .
git add vaani/config.py tests/test_smartflo.py
git commit -m "Add the Smartflo settings, with the secret kept off the wire"
```

---

### Task 2: Frame codec — parsing inbound frames

**Files:**
- Create: `vaani/telephony/smartflo_frames.py`
- Test: `tests/test_smartflo.py`

**Interfaces:**
- Consumes: `ulaw_to_pcm16` from `vaani/audio/resample.py`.
- Produces: `SmartfloEvent` dataclass with fields `kind: str`, `stream_sid: str | None`, `call_sid: str | None`, `pcm: bytes | None`, `caller: str | None`, `called: str | None`, `direction: str | None`, `custom: dict[str, str]`, `digit: str | None`, `mark_name: str | None`; and `parse_frame(raw: str | bytes) -> SmartfloEvent`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_smartflo.py` (and add `import base64`, `import json` to the imports at the top):

```python
from vaani.telephony.smartflo_frames import SmartfloEvent, parse_frame


def _ulaw_silence(ms: int) -> bytes:
    # 0xFF is mu-law zero. 8 bytes per ms at 8 kHz.
    return b"\xff" * (8 * ms)


def test_parse_start_carries_the_caller_and_stream():
    raw = json.dumps(
        {
            "event": "start",
            "streamSid": "MZ123",
            "start": {
                "accountSid": "AC1",
                "streamSid": "MZ123",
                "callSid": "CA9",
                "from": "+919876543210",
                "to": "+911140000000",
                "direction": "inbound",
                "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000},
                "customParameters": {"agent": "grievance"},
            },
        }
    )
    event = parse_frame(raw)
    assert event.kind == "start"
    assert event.stream_sid == "MZ123"
    assert event.call_sid == "CA9"
    assert event.caller == "+919876543210"
    assert event.called == "+911140000000"
    assert event.custom == {"agent": "grievance"}


def test_parse_media_decodes_mulaw_to_pipeline_audio():
    payload = base64.b64encode(_ulaw_silence(100)).decode()
    event = parse_frame(
        json.dumps({"event": "media", "streamSid": "MZ1", "media": {"payload": payload}})
    )
    assert event.kind == "media"
    # 100 ms of 8 kHz mu-law becomes 100 ms of 16 kHz PCM16: 1600 samples, 3200 bytes.
    assert event.pcm is not None and len(event.pcm) == 3200


def test_parse_dtmf_and_mark():
    dtmf = parse_frame(json.dumps({"event": "dtmf", "streamSid": "MZ1", "dtmf": {"digit": "7"}}))
    assert (dtmf.kind, dtmf.digit) == ("dtmf", "7")

    mark = parse_frame(json.dumps({"event": "mark", "streamSid": "MZ1", "mark": {"name": "utt-3"}}))
    assert (mark.kind, mark.mark_name) == ("mark", "utt-3")


def test_parse_connected_and_stop_are_recognised():
    assert parse_frame(json.dumps({"event": "connected"})).kind == "connected"
    assert parse_frame(json.dumps({"event": "stop", "streamSid": "MZ1"})).kind == "stop"


def test_a_bad_frame_is_invalid_rather_than_fatal():
    """One malformed frame from a carrier must not end a call."""
    assert parse_frame("not json").kind == "invalid"
    assert parse_frame(json.dumps(["not", "an", "object"])).kind == "invalid"
    assert parse_frame(json.dumps({"no_event_key": 1})).kind == "invalid"
    assert (
        parse_frame(json.dumps({"event": "media", "media": {"payload": "!!!not base64!!!"}})).kind
        == "invalid"
    )
    assert parse_frame(json.dumps({"event": "unheard_of"})).kind == "invalid"
    assert parse_frame(b"\xff\xfe binary junk").kind == "invalid"
    assert isinstance(parse_frame("not json"), SmartfloEvent)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_smartflo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vaani.telephony.smartflo_frames'`

- [ ] **Step 3: Write minimal implementation**

Create `vaani/telephony/smartflo_frames.py`:

```python
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
    except (json.JSONDecodeError, TypeError, ValueError):
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_smartflo.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
.venv/Scripts/python.exe -m ruff check --fix . && .venv/Scripts/python.exe -m ruff format .
git add vaani/telephony/smartflo_frames.py tests/test_smartflo.py
git commit -m "Parse Smartflo frames without ever raising on a bad one"
```

---

### Task 3: Frame codec — outbound frames

**Files:**
- Modify: `vaani/telephony/smartflo_frames.py`
- Test: `tests/test_smartflo.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `media_frame(stream_sid: str, ulaw: bytes, seq: int) -> str`, `mark_frame(stream_sid: str, name: str, seq: int) -> str`, `clear_frame(stream_sid: str, seq: int) -> str`, and the constant `ULAW_FRAME_BYTES = 160`.

`media_frame` takes **already-encoded mu-law** and only base64s and envelopes it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_smartflo.py`:

```python
from vaani.telephony.smartflo_frames import ULAW_FRAME_BYTES, clear_frame, mark_frame, media_frame


def test_media_frame_envelopes_mulaw_unchanged():
    ulaw = bytes(range(160))
    frame = json.loads(media_frame("MZ1", ulaw, 7))
    assert frame["event"] == "media"
    assert frame["streamSid"] == "MZ1"
    # Twilio-shaped, and Smartflo copies it: the sequence number is a string.
    assert frame["sequenceNumber"] == "7"
    assert base64.b64decode(frame["media"]["payload"]) == ulaw


def test_mark_and_clear_frames_carry_stream_and_sequence():
    mark = json.loads(mark_frame("MZ1", "utt-2", 11))
    assert mark == {
        "event": "mark",
        "streamSid": "MZ1",
        "sequenceNumber": "11",
        "mark": {"name": "utt-2"},
    }

    clear = json.loads(clear_frame("MZ1", 12))
    assert clear == {"event": "clear", "streamSid": "MZ1", "sequenceNumber": "12"}


def test_one_mulaw_frame_is_twenty_milliseconds():
    """160 bytes of mu-law at 8 kHz is exactly 20 ms, which is why Smartflo's
    multiple-of-160 rule lines up with the pipeline's frame cadence."""
    assert ULAW_FRAME_BYTES == 160
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_smartflo.py -v`
Expected: FAIL with `ImportError: cannot import name 'ULAW_FRAME_BYTES'`

- [ ] **Step 3: Write minimal implementation**

Append to `vaani/telephony/smartflo_frames.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_smartflo.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
.venv/Scripts/python.exe -m ruff check --fix . && .venv/Scripts/python.exe -m ruff format .
git add vaani/telephony/smartflo_frames.py tests/test_smartflo.py
git commit -m "Build the three frames the bot sends back to Smartflo"
```

---

### Task 4: SmartfloTransport

**Files:**
- Create: `vaani/telephony/smartflo.py`
- Test: `tests/test_smartflo.py`

**Interfaces:**
- Consumes: `media_frame`, `mark_frame`, `clear_frame`, `ULAW_FRAME_BYTES` from `vaani/telephony/smartflo_frames.py`; `pcm16_to_ulaw` from `vaani/audio/resample.py`.
- Produces: `SmartfloTransport(sender: Callable[[str], Awaitable[None]], stream_sid: str = "")` with `send_audio(pcm: bytes) -> None`, `send_event(event: dict[str, Any]) -> None`, `close() -> None`, `note_mark(name: str | None) -> None`, and attributes `stream_sid: str`, `frames: list[str]` is **not** part of the interface — tests capture through the injected `sender`.

Taking a `sender` callable rather than the `WebSocket` keeps the transport testable without a socket, and mirrors how `AudioSocketTransport` takes a `StreamWriter` rather than reaching for a server.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_smartflo.py`:

```python
import pytest

from vaani.telephony.smartflo import SmartfloTransport


class _Sink:
    """Captures the JSON frames a transport would have put on the wire."""

    def __init__(self) -> None:
        self.frames: list[dict] = []

    async def __call__(self, raw: str) -> None:
        self.frames.append(json.loads(raw))

    def of_event(self, kind: str) -> list[dict]:
        return [f for f in self.frames if f.get("event") == kind]


def _pcm_silence_16k(ms: int) -> bytes:
    return b"\x00\x00" * (16 * ms)


async def test_agent_audio_goes_out_as_multiples_of_160_mulaw_bytes():
    sink = _Sink()
    transport = SmartfloTransport(sink, stream_sid="MZ1")

    # 200 ms at 16 kHz PCM16 is 200 ms of mu-law at 8 kHz: 1600 bytes, ten frames' worth.
    await transport.send_audio(_pcm_silence_16k(200))

    media = sink.of_event("media")
    assert media, "no media frame was sent"
    for frame in media:
        payload = base64.b64decode(frame["media"]["payload"])
        assert len(payload) >= ULAW_FRAME_BYTES, "Smartflo refuses media under 160 bytes"
        assert len(payload) % ULAW_FRAME_BYTES == 0, "media must be a multiple of 160 bytes"
    assert sum(len(base64.b64decode(f["media"]["payload"])) for f in media) == 1600


async def test_a_remainder_is_held_back_rather_than_sent_short():
    sink = _Sink()
    transport = SmartfloTransport(sink, stream_sid="MZ1")

    # 25 ms is 200 mu-law bytes: one whole frame plus 40 bytes that must wait.
    await transport.send_audio(_pcm_silence_16k(25))
    first = sum(len(base64.b64decode(f["media"]["payload"])) for f in sink.of_event("media"))
    assert first == 160, "the 40-byte remainder must not go out short"

    await transport.send_audio(_pcm_silence_16k(25))
    total = sum(len(base64.b64decode(f["media"]["payload"])) for f in sink.of_event("media"))
    assert total == 320, "the held-back remainder must go out with the next audio"


async def test_sequence_numbers_increment_across_every_frame():
    sink = _Sink()
    transport = SmartfloTransport(sink, stream_sid="MZ1")
    await transport.send_audio(_pcm_silence_16k(20))
    await transport.send_event({"type": "speech_end"})
    await transport.send_audio(_pcm_silence_16k(20))

    seqs = [int(f["sequenceNumber"]) for f in sink.frames]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs), f"not strictly increasing: {seqs}"


async def test_barge_in_clears_the_carrier_buffer_and_our_own():
    sink = _Sink()
    transport = SmartfloTransport(sink, stream_sid="MZ1")

    await transport.send_audio(_pcm_silence_16k(25))  # leaves a 40-byte remainder
    await transport.send_event({"type": "barge_in"})
    assert sink.of_event("clear"), "barge-in must flush what the carrier has buffered"

    sent_before = len(sink.of_event("media"))
    # 15 ms is 120 mu-law bytes. Had the stale 40-byte remainder survived, the two
    # together would make 160 and go out immediately.
    await transport.send_audio(_pcm_silence_16k(15))
    assert len(sink.of_event("media")) == sent_before, "stale audio survived the clear"


async def test_speech_end_sends_a_mark_and_the_echo_is_recorded(caplog):
    sink = _Sink()
    transport = SmartfloTransport(sink, stream_sid="MZ1")
    await transport.send_event({"type": "speech_end"})

    marks = sink.of_event("mark")
    assert len(marks) == 1
    name = marks[0]["mark"]["name"]

    with caplog.at_level("INFO"):
        transport.note_mark(name)
    assert any("playback confirmed" in r.message for r in caplog.records)


async def test_sends_are_dropped_once_closed_rather_than_raising():
    sink = _Sink()
    transport = SmartfloTransport(sink, stream_sid="MZ1")
    await transport.close()
    await transport.send_audio(_pcm_silence_16k(200))
    await transport.send_event({"type": "barge_in"})
    assert sink.frames == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_smartflo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vaani.telephony.smartflo'`

- [ ] **Step 3: Write minimal implementation**

Create `vaani/telephony/smartflo.py`:

```python
"""Adapts a CallSession to a Tata Smartflo voice-bot WebSocket.

Smartflo speaks mu-law at 8 kHz; the pipeline speaks PCM16 at 16 kHz. Conversion
happens here and nowhere else, exactly as it does for Asterisk.

Nothing in the session changes for this transport. The two carrier-specific
behaviours ride on events the session already emits: `barge_in` becomes `clear`,
which empties what Smartflo still has buffered for the caller, and `speech_end`
becomes `mark`, whose echo is a genuine playback-complete signal. The echo is
only measured for now — the session keeps pacing playback against the wall clock,
because adopting a carrier's completion signal before a single live call would be
optimising against a document.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

from vaani.audio.resample import pcm16_to_ulaw
from vaani.core.logging import get_logger
from vaani.telephony.smartflo_frames import (
    ULAW_FRAME_BYTES,
    clear_frame,
    mark_frame,
    media_frame,
)

log = get_logger(__name__)

Sender = Callable[[str], Awaitable[None]]


class SmartfloTransport:
    def __init__(self, sender: Sender, stream_sid: str = "") -> None:
        self._send_raw = sender
        self.stream_sid = stream_sid
        self._open = True
        # One writer at a time: audio streaming races event emission constantly.
        self._lock = asyncio.Lock()
        # Agent audio arrives in lumps that do not divide by 160. Smartflo refuses
        # a short frame, so the remainder waits here for the next one.
        self._pending = bytearray()
        self._seq = 0
        self._marks = 0
        self._mark_sent_at: dict[str, float] = {}

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    async def _send(self, frame: str) -> None:
        if not self._open:
            return
        try:
            await self._send_raw(frame)
        except Exception:
            self._open = False

    async def send_audio(self, pcm: bytes) -> None:
        if not self._open or not self.stream_sid:
            return
        async with self._lock:
            self._pending.extend(pcm16_to_ulaw(pcm))
            usable = (len(self._pending) // ULAW_FRAME_BYTES) * ULAW_FRAME_BYTES
            if not usable:
                return
            chunk = bytes(self._pending[:usable])
            del self._pending[:usable]
        await self._send(media_frame(self.stream_sid, chunk, self._next_seq()))

    async def send_event(self, event: dict[str, Any]) -> None:
        kind = event.get("type")

        if kind == "barge_in" and self.stream_sid:
            async with self._lock:
                # Ours as well as theirs: a flush followed by our own stale
                # remainder would put the interrupted sentence back on the line.
                self._pending.clear()
            await self._send(clear_frame(self.stream_sid, self._next_seq()))
        elif kind == "speech_end" and self.stream_sid:
            self._marks += 1
            name = f"utt-{self._marks}"
            self._mark_sent_at[name] = time.monotonic()
            await self._send(mark_frame(self.stream_sid, name, self._next_seq()))

        if kind in ("transcript", "transfer", "call_end", "barge_in"):
            log.info(
                "call event",
                extra={
                    "event_type": kind,
                    "payload": json.dumps(
                        {k: v for k, v in event.items() if k != "record"}, default=str
                    )[:500],
                },
            )

    def note_mark(self, name: str | None) -> None:
        """Smartflo echoing a mark means the caller has heard up to that point.

        Recorded, not acted on. The round trip is the measurement that decides
        whether to make this the playback-complete signal instead of pacing.
        """
        sent = self._mark_sent_at.pop(name or "", None)
        if sent is None:
            return
        log.info(
            "playback confirmed",
            extra={"mark": name, "round_trip_ms": int((time.monotonic() - sent) * 1000)},
        )

    async def close(self) -> None:
        self._open = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_smartflo.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
.venv/Scripts/python.exe -m ruff check --fix . && .venv/Scripts/python.exe -m ruff format .
git add vaani/telephony/smartflo.py tests/test_smartflo.py
git commit -m "Carry a call over Smartflo without touching the pipeline"
```

---

### Task 5: Per-call token

**Files:**
- Create: `vaani/api/smartflo.py`
- Test: `tests/test_smartflo.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `CALL_TOKEN_TTL_S: int`, `mint_call_token(secret: str, call_id: str, *, now: float | None = None) -> str`, `verify_call_token(secret: str, token: str, *, now: float | None = None) -> str | None` (returns the call id, or `None` when the token is malformed, forged or expired).

The call id is base64url-encoded inside the token so a carrier id containing a `.` cannot break the delimiters.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_smartflo.py`:

```python
from vaani.api.smartflo import CALL_TOKEN_TTL_S, mint_call_token, verify_call_token


def test_a_fresh_token_round_trips_to_its_call_id():
    token = mint_call_token("s3cret", "CA9")
    assert verify_call_token("s3cret", token) == "CA9"


def test_a_call_id_with_a_dot_survives_the_encoding():
    token = mint_call_token("s3cret", "call.with.dots")
    assert verify_call_token("s3cret", token) == "call.with.dots"


def test_a_tampered_or_foreign_token_is_refused():
    token = mint_call_token("s3cret", "CA9")
    assert verify_call_token("different-secret", token) is None
    assert verify_call_token("s3cret", token + "x") is None
    assert verify_call_token("s3cret", "garbage") is None
    assert verify_call_token("s3cret", "a.b.c") is None
    assert verify_call_token("s3cret", "") is None


def test_a_token_expires():
    minted_at = 1_000_000.0
    token = mint_call_token("s3cret", "CA9", now=minted_at)
    assert verify_call_token("s3cret", token, now=minted_at + CALL_TOKEN_TTL_S - 1) == "CA9"
    assert verify_call_token("s3cret", token, now=minted_at + CALL_TOKEN_TTL_S + 1) is None


def test_an_empty_secret_never_validates():
    """An unconfigured deployment must refuse, not accept everything."""
    assert verify_call_token("", mint_call_token("", "CA9")) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_smartflo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vaani.api.smartflo'`

- [ ] **Step 3: Write minimal implementation**

Create `vaani/api/smartflo.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_smartflo.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
.venv/Scripts/python.exe -m ruff check --fix . && .venv/Scripts/python.exe -m ruff format .
git add vaani/api/smartflo.py tests/test_smartflo.py
git commit -m "Mint a per-call token so a leaked wss_url cannot reach the API"
```

---

### Task 6: Handshake endpoint and the guard exemption

**Files:**
- Modify: `vaani/api/smartflo.py`
- Modify: `vaani/api/auth.py` (rename `_settings` to `live_settings`, extend `OPEN_PATHS` and the module docstring)
- Modify: `vaani/main.py:164-168` (register the router)
- Test: `tests/test_smartflo.py`

**Interfaces:**
- Consumes: `mint_call_token` from Task 5; `Settings.smartflo_*` from Task 1.
- Produces: `router: APIRouter` in `vaani/api/smartflo.py` exposing `GET|POST /telephony/smartflo/handshake` (mounted under `/api`); `live_settings(app) -> Settings` in `vaani/api/auth.py`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_smartflo.py` (add `from fastapi.testclient import TestClient` and `from vaani.main import create_app` to the imports):

```python
@pytest.fixture
def smartflo_app(monkeypatch):
    """The real app, with Smartflo switched on and a known secret."""
    monkeypatch.setenv("VAANI_ENV", "dev")
    monkeypatch.setenv("VAANI_API_TOKEN", "admin-token")
    monkeypatch.setenv("VAANI_SMARTFLO_ENABLED", "true")
    monkeypatch.setenv("VAANI_SMARTFLO_WEBHOOK_SECRET", "hook-secret")
    monkeypatch.setenv("VAANI_SMARTFLO_PUBLIC_HOST", "voice.example.in")
    monkeypatch.setenv("VAANI_STT_PROVIDER", "mock")
    monkeypatch.setenv("VAANI_LLM_PROVIDER", "mock")
    monkeypatch.setenv("VAANI_TTS_PROVIDER", "mock")
    monkeypatch.setenv("VAANI_VECTOR_STORE", "memory")
    monkeypatch.setenv("VAANI_EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("VAANI_RECORD_CALLS", "false")
    with TestClient(create_app()) as client:
        yield client


HANDSHAKE = "/api/telephony/smartflo/handshake"


def test_handshake_returns_exactly_the_two_keys_smartflo_accepts(smartflo_app):
    """Smartflo refuses a body with any extra key, and drops the call."""
    response = smartflo_app.post(
        HANDSHAKE,
        params={"key": "hook-secret"},
        json={"callId": "CA9", "fromNumber": "+919876543210", "toNumber": "+911140000000"},
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"success", "wss_url"}
    assert body["success"] is True
    assert body["wss_url"].startswith("wss://voice.example.in/ws/smartflo?token=")


def test_handshake_accepts_a_get_with_query_parameters(smartflo_app):
    response = smartflo_app.get(
        HANDSHAKE, params={"key": "hook-secret", "callId": "CA9", "fromNumber": "+911"}
    )
    assert response.status_code == 200
    assert response.json()["success"] is True


def test_handshake_mints_a_token_the_websocket_will_accept(smartflo_app):
    body = smartflo_app.post(
        HANDSHAKE, params={"key": "hook-secret"}, json={"callId": "CA9"}
    ).json()
    token = body["wss_url"].split("token=", 1)[1]
    assert verify_call_token("hook-secret", token) == "CA9"


def test_handshake_refuses_a_wrong_or_missing_secret(smartflo_app):
    assert smartflo_app.post(HANDSHAKE, json={"callId": "CA9"}).status_code == 401
    assert (
        smartflo_app.post(HANDSHAKE, params={"key": "wrong"}, json={"callId": "CA9"}).status_code
        == 401
    )


def test_handshake_answers_well_inside_the_two_second_budget(smartflo_app):
    """Smartflo hangs up at 2000 ms. This path must do no I/O at all."""
    started = time.monotonic()
    smartflo_app.post(HANDSHAKE, params={"key": "hook-secret"}, json={"callId": "CA9"})
    assert (time.monotonic() - started) < 0.5


def test_the_handshake_is_not_behind_the_admin_bearer_token(smartflo_app):
    """It is reached by Tata's servers, which send no Authorization header."""
    response = smartflo_app.post(HANDSHAKE, params={"key": "hook-secret"}, json={"callId": "CA9"})
    assert response.status_code == 200, "the shared guard must not stand in front of this path"
```

Add `import time` to the test file's imports.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_smartflo.py -v -k handshake`
Expected: FAIL with 401 or 404 — the route does not exist and the guard refuses the path.

- [ ] **Step 3: Write minimal implementation**

First, in `vaani/api/auth.py`, rename the private settings accessor so other modules can share it. Find `def _settings(` and rename it to `live_settings`, updating its one call site inside `TokenGuard.__call__`:

```python
        settings = live_settings(scope["app"])
```

Extend `OPEN_PATHS`:

```python
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
    }
)
```

And extend the module docstring's "Three paths stay open by design" paragraph to name the two Smartflo paths and say they are self-guarded rather than open.

Then append to `vaani/api/smartflo.py`:

```python
from typing import Any

from fastapi import APIRouter, Request
from starlette.responses import JSONResponse

from vaani.api.auth import live_settings
from vaani.core.logging import get_logger

log = get_logger(__name__)
router = APIRouter()


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
```

Note: `live_settings` in `auth.py` takes the ASGI `scope["app"]`. Pass whatever that helper expects — read its body before wiring, and if it reads `app.state`, pass `request.app`. Adjust the call above to match; the test will tell you immediately.

Finally, register the router in `vaani/main.py` beside the others:

```python
    app.include_router(smartflo_api.router, prefix="/api")
```

with `from vaani.api import smartflo as smartflo_api` added to the imports at the top.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_smartflo.py -v`
Expected: PASS

Then the whole suite, because `auth.py` and `main.py` changed: `.venv/Scripts/python.exe -m pytest -q`

- [ ] **Step 5: Commit**

```bash
.venv/Scripts/python.exe -m ruff check --fix . && .venv/Scripts/python.exe -m ruff format .
git add vaani/api/smartflo.py vaani/api/auth.py vaani/main.py tests/test_smartflo.py
git commit -m "Answer Smartflo's per-call lookup without opening the control plane"
```

---

### Task 7: The WebSocket endpoint

**Files:**
- Modify: `vaani/api/smartflo.py`
- Modify: `vaani/main.py` (register the WebSocket router — the same `router` object, so only the import from Task 6 is needed; add `app.include_router(smartflo_api.router)` **without** the `/api` prefix for the `/ws` path)
- Test: `tests/test_smartflo.py`

**Interfaces:**
- Consumes: `SmartfloTransport` (Task 4), `parse_frame` (Task 2), `verify_call_token` (Task 5), `CallSession` and `CallCapacityError`.
- Produces: `WS /ws/smartflo?token=<per-call token>`.

Because `APIRouter` prefixes apply to every route on it, put the WebSocket on a **second** router object, `ws_router`, in the same module, and mount that one without a prefix.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_smartflo.py`:

```python
def _start_frame(stream_sid: str = "MZ1", caller: str = "+919876543210") -> str:
    return json.dumps(
        {
            "event": "start",
            "streamSid": stream_sid,
            "start": {
                "streamSid": stream_sid,
                "callSid": "CA9",
                "from": caller,
                "to": "+911140000000",
                "direction": "inbound",
                "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000},
            },
        }
    )


def _media_frame_in(ms: int = 100) -> str:
    return json.dumps(
        {
            "event": "media",
            "streamSid": "MZ1",
            "media": {"payload": base64.b64encode(_ulaw_silence(ms)).decode()},
        }
    )


def test_websocket_refuses_a_bad_token(smartflo_app):
    with pytest.raises(Exception):
        with smartflo_app.websocket_connect("/ws/smartflo?token=forged"):
            pass


def test_websocket_refuses_no_token(smartflo_app):
    with pytest.raises(Exception):
        with smartflo_app.websocket_connect("/ws/smartflo"):
            pass


def test_a_smartflo_call_greets_the_caller(smartflo_app):
    """The whole path: handshake, connect, start, and audio comes back."""
    body = smartflo_app.post(
        HANDSHAKE, params={"key": "hook-secret"}, json={"callId": "CA9"}
    ).json()
    token = body["wss_url"].split("token=", 1)[1]

    with smartflo_app.websocket_connect(f"/ws/smartflo?token={token}") as ws:
        ws.send_text(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
        ws.send_text(_start_frame())
        ws.send_text(_media_frame_in())

        media = []
        for _ in range(40):
            frame = json.loads(ws.receive_text())
            if frame.get("event") == "media":
                media.append(frame)
                break
        assert media, "the agent never sent audio back"
        payload = base64.b64decode(media[0]["media"]["payload"])
        assert len(payload) % ULAW_FRAME_BYTES == 0
        assert media[0]["streamSid"] == "MZ1"

        ws.send_text(json.dumps({"event": "stop", "streamSid": "MZ1"}))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_smartflo.py -v -k websocket or smartflo_call`
Expected: FAIL — `/ws/smartflo` returns 404 because no route is registered.

- [ ] **Step 3: Write minimal implementation**

Append to `vaani/api/smartflo.py`:

```python
import asyncio
import contextlib

from fastapi import WebSocket, WebSocketDisconnect

from vaani.pipeline.manager import CallCapacityError
from vaani.pipeline.session import CallSession
from vaani.telephony.smartflo import SmartfloTransport
from vaani.telephony.smartflo_frames import parse_frame

ws_router = APIRouter()


@ws_router.websocket("/ws/smartflo")
async def smartflo_stream(ws: WebSocket) -> None:
    settings = live_settings(ws.app)
    secret = getattr(settings, "smartflo_webhook_secret", "") or ""
    token = ws.query_params.get("token", "")

    if not getattr(settings, "smartflo_enabled", False) or verify_call_token(secret, token) is None:
        # Closing before accept refuses the handshake outright.
        await ws.close(code=1008)
        return

    await ws.accept()
    transport = SmartfloTransport(ws.send_text)
    services = ws.app.state.services
    manager = ws.app.state.calls

    # The caller's number arrives in `start`, which lands before any audio, so
    # the session is built only once it has been read.
    session: CallSession | None = None
    session_task: asyncio.Task[None] | None = None

    try:
        while True:
            try:
                raw = await ws.receive_text()
            except (WebSocketDisconnect, RuntimeError):
                break

            event = parse_frame(raw)

            if event.kind == "start":
                if session is not None:
                    continue
                transport.stream_sid = event.stream_sid or ""
                session = CallSession(
                    transport=transport,
                    services=services,
                    agent_key=getattr(settings, "smartflo_agent", "default") or "default",
                    caller_number=event.caller,
                    direction=event.direction or "inbound",
                )
                try:
                    await manager.register(session)
                except CallCapacityError as exc:
                    log.warning("smartflo call refused", extra={"reason": str(exc)})
                    break
                session_task = asyncio.create_task(
                    session.run(), name=f"smartflo:{session.call_id}"
                )
            elif event.kind == "media" and session is not None and event.pcm:
                await session.push_audio(event.pcm)
            elif event.kind == "dtmf" and session is not None and event.digit:
                await session.push_dtmf(event.digit)
            elif event.kind == "mark":
                transport.note_mark(event.mark_name)
            elif event.kind == "stop":
                break
    finally:
        if session is not None:
            await session.hangup("caller_disconnected")
            if session_task is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await asyncio.wait_for(session_task, timeout=10)
            await manager.unregister(session)
        await transport.close()
        with contextlib.suppress(Exception):
            await ws.close()
```

Register it in `vaani/main.py`, next to the existing `include_router` calls:

```python
    app.include_router(smartflo_api.ws_router)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_smartflo.py -v`
Expected: PASS

Then the whole suite: `.venv/Scripts/python.exe -m pytest -q`

- [ ] **Step 5: Commit**

```bash
.venv/Scripts/python.exe -m ruff check --fix . && .venv/Scripts/python.exe -m ruff format .
git add vaani/api/smartflo.py vaani/main.py tests/test_smartflo.py
git commit -m "Take a Smartflo call from start to stop"
```

---

### Task 8: Barge-in over the real socket, and the operator documentation

**Files:**
- Modify: `docs/telephony.md`
- Modify: `.env.example`
- Test: `tests/test_smartflo.py`

**Interfaces:**
- Consumes: everything above. Produces no new code interface.

This task proves the one carrier-specific behaviour end to end — that interrupting the agent puts a `clear` on the wire — and writes down what the operator must do in the Tata portal, which no test can cover.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_smartflo.py`:

```python
def test_talking_over_the_agent_puts_a_clear_on_the_wire(smartflo_app):
    """Barge-in on this carrier means `clear`: without it the audio Smartflo has
    already buffered keeps playing after the agent has been cut off."""
    body = smartflo_app.post(
        HANDSHAKE, params={"key": "hook-secret"}, json={"callId": "CA9"}
    ).json()
    token = body["wss_url"].split("token=", 1)[1]

    # Real speech, not silence: the barge-in detector needs sustained voiced audio.
    speech = pcm16_to_ulaw(_tone_16k(600))

    with smartflo_app.websocket_connect(f"/ws/smartflo?token={token}") as ws:
        ws.send_text(_start_frame())

        # Wait until the greeting is genuinely flowing before talking over it.
        for _ in range(40):
            if json.loads(ws.receive_text()).get("event") == "media":
                break

        for offset in range(0, len(speech), 800):  # 100 ms lumps, as Smartflo sends
            chunk = speech[offset : offset + 800]
            ws.send_text(
                json.dumps(
                    {
                        "event": "media",
                        "streamSid": "MZ1",
                        "media": {"payload": base64.b64encode(chunk).decode()},
                    }
                )
            )

        seen = []
        for _ in range(80):
            seen.append(json.loads(ws.receive_text()).get("event"))
            if "clear" in seen:
                break
        assert "clear" in seen, f"barge-in never reached the carrier; saw {seen}"

        ws.send_text(json.dumps({"event": "stop", "streamSid": "MZ1"}))
```

Add to the test file's imports: `import math`, `import struct`, and `from vaani.audio.resample import pcm16_to_ulaw`. Define the tone helper locally rather than importing it from `tests/test_pipeline.py` — `tests/` has no `__init__.py`, so a cross-module test import is not reliable under pytest's rootdir handling:

```python
def _tone_16k(ms: int, freq: float = 220.0, amplitude: float = 0.5) -> bytes:
    n = 16 * ms
    return struct.pack(
        f"<{n}h",
        *(int(amplitude * 32767 * math.sin(2 * math.pi * freq * i / 16000)) for i in range(n)),
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_smartflo.py::test_talking_over_the_agent_puts_a_clear_on_the_wire -v`
Expected: PASS is possible here if Tasks 4 and 7 are both correct — this is an integration check over already-built behaviour rather than new code. If it fails, the failure is the bug: read whether `clear` is missing (transport wiring) or no media ever arrived (socket wiring), and fix that rather than the test.

- [ ] **Step 3: Write the operator documentation**

Append a section to `docs/telephony.md`:

```markdown
## Tata Smartflo (voice bot streaming)

Smartflo is the hosted path: Tata holds the licence and runs the media servers
in India, and Vaani is reached as a WebSocket bot endpoint. Asterisk is not
involved. Audio is G.711 mu-law at 8 kHz, base64 inside JSON.

### What Tata must do (they cannot be done from here)

1. Enable **Channels Hub** on the account.
2. Register the bot under **Settings → Channels → Voice Bot**: name, description,
   and the dynamic endpoint URL below.
3. For inbound, map the Voice Bot to a DID with **Configure Destination** in
   **My Numbers**.

Broadcast and outreach campaigns are not supported by Smartflo streaming, which
suits the inbound-first plan.

### What to configure here

On `/settings`, under Telephony:

| Setting | Value |
|---|---|
| Tata Smartflo voice bot | on |
| Smartflo webhook secret | a long random string; also put it in the endpoint URL |
| Smartflo public host | the public hostname, e.g. `voice.example.in` |
| Smartflo agent profile | which agent answers |

The dynamic endpoint URL given to Tata is:

    https://<your host>/api/telephony/smartflo/handshake?key=<webhook secret>

It answers with `{"success": true, "wss_url": "..."}` and nothing else — Smartflo
refuses any extra key and drops the call — inside its 2000 ms budget.

### Why the secret is not the API token

That URL sits in Tata's configuration and its logs. The webhook secret grants
only the ability to ask for a socket URL, and the per-call token in the reply is
valid for two minutes and can open exactly one call. Neither can reach the
settings API, the knowledge base, or a live call.
```

Add to `.env.example`:

```bash
# --- Tata Smartflo voice bot (hosted PSTN; see docs/telephony.md) -------------
VAANI_SMARTFLO_ENABLED=false
VAANI_SMARTFLO_WEBHOOK_SECRET=
VAANI_SMARTFLO_PUBLIC_HOST=
VAANI_SMARTFLO_AGENT=default
```

- [ ] **Step 4: Run the whole suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS, and still about a second — no test here touches the network.

- [ ] **Step 5: Commit**

```bash
.venv/Scripts/python.exe -m ruff check --fix . && .venv/Scripts/python.exe -m ruff format .
git add tests/test_smartflo.py docs/telephony.md .env.example
git commit -m "Prove barge-in reaches the carrier, and write down the portal steps"
```

---

## Self-review notes

**Spec coverage.** Every section of the design maps to a task: settings → 1; codec parse → 2; codec emit → 3; transport, `clear`, `mark` instrumentation → 4; per-call token → 5; handshake, `OPEN_PATHS`, webhook secret → 6; WebSocket, `start`/`media`/`dtmf`/`stop`, capacity → 7; barge-in integration and the operator steps → 8. The spec's "out of scope" list (outbound, `mark` as load-bearing, per-DID routing, removing AudioSocket) is implemented by no task, which is correct.

**Two places the implementer must verify rather than trust this plan:**

1. `live_settings` in `auth.py` — Task 6 renames a private helper without having read its body. Read it before wiring and pass whatever it actually expects (`scope["app"]` versus `app.state`). The handshake test fails loudly if this is wrong.
2. Task 8 Step 2 may pass immediately, because it integrates behaviour built in Tasks 4 and 7 rather than introducing new code. That is expected and noted in the step; do not manufacture a failure to satisfy the red-green ritual.
