"""Tata Smartflo voice-bot streaming: codec, transport and endpoints.

Everything here runs against mock providers and a fake Smartflo client, so the
wire format is exercised without a Tata account, a phone line or a network.
"""

from __future__ import annotations

import asyncio
import base64
import json

from vaani.config import Settings
from vaani.telephony.smartflo import SmartfloTransport
from vaani.telephony.smartflo_frames import (
    ULAW_FRAME_BYTES,
    SmartfloEvent,
    clear_frame,
    mark_frame,
    media_frame,
    parse_frame,
)


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
    # A deeply nested payload overflows json.loads' call stack with RecursionError,
    # not JSONDecodeError — a carrier flooding one bad frame must not kill the call.
    assert parse_frame("[" * 100_000 + "]" * 100_000).kind == "invalid"


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


class _StallingSink(_Sink):
    """A sender whose first frame stalls, as a socket under backpressure does.

    Counts overlapping sends: a transport that hands frame N+1 to the sender while
    N is still in flight has already lost sequence order, whatever the sender then
    does with them.
    """

    def __init__(self) -> None:
        super().__init__()
        self.in_flight = 0
        self.max_in_flight = 0
        self._stalled = False

    async def __call__(self, raw: str) -> None:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        if not self._stalled:
            self._stalled = True
            await asyncio.sleep(0.02)
        self.in_flight -= 1
        await super().__call__(raw)


async def test_a_barge_in_cannot_overtake_a_media_frame_still_in_flight():
    sink = _StallingSink()
    transport = SmartfloTransport(sink, stream_sid="MZ1")

    audio = asyncio.create_task(transport.send_audio(_pcm_silence_16k(20)))
    await asyncio.sleep(0)  # far enough for the media frame to be stalled inside the sender
    await transport.send_event({"type": "barge_in"})
    await audio

    assert sink.max_in_flight == 1, "a second frame was handed over while the first was in flight"
    seqs = [int(f["sequenceNumber"]) for f in sink.frames]
    assert seqs == [1, 2], f"frames reached the sender out of sequence order: {seqs}"
