"""Tata Smartflo voice-bot streaming: codec, transport and endpoints.

Everything here runs against mock providers and a fake Smartflo client, so the
wire format is exercised without a Tata account, a phone line or a network.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import logging
import math
import queue
import struct
import threading
import time

import httpx
import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from vaani.api.smartflo import CALL_TOKEN_TTL_S, mint_call_token, verify_call_token
from vaani.audio.resample import pcm16_to_ulaw
from vaani.config import Settings
from vaani.core.logging import configure_logging
from vaani.main import create_app
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


async def test_agent_audio_goes_out_one_twenty_millisecond_frame_per_message():
    """One media message per 160 bytes, as the carrier itself sends.

    Sarvam yields network-sized HTTP chunks, so the whole buffer in one message
    put two seconds of audio in a single ~22 KB base64 frame — a size no Tata
    document describes, against a carrier that drops a call over one extra key
    in a JSON body. The audio delivered is identical either way, so it goes out
    in the shape the carrier is known to accept.
    """
    sink = _Sink()
    transport = SmartfloTransport(sink, stream_sid="MZ1")

    await transport.send_audio(_pcm_silence_16k(200))  # ten 20 ms frames' worth

    media = sink.of_event("media")
    sizes = [len(base64.b64decode(f["media"]["payload"])) for f in media]
    assert sizes == [ULAW_FRAME_BYTES] * 10, f"expected ten 160-byte frames, got {sizes}"
    # Still allocated in send order, still inside the lock.
    assert [int(f["sequenceNumber"]) for f in media] == list(range(1, 11))


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


# -- Per-call token -------------------------------------------------------------


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


def test_a_token_with_non_ascii_or_unencodable_parts_is_refused_not_raised():
    """`hmac.compare_digest` raises on non-ASCII str and `str.encode` on a lone
    surrogate. A stranger's token must land on None, never on a traceback."""
    assert verify_call_token("s3cret", "AAA.100.é") is None
    assert verify_call_token("s3cret", "AAA.100.\ud800") is None
    assert verify_call_token("s3cret", "\ud800.100.abc") is None
    assert verify_call_token("s3cret", "é.100.abc") is None


# -- Handshake endpoint ---------------------------------------------------------


def _handshake_app(settings):
    """The app the handshake tests drive. Separate from the fixture below so the
    raw-ASGI tests can reach the app object rather than an httpx client."""
    return create_app(
        settings.model_copy(
            update={
                "env": "dev",
                "api_token": "admin-token",
                "smartflo_enabled": True,
                "smartflo_webhook_secret": "hook-secret",
                "smartflo_public_host": "voice.example.in",
            }
        )
    )


@pytest.fixture
async def smartflo_app(settings):
    """The real app, with Smartflo switched on and a known secret.

    The settings go in as an object, not through the environment: `create_app()`
    without one reads the SettingsStore, which layers the developer's untracked
    data/settings.json over the env and would make this fixture behave
    differently on a clean checkout. ASGITransport never runs lifespan, which
    would build that very store and replace the settings passed here. The
    handshake must touch no provider, so none is built: a regression there
    fails loudly on a missing attribute instead of passing against a mock.
    """
    app = _handshake_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


HANDSHAKE = "/api/telephony/smartflo/handshake"


async def test_handshake_returns_exactly_the_two_keys_smartflo_accepts(smartflo_app):
    """Smartflo refuses a body with any extra key, and drops the call."""
    response = await smartflo_app.post(
        HANDSHAKE,
        params={"key": "hook-secret"},
        json={"callId": "CA9", "fromNumber": "+919876543210", "toNumber": "+911140000000"},
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"success", "wss_url"}
    assert body["success"] is True
    assert body["wss_url"].startswith("wss://voice.example.in/ws/smartflo?token=")


async def test_handshake_accepts_a_get_with_query_parameters(smartflo_app):
    response = await smartflo_app.get(
        HANDSHAKE, params={"key": "hook-secret", "callId": "CA9", "fromNumber": "+911"}
    )
    assert response.status_code == 200
    assert response.json()["success"] is True


async def test_handshake_mints_a_token_the_websocket_will_accept(smartflo_app):
    body = (
        await smartflo_app.post(HANDSHAKE, params={"key": "hook-secret"}, json={"callId": "CA9"})
    ).json()
    token = body["wss_url"].split("token=", 1)[1]
    assert verify_call_token("hook-secret", token) == "CA9"


async def test_handshake_refuses_a_wrong_or_missing_secret(smartflo_app):
    assert (await smartflo_app.post(HANDSHAKE, json={"callId": "CA9"})).status_code == 401
    assert (
        await smartflo_app.post(HANDSHAKE, params={"key": "wrong"}, json={"callId": "CA9"})
    ).status_code == 401


async def test_handshake_answers_well_inside_the_two_second_budget(smartflo_app):
    """Smartflo hangs up at 2000 ms. This path must do no I/O at all."""
    started = time.monotonic()
    await smartflo_app.post(HANDSHAKE, params={"key": "hook-secret"}, json={"callId": "CA9"})
    assert (time.monotonic() - started) < 0.5


async def test_the_handshake_is_not_behind_the_admin_bearer_token(smartflo_app):
    """It is reached by Tata's servers, which send no Authorization header."""
    response = await smartflo_app.post(
        HANDSHAKE, params={"key": "hook-secret"}, json={"callId": "CA9"}
    )
    assert response.status_code == 200, "the shared guard must not stand in front of this path"


async def test_handshake_refuses_an_unencodable_secret_instead_of_crashing(smartflo_app):
    """A JSON body can carry a lone surrogate as "\ud800". Before the fix
    `presented.encode()` raised ahead of the comparison, turning a public
    refusal into an unhandled 500."""
    response = await smartflo_app.post(
        HANDSHAKE,
        content=b'{"key": "\ud800", "callId": "CA9"}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 401


async def test_handshake_refuses_an_unencodable_call_id_instead_of_crashing(smartflo_app):
    response = await smartflo_app.post(
        HANDSHAKE,
        params={"key": "hook-secret"},
        content=b'{"callId": "\ud800"}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400
    assert "wss_url" not in response.text


async def test_a_missing_call_id_gets_a_random_id_not_a_shared_one(smartflo_app):
    """A fixed "unknown" would be one credential for every callId-less caller,
    and a real Smartflo callback with no callId must still get a socket."""
    ids = []
    for _ in range(2):
        body = (await smartflo_app.post(HANDSHAKE, params={"key": "hook-secret"}, json={})).json()
        ids.append(verify_call_token("hook-secret", body["wss_url"].split("token=", 1)[1]))
    assert all(ids), "each token must still verify"
    assert "unknown" not in ids
    assert ids[0] != ids[1], "two callId-less handshakes must not share a credential"


def test_the_access_log_redacts_the_handshake_secret():
    """uvicorn logs the query string of every HTTP request. The webhook secret
    is also the HMAC key that mints per-call tokens, so writing it to disk on
    every inbound call would let anyone with the log mint tokens at will."""
    configure_logging("INFO")
    stream = io.StringIO()
    logging.getLogger().handlers[0].setStream(stream)

    secret = "hs-9f3e-topsecret"
    access = logging.getLogger("uvicorn.access")
    for query in (f"key={secret}&callId=CA9", f"callId=CA9&key={secret}"):
        access.info(
            '%s - "%s %s HTTP/%s" %d', "10.0.0.1:1", "POST", f"{HANDSHAKE}?{query}", "1.1", 200
        )

    out = stream.getvalue()
    assert secret not in out
    assert "callId=CA9" in out, "the rest of the line survives"


def test_a_broken_configured_secret_never_verifies():
    """`_sign` encodes the secret. A lone surrogate saved as the secret used to
    raise there, on the socket path, where nothing had guarded it."""
    assert verify_call_token("\ud800", "AAA.100.abc") is None


# -- What the handshake will read before it decides anything --------------------

OVERSIZED = 8 * 1024 * 1024  # far over the ceiling; small enough that a regression is survivable


async def test_a_handshake_body_over_the_ceiling_is_refused_before_it_is_read(settings, asgi_post):
    """This path is exempt from the bearer guard, so a stranger reaches it with
    no credential at all. Reading first meant a POST that declared two gigabytes
    became two gigabytes of RSS on a VM that shares its memory with a government
    website — and a second of parsing on the loop every live call is paced from.
    """
    status, read = await asgi_post(
        _handshake_app(settings),
        HANDSHAKE,
        query="key=hook-secret",
        headers={"content-type": "application/json", "content-length": str(OVERSIZED)},
        body_bytes=OVERSIZED,
    )
    assert status == 413
    assert read == 0, "the body was buffered before its declared size was checked"


async def test_a_handshake_that_declares_no_length_is_refused_before_it_is_read(
    settings, asgi_post
):
    """Chunked transfer is a Content-Length-shaped hole: the size is only known
    once the whole body has arrived, which is the read the ceiling exists to
    prevent. Neither Tata nor a browser needs to stream a handful of fields."""
    status, read = await asgi_post(
        _handshake_app(settings),
        HANDSHAKE,
        query="key=hook-secret",
        headers={"content-type": "application/json", "transfer-encoding": "chunked"},
        body_bytes=OVERSIZED,
    )
    assert status == 411
    assert read == 0, "an undeclared body was buffered anyway"


async def test_a_wrong_secret_in_the_query_is_refused_before_the_body_is_read(settings, asgi_post):
    """The documented endpoint URL carries the secret in the query string, so in
    the normal case the secret can be settled without touching the body at all.
    Under the ceiling on purpose: this is the authenticate-first half, not the
    size half."""
    status, read = await asgi_post(
        _handshake_app(settings),
        HANDSHAKE,
        query="key=wrong",
        headers={"content-type": "application/json", "content-length": "4096"},
        body_bytes=4096,
    )
    assert status == 401
    assert read == 0, "a stranger's body was parsed before their secret was checked"


async def test_a_body_under_the_ceiling_is_still_read(settings, asgi_post):
    """The guard must refuse size, not bodies. A POST whose callId is in the body
    still has to be read, or every real Smartflo callback loses its call id."""
    status, read = await asgi_post(
        _handshake_app(settings),
        HANDSHAKE,
        query="key=hook-secret",
        headers={"content-type": "application/json", "content-length": "2"},
        body_bytes=2,
    )
    assert status == 200
    assert read == 2


async def test_a_malformed_form_body_is_refused_not_a_server_error(smartflo_app):
    """python-multipart raises on a body that is not the form it claims to be.
    Only the JSON branch was guarded, so this escaped as a 500 and a logged
    traceback on a public, unauthenticated path."""
    response = await smartflo_app.post(
        HANDSHAKE,
        content=b"--zz\r\ngarbage\r\n",
        headers={"content-type": "multipart/form-data; boundary=zz"},
    )
    assert response.status_code == 401, "an unparseable form must refuse, not crash"


# -- The call itself, over the WebSocket ----------------------------------------


class _PinnedStore:
    """Stands in for the SettingsStore that lifespan builds.

    `create_app(settings)` alone stops being hermetic once lifespan runs: it
    constructs a SettingsStore, which layers the developer's untracked
    data/settings.json over the environment and *replaces* the settings passed
    in. That file exists on some machines and not on a clean checkout. Pinning
    the store to the object under test is what makes these tests see the same
    settings on both.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings


@contextlib.contextmanager
def _live_app(settings, monkeypatch, tmp_path, **overrides):
    """The real app with lifespan running, so the services and the call manager
    the socket needs exist. httpx's ASGITransport speaks no WebSocket and never
    runs lifespan, hence Starlette's TestClient here and not the fixture above."""
    pinned = settings.model_copy(
        update={
            "env": "dev",
            "api_token": "admin-token",
            "smartflo_enabled": True,
            "smartflo_webhook_secret": "hook-secret",
            "smartflo_public_host": "voice.example.in",
            "database_url": f"sqlite+aiosqlite:///{tmp_path / 'calls.db'}",
            **overrides,
        }
    )
    monkeypatch.setattr("vaani.main.SettingsStore", lambda: _PinnedStore(pinned))
    # Boot also indexes ./knowledge, and the checkout has one. Nothing in these
    # tests should depend on what a developer keeps there.
    monkeypatch.chdir(tmp_path)
    with TestClient(create_app(pinned)) as client:
        yield client


@pytest.fixture
def smartflo_live(settings, monkeypatch, tmp_path):
    with _live_app(settings, monkeypatch, tmp_path) as client:
        yield client


def _call_token(client: TestClient, call_id: str = "CA9") -> str:
    body = client.post(HANDSHAKE, params={"key": "hook-secret"}, json={"callId": call_id}).json()
    return body["wss_url"].split("token=", 1)[1]


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


def _stop_frame(stream_sid: str = "MZ1") -> str:
    return json.dumps({"event": "stop", "streamSid": stream_sid})


# Every wait on a frame is bounded by this. Generous, because the session paces
# playback against the wall clock and CI is slower than a laptop; the number
# matters far less than having one, since an unbounded wait turns a regression
# into a hung job instead of a failure.
FRAME_BUDGET_S = 10.0


def _receive_text_within(ws, budget_s: float, waiting_for: str) -> str:
    """One `ws.receive_text()`, bounded to a wall-clock budget.

    `WebSocketTestSession.receive_text` blocks on an unbounded anyio memory
    stream, so every regression these tests exist to catch — a `clear` that
    never reaches the carrier, a socket the endpoint never closes, a `start`
    deadline that stops firing — would hang CI rather than fail it. A daemon
    thread turns that into something `queue.Queue.get` can time out on; the
    thread is never joined; it dies on its own once the WebSocket session's
    streams close at the end of the `with` block, or is abandoned with the
    process at worst, since it is a daemon.
    """

    def _recv() -> None:
        try:
            box.put(ws.receive_text())
        except BaseException as exc:  # handed to the waiting side, not swallowed
            box.put(exc)

    box: queue.Queue[object] = queue.Queue(maxsize=1)
    threading.Thread(target=_recv, daemon=True).start()
    try:
        outcome = box.get(timeout=budget_s)
    except queue.Empty:
        pytest.fail(f"{waiting_for}: nothing arrived within {budget_s}s")
    if isinstance(outcome, BaseException):
        raise outcome
    return outcome  # type: ignore[return-value]


def _first_media(ws) -> dict:
    """The transport puts only media, mark and clear on the wire, so the
    greeting's first media frame is the first thing to arrive."""
    for _ in range(40):
        raw = _receive_text_within(ws, FRAME_BUDGET_S, "waiting for the agent's first media frame")
        frame = json.loads(raw)
        if frame.get("event") == "media":
            return frame
    pytest.fail("the agent never sent audio back")


def _until_closed(ws) -> None:
    """Drain until the server closes the socket, and fail if it never does.
    The transport leaves the socket open on purpose; closing it is the
    endpoint's job, and Smartflo keeps the line up until it happens.

    The cap only has to sit above one whole greeting, which is now one message
    per 20 ms of audio: a starved loop catches up on pacing by bursting the rest
    of the utterance ahead of the close."""
    for _ in range(4000):
        try:
            _receive_text_within(ws, FRAME_BUDGET_S, "waiting for the endpoint to close the socket")
        except WebSocketDisconnect:
            return
    pytest.fail("the endpoint never closed the socket")


def _expect_close_within(ws, budget_s: float = FRAME_BUDGET_S) -> None:
    """The endpoint must close this socket, and must do it in bounded time."""
    with pytest.raises(WebSocketDisconnect):
        _receive_text_within(ws, budget_s, "waiting for the endpoint to close the socket")


def test_websocket_refuses_a_bad_token(smartflo_live):
    with pytest.raises(WebSocketDisconnect) as refused:
        with smartflo_live.websocket_connect("/ws/smartflo?token=forged"):
            pass
    # 1008 is the route's own refusal; an unrouted path closes with 1000.
    assert refused.value.code == 1008


def test_websocket_refuses_no_token(smartflo_live):
    with pytest.raises(WebSocketDisconnect) as refused:
        with smartflo_live.websocket_connect("/ws/smartflo"):
            pass
    assert refused.value.code == 1008


def test_websocket_refuses_a_genuine_token_while_smartflo_is_off(settings, monkeypatch, tmp_path):
    with _live_app(settings, monkeypatch, tmp_path, smartflo_enabled=False) as client:
        token = mint_call_token("hook-secret", "CA9")
        with pytest.raises(WebSocketDisconnect) as refused:
            with client.websocket_connect(f"/ws/smartflo?token={token}"):
                pass
        assert refused.value.code == 1008


def test_websocket_refuses_when_the_configured_secret_is_broken(settings, monkeypatch, tmp_path):
    """The handshake already guards a secret that cannot be encoded. The socket
    must refuse the same way, not raise inside the handler."""
    with _live_app(settings, monkeypatch, tmp_path, smartflo_webhook_secret="\ud800") as client:
        with pytest.raises(WebSocketDisconnect) as refused:
            with client.websocket_connect("/ws/smartflo?token=AAA.100.abc"):
                pass
        assert refused.value.code == 1008


def test_a_smartflo_call_greets_the_caller(smartflo_live):
    """The whole path: handshake, connect, start, audio back, stop — and then
    the server closes the socket, which the transport cannot do for it."""
    token = _call_token(smartflo_live)
    with smartflo_live.websocket_connect(f"/ws/smartflo?token={token}") as ws:
        ws.send_text(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
        ws.send_text(_start_frame())
        ws.send_text(_media_frame_in())

        media = _first_media(ws)
        payload = base64.b64decode(media["media"]["payload"])
        assert len(payload) % ULAW_FRAME_BYTES == 0
        assert media["streamSid"] == "MZ1"

        ws.send_text(_stop_frame())
        _until_closed(ws)
    assert smartflo_live.app.state.calls.live_count == 0


def test_an_operator_hangup_closes_the_socket(smartflo_live):
    """When the session ends first — the supervisor's hang-up button, or the
    agent saying goodbye — Smartflo has to see the socket close, or the line
    stays up with dead air until the caller gives up."""
    manager = smartflo_live.app.state.calls
    token = _call_token(smartflo_live)
    with smartflo_live.websocket_connect(f"/ws/smartflo?token={token}") as ws:
        ws.send_text(_start_frame())
        _first_media(ws)  # registered and running

        [live] = smartflo_live.portal.call(manager.live)
        assert live["caller_number"] == "+919876543210"
        smartflo_live.portal.call(manager.hangup, live["call_id"])
        _until_closed(ws)
    assert manager.live_count == 0


def test_a_call_past_capacity_is_refused_at_start(settings, monkeypatch, tmp_path):
    """Smartflo's protocol has no 'rejected' frame; closing the socket is the
    only refusal it understands."""
    with _live_app(settings, monkeypatch, tmp_path, max_concurrent_calls=1) as client:
        busy_token = _call_token(client, "CA1")
        with client.websocket_connect(f"/ws/smartflo?token={busy_token}") as busy:
            busy.send_text(_start_frame(stream_sid="MZ1"))
            _first_media(busy)  # the one line is now taken

            spare_token = _call_token(client, "CA2")
            with client.websocket_connect(f"/ws/smartflo?token={spare_token}") as ws:
                ws.send_text(_start_frame(stream_sid="MZ2"))
                _expect_close_within(ws)

            busy.send_text(_stop_frame("MZ1"))
            _until_closed(busy)


def test_the_live_fixture_ignores_a_settings_file_on_disk(settings, monkeypatch, tmp_path):
    """Lifespan builds a SettingsStore from ./data/settings.json. A developer's
    untracked copy that switches Smartflo off, and a clean checkout with none,
    must both leave these tests seeing the fixture's values. This plants the
    hostile version where the store would look and checks the call still opens."""
    hostile = tmp_path / "data" / "settings.json"
    hostile.parent.mkdir()
    hostile.write_text(
        json.dumps(
            {"smartflo_enabled": False, "smartflo_webhook_secret": "other", "api_token": "x"}
        )
    )
    with _live_app(settings, monkeypatch, tmp_path) as client:
        assert client.app.state.settings.smartflo_webhook_secret == "hook-secret"
        with client.websocket_connect(f"/ws/smartflo?token={_call_token(client)}") as ws:
            ws.send_text(_start_frame())
            _first_media(ws)
            ws.send_text(_stop_frame())
            _until_closed(ws)


class _EndpointLog(logging.Handler):
    """Captures the endpoint's own records.

    Not caplog: `create_app` calls `configure_logging`, which replaces the root
    handlers and takes pytest's capture handler with them, so anything logged
    after the app boots would be lost. A handler on the module's logger survives
    that, because `configure_logging` never touches non-root loggers' handlers.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextlib.contextmanager
def _endpoint_log():
    logger = logging.getLogger("vaani.api.smartflo")
    handler = _EndpointLog()
    logger.addHandler(handler)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)


def _wait_for_record(records, predicate, timeout: float = 2.0) -> logging.LogRecord | None:
    """The handler runs on the TestClient's loop thread, so a match may not have
    landed by the time the sending call returns."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for record in list(records):
            if predicate(record):
                return record
        time.sleep(0.02)
    return None


def test_a_socket_that_never_sends_start_is_closed_not_held(settings, monkeypatch, tmp_path):
    """A valid token must not buy an untracked socket.

    Before `start` the call is not registered, so the socket is invisible to the
    concurrency cap, to `live()` and to `drain()`: a peer that connects and then
    says nothing would otherwise hold one open for as long as it liked.
    """
    monkeypatch.setattr("vaani.api.smartflo.START_DEADLINE_S", 0.2)
    with _live_app(settings, monkeypatch, tmp_path) as client:
        token = _call_token(client, "CA-silent")
        with _endpoint_log() as records:
            with client.websocket_connect(f"/ws/smartflo?token={token}") as ws:
                _expect_close_within(ws)  # nothing sent; the deadline must close it
        assert client.app.state.calls.live_count == 0

    warned = _wait_for_record(records, lambda r: "start" in r.getMessage())
    assert warned is not None, "an abandoned socket must say so in the log"
    assert warned.levelno >= logging.WARNING
    # The call id from the token is the only handle an operator has here: no
    # session exists yet, so nothing else ties the socket to a call.
    assert getattr(warned, "token_call_id", None) == "CA-silent"


def test_a_socket_that_leaves_before_start_says_so(smartflo_live):
    """A carrier that hangs up before `start` must not close the log with it.

    Regression for a real outage: Tata's first live call connected and went away
    inside the deadline, so the timeout warning never fired and the only trace
    left was uvicorn's "connection open". Whether the peer waits for the deadline
    or leaves early decides nothing about the operator's need to see it.
    """
    token = _call_token(smartflo_live, "CA-earlyexit")
    with _endpoint_log() as records:
        with smartflo_live.websocket_connect(f"/ws/smartflo?token={token}") as ws:
            # One frame the parser cannot place, then gone — the shape a carrier
            # speaking a dialect we do not parse would leave behind.
            ws.send_text(json.dumps({"event": "hello", "streamSid": "MZ1"}))

    left = _wait_for_record(records, lambda r: "without" in r.getMessage())
    assert left is not None, "a peer that leaves before start must say so in the log"
    assert left.levelno >= logging.WARNING
    assert getattr(left, "token_call_id", None) == "CA-earlyexit"
    # What did arrive is the whole diagnosis: if `start` came in a shape the
    # parser missed, this is the only place it shows up.
    assert getattr(left, "frame_counts", None) == {"invalid": 1}


def test_the_close_code_a_carrier_hangs_up_with_is_recorded(smartflo_live):
    """Why the peer left is the whole diagnosis when it sends nothing.

    Tata's calls connect and close in the same second with `frame_counts: {}`.
    Without the close code there is no way to tell a protocol objection (1002)
    from a policy refusal (1008) from an ordinary goodbye (1000), and each of
    those points somewhere completely different.
    """
    token = _call_token(smartflo_live, "CA-closecode")
    with _endpoint_log() as records:
        with smartflo_live.websocket_connect(f"/ws/smartflo?token={token}") as ws:
            ws.close(code=1002, reason="protocol error")

    left = _wait_for_record(records, lambda r: "without" in r.getMessage())
    assert left is not None, "a peer that leaves before start must say so"
    assert getattr(left, "close_code", None) == 1002
    assert getattr(left, "close_reason", None) == "protocol error"


def test_a_requested_subprotocol_is_echoed_back(smartflo_live):
    """A client that asks for a sub-protocol and is answered with none may hang up.

    Tata's live calls opened the socket and vanished in the same second having
    sent nothing at all — `frame_counts: {}` — which is what a client does when
    the negotiated sub-protocol is not the one it asked for. Echoing what was
    requested costs nothing and removes the possibility.
    """
    token = _call_token(smartflo_live)
    with smartflo_live.websocket_connect(
        f"/ws/smartflo?token={token}", subprotocols=["audio.smartflo.tata"]
    ) as ws:
        assert ws.accepted_subprotocol == "audio.smartflo.tata"


def test_no_subprotocol_requested_means_none_negotiated(smartflo_live):
    """The ordinary case must not start claiming a protocol nobody asked for."""
    token = _call_token(smartflo_live)
    with smartflo_live.websocket_connect(f"/ws/smartflo?token={token}") as ws:
        assert ws.accepted_subprotocol is None


def test_a_start_without_a_stream_id_is_refused_loudly(smartflo_live):
    """No stream id means `send_audio` returns early for the whole call.

    Left to run, that is a registered, connected line with dead air on it and
    nothing in the log to explain why. It is refused instead, so the slot goes
    back and the cause is on the record.
    """
    token = _call_token(smartflo_live)
    with _endpoint_log() as records:
        with smartflo_live.websocket_connect(f"/ws/smartflo?token={token}") as ws:
            ws.send_text(
                json.dumps(
                    {
                        "event": "start",
                        "start": {
                            "callSid": "CA9",
                            "from": "+919876543210",
                            "to": "+911140000000",
                            "direction": "inbound",
                        },
                    }
                )
            )
            complained = _wait_for_record(
                records,
                lambda r: r.levelno >= logging.ERROR and "streamSid" in r.getMessage(),
            )
            assert complained is not None, "a silent line must be loud in the log"
            _expect_close_within(ws)

    assert smartflo_live.app.state.calls.live_count == 0


def test_frames_the_parser_cannot_read_are_counted_not_discarded_in_silence(smartflo_live):
    """The likeliest first-live-call failure, and the hardest to diagnose.

    If Tata's media arrives in a shape `parse_frame` does not match, every frame
    comes back `invalid`, the caller talks into silence, and the idle hangup ends
    the call ninety seconds later with a log containing the greeting and nothing
    else. One count is the whole diagnosis — and at ten frames a second it has to
    be a count, not nine hundred lines.
    """
    token = _call_token(smartflo_live)
    strange = json.dumps({"type": "audio", "data": "body-marker-that-must-not-be-logged"})

    with _endpoint_log() as records:
        with smartflo_live.websocket_connect(f"/ws/smartflo?token={token}") as ws:
            ws.send_text(_start_frame())
            _first_media(ws)
            for _ in range(30):
                ws.send_text(strange)
            ws.send_text(_stop_frame())
            _until_closed(ws)

        summary = _wait_for_record(records, lambda r: "did not act on" in r.getMessage())

    assert summary is not None, "a stream nothing could read must reach the log"
    assert summary.levelno >= logging.WARNING, "an operator watching at INFO must see it"
    assert getattr(summary, "frame_counts", {}).get("invalid") == 30

    sightings = [r for r in records if "does not recognise" in r.getMessage()]
    assert len(sightings) == 1, f"thirty frames must not write thirty lines; wrote {len(sightings)}"
    # The keys are the diagnosis: "data,type" is not "event,media".
    assert getattr(sightings[0], "frame_shape", "") == "data,type"
    assert not any(
        "body-marker" in str(record.__dict__)
        for record in records
        if record.levelno > logging.DEBUG
    ), "the raw frame body must not be logged above debug"


def test_frames_seen_while_waiting_for_start_are_reported_with_the_timeout(
    settings, monkeypatch, tmp_path
):
    """`_await_start` drops everything that is not start or stop. If Tata's start
    frame is shaped differently the operator's only clue was a socket that said
    nothing; what did arrive is the diagnosis, so it goes in the same warning."""
    monkeypatch.setattr("vaani.api.smartflo.START_DEADLINE_S", 1.0)
    with _live_app(settings, monkeypatch, tmp_path) as client:
        token = _call_token(client, "CA-odd-start")
        with _endpoint_log() as records:
            with client.websocket_connect(f"/ws/smartflo?token={token}") as ws:
                ws.send_text(json.dumps({"event": "connected", "protocol": "Call"}))
                for _ in range(5):
                    ws.send_text(json.dumps({"type": "start", "streamId": "MZ1"}))
                _expect_close_within(ws)

    warned = _wait_for_record(records, lambda r: "no start frame" in r.getMessage())
    assert warned is not None, "an abandoned socket must say so in the log"
    counts = getattr(warned, "frame_counts", {})
    assert counts.get("invalid") == 5, f"the frames that did arrive must be on the record: {counts}"
    assert counts.get("connected") == 1


# -- Barge-in reaching the carrier -----------------------------------------------


def _tone_16k(ms: int, freq: float = 220.0, amplitude: float = 0.5) -> bytes:
    n = 16 * ms
    return struct.pack(
        f"<{n}h",
        *(int(amplitude * 32767 * math.sin(2 * math.pi * freq * i / 16000)) for i in range(n)),
    )


def _receive_event_within(ws, budget_s: float, seen: list[str]) -> str:
    """The name of the next frame's event, bounded the same way."""
    raw = _receive_text_within(ws, budget_s, f"waiting for a frame; saw {seen}")
    return json.loads(raw).get("event")


def test_talking_over_the_agent_puts_a_clear_on_the_wire(smartflo_live):
    """Barge-in on this carrier means `clear`: without it the audio Smartflo has
    already buffered keeps playing after the agent has been cut off."""
    token = _call_token(smartflo_live)

    # Real speech, not silence: the barge-in detector needs sustained voiced audio.
    speech = pcm16_to_ulaw(_tone_16k(600))

    with smartflo_live.websocket_connect(f"/ws/smartflo?token={token}") as ws:
        ws.send_text(_start_frame())

        # Wait until the greeting is genuinely flowing before talking over it.
        _first_media(ws)

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

        seen: list[str] = []
        for _ in range(80):
            seen.append(_receive_event_within(ws, 2.0, seen))
            if "clear" in seen:
                break
        assert "clear" in seen, f"barge-in never reached the carrier; saw {seen}"

        ws.send_text(_stop_frame())
        _until_closed(ws)
