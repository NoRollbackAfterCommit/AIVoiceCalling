"""Saarika realtime streaming contract, offline.

The connection is injected, so the whole protocol — query string, audio framing,
finalisation and every failure path — is exercised with no network.
"""

from __future__ import annotations

import asyncio
import base64
import json
from urllib.parse import parse_qs, urlparse

import pytest

from vaani.config import SAMPLE_RATE
from vaani.providers.stt.sarvam_realtime import SarvamRealtimeSession, build_url


class FakeWS:
    """Records what was sent and replays scripted server messages."""

    def __init__(self, script: list[dict] | None = None, fail_on_send: bool = False) -> None:
        self.sent: list[dict] = []
        self.closed = False
        self._fail_on_send = fail_on_send
        self._inbox: asyncio.Queue = asyncio.Queue()
        for message in script or []:
            self._inbox.put_nowait(json.dumps(message))

    async def send(self, raw: str) -> None:
        if self._fail_on_send:
            raise ConnectionError("socket gone")
        self.sent.append(json.loads(raw))

    async def close(self) -> None:
        self.closed = True

    def push(self, message: dict) -> None:
        self._inbox.put_nowait(json.dumps(message))

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        return await self._inbox.get()


def _session(ws: FakeWS, **kw) -> SarvamRealtimeSession:
    async def connect(url, **_):
        ws.url = url
        return ws

    return SarvamRealtimeSession(api_key="k", connect=connect, **kw)


def _pcm(ms: int) -> bytes:
    return b"\x07\x00" * int(SAMPLE_RATE * ms / 1000)


# -- url ------------------------------------------------------------------


def test_url_declares_the_pipeline_audio_format():
    q = parse_qs(urlparse(build_url(language="hi-IN")).query)
    assert q["encoding"] == ["linear16"]
    assert q["sample_rate"] == [str(SAMPLE_RATE)]


def test_endpointing_is_vad_because_manual_does_not_work():
    """Measured 2026-08-08: the service accepts endpointing=manual and echoes
    turn_detection: manual, then emits no speech events at all and ignores both
    flush and speech_end. This is a constraint, not a preference."""
    q = parse_qs(urlparse(build_url(language="hi-IN")).query)
    assert q["endpointing"] == ["vad"]


def test_the_silence_window_is_short_enough_to_beat_the_batch_endpoint():
    """The service default of 1000 ms is slower than batch transcription.
    Measured end-of-speech to final: 1000 -> 1234 ms, 500 -> 755, 300 -> 577,
    against a batch baseline of 966."""
    q = parse_qs(urlparse(build_url(language="hi-IN")).query)
    assert int(q["silence_duration_ms"][0]) <= 500


def test_the_silence_window_is_configurable():
    q = parse_qs(urlparse(build_url(language="hi-IN", silence_duration_ms=250)).query)
    assert q["silence_duration_ms"] == ["250"]


def test_no_language_means_auto_detect():
    q = parse_qs(urlparse(build_url(language=None)).query)
    assert q["language_code"] == ["auto"]


# -- happy path -----------------------------------------------------------


async def test_audio_is_sent_base64_encoded():
    ws = FakeWS()
    s = _session(ws)
    await s.open()
    try:
        await s.feed(_pcm(20))
        assert ws.sent[0]["event"] == "audio_input"
        assert base64.b64decode(ws.sent[0]["audio"]) == _pcm(20)
    finally:
        await s.close()


async def test_awaiting_a_final_returns_the_transcript():
    ws = FakeWS()
    s = _session(ws)
    await s.open()
    try:
        await s.feed(_pcm(200))
        ws.push({
            "event": "transcript.final",
            "text": "मेरा बिजली का बिल",
            "language": "hi-IN",
            "language_confidence": "0.91",
            "start_s": 0.2,
            "end_s": 1.7,
        })
        result = await s.await_final(timeout=2.0)
        assert result is not None
        assert result.text == "मेरा बिजली का बिल"
        assert result.language == "hi-IN"
        assert result.confidence == pytest.approx(0.91)
        assert result.duration_s == pytest.approx(1.5)
        assert result.is_final is True
        # Finalisation is the service's VAD, not a message we send.
        assert {m["event"] for m in ws.sent} == {"audio_input"}
    finally:
        await s.close()


async def test_partials_are_collected_but_are_not_the_result():
    """Partials drive the live caption. Only finals reach the agent."""
    ws = FakeWS()
    s = _session(ws)
    await s.open()
    try:
        ws.push({"event": "transcript.partial", "text": "मेरा"})
        ws.push({"event": "transcript.partial", "text": "मेरा बिजली"})
        await asyncio.sleep(0.05)
        assert s.partials == ["मेरा", "मेरा बिजली"]
    finally:
        await s.close()


async def test_close_ends_the_session_politely():
    ws = FakeWS()
    s = _session(ws)
    await s.open()
    await s.close()
    assert ws.sent[-1]["event"] == "end"
    assert ws.closed is True


# -- failure paths --------------------------------------------------------


async def test_a_dead_socket_degrades_instead_of_raising():
    """A broken stream must fall back to the batch path, never end the call."""
    ws = FakeWS(fail_on_send=True)
    s = _session(ws)
    await s.open()
    try:
        await s.feed(_pcm(20))          # must not raise
        assert s.failed is True
        assert await s.await_final(timeout=0.2) is None
    finally:
        await s.close()


async def test_a_missing_final_times_out_rather_than_hanging():
    """Waiting forever would strand the caller in silence."""
    ws = FakeWS()
    s = _session(ws)
    await s.open()
    try:
        assert await s.await_final(timeout=0.2) is None
    finally:
        await s.close()


async def test_a_fatal_error_marks_the_session_failed():
    ws = FakeWS()
    s = _session(ws)
    await s.open()
    try:
        ws.push({"event": "error", "code": "quota", "is_fatal": True, "message": "x"})
        await asyncio.sleep(0.05)
        assert s.failed is True
    finally:
        await s.close()


async def test_a_non_fatal_error_leaves_the_session_usable():
    ws = FakeWS()
    s = _session(ws)
    await s.open()
    try:
        ws.push({"event": "error", "code": "noise", "is_fatal": False, "message": "x"})
        await asyncio.sleep(0.05)
        assert s.failed is False
    finally:
        await s.close()


async def test_unparseable_frames_are_ignored():
    ws = FakeWS()
    s = _session(ws)
    await s.open()
    try:
        ws._inbox.put_nowait("not json")
        ws.push({"event": "transcript.final", "text": "ठीक है"})
        result = await s.await_final(timeout=2.0)
        assert result.text == "ठीक है"
    finally:
        await s.close()


async def test_feeding_empty_audio_sends_nothing():
    ws = FakeWS()
    s = _session(ws)
    await s.open()
    try:
        await s.feed(b"")
        assert ws.sent == []
    finally:
        await s.close()
