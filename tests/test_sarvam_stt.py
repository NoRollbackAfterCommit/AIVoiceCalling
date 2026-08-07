"""Sarvam STT contract tests.

Exercised against a stubbed transport rather than the live API: the default
suite must stay offline. Real-audio accuracy is covered by the golden set in
tests/test_golden_audio.py, which is marked and opt-in.
"""

from __future__ import annotations

import httpx
import pytest

from vaani.config import SAMPLE_RATE
from vaani.providers.base import STTProvider
from vaani.providers.stt.sarvam import SarvamSTT


def _pcm(ms: int) -> bytes:
    return b"\x01\x00" * int(SAMPLE_RATE * ms / 1000)


def _stub(handler) -> httpx.AsyncClient:
    """Mirrors the client start() builds, minus the network."""
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.sarvam.ai",
        headers={"api-subscription-key": "k"},
    )


async def test_satisfies_the_protocol():
    assert isinstance(SarvamSTT(api_key="k"), STTProvider)


async def test_transcribes_and_reports_language():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["api-subscription-key"] == "k"
        return httpx.Response(
            200, json={"transcript": "बिजली का बिल", "language_code": "hi-IN"}
        )

    stt = SarvamSTT(api_key="k")
    stt._http = _stub(handler)
    result = await stt.transcribe(_pcm(800))

    assert result.text == "बिजली का बिल"
    assert result.language == "hi-IN"
    assert result.is_final is True
    assert result.duration_s == pytest.approx(0.8, rel=0.01)


async def test_short_audio_is_not_sent_to_the_api():
    """Under 150 ms is a click or a breath. Sending it wastes a round trip
    and returns hallucinated text."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not call the API for sub-threshold audio")

    stt = SarvamSTT(api_key="k")
    stt._http = _stub(handler)
    result = await stt.transcribe(_pcm(80))

    assert result.text == ""
    assert result.is_final is True


async def test_missing_transcript_field_yields_empty_text_not_a_crash():
    stt = SarvamSTT(api_key="k")
    stt._http = _stub(lambda r: httpx.Response(200, json={}))
    result = await stt.transcribe(_pcm(500))
    assert result.text == ""


async def test_stream_buffers_then_yields_one_final():
    """TurnDetector already segments utterances, so streaming reduces to a
    single transcription of the completed turn."""

    stt = SarvamSTT(api_key="k")
    stt._http = _stub(
        lambda r: httpx.Response(200, json={"transcript": "हाँ", "language_code": "hi-IN"})
    )

    async def audio():
        yield _pcm(300)
        yield _pcm(300)

    results = [t async for t in stt.stream(audio())]
    assert len(results) == 1
    assert results[0].is_final is True
    assert results[0].text == "हाँ"


async def test_start_without_api_key_fails_loudly():
    with pytest.raises(ValueError, match="API key"):
        await SarvamSTT(api_key="").start()


async def test_start_sends_the_key_as_a_subscription_header():
    """Sarvam authenticates on api-subscription-key, not a bearer token."""
    stt = SarvamSTT(api_key="secret-key")
    await stt.start()
    try:
        assert stt._http.headers["api-subscription-key"] == "secret-key"
        assert str(stt._http.base_url) == "https://api.sarvam.ai"
    finally:
        await stt.close()
