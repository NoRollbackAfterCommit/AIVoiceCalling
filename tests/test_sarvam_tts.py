"""Sarvam TTS contract tests, offline."""

from __future__ import annotations

import base64
import io
import json
import struct
import wave

import httpx
import pytest

from vaani.config import SAMPLE_RATE
from vaani.providers.base import TTSProvider
from vaani.providers.tts.sarvam import SarvamTTS


def _wav_b64(seconds: float = 0.5, rate: int = 22050) -> str:
    """A silent WAV at a rate deliberately different from SAMPLE_RATE, so the
    test proves the provider resamples rather than passing bytes through."""
    n = int(rate * seconds)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(struct.pack(f"<{n}h", *([0] * n)))
    return base64.b64encode(buffer.getvalue()).decode()


def _stub(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.sarvam.ai",
        headers={"api-subscription-key": "k"},
    )


async def test_satisfies_the_protocol():
    assert isinstance(SarvamTTS(api_key="k"), TTSProvider)


async def test_synthesize_returns_pcm_at_the_pipeline_rate():
    tts = SarvamTTS(api_key="k")
    tts._http = _stub(lambda r: httpx.Response(200, json={"audios": [_wav_b64(0.5, 22050)]}))
    pcm = await tts.synthesize("नमस्ते")

    # 0.5 s at SAMPLE_RATE, 2 bytes per sample, within resampler rounding.
    assert len(pcm) == pytest.approx(SAMPLE_RATE * 2 * 0.5, rel=0.02)
    assert len(pcm) % 2 == 0


async def test_voice_and_language_reach_the_request():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"audios": [_wav_b64()]})

    tts = SarvamTTS(api_key="k")
    tts._http = _stub(handler)
    await tts.synthesize("नमस्ते", voice="hi-IN:meera")

    assert seen["speaker"] == "meera"
    assert seen["target_language_code"] == "hi-IN"


async def test_stream_yields_frame_aligned_chunks():
    tts = SarvamTTS(api_key="k")
    tts._http = _stub(lambda r: httpx.Response(200, json={"audios": [_wav_b64(1.0, 22050)]}))
    chunks = [c async for c in tts.stream("एक दो तीन")]

    assert chunks, "expected at least one chunk"
    # Misaligned chunks put half a sample on the wire and click audibly.
    assert all(len(c) % 2 == 0 for c in chunks)
    assert sum(len(c) for c in chunks) == pytest.approx(SAMPLE_RATE * 2, rel=0.02)


async def test_empty_text_produces_no_audio_and_no_request():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not call the API for empty text")

    tts = SarvamTTS(api_key="k")
    tts._http = _stub(handler)
    assert await tts.synthesize("   ") == b""
    assert [c async for c in tts.stream("")] == []


async def test_a_bare_voice_name_keeps_the_hindi_default():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"audios": [_wav_b64()]})

    tts = SarvamTTS(api_key="k", default_voice="anushka")
    tts._http = _stub(handler)
    await tts.synthesize("नमस्ते")

    assert seen["speaker"] == "anushka"
    assert seen["target_language_code"] == "hi-IN"


async def test_no_audio_in_the_response_is_survivable():
    """An empty result must leave the caller in silence for one turn, not
    tear the call down."""
    tts = SarvamTTS(api_key="k")
    tts._http = _stub(lambda r: httpx.Response(200, json={"audios": []}))
    assert await tts.synthesize("नमस्ते") == b""
