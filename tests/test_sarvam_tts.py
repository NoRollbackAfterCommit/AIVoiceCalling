"""Sarvam TTS contract tests, offline.

Synthesis goes through /text-to-speech/stream asking for linear16 at
SAMPLE_RATE, so the bytes arriving are already the pipeline's format: no WAV
container to parse and no resampling. What matters here is that the first chunk
leaves early and that no chunk ever splits a sample.
"""

from __future__ import annotations

import json

import httpx
import pytest

from vaani.config import SAMPLE_RATE
from vaani.providers.base import TTSProvider
from vaani.providers.tts.sarvam import SarvamTTS, _aligned


def _stub(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.sarvam.ai",
        headers={"api-subscription-key": "k"},
    )


def _pcm(n_samples: int) -> bytes:
    return b"\x11\x22" * n_samples


async def _iter(*chunks: bytes):
    for c in chunks:
        yield c


# -- chunk alignment -------------------------------------------------------


async def test_alignment_never_splits_a_sample():
    """A transport may split anywhere. Half a sample on the wire is an audible
    click, so odd bytes carry into the next chunk."""
    out = [c async for c in _aligned(_iter(b"\x01", b"\x02\x03", b"\x04"))]
    assert all(len(c) % 2 == 0 for c in out)
    assert b"".join(out) == b"\x01\x02\x03\x04"


async def test_alignment_preserves_every_byte_of_an_even_stream():
    source = [_pcm(100), _pcm(37), _pcm(1)]
    out = [c async for c in _aligned(_iter(*source))]
    assert b"".join(out) == b"".join(source)


async def test_a_trailing_half_sample_is_dropped_not_emitted():
    out = [c async for c in _aligned(_iter(b"\x01\x02", b"\x03"))]
    assert b"".join(out) == b"\x01\x02"


async def test_alignment_emits_nothing_for_an_empty_stream():
    assert [c async for c in _aligned(_iter())] == []


# -- provider --------------------------------------------------------------


async def test_satisfies_the_protocol():
    assert isinstance(SarvamTTS(api_key="k"), TTSProvider)


async def test_stream_requests_pipeline_format_from_the_streaming_endpoint():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=_pcm(1600))

    tts = SarvamTTS(api_key="k")
    tts._http = _stub(handler)
    [c async for c in tts.stream("नमस्ते", voice="hi-IN:priya")]

    assert seen["path"] == "/text-to-speech/stream"
    body = seen["body"]
    # The streaming endpoint names these differently from the REST one:
    # `text` rather than `inputs`, `language_code` rather than
    # `target_language_code`. Getting it wrong is a 400.
    assert body["text"] == "नमस्ते"
    assert body["language_code"] == "hi-IN"
    assert body["speaker"] == "priya"
    assert body["speech_sample_rate"] == SAMPLE_RATE
    assert body["output_audio_codec"] == "linear16"
    assert "inputs" not in body and "target_language_code" not in body


async def test_streamed_chunks_are_sample_aligned_and_complete():
    payload = _pcm(4000)
    tts = SarvamTTS(api_key="k")
    tts._http = _stub(lambda r: httpx.Response(200, content=payload))

    chunks = [c async for c in tts.stream("एक दो तीन")]
    assert chunks
    assert all(len(c) % 2 == 0 for c in chunks)
    assert b"".join(chunks) == payload


async def test_synthesize_accumulates_the_stream():
    payload = _pcm(2000)
    tts = SarvamTTS(api_key="k")
    tts._http = _stub(lambda r: httpx.Response(200, content=payload))
    assert await tts.synthesize("नमस्ते") == payload


async def test_a_bare_voice_name_keeps_the_hindi_default():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, content=_pcm(10))

    tts = SarvamTTS(api_key="k", default_voice="priya")
    tts._http = _stub(handler)
    await tts.synthesize("नमस्ते")

    assert seen["speaker"] == "priya"
    assert seen["language_code"] == "hi-IN"


async def test_empty_text_produces_no_audio_and_no_request():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not call the API for empty text")

    tts = SarvamTTS(api_key="k")
    tts._http = _stub(handler)
    assert await tts.synthesize("   ") == b""
    assert [c async for c in tts.stream("")] == []


async def test_an_api_error_is_raised_not_silently_empty():
    """A 400 from a bad speaker must not read as a successful silent turn."""
    tts = SarvamTTS(api_key="k")
    tts._http = _stub(lambda r: httpx.Response(400, json={"error": {"message": "bad speaker"}}))
    with pytest.raises(httpx.HTTPStatusError):
        [c async for c in tts.stream("नमस्ते")]


async def test_start_without_api_key_fails_loudly():
    with pytest.raises(ValueError, match="API key"):
        await SarvamTTS(api_key="").start()
