"""OpenAI hosted speech and embedding providers.

The OpenAI *chat* model is already reachable through `openai_compat` — it is the
protocol that client speaks — so only the pieces with their own endpoint shape
live here: transcription, speech synthesis and embeddings.

Everything crossing into the pipeline is converted to its 16 kHz PCM16 contract
at this boundary: Whisper wants a container format rather than raw samples, and
the speech endpoint emits 24 kHz.
"""

from __future__ import annotations

import io
import wave
from collections.abc import AsyncIterator
from typing import Any, ClassVar

import httpx

from vaani.audio.resample import resample_pcm16
from vaani.config import FRAME_SAMPLES, SAMPLE_RATE
from vaani.core.logging import get_logger
from vaani.providers.base import Transcript

log = get_logger(__name__)

_OPENAI_TTS_RATE = 24_000
_CHUNK_SAMPLES = FRAME_SAMPLES * 10  # 200 ms


def _client(api_key: str, base_url: str, timeout: float) -> httpx.AsyncClient:
    if not api_key:
        raise ValueError("An OpenAI API key is required. Set it in Settings.")
    return httpx.AsyncClient(
        base_url=base_url.rstrip("/"),
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=httpx.Timeout(timeout, connect=5.0),
        limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
    )


def _to_wav(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes:
    """Whisper needs a recognised container, not bare samples."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


# ---------------------------------------------------------------------------


class OpenAISTT:
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "whisper-1",
        base_url: str = "https://api.openai.com/v1",
        language: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._language = language
        self._timeout = timeout_s
        self._http: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self._http = _client(self._api_key, self._base_url, self._timeout)

    async def transcribe(self, pcm: bytes, *, language: str | None = None) -> Transcript:
        if self._http is None:
            raise RuntimeError("OpenAISTT.start() was not awaited")
        duration = len(pcm) / (SAMPLE_RATE * 2)
        if duration < 0.15:  # under 150 ms is noise, not speech
            return Transcript(text="", is_final=True, duration_s=duration)

        data: dict[str, Any] = {"model": self._model, "response_format": "json"}
        if lang := (language or self._language):
            data["language"] = lang

        response = await self._http.post(
            "/audio/transcriptions",
            files={"file": ("audio.wav", _to_wav(pcm), "audio/wav")},
            data=data,
        )
        response.raise_for_status()
        payload = response.json()
        return Transcript(
            text=(payload.get("text") or "").strip(),
            is_final=True,
            language=payload.get("language") or language or self._language,
            duration_s=duration,
        )

    async def stream(
        self, audio: AsyncIterator[bytes], *, language: str | None = None
    ) -> AsyncIterator[Transcript]:
        buffer = bytearray()
        async for chunk in audio:
            buffer.extend(chunk)
        yield await self.transcribe(bytes(buffer), language=language)

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None


# ---------------------------------------------------------------------------


class OpenAITTS:
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "tts-1",
        voice: str = "alloy",
        base_url: str = "https://api.openai.com/v1",
        speed: float = 1.0,
        timeout_s: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._voice = voice
        self._base_url = base_url
        self._speed = speed
        self._timeout = timeout_s
        self._http: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self._http = _client(self._api_key, self._base_url, self._timeout)

    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        if self._http is None:
            raise RuntimeError("OpenAITTS.start() was not awaited")
        if not text.strip():
            return b""
        response = await self._http.post(
            "/audio/speech",
            json={
                "model": self._model,
                "input": text,
                "voice": voice or self._voice,
                # Raw PCM rather than mp3: decoding a container per utterance
                # would add latency the caller hears as a pause.
                "response_format": "pcm",
                "speed": self._speed,
            },
        )
        response.raise_for_status()
        return resample_pcm16(response.content, _OPENAI_TTS_RATE, SAMPLE_RATE)

    async def stream(self, text: str, *, voice: str | None = None) -> AsyncIterator[bytes]:
        pcm = await self.synthesize(text, voice=voice)
        step = _CHUNK_SAMPLES * 2
        for offset in range(0, len(pcm), step):
            yield pcm[offset : offset + step]

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None


# ---------------------------------------------------------------------------


class OpenAIEmbedding:
    name = "openai"
    # Normalised cosine scores from these models sit on the same scale as BGE.
    similarity_floor = 0.25

    _DIMS: ClassVar[dict[str, int]] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        timeout_s: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._timeout = timeout_s
        self._http: httpx.AsyncClient | None = None
        self.dim = self._DIMS.get(model, 1536)

    async def start(self) -> None:
        self._http = _client(self._api_key, self._base_url, self._timeout)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self._http is None:
            raise RuntimeError("OpenAIEmbedding.start() was not awaited")
        if not texts:
            return []
        response = await self._http.post("/embeddings", json={"model": self._model, "input": texts})
        response.raise_for_status()
        payload = response.json()
        # The API does not promise input order; index is authoritative.
        rows = sorted(payload["data"], key=lambda d: d["index"])
        vectors = [row["embedding"] for row in rows]
        if vectors:
            self.dim = len(vectors[0])
        return vectors

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None
