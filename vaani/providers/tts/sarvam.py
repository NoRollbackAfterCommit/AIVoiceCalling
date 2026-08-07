"""Sarvam Bulbul text to speech.

Bulbul is the reason this provider exists: OpenAI's and most Western vendors'
voices cannot pronounce Indian languages, and an English-accented model reading
Devanagari is unusable on a citizen helpline. Bulbul covers all eleven of the
languages this platform targets, including Odia.

Output is chunked to roughly 200 ms and resampled to SAMPLE_RATE here, so the
pipeline continues to see exactly one audio format.
"""

from __future__ import annotations

import base64
import io
import wave
from collections.abc import AsyncIterator
from typing import Any

from vaani.audio.resample import resample_pcm16
from vaani.config import FRAME_SAMPLES, SAMPLE_RATE
from vaani.core.logging import get_logger

log = get_logger(__name__)

_CHUNK_BYTES = FRAME_SAMPLES * 10 * 2  # 200 ms of PCM16


class SarvamTTS:
    name = "sarvam"

    def __init__(
        self,
        api_key: str,
        model: str = "bulbul:v3",
        default_voice: str = "anushka",
        base_url: str = "https://api.sarvam.ai",
        speed: float = 1.0,
        timeout_s: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._default_voice = default_voice
        self._base_url = base_url
        self._speed = speed
        self._timeout = timeout_s
        self._http: Any = None

    async def start(self) -> None:
        import httpx

        if not self._api_key:
            raise ValueError("A Sarvam API key is required. Set it in Settings.")
        self._http = httpx.AsyncClient(
            base_url=self._base_url.rstrip("/"),
            headers={"api-subscription-key": self._api_key},
            timeout=httpx.Timeout(self._timeout, connect=5.0),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
        )
        log.info("sarvam tts ready", extra={"model": self._model})

    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        if not text.strip():
            return b""
        if self._http is None:
            raise RuntimeError("SarvamTTS.start() was not awaited")

        language, speaker = _split_voice(voice or self._default_voice)
        response = await self._http.post(
            "/text-to-speech",
            json={
                "inputs": [text],
                "target_language_code": language,
                "speaker": speaker,
                "model": self._model,
                "pace": self._speed,
            },
        )
        response.raise_for_status()
        audios = response.json().get("audios") or []
        if not audios:
            # One silent turn is recoverable; an exception here would drop a
            # live call over a transient synthesis failure.
            log.warning("sarvam tts returned no audio", extra={"chars": len(text)})
            return b""
        return _wav_to_pipeline_pcm(base64.b64decode(audios[0]))

    async def stream(self, text: str, *, voice: str | None = None) -> AsyncIterator[bytes]:
        pcm = await self.synthesize(text, voice=voice)
        for start in range(0, len(pcm), _CHUNK_BYTES):
            yield pcm[start : start + _CHUNK_BYTES]

    async def close(self) -> None:
        http, self._http = self._http, None
        if http is not None:
            await http.aclose()


def _split_voice(voice: str) -> tuple[str, str]:
    """Voices are configured as "hi-IN:meera" so one profile field carries both
    the language and the speaker. A bare name keeps the Hindi default."""
    if ":" in voice:
        language, speaker = voice.split(":", 1)
        return language, speaker
    return "hi-IN", voice


def _wav_to_pipeline_pcm(raw: bytes) -> bytes:
    with wave.open(io.BytesIO(raw), "rb") as wav:
        rate = wav.getframerate()
        pcm = wav.readframes(wav.getnframes())
    return resample_pcm16(pcm, rate, SAMPLE_RATE) if rate != SAMPLE_RATE else pcm
