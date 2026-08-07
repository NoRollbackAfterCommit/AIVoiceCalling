"""Sarvam Saarika speech to text.

Chosen over Whisper for Indian deployments because it is trained on code-mixed
Indian speech. A caller saying "मेरा electricity bill pending hai" is the normal
case on an Indian helpline, not an edge case, and engines configured for a single
language degrade badly on it. Saarika also covers od-IN, which most vendors do
not.

Transcription is per completed utterance rather than streaming: TurnDetector has
already decided where the turn ended, so a streaming socket would add complexity
without removing latency from the critical path.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from vaani.config import SAMPLE_RATE
from vaani.core.logging import get_logger
from vaani.providers.base import Transcript

log = get_logger(__name__)

# Below this, the buffer is a click, a breath or line noise. Sending it wastes a
# round trip and these models return confident nonsense for near-silence.
_MIN_UTTERANCE_S = 0.15


class SarvamSTT:
    name = "sarvam"

    def __init__(
        self,
        api_key: str,
        model: str = "saaras:v3",
        base_url: str = "https://api.sarvam.ai",
        language: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._language = language
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
        log.info("sarvam stt ready", extra={"model": self._model})

    async def transcribe(self, pcm: bytes, *, language: str | None = None) -> Transcript:
        if self._http is None:
            raise RuntimeError("SarvamSTT.start() was not awaited")

        duration = len(pcm) / (SAMPLE_RATE * 2)
        if duration < _MIN_UTTERANCE_S:
            return Transcript(text="", is_final=True, duration_s=duration)

        data: dict[str, Any] = {"model": self._model}
        # Omitting language_code lets Saarika auto-detect, which is what makes a
        # caller switching language mid-call work at all.
        if lang := (language or self._language):
            data["language_code"] = lang

        response = await self._http.post(
            "/speech-to-text",
            files={"file": ("audio.wav", _to_wav(pcm), "audio/wav")},
            data=data,
        )
        response.raise_for_status()
        payload = response.json()
        return Transcript(
            text=(payload.get("transcript") or "").strip(),
            is_final=True,
            language=payload.get("language_code") or language or self._language,
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
        http, self._http = self._http, None
        if http is not None:
            await http.aclose()


def _to_wav(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes:
    """The API wants a recognised container, not bare samples."""
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return buffer.getvalue()
