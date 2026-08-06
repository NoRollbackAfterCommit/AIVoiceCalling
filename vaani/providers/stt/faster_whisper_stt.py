"""faster-whisper backed speech recognition.

Whisper is not a streaming model: it wants a complete utterance. The pipeline
therefore uses voice activity detection to cut turns, and calls transcribe() once
per turn. `stream()` emits interim results by re-running on a growing buffer at a
coarse cadence, which costs GPU but gives the supervisor console live captions —
turn it off (interim_interval_s <= 0) on constrained hardware.

Inference is blocking C++, so every call goes to a thread to keep the event loop
free for the other concurrent calls on the box.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator

import numpy as np

from vaani.config import SAMPLE_RATE, SAMPLE_WIDTH
from vaani.core.logging import get_logger
from vaani.providers.base import Transcript

log = get_logger(__name__)


def _pcm_to_float32(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


class FasterWhisperSTT:
    name = "faster_whisper"

    def __init__(
        self,
        model: str = "small",
        device: str = "auto",
        compute_type: str = "default",
        default_language: str | None = None,
        interim_interval_s: float = 0.0,
    ) -> None:
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._default_language = default_language
        self._interim_interval_s = interim_interval_s
        self._model: Any = None
        # One inference at a time per worker. Whisper saturates the GPU already;
        # concurrency comes from running more workers, not more threads.
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        from faster_whisper import WhisperModel

        device = self._device
        compute = self._compute_type
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        if compute == "default":
            compute = "float16" if device == "cuda" else "int8"

        log.info(
            "loading whisper", extra={"model": self._model_name, "device": device, "compute": compute}
        )
        started = time.monotonic()
        self._model = await asyncio.to_thread(
            WhisperModel, self._model_name, device=device, compute_type=compute
        )
        log.info("whisper ready", extra={"load_seconds": round(time.monotonic() - started, 2)})

    def _transcribe_sync(self, audio: np.ndarray, language: str | None) -> Transcript:
        segments, info = self._model.transcribe(
            audio,
            language=language,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,  # stops hallucinated carry-over between turns
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return Transcript(
            text=text,
            is_final=True,
            language=info.language,
            confidence=getattr(info, "language_probability", None),
            duration_s=len(audio) / SAMPLE_RATE,
        )

    async def transcribe(self, pcm: bytes, *, language: str | None = None) -> Transcript:
        if self._model is None:
            raise RuntimeError("FasterWhisperSTT.start() was not awaited")
        if len(pcm) < SAMPLE_RATE * SAMPLE_WIDTH // 10:  # under 100 ms is noise
            return Transcript(text="", is_final=True, duration_s=0.0)
        audio = _pcm_to_float32(pcm)
        async with self._lock:
            return await asyncio.to_thread(
                self._transcribe_sync, audio, language or self._default_language
            )

    async def stream(
        self, audio: AsyncIterator[bytes], *, language: str | None = None
    ) -> AsyncIterator[Transcript]:
        buffer = bytearray()
        last_interim = 0.0
        async for chunk in audio:
            buffer.extend(chunk)
            if self._interim_interval_s <= 0:
                continue
            now = time.monotonic()
            if now - last_interim >= self._interim_interval_s:
                last_interim = now
                partial = await self.transcribe(bytes(buffer), language=language)
                if partial.text:
                    partial.is_final = False
                    yield partial
        yield await self.transcribe(bytes(buffer), language=language)

    async def close(self) -> None:
        self._model = None
