"""Piper text to speech.

Piper is the right default for this platform: it is fast enough to run on CPU
next to the SIP gateway, ships permissively licensed voices for Indian and
European languages, and has no network dependency — which is what makes an
on-premise government deployment possible at all.

Voices are .onnx + .onnx.json pairs under `voices_dir`, one per language. Piper
emits at its own sample rate (usually 22050); we resample to the pipeline's
16 kHz here so nothing downstream has to care.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np

from vaani.audio.resample import resample_pcm16
from vaani.config import FRAME_SAMPLES, SAMPLE_RATE
from vaani.core.logging import get_logger

log = get_logger(__name__)

_CHUNK_SAMPLES = FRAME_SAMPLES * 10  # 200 ms of 16 kHz audio


class PiperTTS:
    name = "piper"

    def __init__(
        self,
        voices_dir: str = "./models/piper",
        default_voice: str = "en_US-lessac-medium",
        speed: float = 1.0,
    ) -> None:
        self._dir = Path(voices_dir)
        self._default_voice = default_voice
        self._speed = speed
        self._voices: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    def _load(self, voice: str) -> Any:
        from piper import PiperVoice

        model = self._dir / f"{voice}.onnx"
        if not model.exists():
            raise FileNotFoundError(
                f"Piper voice {voice!r} not found at {model}. "
                f"Download voices with scripts/fetch_voices.sh"
            )
        log.info("loading piper voice", extra={"voice": voice})
        return PiperVoice.load(str(model))

    async def start(self) -> None:
        # Preload the default so the first caller does not pay the load cost.
        self._voices[self._default_voice] = await asyncio.to_thread(
            self._load, self._default_voice
        )

    async def _voice_for(self, voice: str | None) -> Any:
        key = voice or self._default_voice
        if key not in self._voices:
            async with self._lock:
                if key not in self._voices:
                    self._voices[key] = await asyncio.to_thread(self._load, key)
        return self._voices[key]

    def _synth_sync(self, engine: Any, text: str) -> bytes:
        native_rate = engine.config.sample_rate
        chunks: list[np.ndarray] = []
        # length_scale > 1 slows speech down; invert so `speed` reads naturally.
        for audio in engine.synthesize_stream_raw(
            text, length_scale=1.0 / max(self._speed, 0.1)
        ):
            chunks.append(np.frombuffer(audio, dtype=np.int16))
        if not chunks:
            return b""
        pcm = np.concatenate(chunks).tobytes()
        return resample_pcm16(pcm, native_rate, SAMPLE_RATE)

    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        if not text.strip():
            return b""
        engine = await self._voice_for(voice)
        return await asyncio.to_thread(self._synth_sync, engine, text)

    async def stream(self, text: str, *, voice: str | None = None) -> AsyncIterator[bytes]:
        # Piper synthesises a whole sentence at a time, so split on sentence
        # boundaries: the caller hears sentence one while sentence two renders.
        for sentence in _split_sentences(text):
            pcm = await self.synthesize(sentence, voice=voice)
            step = _CHUNK_SAMPLES * 2
            for offset in range(0, len(pcm), step):
                yield pcm[offset : offset + step]

    async def close(self) -> None:
        self._voices.clear()


def _split_sentences(text: str) -> list[str]:
    import re

    parts = re.split(r"(?<=[.!?।])\s+", text.strip())  # । is the Devanagari danda
    return [p for p in parts if p.strip()]
