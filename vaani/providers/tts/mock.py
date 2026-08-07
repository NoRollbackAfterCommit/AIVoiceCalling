"""Synthetic speech that is audible but not intelligible.

Generates a soft formant-ish tone whose length tracks the text, so a demo call
has real audio flowing at real timing: you can hear the agent start, hear it get
cut off by barge-in, and measure time-to-first-audio — all without a voice model.
"""

from __future__ import annotations

import asyncio
import math
import struct
from collections.abc import AsyncIterator

from vaani.config import FRAME_SAMPLES, SAMPLE_RATE

# Roughly natural speaking rate, used to size the output.
_CHARS_PER_SECOND = 14.0
_CHUNK_SAMPLES = FRAME_SAMPLES * 10  # 200 ms


class MockTTS:
    name = "mock"

    def __init__(self, voice: str = "mock-neutral") -> None:
        self.voice = voice

    async def start(self) -> None:
        return None

    def _render(self, text: str) -> bytes:
        seconds = max(0.35, len(text) / _CHARS_PER_SECOND)
        total = int(seconds * SAMPLE_RATE)
        # Two detuned partials plus a slow amplitude wobble reads as "speech-like"
        # rather than a test tone, which makes barge-in demos less confusing.
        samples = []
        for n in range(total):
            t = n / SAMPLE_RATE
            envelope = 0.5 + 0.5 * math.sin(2 * math.pi * 3.1 * t)
            value = 0.18 * envelope * (
                math.sin(2 * math.pi * 165 * t) + 0.4 * math.sin(2 * math.pi * 330 * t)
            )
            # Fade the edges so chunk boundaries do not click.
            if n < 240:
                value *= n / 240
            elif n > total - 240:
                value *= max(0, (total - n) / 240)
            samples.append(int(value * 32767))
        return struct.pack(f"<{len(samples)}h", *samples)

    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        await asyncio.sleep(0.03)
        return self._render(text)

    async def stream(self, text: str, *, voice: str | None = None) -> AsyncIterator[bytes]:
        pcm = await self.synthesize(text, voice=voice)
        step = _CHUNK_SAMPLES * 2
        for offset in range(0, len(pcm), step):
            yield pcm[offset : offset + step]

    async def close(self) -> None:
        return None
