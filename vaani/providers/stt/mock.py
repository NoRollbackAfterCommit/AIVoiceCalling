"""Deterministic STT for development, CI and load tests.

It does not recognise speech. It reports how much audio it was given and returns
a canned line, which is enough to exercise turn detection, barge-in, the agent
loop, tool calling and TTS without a GPU in the room.
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import AsyncIterator

from vaani.config import SAMPLE_RATE, SAMPLE_WIDTH
from vaani.providers.base import Transcript

# Cycled so a scripted demo call walks through a plausible conversation.
_SCRIPT = [
    "Hello, I want to check my electricity bill.",
    "My consumer number is nine one two three four five.",
    "When is the last date to pay?",
    "Can you register a complaint about voltage fluctuation?",
    "Thank you, that is all.",
]


class MockSTT:
    name = "mock"

    def __init__(self, script: list[str] | None = None) -> None:
        self._script = script or _SCRIPT
        self._cycle = itertools.cycle(self._script)

    async def start(self) -> None:  # pragma: no cover - nothing to load
        return None

    async def transcribe(self, pcm: bytes, *, language: str | None = None) -> Transcript:
        duration = len(pcm) / (SAMPLE_RATE * SAMPLE_WIDTH)
        # Silence in, silence out: keeps the turn machine honest under test.
        if duration < 0.15:
            return Transcript(
                text="", is_final=True, language=language or "en", duration_s=duration
            )
        await asyncio.sleep(0.05)  # stand in for inference latency
        return Transcript(
            text=next(self._cycle),
            is_final=True,
            language=language or "en",
            confidence=0.95,
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
        return None
