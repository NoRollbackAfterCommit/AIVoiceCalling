"""Comfort audio under the thinking pause.

A silent line reads as a dropped call: callers say "hello?" into the gap, the
turn detector treats that as a fresh utterance, and the agent answers a question
nobody asked.
"""

from __future__ import annotations

import asyncio
import struct
from dataclasses import replace

import pytest

from vaani.audio.hold import hold_loop
from vaani.config import FRAME_BYTES, SAMPLE_RATE, Settings
from vaani.core.registry import build_services
from vaani.pipeline.session import CallSession, CallState

from .test_pipeline import FakeTransport, silence, tone


def _peak(pcm: bytes) -> float:
    n = len(pcm) // 2
    return max(abs(v) for v in struct.unpack(f"<{n}h", pcm)) / 32767


def test_chunks_are_sample_aligned_and_a_sensible_size():
    loop = hold_loop()
    chunk = next(loop)
    assert len(chunk) % 2 == 0
    assert len(chunk) == FRAME_BYTES * 10          # 200 ms
    assert len(chunk) / 2 / SAMPLE_RATE == pytest.approx(0.2)


def test_it_stays_well_below_speech_level():
    """Loud hold music over a pause is worse than silence."""
    loop = hold_loop()
    peak = max(_peak(next(loop)) for _ in range(20))
    assert 0.0 < peak < 0.15, f"peak {peak:.3f} is too loud to sit under a voice"


def test_it_actually_produces_sound():
    loop = hold_loop()
    assert max(_peak(next(loop)) for _ in range(10)) > 0.01


def test_it_loops_without_running_out():
    loop = hold_loop()
    total = sum(len(next(loop)) for _ in range(200))   # 40 s
    assert total == FRAME_BYTES * 10 * 200


@pytest.fixture
def settings() -> Settings:
    return Settings(
        stt_provider="mock", llm_provider="mock", tts_provider="mock",
        vector_store="memory", embedding_provider="hash", record_calls=False,
        end_of_turn_silence_ms=200, idle_prompt_after_s=120, idle_hangup_after_s=600,
    )


@pytest.fixture
async def services(settings: Settings):
    svc = build_services(settings)
    svc.profiles["default"] = replace(svc.profiles["default"], ask_language=False)
    await svc.start()
    yield svc
    await svc.close()


async def test_hold_plays_while_thinking_and_stops_when_the_agent_speaks(services):
    transport = FakeTransport()
    session = CallSession(transport, services)
    task = asyncio.create_task(session.run())
    await asyncio.sleep(0.3)

    before = len(transport.audio)
    for chunk in (tone(600), silence(500)):
        for i in range(0, len(chunk) - FRAME_BYTES + 1, FRAME_BYTES):
            await session.push_audio(chunk[i : i + FRAME_BYTES])
    await asyncio.sleep(1.2)

    assert len(transport.audio) > before, "the caller heard nothing during the pause"
    # Whatever happened, the hold task must not outlive the turn.
    assert session._hold is None or session._hold.done()

    await session.hangup()
    await asyncio.wait_for(task, timeout=10)


async def test_hold_is_cancelled_when_the_call_ends(services):
    transport = FakeTransport()
    session = CallSession(transport, services)
    task = asyncio.create_task(session.run())
    await asyncio.sleep(0.3)

    session._start_hold()
    assert session._hold is not None
    await session.hangup()
    await asyncio.wait_for(task, timeout=10)

    assert session._hold is None
    assert session.state == CallState.ENDED
