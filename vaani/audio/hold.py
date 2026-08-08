"""Comfort audio for the gap between a caller finishing and the agent replying.

A phone line that goes completely silent reads as a dropped call. Callers say
"hello? hello?" into the gap, which the turn detector then treats as a fresh
utterance, and the agent ends up answering a question nobody asked. A quiet tone
under the pause tells them the line is alive and buys the pipeline its thinking
time without changing anything about how fast it actually is.

Generated rather than shipped as an asset, so the base install stays free of
binary files and this works in an air-gapped deployment with nothing to fetch.
It is deliberately unobtrusive: a soft two-note figure well below speech level,
because the caller is waiting, not being entertained.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Iterator

from vaani.config import FRAME_SAMPLES, SAMPLE_RATE

# Well under speech level. Loud hold music over a pause is worse than silence.
_AMPLITUDE = 0.06
# A minor third, low in the register so it sits under a voice rather than
# competing with it.
_NOTES = (294.0, 349.0)
_NOTE_S = 1.1
_CHUNK_SAMPLES = FRAME_SAMPLES * 10  # 200 ms


def _sample(n: int) -> float:
    t = n / SAMPLE_RATE
    note = _NOTES[int(t / _NOTE_S) % len(_NOTES)]
    # Fade each note in and out so the change of pitch does not click.
    phase_in_note = (t % _NOTE_S) / _NOTE_S
    envelope = math.sin(math.pi * phase_in_note) ** 2
    return _AMPLITUDE * envelope * math.sin(2 * math.pi * note * t)


def hold_loop() -> Iterator[bytes]:
    """Yield 200 ms PCM16 chunks forever. The caller cancels it."""
    period = int(SAMPLE_RATE * _NOTE_S * len(_NOTES))
    n = 0
    while True:
        values = [int(32767 * _sample((n + i) % period)) for i in range(_CHUNK_SAMPLES)]
        yield struct.pack(f"<{_CHUNK_SAMPLES}h", *values)
        n = (n + _CHUNK_SAMPLES) % period
