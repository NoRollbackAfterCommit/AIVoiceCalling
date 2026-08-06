"""Sample rate and codec conversion at the edges of the pipeline.

Telephony hands us 8 kHz mu-law or A-law; Piper emits 22.05 kHz; browsers emit
whatever the sound card felt like. Everything is normalised to 16 kHz PCM16 here
so the pipeline only ever sees one format.

Linear interpolation is used rather than a windowed-sinc filter: for speech that
is already band-limited to 8 kHz the aliasing is inaudible over a phone line, and
this runs with no scipy dependency.
"""

from __future__ import annotations

import numpy as np

# audioop was removed from the stdlib in Python 3.13; the `audioop-lts` backport
# provides the identical module. Only the G.711 codecs and rms() are used.
try:
    import audioop  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - 3.13+
    import audioop_lts as audioop  # type: ignore[import-not-found, no-redef]


def resample_pcm16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    if src_rate == dst_rate or not pcm:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16)
    if samples.size == 0:
        return b""
    out_len = int(round(samples.size * dst_rate / src_rate))
    if out_len <= 0:
        return b""
    src_idx = np.linspace(0, samples.size - 1, num=out_len, dtype=np.float64)
    resampled = np.interp(src_idx, np.arange(samples.size), samples.astype(np.float64))
    return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()


def ulaw_to_pcm16(data: bytes, src_rate: int = 8000, dst_rate: int = 16000) -> bytes:
    """G.711 mu-law (North American / most SIP trunks) to pipeline PCM."""
    return resample_pcm16(audioop.ulaw2lin(data, 2), src_rate, dst_rate)


def pcm16_to_ulaw(pcm: bytes, src_rate: int = 16000, dst_rate: int = 8000) -> bytes:
    return audioop.lin2ulaw(resample_pcm16(pcm, src_rate, dst_rate), 2)


def alaw_to_pcm16(data: bytes, src_rate: int = 8000, dst_rate: int = 16000) -> bytes:
    """G.711 A-law (European / Indian PSTN)."""
    return resample_pcm16(audioop.alaw2lin(data, 2), src_rate, dst_rate)


def pcm16_to_alaw(pcm: bytes, src_rate: int = 16000, dst_rate: int = 8000) -> bytes:
    return audioop.lin2alaw(resample_pcm16(pcm, src_rate, dst_rate), 2)


def stereo_to_mono(pcm: bytes) -> bytes:
    return audioop.tomono(pcm, 2, 0.5, 0.5)


def rms_dbfs(pcm: bytes) -> float:
    """Loudness in dBFS. Used for the level meter and for silence heuristics.

    Returns a builtin float, not np.float64: comparisons on the numpy scalar
    yield np.bool_, which is truthy but is not `True`, and that difference has a
    habit of surfacing as a baffling bug three layers away.
    """
    if not pcm:
        return -96.0
    rms = audioop.rms(pcm, 2)
    if rms <= 0:
        return -96.0
    return float(20.0 * np.log10(rms / 32768.0))
