"""Voice activity detection and end-of-turn segmentation.

This module decides *when the caller has stopped talking*, which is the single
biggest determinant of whether a voice agent feels natural or maddening. Cut too
early and you talk over someone drawing breath mid-sentence; cut too late and
every exchange has a dead second in it.

Two detectors, one interface:

  WebRTCVAD    — the webrtcvad C extension. Trained on telephony, cheap, good.
  EnergyVAD    — adaptive noise-floor gate. No dependency, degrades gracefully
                 on a noisy line, and is the automatic fallback.

`TurnDetector` wraps whichever is available with hangover logic: a turn ends only
after `silence_ms` of continuous non-speech *following* real speech, so a pause
between words does not end the turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from vaani.audio.resample import rms_dbfs
from vaani.config import FRAME_BYTES, FRAME_MS, SAMPLE_RATE
from vaani.core.logging import get_logger

log = get_logger(__name__)


class VAD(Protocol):
    def is_speech(self, frame: bytes) -> bool: ...
    def reset(self) -> None: ...


class EnergyVAD:
    """Adaptive gate: tracks the quietest recent frames as the noise floor and
    calls anything sufficiently above it speech.

    The margin is deliberately wide. This is the fallback for when webrtcvad is
    missing, and a narrow one lets room hum, an air conditioner and a second
    person talking behind the caller all read as speech — the agent then answers
    the room instead of the caller. Speech on a phone line sits well clear of
    18 dB above the floor; background noise mostly does not.
    """

    def __init__(self, margin_db: float = 18.0, floor_db: float = -50.0) -> None:
        self._margin = margin_db
        self._initial_floor = floor_db
        self._floor = floor_db

    def is_speech(self, frame: bytes) -> bool:
        level = rms_dbfs(frame)
        speech = bool(level > self._floor + self._margin)
        # Track the floor downward fast and upward slowly, so a sudden loud room
        # does not permanently desensitise the detector.
        if not speech:
            self._floor = min(self._floor + 0.05, max(level, -70.0))
        return speech

    def reset(self) -> None:
        self._floor = self._initial_floor


class WebRTCVAD:
    """webrtcvad. Aggressiveness 0 (permissive) to 3 (aggressive); 2 is the
    right trade-off for a phone line with background noise."""

    def __init__(self, aggressiveness: int = 2) -> None:
        import webrtcvad

        self._vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, frame: bytes) -> bool:
        # webrtcvad only accepts exactly 10/20/30 ms frames at 8/16/32/48 kHz.
        if len(frame) != FRAME_BYTES:
            return False
        try:
            return self._vad.is_speech(frame, SAMPLE_RATE)
        except Exception:  # malformed frame; treat as silence
            return False

    def reset(self) -> None:
        return None


def build_vad(aggressiveness: int = 2) -> VAD:
    try:
        return WebRTCVAD(aggressiveness)
    except ImportError:
        log.info("webrtcvad unavailable, using energy VAD")
        return EnergyVAD()


class TurnState:
    IDLE = "idle"        # nothing heard yet
    SPEAKING = "speaking"  # caller is mid-utterance
    ENDED = "ended"      # utterance complete, ready to transcribe


@dataclass(slots=True)
class TurnEvent:
    state: str
    audio: bytes = b""
    duration_s: float = 0.0


class TurnDetector:
    """Feeds on 20 ms frames, emits a completed utterance."""

    def __init__(
        self,
        vad: VAD | None = None,
        silence_ms: int = 700,
        min_speech_ms: int = 200,
        max_utterance_s: float = 30.0,
        pre_roll_ms: int = 300,
    ) -> None:
        self._vad = vad or build_vad()
        self._silence_frames = max(1, silence_ms // FRAME_MS)
        self._min_speech_frames = max(1, min_speech_ms // FRAME_MS)
        self._max_frames = int(max_utterance_s * 1000 / FRAME_MS)
        # Frames kept from before speech was detected. VAD always trips a frame
        # or two late, and without pre-roll Whisper loses the first consonant.
        self._pre_roll_frames = max(0, pre_roll_ms // FRAME_MS)

        self._buffer: list[bytes] = []
        self._pre_roll: list[bytes] = []
        self._speech_frames = 0
        self._silence_run = 0
        self._state = TurnState.IDLE

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state == TurnState.SPEAKING

    def push(self, frame: bytes) -> TurnEvent | None:
        """Returns a TurnEvent when an utterance completes, else None."""
        speech = self._vad.is_speech(frame)

        if self._state == TurnState.IDLE:
            if speech:
                self._state = TurnState.SPEAKING
                self._buffer = [*self._pre_roll, frame]
                self._pre_roll.clear()
                self._speech_frames = 1
                self._silence_run = 0
                return TurnEvent(state=TurnState.SPEAKING)
            self._pre_roll.append(frame)
            if len(self._pre_roll) > self._pre_roll_frames:
                self._pre_roll.pop(0)
            return None

        # SPEAKING
        self._buffer.append(frame)
        if speech:
            self._speech_frames += 1
            self._silence_run = 0
        else:
            self._silence_run += 1

        too_long = len(self._buffer) >= self._max_frames
        settled = self._silence_run >= self._silence_frames

        if not (settled or too_long):
            return None

        audio = b"".join(self._buffer)
        enough_speech = self._speech_frames >= self._min_speech_frames
        self._reset_buffers()

        if not enough_speech:
            # A cough or a door slam. Drop it rather than sending noise to STT.
            return None
        return TurnEvent(
            state=TurnState.ENDED,
            audio=audio,
            duration_s=len(audio) / (SAMPLE_RATE * 2),
        )

    def _reset_buffers(self) -> None:
        self._buffer = []
        self._pre_roll = []
        self._speech_frames = 0
        self._silence_run = 0
        self._state = TurnState.IDLE

    def reset(self) -> None:
        self._reset_buffers()
        self._vad.reset()


class BargeInDetector:
    """Watches for the caller talking over the agent.

    Requires `trigger_ms` of *continuous* speech before firing. A single noisy
    frame — a keyboard, a car horn, the agent's own voice leaking back through a
    speakerphone — must not cut the agent off mid-sentence.
    """

    def __init__(self, trigger_ms: int = 240, vad: VAD | None = None) -> None:
        self._vad = vad or build_vad(aggressiveness=3)  # strict while agent speaks
        self._needed = max(1, trigger_ms // FRAME_MS)
        self._run = 0

    def push(self, frame: bytes) -> bool:
        if self._vad.is_speech(frame):
            self._run += 1
            if self._run >= self._needed:
                self._run = 0
                return True
        else:
            self._run = 0
        return False

    def reset(self) -> None:
        self._run = 0
        self._vad.reset()


def iter_frames(pcm: bytes, frame_bytes: int = FRAME_BYTES):
    """Split a buffer into exact frames, discarding any trailing partial."""
    for offset in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
        yield pcm[offset : offset + frame_bytes]
