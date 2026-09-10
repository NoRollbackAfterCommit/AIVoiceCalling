"""Adapts a CallSession to a Tata Smartflo voice-bot WebSocket.

Smartflo speaks mu-law at 8 kHz; the pipeline speaks PCM16 at 16 kHz. Conversion
happens here and nowhere else, exactly as it does for Asterisk.

Nothing in the session changes for this transport. The two carrier-specific
behaviours ride on events the session already emits: `barge_in` becomes `clear`,
which empties what Smartflo still has buffered for the caller, and `speech_end`
becomes `mark`, whose echo is a genuine playback-complete signal. The echo is
only measured for now — the session keeps pacing playback against the wall clock,
because adopting a carrier's completion signal before a single live call would be
optimising against a document.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

from vaani.audio.resample import pcm16_to_ulaw
from vaani.core.logging import get_logger
from vaani.telephony.smartflo_frames import (
    ULAW_FRAME_BYTES,
    clear_frame,
    mark_frame,
    media_frame,
)

log = get_logger(__name__)

Sender = Callable[[str], Awaitable[None]]


class SmartfloTransport:
    def __init__(self, sender: Sender, stream_sid: str = "") -> None:
        self._send_raw = sender
        self.stream_sid = stream_sid
        self._open = True
        # One writer at a time, held across the socket await: taking a sequence
        # number and putting that frame on the wire is one step. Release between
        # them and a `clear` can reach the wire ahead of the media it was meant to
        # flush whenever the socket stalls on a frame; the carrier then sees
        # sequence numbers out of order, and what it does with the stale audio is
        # its call, not ours. Per transport, so one call never waits on another.
        self._lock = asyncio.Lock()
        # Agent audio arrives in lumps that do not divide by 160. Smartflo refuses
        # a short frame, so the remainder waits here for the next one.
        self._pending = bytearray()
        self._seq = 0
        self._marks = 0
        self._mark_sent_at: dict[str, float] = {}

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    async def _send(self, frame: str) -> None:
        if not self._open:
            return
        try:
            await self._send_raw(frame)
        except Exception:
            self._open = False

    async def send_audio(self, pcm: bytes) -> None:
        if not self._open or not self.stream_sid:
            return
        async with self._lock:
            self._pending.extend(pcm16_to_ulaw(pcm))
            usable = (len(self._pending) // ULAW_FRAME_BYTES) * ULAW_FRAME_BYTES
            if not usable:
                return
            chunk = bytes(self._pending[:usable])
            del self._pending[:usable]
            await self._send(media_frame(self.stream_sid, chunk, self._next_seq()))

    async def send_event(self, event: dict[str, Any]) -> None:
        kind = event.get("type")

        if kind == "barge_in" and self.stream_sid:
            async with self._lock:
                # Ours as well as theirs: a flush followed by our own stale
                # remainder would put the interrupted sentence back on the line.
                self._pending.clear()
                await self._send(clear_frame(self.stream_sid, self._next_seq()))
        elif kind == "speech_end" and self.stream_sid:
            async with self._lock:
                self._marks += 1
                name = f"utt-{self._marks}"
                # Stamped just before the send, not before the wait for the lock,
                # so a queued media frame does not inflate the measured round trip.
                self._mark_sent_at[name] = time.monotonic()
                await self._send(mark_frame(self.stream_sid, name, self._next_seq()))

        if kind in ("transcript", "transfer", "call_end", "barge_in"):
            log.info(
                "call event",
                extra={
                    "event_type": kind,
                    "payload": json.dumps(
                        {k: v for k, v in event.items() if k != "record"}, default=str
                    )[:500],
                },
            )

    def note_mark(self, name: str | None) -> None:
        """Smartflo echoing a mark means the caller has heard up to that point.

        Recorded, not acted on. The round trip is the measurement that decides
        whether to make this the playback-complete signal instead of pacing.
        """
        sent = self._mark_sent_at.pop(name or "", None)
        if sent is None:
            return
        log.info(
            "playback confirmed",
            extra={"mark": name, "round_trip_ms": int((time.monotonic() - sent) * 1000)},
        )

    async def close(self) -> None:
        self._open = False
