"""Saarika realtime speech to text over a WebSocket.

The batch endpoint only starts working once the caller has stopped talking, so
its whole cost lands in the silence the caller is sitting in — measured at about
840 ms of dead air per turn. Streaming spends that time while they are still
speaking, and the final transcript arrives shortly after they stop.

**Endpointing is server-side, and that is a constraint rather than a choice.**
`endpointing=manual` is documented and the service accepts it — the session echo
reports `turn_detection: manual` — but it then emits no speech events at all and
ignores both `flush` and `speech_end`. Measured 2026-08-08. So turn segmentation
comes from the service's own VAD, and `silence_duration_ms` is the knob that
decides how long a caller's pause may be before the turn is closed.

That makes it the second turn detector on the call, which is a real design
tension: `end_of_turn_silence_ms` in Settings and this value must be kept
consistent or the two will disagree about when the caller finished. Wiring this
into CallSession therefore means deciding which of the two is authoritative —
see docs/superpowers/specs for that decision.

Audio must be sent in chunks of about 100 ms. Measured: 20 ms frames produce no
speech events whatsoever, silently.

A session is per call, not per turn: the handshake costs a round trip, and paying
it on every utterance would give back most of what streaming buys.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from typing import Any
from urllib.parse import urlencode

from vaani.config import SAMPLE_RATE
from vaani.core.logging import get_logger
from vaani.providers.base import Transcript

log = get_logger(__name__)

_WS_URL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"
_MODEL = "saaras:v3-realtime"


def build_url(
    *,
    language: str | None,
    model: str = _MODEL,
    sample_rate: int = SAMPLE_RATE,
    silence_duration_ms: int = 300,
) -> str:
    """`auto` rather than a fixed language: a caller switching mid-call is the
    normal case on an Indian helpline, and code-mixing is why Saarika is here."""
    query = urlencode(
        {
            "language_code": language or "auto",
            "model": model,
            "encoding": "linear16",
            "sample_rate": sample_rate,
            # Manual endpointing is accepted but non-functional. See the module
            # docstring; this is not a preference.
            "endpointing": "vad",
            "mode": "transcribe",
            # The default of 1000 ms makes streaming slower than the batch
            # endpoint. Measured end-of-speech to final: 1000 ms -> 1234 ms,
            # 500 -> 755, 300 -> 577, against a batch baseline of 966.
            "silence_duration_ms": silence_duration_ms,
        }
    )
    return f"{_WS_URL}?{query}"


class SarvamRealtimeSession:
    """One streaming recognition session, spanning a whole call."""

    def __init__(
        self,
        api_key: str,
        *,
        language: str | None = None,
        model: str = _MODEL,
        sample_rate: int = SAMPLE_RATE,
        silence_duration_ms: int = 300,
        connect: Any = None,
    ) -> None:
        self._api_key = api_key
        self._language = language
        self._model = model
        self._sample_rate = sample_rate
        self._silence_duration_ms = silence_duration_ms
        # Injectable so the contract can be tested without a network.
        self._connect = connect
        self._ws: Any = None
        self._reader: asyncio.Task[None] | None = None
        self._finals: asyncio.Queue[Transcript] = asyncio.Queue()
        self.partials: list[str] = []
        self.failed = False

    async def open(self) -> None:
        connect = self._connect
        if connect is None:
            from websockets.asyncio.client import connect as ws_connect

            connect = ws_connect

        url = build_url(
            language=self._language,
            model=self._model,
            sample_rate=self._sample_rate,
            silence_duration_ms=self._silence_duration_ms,
        )
        self._ws = await connect(
            url, additional_headers={"API-SUBSCRIPTION-KEY": self._api_key}
        )
        self._reader = asyncio.create_task(self._read(), name="saarika-reader")
        log.info("saarika realtime open", extra={"model": self._model})

    async def feed(self, pcm: bytes) -> None:
        """Send caller audio. Never raises: a dead socket must degrade the call
        to the batch path, not end it."""
        if self._ws is None or self.failed or not pcm:
            return
        try:
            await self._ws.send(
                json.dumps(
                    {"event": "audio_input", "audio": base64.b64encode(pcm).decode()}
                )
            )
        except Exception:
            self.failed = True
            log.warning("saarika realtime send failed; falling back to batch")

    async def await_final(self, timeout: float = 3.0) -> Transcript | None:
        """Wait for the service to finalise the current utterance.

        Finalisation is triggered by the service's own VAD after
        `silence_duration_ms` of quiet, not by us — see the module docstring.
        Keep feeding audio (including silence) while waiting, or it never fires.

        Returns None on timeout or failure, which the caller reads as "use the
        batch path instead" — the buffered audio is still there.
        """
        if self._ws is None or self.failed:
            return None
        try:
            return await asyncio.wait_for(self._finals.get(), timeout=timeout)
        except TimeoutError:
            log.warning("saarika realtime final timed out", extra={"timeout_s": timeout})
            return None
        except Exception:
            self.failed = True
            log.warning("saarika realtime finalisation failed")
            return None

    async def close(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.send(json.dumps({"event": "end"}))
            with contextlib.suppress(Exception):
                await ws.close()
        if self._reader is not None:
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader
            self._reader = None

    async def _read(self) -> None:
        try:
            async for raw in self._ws:
                self._handle(raw)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.failed = True
            log.warning("saarika realtime reader stopped")

    def _handle(self, raw: str | bytes) -> None:
        try:
            message = json.loads(raw)
        except (ValueError, TypeError):
            return
        event = message.get("event")

        if event == "transcript.partial":
            # Partials drive the live caption only; the agent never sees them.
            if text := (message.get("text") or "").strip():
                self.partials.append(text)

        elif event == "transcript.final":
            self._finals.put_nowait(
                Transcript(
                    text=(message.get("text") or "").strip(),
                    is_final=True,
                    language=message.get("language"),
                    confidence=_as_float(message.get("language_confidence")),
                    duration_s=_span(message),
                )
            )

        elif event == "error":
            # Non-fatal errors are per-utterance; a fatal one kills the session.
            if message.get("is_fatal"):
                self.failed = True
            log.warning(
                "saarika realtime error",
                extra={"code": message.get("code"), "fatal": message.get("is_fatal")},
            )


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _span(message: dict[str, Any]) -> float:
    start, end = message.get("start_s"), message.get("end_s")
    try:
        return max(0.0, float(end) - float(start))
    except (TypeError, ValueError):
        return 0.0
