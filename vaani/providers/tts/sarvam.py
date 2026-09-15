"""Sarvam Bulbul text to speech.

Bulbul is the reason this provider exists: OpenAI's and most Western vendors'
voices cannot pronounce Indian languages, and an English-accented model reading
Devanagari is unusable on a citizen helpline. Bulbul covers all eleven of the
languages this platform targets, including Odia.

Synthesis goes through the streaming endpoint, not the REST one, and that choice
is the whole point of this module. On a phone call the metric is time to *first*
audio, not total render time: measured against the live API, rendering a short
Hindi sentence in full took about 1.9 seconds, which is dead air the caller sits
through before hearing a syllable. Streaming lets playback start while the rest
is still being generated, and lets barge-in cancel mid-sentence.

Two things make this cheaper than it looks. The endpoint accepts
`speech_sample_rate`, so audio arrives already at SAMPLE_RATE with no resampling,
and `output_audio_codec=linear16` means raw PCM16 with no WAV container to parse.

Note the streaming endpoint does not share the REST endpoint's field names: it
takes `text` rather than `inputs`, and `language_code` rather than
`target_language_code`. Sending the REST shape here is a 400.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from vaani.config import SAMPLE_RATE
from vaani.core.logging import get_logger

log = get_logger(__name__)

# One attempt plus two retries. Measured against production on 2026-09-15:
# fifty simultaneous calls drew twenty-one 429s from this endpoint, and each one
# became a caller holding a live line in silence — the session logs "no audio
# produced for a reply" and moves on. Thirty concurrent calls was clean, so the
# burst that breaks it is short, and riding it out costs under a second.
# Bounded, because a caller listening to dead air is worse than a turn that
# fails and recovers.
_MAX_ATTEMPTS = 3


class SarvamTTS:
    name = "sarvam"

    def __init__(
        self,
        api_key: str,
        model: str = "bulbul:v3",
        default_voice: str = "priya",
        base_url: str = "https://api.sarvam.ai",
        speed: float = 1.0,
        timeout_s: float = 30.0,
        retry_backoff_s: float = 0.25,
    ) -> None:
        self._retry_backoff = retry_backoff_s
        self._api_key = api_key
        self._model = model
        self._default_voice = default_voice
        self._base_url = base_url
        self._speed = speed
        self._timeout = timeout_s
        self._http: Any = None

    async def start(self) -> None:
        import httpx

        if not self._api_key:
            raise ValueError("A Sarvam API key is required. Set it in Settings.")
        self._http = httpx.AsyncClient(
            base_url=self._base_url.rstrip("/"),
            headers={"api-subscription-key": self._api_key},
            # No overall read timeout on synthesis: the point of streaming is a
            # long-lived response body. connect is what should fail fast.
            timeout=httpx.Timeout(self._timeout, connect=5.0),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
        )
        log.info("sarvam tts ready", extra={"model": self._model})

    async def stream(self, text: str, *, voice: str | None = None) -> AsyncIterator[bytes]:
        if not text.strip():
            return
        if self._http is None:
            raise RuntimeError("SarvamTTS.start() was not awaited")

        language, speaker = _split_voice(voice or self._default_voice)
        payload = {
            "text": text,
            "language_code": language,
            "speaker": speaker,
            "model": self._model,
            "pace": self._speed,
            "speech_sample_rate": SAMPLE_RATE,
            "output_audio_codec": "linear16",
        }

        for attempt in range(_MAX_ATTEMPTS):
            async with self._http.stream(
                "POST", "/text-to-speech/stream", json=payload
            ) as response:
                # Only the rate limit is worth retrying: a 400 from a bad speaker
                # fails identically every time, and retrying it spends the
                # caller's silence on a certainty.
                if response.status_code == 429 and attempt < _MAX_ATTEMPTS - 1:
                    await response.aread()
                    wait = _retry_after(response)
                    if wait is None:
                        wait = self._retry_backoff * (2**attempt)
                    log.warning(
                        "sarvam tts rate limited; retrying",
                        extra={"attempt": attempt + 1, "wait_s": round(wait, 2)},
                    )
                    await asyncio.sleep(wait)
                    continue
                if response.status_code != 200:
                    # The body has to be read before the message is useful, and a
                    # silent empty turn would hide a bad speaker or language code.
                    await response.aread()
                    response.raise_for_status()
                async for chunk in _aligned(response.aiter_bytes()):
                    yield chunk
                return

    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        """Whole utterance at once. Correct, but it forfeits the latency win —
        prefer stream() anywhere a caller is waiting."""
        buffer = bytearray()
        async for chunk in self.stream(text, voice=voice):
            buffer.extend(chunk)
        return bytes(buffer)

    async def close(self) -> None:
        http, self._http = self._http, None
        if http is not None:
            await http.aclose()


async def _aligned(source: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """Re-cut an arbitrary byte stream onto sample boundaries.

    HTTP chunks fall wherever the network puts them, and PCM16 samples are two
    bytes. Forwarding an odd-length chunk shifts every following sample by one
    byte, which is heard as a burst of noise rather than speech.
    """
    carry = b""
    async for chunk in source:
        buffer = carry + chunk
        cut = len(buffer) - (len(buffer) % 2)
        if cut:
            yield buffer[:cut]
        carry = buffer[cut:]
    # A single trailing byte is half a sample. Dropping it loses 1/32000th of a
    # second; emitting it would click.
    if carry:
        log.debug("dropped a trailing half sample", extra={"bytes": len(carry)})


def _retry_after(response: Any) -> float | None:
    """Seconds the vendor asked us to wait, if it said.

    Ignoring a `Retry-After` and retrying sooner is how one rate-limited request
    becomes three. Only the delta-seconds form is honoured; the HTTP-date form is
    treated as absent rather than parsed, because a clock skew between us and the
    vendor would turn it into an unbounded wait on a live call.
    """
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        seconds = float(raw.strip())
    except (TypeError, ValueError):
        return None
    # A limiter asking for longer than the caller will wait is a refusal, not a
    # delay; fall back to our own backoff and let the attempts run out.
    return seconds if 0 <= seconds <= 5 else None


def _split_voice(voice: str) -> tuple[str, str]:
    """Voices are configured as "hi-IN:priya" so one profile field carries both
    the language and the speaker. A bare name keeps the Hindi default."""
    if ":" in voice:
        language, speaker = voice.split(":", 1)
        return language, speaker
    return "hi-IN", voice
