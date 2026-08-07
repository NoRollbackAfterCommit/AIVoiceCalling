"""<Vendor> speech-to-text.

One-paragraph note on why this provider exists and what it trades away — hosted
latency vs. on-premise privacy, model size vs. GPU cost. That is the thing a
reader is deciding between when they land here.

Replace the STT shape below with LLMProvider / TTSProvider / EmbeddingProvider
as needed; the contracts live in vaani/providers/base.py.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from vaani.config import SAMPLE_RATE
from vaani.core.logging import get_logger
from vaani.providers.base import Transcript

log = get_logger(__name__)


class VendorSTT:
    name = "vendor"

    def __init__(self, api_key: str, model: str, *, timeout_s: float = 30.0) -> None:
        # Store config only. No SDK import, no network, no model load here —
        # __init__ runs during build_services() before the event loop is warm.
        self._api_key = api_key
        self._model = model
        self._timeout_s = timeout_s
        self._client: object | None = None

    async def start(self) -> None:
        """Load models / open clients. Called once at boot, not per call."""
        # Import inside the method so the dependency stays optional and an
        # air-gapped install without this extra genuinely cannot reach out.
        from vendor_sdk import AsyncClient  # type: ignore[import-not-found]

        self._client = AsyncClient(api_key=self._api_key, timeout=self._timeout_s)
        log.info("vendor stt ready", extra={"model": self._model})

    async def transcribe(self, pcm: bytes, *, language: str | None = None) -> Transcript:
        """Transcribe one complete utterance.

        `pcm` is PCM16 mono at SAMPLE_RATE. If the vendor wants something else,
        convert here — the pipeline never sees another format.
        """
        # A synchronous SDK must not run on the event loop: one blocking call
        # stalls every concurrent call on this worker, not just this one.
        #   result = await asyncio.to_thread(self._client.transcribe, pcm)
        raise NotImplementedError

    async def stream(
        self, audio: AsyncIterator[bytes], *, language: str | None = None
    ) -> AsyncIterator[Transcript]:
        """Yield partials as they arrive, then exactly one is_final=True result.

        Partials drive the live caption in the supervisor console; only the final
        reaches the agent. If the vendor has no streaming API, buffer the frames
        and yield a single final — correct, just less responsive.
        """
        raise NotImplementedError
        yield  # pragma: no cover - makes this an async generator

    async def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            await client.aclose()
