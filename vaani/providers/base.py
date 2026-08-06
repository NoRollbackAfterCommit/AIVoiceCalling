"""Provider contracts.

Everything the platform does that needs a model sits behind one of these four
protocols. Swapping Whisper for NeMo, or Llama for Qwen, is a config change and a
new file in the matching subpackage — the pipeline never imports a concrete
provider.

All audio crossing these boundaries is PCM16 mono at vaani.config.SAMPLE_RATE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Speech to text
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Transcript:
    text: str
    # False while the utterance is still being spoken. Partials drive the live
    # caption in the supervisor console; only finals reach the agent.
    is_final: bool = True
    language: str | None = None
    confidence: float | None = None
    duration_s: float = 0.0


@runtime_checkable
class STTProvider(Protocol):
    name: str

    async def start(self) -> None:
        """Load models. Called once at boot, not per call."""

    async def transcribe(self, pcm: bytes, *, language: str | None = None) -> Transcript:
        """Transcribe one complete utterance."""

    async def stream(
        self, audio: AsyncIterator[bytes], *, language: str | None = None
    ) -> AsyncIterator[Transcript]:
        """Transcribe incrementally, yielding partials then a final."""

    async def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Text to speech
# ---------------------------------------------------------------------------


@runtime_checkable
class TTSProvider(Protocol):
    name: str

    async def start(self) -> None: ...

    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        """Render the whole utterance. Convenient, but adds latency."""

    async def stream(self, text: str, *, voice: str | None = None) -> AsyncIterator[bytes]:
        """Yield PCM chunks as they are produced.

        This is the one that matters: streaming lets the caller hear the first
        syllable while the rest is still rendering, and lets barge-in cancel
        mid-sentence.
        """

    async def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Large language model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class Message:
    role: str  # system | user | assistant | tool
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def to_wire(self) -> dict[str, Any]:
        """OpenAI chat-completions wire format."""
        msg: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": _dumps(tc.arguments)},
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name
        return msg


def _dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)


@dataclass(slots=True)
class Completion:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    async def start(self) -> None: ...

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion: ...

    async def stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield text deltas. Used for the final spoken answer so TTS can start
        before the model has finished thinking."""

    async def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str
    dim: int
    # Cosine scores are NOT comparable across embedding models: a strong match
    # scores ~0.75 with BGE and ~0.10 with a lexical hash. A relevance threshold
    # therefore belongs to the model, not to the retriever. Each provider
    # declares the similarity below which a hit is noise.
    similarity_floor: float

    async def start(self) -> None: ...

    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def close(self) -> None: ...
