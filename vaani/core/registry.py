"""Service container.

Models are expensive to load and must be shared across every concurrent call, so
they are constructed once at startup and handed to sessions by reference. This is
also the single place where a config string becomes a concrete class — the one
spot to touch when adding a provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vaani.agent.prompt import DEFAULT_PROFILE, AgentProfile
from vaani.agent.tools.base import ToolRegistry
from vaani.agent.tools.builtin import registry as builtin_tools
from vaani.config import Settings
from vaani.core.logging import get_logger
from vaani.providers.base import EmbeddingProvider, LLMProvider, STTProvider, TTSProvider
from vaani.rag.retriever import Retriever
from vaani.rag.store import MemoryVectorStore, QdrantVectorStore, VectorStore

log = get_logger(__name__)


def build_stt(s: Settings) -> STTProvider:
    if s.stt_provider == "faster_whisper":
        from vaani.providers.stt.faster_whisper_stt import FasterWhisperSTT

        return FasterWhisperSTT(
            model=s.stt_model,
            device=s.stt_device,
            compute_type=s.stt_compute_type,
            default_language=s.stt_language,
        )
    from vaani.providers.stt.mock import MockSTT

    return MockSTT()


def build_llm(s: Settings) -> LLMProvider:
    if s.llm_provider == "openai_compat":
        from vaani.providers.llm.openai_compat import OpenAICompatLLM

        return OpenAICompatLLM(
            base_url=s.llm_base_url,
            model=s.llm_model,
            api_key=s.llm_api_key,
            temperature=s.llm_temperature,
            max_tokens=s.llm_max_tokens,
            timeout_s=s.llm_timeout_s,
        )
    from vaani.providers.llm.mock import MockLLM

    return MockLLM()


def build_tts(s: Settings) -> TTSProvider:
    if s.tts_provider == "piper":
        from vaani.providers.tts.piper import PiperTTS

        return PiperTTS(
            voices_dir=s.tts_voices_dir, default_voice=s.tts_voice, speed=s.tts_speed
        )
    from vaani.providers.tts.mock import MockTTS

    return MockTTS()


def build_embedder(s: Settings) -> EmbeddingProvider:
    if s.embedding_provider == "sentence_transformers":
        from vaani.providers.embeddings.providers import SentenceTransformerEmbedding

        return SentenceTransformerEmbedding(model=s.embedding_model)
    from vaani.providers.embeddings.providers import HashEmbedding

    return HashEmbedding(dim=s.embedding_dim)


def build_store(s: Settings) -> VectorStore:
    if s.vector_store == "qdrant":
        return QdrantVectorStore(url=s.qdrant_url, api_key=s.qdrant_api_key)
    return MemoryVectorStore()


@dataclass
class Services:
    settings: Settings
    stt: STTProvider
    llm: LLMProvider
    tts: TTSProvider
    retriever: Retriever
    tools: ToolRegistry
    profiles: dict[str, AgentProfile]

    def profile(self, key: str) -> AgentProfile:
        return self.profiles.get(key) or self.profiles["default"]

    def as_tool_services(self) -> dict[str, Any]:
        """The subset tools are allowed to reach. Tools get the retriever, not
        the LLM — a tool that calls back into the model is a recursion bug."""
        return {"retriever": self.retriever, "settings": self.settings}

    async def start(self) -> None:
        log.info(
            "starting providers",
            extra={
                "stt": self.settings.stt_provider,
                "llm": self.settings.llm_provider,
                "tts": self.settings.tts_provider,
                "vector_store": self.settings.vector_store,
                "embeddings": self.settings.embedding_provider,
            },
        )
        await self.stt.start()
        await self.llm.start()
        await self.tts.start()
        await self.retriever.start()
        log.info("providers ready")

    async def close(self) -> None:
        for provider in (self.stt, self.llm, self.tts):
            try:
                await provider.close()
            except Exception:
                log.exception("provider shutdown failed")


def build_services(settings: Settings) -> Services:
    retriever = Retriever(
        store=build_store(settings),
        embedder=build_embedder(settings),
        top_k=settings.rag_top_k,
        min_score=settings.rag_min_score,
    )
    return Services(
        settings=settings,
        stt=build_stt(settings),
        llm=build_llm(settings),
        tts=build_tts(settings),
        retriever=retriever,
        tools=builtin_tools,
        profiles={"default": DEFAULT_PROFILE},
    )
