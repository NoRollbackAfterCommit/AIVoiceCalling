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
    if s.stt_provider == "openai":
        from vaani.providers.openai_hosted import OpenAISTT

        return OpenAISTT(
            api_key=s.openai_api_key or "",
            model=s.stt_model if s.stt_model.startswith(("whisper", "gpt-4o")) else "whisper-1",
            base_url=s.openai_base_url,
            language=s.stt_language,
            timeout_s=s.llm_timeout_s,
        )
    if s.stt_provider == "sarvam":
        from vaani.providers.stt.sarvam import SarvamSTT

        return SarvamSTT(
            api_key=s.sarvam_api_key or "",
            model=s.sarvam_stt_model,
            language=s.stt_language,
            timeout_s=s.llm_timeout_s,
        )
    from vaani.providers.stt.mock import MockSTT

    return MockSTT()


def build_llm(s: Settings) -> LLMProvider:
    if s.llm_provider == "anthropic":
        from vaani.providers.llm.anthropic_llm import AnthropicLLM

        return AnthropicLLM(
            api_key=s.anthropic_api_key or "",
            model=s.anthropic_model,
            max_tokens=s.llm_max_tokens,
            temperature=s.llm_temperature,
            effort=s.anthropic_effort,
            timeout_s=s.llm_timeout_s,
        )
    if s.llm_provider in ("openai", "openai_compat"):
        from vaani.providers.llm.openai_compat import OpenAICompatLLM

        # OpenAI's own API and every self-hosted server speak the same protocol,
        # so one client covers both; only the endpoint and credential differ.
        hosted = s.llm_provider == "openai"
        return OpenAICompatLLM(
            base_url=s.openai_base_url if hosted else s.llm_base_url,
            model=s.openai_model if hosted else s.llm_model,
            api_key=(s.openai_api_key or "") if hosted else "not-needed",
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
    if s.tts_provider == "openai":
        from vaani.providers.openai_hosted import OpenAITTS

        # Piper voice names mean nothing to OpenAI; fall back to a valid one.
        voice = s.tts_voice if "-" not in s.tts_voice else "alloy"
        return OpenAITTS(
            api_key=s.openai_api_key or "",
            model=s.tts_model,
            voice=voice,
            base_url=s.openai_base_url,
            speed=s.tts_speed,
            timeout_s=s.llm_timeout_s,
        )
    if s.tts_provider == "sarvam":
        from vaani.providers.tts.sarvam import SarvamTTS

        return SarvamTTS(
            api_key=s.sarvam_api_key or "",
            model=s.sarvam_tts_model,
            default_voice=s.sarvam_voice,
            speed=s.tts_speed,
        )
    from vaani.providers.tts.mock import MockTTS

    return MockTTS()


def build_embedder(s: Settings) -> EmbeddingProvider:
    if s.embedding_provider == "sentence_transformers":
        from vaani.providers.embeddings.providers import SentenceTransformerEmbedding

        return SentenceTransformerEmbedding(model=s.embedding_model)
    if s.embedding_provider == "openai":
        from vaani.providers.openai_hosted import OpenAIEmbedding

        model = s.embedding_model if "embedding" in s.embedding_model else "text-embedding-3-small"
        return OpenAIEmbedding(
            api_key=s.openai_api_key or "",
            model=model,
            base_url=s.openai_base_url,
            timeout_s=s.llm_timeout_s,
        )
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

    async def reload(self, new: Settings) -> list[str]:
        """Apply changed settings in place, rebuilding only what they affect.

        Selective rather than wholesale, because the in-memory vector store lives
        inside the retriever: rebuilding everything to change a TTS voice would
        silently wipe every document the operator had uploaded. Each provider is
        started before the old one is swapped out and closed, so a bad API key
        surfaces as a failed save rather than a platform left with no LLM.
        """
        old = self.settings
        rebuilt: list[str] = []

        def changed(*keys: str) -> bool:
            return any(getattr(old, k) != getattr(new, k) for k in keys)

        if changed("stt_provider", "stt_model", "stt_device", "stt_compute_type",
                   "stt_language", "openai_api_key", "openai_base_url",
                   "sarvam_api_key", "sarvam_stt_model"):
            provider = build_stt(new)
            await provider.start()
            previous, self.stt = self.stt, provider
            await _quiet_close(previous)
            rebuilt.append("stt")

        if changed("llm_provider", "llm_base_url", "llm_model", "llm_temperature",
                   "llm_max_tokens", "llm_timeout_s", "anthropic_api_key",
                   "anthropic_model", "anthropic_effort", "openai_api_key",
                   "openai_model", "openai_base_url"):
            provider = build_llm(new)
            await provider.start()
            previous, self.llm = self.llm, provider
            await _quiet_close(previous)
            rebuilt.append("llm")

        if changed("tts_provider", "tts_voice", "tts_voices_dir", "tts_speed",
                   "tts_model", "openai_api_key", "openai_base_url",
                   "sarvam_api_key", "sarvam_tts_model", "sarvam_voice"):
            provider = build_tts(new)
            await provider.start()
            previous, self.tts = self.tts, provider
            await _quiet_close(previous)
            rebuilt.append("tts")

        if changed("vector_store", "qdrant_url", "qdrant_api_key",
                   "embedding_provider", "embedding_model", "embedding_dim"):
            # This one does discard the in-memory index — the vectors are not
            # portable across embedding models, so there is nothing to carry.
            retriever = Retriever(
                store=build_store(new), embedder=build_embedder(new),
                top_k=new.rag_top_k, min_score=new.rag_min_score,
            )
            await retriever.start()
            self.retriever = retriever
            rebuilt.append("knowledge")
            log.warning("knowledge store rebuilt; re-ingest documents if it was in-memory")
        else:
            # Cheap knobs that need no rebuild.
            self.retriever._top_k = new.rag_top_k
            self.retriever._min_score = min(
                new.rag_min_score,
                getattr(self.retriever._embedder, "similarity_floor", new.rag_min_score),
            )

        self.settings = new
        log.info("settings applied", extra={"rebuilt": rebuilt})
        return rebuilt


async def _quiet_close(provider: Any) -> None:
    try:
        await provider.close()
    except Exception:
        log.exception("failed to close replaced provider")


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
