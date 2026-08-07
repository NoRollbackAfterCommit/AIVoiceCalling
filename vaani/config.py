"""Central configuration.

Every provider in the platform is selected by a string here, so a deployment is
reconfigured without code changes: a laptop demo (all mocks, no GPU), a hosted
pilot (Claude or OpenAI), or an air-gapped government cluster (Whisper + Llama +
Piper + Qdrant on local GPUs) are the same binary with different settings.

Each field carries UI metadata alongside its type. That metadata is the single
source of truth for three things at once — validation, the `/api/settings/schema`
descriptor, and the rendering of the admin settings page — so adding a knob means
editing one line here and nothing else.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# The pipeline works in one audio format end to end: signed 16-bit little-endian
# PCM, mono, 16 kHz. Telephony codecs (8 kHz mu-law) are converted at the edge in
# vaani.telephony, never inside the pipeline.
SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2
CHANNELS = 1
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 320
FRAME_BYTES = FRAME_SAMPLES * SAMPLE_WIDTH  # 640


def cfg(
    default: Any,
    *,
    group: str,
    label: str,
    help: str = "",
    secret: bool = False,
    options: list[dict[str, str]] | None = None,
    depends_on: dict[str, list[str]] | None = None,
    restart: bool = True,
    **kwargs: Any,
) -> Any:
    """A settings field plus everything the admin UI needs to render it.

    `secret` fields are never returned by the API in cleartext — only a masked
    hint — so an API key entered in the browser cannot be read back out of it.
    `depends_on` lets the UI hide irrelevant fields (an Anthropic API key means
    nothing when the LLM provider is Piper), and `restart` marks settings that
    require rebuilding the provider objects rather than taking effect per call.
    """
    return Field(
        default,
        json_schema_extra={
            "group": group,
            "label": label,
            "help": help,
            "secret": secret,
            "options": options,
            "depends_on": depends_on,
            "restart": restart,
        },
        **kwargs,
    )


def _opts(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"value": v, "label": lbl} for v, lbl in pairs]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VAANI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- service ----------------------------------------------------------
    env: Literal["dev", "staging", "prod"] = cfg(
        "dev", group="Service", label="Environment",
        options=_opts(("dev", "Development"), ("staging", "Staging"), ("prod", "Production")),
    )
    host: str = cfg("0.0.0.0", group="Service", label="Bind address")
    port: int = cfg(8080, group="Service", label="Port", ge=1, le=65535)
    log_level: str = cfg(
        "INFO", group="Service", label="Log level", restart=False,
        options=_opts(("DEBUG", "DEBUG"), ("INFO", "INFO"), ("WARNING", "WARNING"),
                      ("ERROR", "ERROR")),
    )
    cors_origins: str = cfg(
        "*", group="Service", label="CORS origins",
        help="Comma separated. '*' is fine for a dev box, never for production.",
    )

    # ---- persistence ------------------------------------------------------
    database_url: str = cfg(
        "sqlite+aiosqlite:///./data/vaani.db", group="Service", label="Database URL",
        help="SQLite by default so the platform runs with nothing else installed.",
    )
    redis_url: str | None = cfg(None, group="Service", label="Redis URL")

    # ---- speech to text ---------------------------------------------------
    stt_provider: Literal["mock", "faster_whisper", "openai", "sarvam"] = cfg(
        "mock", group="Speech to text", label="Provider",
        options=_opts(
            ("mock", "Mock — no model, for development"),
            ("faster_whisper", "Faster-Whisper — self-hosted, offline"),
            ("openai", "OpenAI Whisper API — hosted"),
            ("sarvam", "Sarvam Saarika — Indian languages, hosted in India"),
        ),
    )
    stt_model: str = cfg(
        "small", group="Speech to text", label="Model",
        help="Self-hosted: tiny | base | small | medium | large-v3. "
             "OpenAI: whisper-1 | gpt-4o-transcribe.",
        depends_on={"stt_provider": ["faster_whisper", "openai"]},
    )
    stt_device: Literal["auto", "cpu", "cuda"] = cfg(
        "auto", group="Speech to text", label="Device",
        options=_opts(("auto", "Auto-detect"), ("cpu", "CPU"), ("cuda", "NVIDIA GPU")),
        depends_on={"stt_provider": ["faster_whisper"]},
    )
    stt_compute_type: str = cfg(
        "default", group="Speech to text", label="Compute type",
        help="int8 (CPU) | float16 (GPU) | default (pick automatically).",
        depends_on={"stt_provider": ["faster_whisper"]},
    )
    stt_language: str | None = cfg(
        None, group="Speech to text", label="Force language",
        help="Leave blank to auto-detect the language on every utterance.",
    )
    sarvam_api_key: str | None = cfg(
        None, group="Speech to text", label="Sarvam API key", secret=True,
        help="One key covers both Saarika speech recognition and Bulbul speech "
             "synthesis. Sarvam hosts in India, which is what makes it usable on "
             "a deployment with data residency obligations.",
        depends_on={"stt_provider": ["sarvam"], "tts_provider": ["sarvam"]},
    )
    sarvam_stt_model: str = cfg(
        "saaras:v3", group="Speech to text", label="Saarika model",
        depends_on={"stt_provider": ["sarvam"]},
    )

    # ---- large language model --------------------------------------------
    llm_provider: Literal["mock", "openai_compat", "anthropic", "openai"] = cfg(
        "mock", group="Language model", label="Provider",
        options=_opts(
            ("mock", "Mock — rule based, for development"),
            ("openai_compat", "Self-hosted — vLLM / Ollama / llama.cpp"),
            ("anthropic", "Anthropic Claude — hosted"),
            ("openai", "OpenAI — hosted"),
        ),
        help="Self-hosted keeps the deployment air-gapped. The hosted options send "
             "call transcripts to a third party — check this is permitted before "
             "enabling it on a government or healthcare deployment.",
    )

    # -- self-hosted (OpenAI-compatible servers)
    llm_base_url: str = cfg(
        "http://localhost:11434/v1", group="Language model", label="Server URL",
        help="Ollama: http://localhost:11434/v1 · vLLM: http://localhost:8000/v1",
        depends_on={"llm_provider": ["openai_compat"]},
    )
    llm_model: str = cfg(
        "qwen2.5:7b-instruct", group="Language model", label="Model",
        depends_on={"llm_provider": ["openai_compat"]},
    )

    # -- Anthropic
    anthropic_api_key: str | None = cfg(
        None, group="Language model", label="Anthropic API key", secret=True,
        help="Starts with sk-ant-. Stored on this server only, never returned by the API.",
        depends_on={"llm_provider": ["anthropic"]},
    )
    anthropic_model: str = cfg(
        "claude-opus-5", group="Language model", label="Claude model",
        options=_opts(
            ("claude-opus-5", "Claude Opus 5 — most capable"),
            ("claude-sonnet-5", "Claude Sonnet 5 — balanced"),
            ("claude-haiku-4-5", "Claude Haiku 4.5 — fastest, best for voice"),
            ("claude-opus-4-8", "Claude Opus 4.8"),
            ("claude-fable-5", "Claude Fable 5 — deepest reasoning"),
        ),
        help="On a live phone call latency is the product. Haiku 4.5 answers "
             "fastest; Opus 5 reasons best but adds noticeable silence per turn.",
        depends_on={"llm_provider": ["anthropic"]},
    )
    anthropic_effort: Literal["low", "medium", "high", "xhigh", "max"] = cfg(
        "low", group="Language model", label="Reasoning effort",
        options=_opts(("low", "Low — fastest, best for live calls"),
                      ("medium", "Medium"), ("high", "High"),
                      ("xhigh", "Extra high"), ("max", "Maximum — slowest")),
        help="Controls how long Claude thinks before answering. Low is the right "
             "default for telephony; raise it only for non-realtime workflows.",
        depends_on={"llm_provider": ["anthropic"]},
    )

    # -- OpenAI
    openai_api_key: str | None = cfg(
        None, group="Language model", label="OpenAI API key", secret=True,
        help="Starts with sk-. Also used by the OpenAI speech and voice providers.",
        depends_on={"llm_provider": ["openai"], "stt_provider": ["openai"],
                    "tts_provider": ["openai"]},
    )
    openai_model: str = cfg(
        "gpt-4o", group="Language model", label="OpenAI model",
        options=_opts(("gpt-4o", "GPT-4o"), ("gpt-4o-mini", "GPT-4o mini — fastest"),
                      ("gpt-4.1", "GPT-4.1"), ("gpt-4.1-mini", "GPT-4.1 mini")),
        depends_on={"llm_provider": ["openai"]},
    )
    openai_base_url: str = cfg(
        "https://api.openai.com/v1", group="Language model", label="OpenAI base URL",
        help="Change only for an Azure OpenAI or proxy endpoint.",
        depends_on={"llm_provider": ["openai"]},
    )

    # -- shared generation controls
    llm_temperature: float = cfg(
        0.3, group="Language model", label="Temperature", ge=0.0, le=2.0,
        help="Ignored by current Claude models, which reject sampling parameters "
             "outright — steer their tone through the agent prompt instead.",
    )
    llm_max_tokens: int = cfg(
        512, group="Language model", label="Max reply tokens", ge=32, le=8192,
        help="A spoken reply is one to three sentences; a large value here only "
             "buys the model room to ramble.",
    )
    llm_timeout_s: float = cfg(
        30.0, group="Language model", label="Request timeout (s)", ge=1.0, le=300.0,
    )

    # ---- text to speech ---------------------------------------------------
    tts_provider: Literal["mock", "piper", "openai", "sarvam"] = cfg(
        "mock", group="Text to speech", label="Provider",
        options=_opts(
            ("mock", "Mock — tone generator, for development"),
            ("piper", "Piper — self-hosted, offline"),
            ("openai", "OpenAI speech — hosted"),
            ("sarvam", "Sarvam Bulbul — Indian languages, hosted in India"),
        ),
    )
    sarvam_tts_model: str = cfg(
        "bulbul:v3", group="Text to speech", label="Bulbul model",
        depends_on={"tts_provider": ["sarvam"]},
    )
    sarvam_voice: str = cfg(
        "hi-IN:anushka", group="Text to speech", label="Sarvam voice",
        help="Written as language:speaker, for example hi-IN:meera or "
             "bn-IN:anushka. The language half selects pronunciation.",
        depends_on={"tts_provider": ["sarvam"]},
    )
    tts_voice: str = cfg(
        "en_US-lessac-medium", group="Text to speech", label="Voice",
        help="Piper: a voice name under the voices directory. "
             "OpenAI: alloy | echo | fable | onyx | nova | shimmer.",
    )
    tts_voices_dir: str = cfg(
        "./models/piper", group="Text to speech", label="Piper voices directory",
        depends_on={"tts_provider": ["piper"]},
    )
    tts_speed: float = cfg(
        1.0, group="Text to speech", label="Speaking rate", ge=0.5, le=2.0,
    )
    tts_model: str = cfg(
        "tts-1", group="Text to speech", label="OpenAI speech model",
        options=_opts(("tts-1", "tts-1 — lowest latency"), ("tts-1-hd", "tts-1-hd")),
        depends_on={"tts_provider": ["openai"]},
    )

    # ---- retrieval --------------------------------------------------------
    vector_store: Literal["memory", "qdrant"] = cfg(
        "memory", group="Knowledge", label="Vector store",
        options=_opts(("memory", "In-memory — no infrastructure"),
                      ("qdrant", "Qdrant — persistent, scalable")),
    )
    qdrant_url: str = cfg(
        "http://localhost:6333", group="Knowledge", label="Qdrant URL",
        depends_on={"vector_store": ["qdrant"]},
    )
    qdrant_api_key: str | None = cfg(
        None, group="Knowledge", label="Qdrant API key", secret=True,
        depends_on={"vector_store": ["qdrant"]},
    )
    embedding_provider: Literal["hash", "sentence_transformers", "openai"] = cfg(
        "hash", group="Knowledge", label="Embeddings",
        options=_opts(
            ("hash", "Lexical hash — no model, development only"),
            ("sentence_transformers", "BGE / E5 — self-hosted"),
            ("openai", "OpenAI embeddings — hosted"),
        ),
    )
    embedding_model: str = cfg(
        "BAAI/bge-m3", group="Knowledge", label="Embedding model",
        help="Self-hosted: BAAI/bge-m3. OpenAI: text-embedding-3-small.",
        depends_on={"embedding_provider": ["sentence_transformers", "openai"]},
    )
    embedding_dim: int = cfg(
        384, group="Knowledge", label="Embedding dimensions", ge=64, le=4096,
        depends_on={"embedding_provider": ["hash"]},
    )
    rag_top_k: int = cfg(
        5, group="Knowledge", label="Passages per answer", ge=1, le=20, restart=False,
        help="More passages means better recall and a longer, slower answer.",
    )
    rag_min_score: float = cfg(
        0.25, group="Knowledge", label="Relevance threshold", ge=0.0, le=1.0,
        restart=False,
        help="An upper bound only — an embedding model whose scores sit on a lower "
             "scale pulls this down to its own floor automatically.",
    )

    # ---- conversation behaviour ------------------------------------------
    end_of_turn_silence_ms: int = cfg(
        700, group="Conversation", label="End-of-turn silence (ms)", ge=200, le=3000,
        restart=False,
        help="Trailing silence that ends the caller's turn. Too low interrupts "
             "someone drawing breath; too high leaves a dead pause. 500-900 works.",
    )
    barge_in_ms: int = cfg(
        240, group="Conversation", label="Barge-in threshold (ms)", ge=80, le=1000,
        restart=False,
        help="Continuous caller speech required to cut the agent off mid-sentence.",
    )
    playout_lead_ms: int = cfg(
        300, group="Conversation", label="Playout lead (ms)", ge=0, le=2000,
        restart=False,
        help="How far ahead of real time agent audio may be sent. Enough slack that "
             "a jittery line never starves, small enough that barge-in stays accurate.",
    )
    max_turn_audio_s: float = cfg(
        30.0, group="Conversation", label="Max utterance (s)", ge=5.0, le=120.0,
        restart=False,
    )
    max_call_duration_s: float = cfg(
        900.0, group="Conversation", label="Max call duration (s)", ge=30.0, le=7200.0,
        restart=False,
    )
    max_concurrent_calls: int = cfg(
        50, group="Conversation", label="Concurrent call limit", ge=1, le=1000,
        help="GPU inference degrades catastrophically rather than gracefully, so "
             "the caller past this limit gets a polite 'all lines busy'.",
    )
    idle_prompt_after_s: float = cfg(
        8.0, group="Conversation", label="Prompt after silence (s)", ge=2.0, le=120.0,
        restart=False,
    )
    idle_hangup_after_s: float = cfg(
        25.0, group="Conversation", label="Hang up after silence (s)", ge=5.0, le=600.0,
        restart=False,
    )

    # ---- storage ----------------------------------------------------------
    recordings_dir: str = cfg(
        "./data/recordings", group="Storage", label="Recordings directory",
        restart=False,
    )
    record_calls: bool = cfg(
        True, group="Storage", label="Record calls", restart=False,
        help="Recordings are the audit trail for a government deployment. Confirm "
             "the caller notification requirements in your jurisdiction.",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # -- helpers used by the settings API -----------------------------------

    @classmethod
    def field_meta(cls, name: str) -> dict[str, Any]:
        field = cls.model_fields[name]
        return dict(getattr(field, "json_schema_extra", None) or {})

    @classmethod
    def secret_fields(cls) -> set[str]:
        return {n for n in cls.model_fields if cls.field_meta(n).get("secret")}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Environment-only settings, used before the runtime store is available."""
    return Settings()
