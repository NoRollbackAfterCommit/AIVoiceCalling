"""Central configuration.

Every provider in the platform is selected by a string here, so a deployment is
reconfigured by environment variables alone — no code changes between a laptop
demo (all mocks, no GPU) and an air-gapped government cluster (Whisper + Llama +
Piper + Qdrant on local GPUs).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VAANI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- service ----------------------------------------------------------
    env: Literal["dev", "staging", "prod"] = "dev"
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"
    # Comma separated. "*" is fine for a dev box, never for production.
    cors_origins: str = "*"

    # ---- persistence ------------------------------------------------------
    # SQLite by default so `vaani serve` works with nothing else installed.
    database_url: str = "sqlite+aiosqlite:///./data/vaani.db"
    redis_url: str | None = None

    # ---- speech to text ---------------------------------------------------
    stt_provider: Literal["mock", "faster_whisper"] = "mock"
    stt_model: str = "small"  # tiny | base | small | medium | large-v3
    stt_device: Literal["auto", "cpu", "cuda"] = "auto"
    stt_compute_type: str = "default"  # int8 | float16 | default
    # None = auto-detect per utterance (Whisper's own language ID).
    stt_language: str | None = None

    # ---- large language model --------------------------------------------
    # "openai_compat" talks to anything exposing /v1/chat/completions:
    # vLLM, Ollama, llama.cpp server, TGI, LM Studio, SGLang.
    llm_provider: Literal["mock", "openai_compat"] = "mock"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "not-needed"
    llm_model: str = "qwen2.5:7b-instruct"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 512
    llm_timeout_s: float = 30.0

    # ---- text to speech ---------------------------------------------------
    tts_provider: Literal["mock", "piper"] = "mock"
    tts_voice: str = "en_US-lessac-medium"
    tts_voices_dir: str = "./models/piper"
    tts_speed: float = 1.0

    # ---- retrieval --------------------------------------------------------
    vector_store: Literal["memory", "qdrant"] = "memory"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    embedding_provider: Literal["hash", "sentence_transformers"] = "hash"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 384
    rag_top_k: int = 5
    rag_min_score: float = 0.25

    # ---- conversation behaviour ------------------------------------------
    # Trailing silence that ends a caller's turn. Too low and the agent
    # interrupts a thinking caller; too high and it feels sluggish.
    end_of_turn_silence_ms: int = 700
    # A caller speaking over the agent cancels playback after this much voice.
    barge_in_ms: int = 240
    max_turn_audio_s: float = 30.0
    # How far ahead of real time agent audio may be sent. Enough slack that the
    # far end never starves on a jittery link, small enough that the session
    # leaves SPEAKING at roughly the moment the caller stops hearing audio —
    # which is what makes barge-in detection correct.
    playout_lead_ms: int = 300
    # Hard ceiling on a single call; protects the GPU pool from stuck sessions.
    max_call_duration_s: float = 900.0
    max_concurrent_calls: int = 50
    # Silence from the caller before the agent prompts, then hangs up.
    idle_prompt_after_s: float = 8.0
    idle_hangup_after_s: float = 25.0

    # ---- storage ----------------------------------------------------------
    recordings_dir: str = "./data/recordings"
    record_calls: bool = True

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
