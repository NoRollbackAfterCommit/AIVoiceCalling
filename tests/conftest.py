"""Fixtures shared across test modules.

`test_pipeline.py` predates this file and keeps its own copies; these exist so
newer modules do not re-declare the mock-tier settings every time. Same values,
same reasoning: ceilings the admin UI allows, timers that never fire inside a
one-second suite.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from vaani.config import Settings
from vaani.core.registry import build_services


@pytest.fixture
def settings() -> Settings:
    return Settings(
        stt_provider="mock",
        llm_provider="mock",
        tts_provider="mock",
        vector_store="memory",
        embedding_provider="hash",
        record_calls=False,
        end_of_turn_silence_ms=200,
        idle_prompt_after_s=120,
        idle_hangup_after_s=600,
    )


@pytest.fixture
async def services(settings: Settings):
    svc = build_services(settings)
    # Language selection has its own tests; with it on, the first caller turn
    # answers "which language?" instead of reaching the agent.
    svc.profiles["default"] = replace(svc.profiles["default"], ask_language=False)
    await svc.start()
    yield svc
    await svc.close()
