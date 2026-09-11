"""Fixtures shared across test modules.

`test_pipeline.py` predates this file and keeps its own copies; these exist so
newer modules do not re-declare the mock-tier settings every time. Same values,
same reasoning: ceilings the admin UI allows, timers that never fire inside a
one-second suite.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

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
def asgi_post():
    """POST through raw ASGI, reporting how many body bytes the app pulled.

    Neither httpx nor TestClient will send a Content-Length that disagrees with
    the body it holds, and both hand the whole body over before the app runs, so
    neither can show that a route refused *before* reading. This manufactures the
    body a chunk at a time on demand: a route that reads it reports a byte count,
    and one that refuses first never allocates a thing.
    """

    async def post(
        app: Any,
        path: str,
        *,
        query: str = "",
        headers: dict[str, str] | None = None,
        body_bytes: int = 0,
    ) -> tuple[int, int]:
        read = 0
        status = 0

        async def receive() -> dict[str, Any]:
            nonlocal read
            if read >= body_bytes:
                return {"type": "http.request", "body": b"", "more_body": False}
            chunk = min(65536, body_bytes - read)
            read += chunk
            return {"type": "http.request", "body": b"x" * chunk, "more_body": read < body_bytes}

        async def send(message: dict[str, Any]) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]

        await app(
            {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.3"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": path,
                "raw_path": path.encode(),
                "root_path": "",
                "query_string": query.encode(),
                "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
                "client": ("203.0.113.9", 51234),
                "server": ("testserver", 80),
            },
            receive,
            send,
        )
        return status, read

    return post


@pytest.fixture
async def services(settings: Settings):
    svc = build_services(settings)
    # Language selection has its own tests; with it on, the first caller turn
    # answers "which language?" instead of reaching the agent.
    svc.profiles["default"] = replace(svc.profiles["default"], ask_language=False)
    await svc.start()
    yield svc
    await svc.close()
