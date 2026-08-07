"""LLM access over the OpenAI chat-completions wire format.

This one client covers every self-hosted server worth deploying — vLLM, Ollama,
llama.cpp, TGI, SGLang, LM Studio — because they all speak this protocol. Nothing
here talks to OpenAI unless base_url is pointed there, so an air-gapped install
stays air-gapped.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from vaani.core.logging import get_logger
from vaani.providers.base import Completion, Message, ToolCall

log = get_logger(__name__)


class OpenAICompatLLM:
    name = "openai_compat"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "not-needed",
        temperature: float = 0.3,
        max_tokens: int = 512,
        timeout_s: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout_s
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=httpx.Timeout(self._timeout, connect=5.0),
            # A busy call centre keeps this pool hot; without limits httpx opens
            # a fresh connection per turn and the handshake shows up as latency.
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
        )

    def _payload(self, messages: list[Message], **overrides: Any) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": [m.to_wire() for m in messages],
            "temperature": overrides.get("temperature") or self._temperature,
            "max_tokens": overrides.get("max_tokens") or self._max_tokens,
        }

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        if self._client is None:
            raise RuntimeError("OpenAICompatLLM.start() was not awaited")
        payload = self._payload(messages, temperature=temperature, max_tokens=max_tokens)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        resp = await self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        msg = choice.get("message", {})

        calls: list[ToolCall] = []
        for raw in msg.get("tool_calls") or []:
            fn = raw.get("function", {})
            calls.append(
                ToolCall(
                    id=raw.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                    name=fn.get("name", ""),
                    arguments=_parse_arguments(fn.get("arguments")),
                )
            )

        usage = data.get("usage") or {}
        return Completion(
            text=(msg.get("content") or "").strip(),
            tool_calls=calls,
            finish_reason=choice.get("finish_reason", "stop"),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        if self._client is None:
            raise RuntimeError("OpenAICompatLLM.start() was not awaited")
        payload = self._payload(messages, temperature=temperature, max_tokens=max_tokens)
        payload["stream"] = True

        async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if not body or body == "[DONE]":
                    continue
                try:
                    delta = json.loads(body)["choices"][0].get("delta", {})
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                piece = delta.get("content")
                if piece:
                    yield piece

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _parse_arguments(raw: Any) -> dict[str, Any]:
    """Tool arguments arrive as a JSON *string*, and small models get it wrong
    often enough that a parse failure must not kill the call."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        log.warning("unparseable tool arguments", extra={"raw": str(raw)[:200]})
        return {}
