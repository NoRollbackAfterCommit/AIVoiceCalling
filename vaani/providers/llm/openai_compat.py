"""LLM access over the OpenAI chat-completions wire format.

This one client covers every self-hosted server worth deploying — vLLM, Ollama,
llama.cpp, TGI, SGLang, LM Studio — because they all speak this protocol. Nothing
here talks to OpenAI unless base_url is pointed there, so an air-gapped install
stays air-gapped.
"""

from __future__ import annotations

import asyncio
import json
import random
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from vaani.core.logging import get_logger
from vaani.providers.base import Completion, Message, ToolCall

log = get_logger(__name__)

# One attempt plus two retries, for the refusals that pass: a rate limit and a
# server-side hiccup. Bounded because the caller is listening to the hold tone
# the whole time. Production, 30 Sep: one unretried 429 on the first question
# after the language choice left a caller in silence until they hung up.
_MAX_ATTEMPTS = 3
_RETRYABLE = {429, 500, 502, 503, 504}
_MAX_RETRY_AFTER_S = 2.0
# A 429 carrying one of these is billing, not load. It fails identically every
# time, so retrying it only lengthens the silence.
_PERMANENT_CODES = {"insufficient_quota", "billing_hard_limit_reached"}


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
        retry_backoff_s: float = 0.25,
    ) -> None:
        self._retry_backoff = retry_backoff_s
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

        resp = await self._post(payload)
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

    async def _post(self, payload: dict[str, Any]) -> httpx.Response:
        assert self._client is not None
        for attempt in range(_MAX_ATTEMPTS):
            resp = await self._client.post("/chat/completions", json=payload)
            if resp.status_code < 400:
                return resp
            code = _error_code(resp)
            last = attempt == _MAX_ATTEMPTS - 1
            asked = _retry_after(resp)
            # A vendor asking for longer than a caller can wait is a refusal;
            # retrying sooner than it asked only gets refused again.
            too_long = asked is not None and asked > _MAX_RETRY_AFTER_S
            retryable = resp.status_code in _RETRYABLE and code not in _PERMANENT_CODES
            if retryable and not too_long and not last:
                # Jittered, or every call caught in one burst retries at the
                # same instant and meets the same limit again.
                wait = asked if asked is not None else self._retry_backoff * (2**attempt)
                wait *= random.uniform(1.0, 1.5)
                log.warning(
                    "llm refused; retrying",
                    extra={"status": resp.status_code, "code": code, "attempt": attempt + 1},
                )
                await asyncio.sleep(wait)
                continue
            # Said once, with the vendor's own reason: "429" alone cannot tell a
            # traffic burst from an account that has run out of credit.
            log.error("llm request failed", extra={"status": resp.status_code, "code": code})
            resp.raise_for_status()
        raise AssertionError("unreachable")  # pragma: no cover

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


def _error_code(resp: httpx.Response) -> str | None:
    """The machine-readable reason in an OpenAI-shaped error body, if any."""
    try:
        error = resp.json().get("error") or {}
    except ValueError:
        return None
    if not isinstance(error, dict):
        return None
    return error.get("code") or error.get("type")


def _retry_after(resp: httpx.Response) -> float | None:
    """Seconds the vendor asked us to wait, in the delta-seconds form only."""
    raw = resp.headers.get("retry-after")
    try:
        seconds = float(raw) if raw else None
    except ValueError:
        return None
    return seconds if seconds is not None and seconds >= 0 else None
