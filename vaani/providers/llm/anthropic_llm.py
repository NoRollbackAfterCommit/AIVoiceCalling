"""Anthropic Claude, via the official SDK.

Three things about the Messages API differ from the OpenAI-compatible shape and
each one is a hard error rather than a degradation, so they are handled here
rather than left to the caller:

**The system prompt is a top-level field, not a message with `role: "system"`.**

**Current models reject sampling parameters.** `temperature` returns a 400 on
Opus 5, Opus 4.8/4.7 and Fable 5. Tone is steered through the agent prompt
instead — see `_ACCEPTS_TEMPERATURE`.

**Thinking must stay on.** With `thinking: {"type": "disabled"}` Opus 5 can write
a tool call into its visible text instead of emitting a `tool_use` block: the
turn succeeds, the call never runs, and nothing raises. For an agent whose whole
design is "the model picks a tool and deterministic code executes it", that is
the worst available failure — a caller told their complaint was registered when
no complaint was created. Adaptive thinking at `low` effort keeps tool calls
structured while staying fast enough for a live call.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from vaani.core.logging import get_logger
from vaani.providers.base import Completion, Message, ToolCall

log = get_logger(__name__)

# Models taking `thinking: {"type": "adaptive"}` and `output_config.effort`.
# Older models use a fixed thinking budget instead and reject both.
_ADAPTIVE_THINKING = {
    "claude-opus-5", "claude-sonnet-5", "claude-fable-5", "claude-mythos-5",
    "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6",
}

# Everything else returns 400 if `temperature` is present at all.
_ACCEPTS_TEMPERATURE = {"claude-haiku-4-5"}


class AnthropicLLM:
    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str = "claude-opus-5",
        max_tokens: int = 512,
        temperature: float = 0.3,
        effort: str = "low",
        timeout_s: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("An Anthropic API key is required. Set it in Settings.")
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._effort = effort
        self._timeout = timeout_s
        self._client: Any = None

    async def start(self) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise RuntimeError(
                "Claude support needs the Anthropic SDK: pip install 'vaani[cloud]'"
            ) from exc

        self._client = AsyncAnthropic(
            api_key=self._api_key,
            timeout=self._timeout,
            max_retries=2,  # SDK retries 429 and 5xx with backoff
        )
        log.info("anthropic ready", extra={"llm_model": self._model, "effort": self._effort})

    # -- request shaping ----------------------------------------------------

    def _tuning(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if self._model in _ADAPTIVE_THINKING:
            params["thinking"] = {"type": "adaptive"}
            params["output_config"] = {"effort": self._effort}
        if self._model in _ACCEPTS_TEMPERATURE:
            params["temperature"] = self._temperature
        return params

    @staticmethod
    def _split(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        """Lift system turns out of the history and convert the rest.

        Anthropic wants tool results as `tool_result` blocks in a *user* turn,
        not a dedicated `tool` role — the single biggest shape difference from
        the OpenAI protocol.
        """
        system_parts: list[str] = []
        turns: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                if msg.content:
                    system_parts.append(msg.content)
            elif msg.role == "user":
                turns.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                blocks: list[dict[str, Any]] = []
                if msg.content:
                    blocks.append({"type": "text", "text": msg.content})
                for call in msg.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        }
                    )
                if blocks:
                    turns.append({"role": "assistant", "content": blocks})
            elif msg.role == "tool":
                turns.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id or "",
                                "content": msg.content or "(no output)",
                            }
                        ],
                    }
                )

        # A conversation must open with a user turn.
        while turns and turns[0]["role"] != "user":
            turns.pop(0)
        return "\n\n".join(system_parts), turns

    @staticmethod
    def _tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        """OpenAI nests the schema under `function`; Anthropic keeps it flat."""
        converted = []
        for tool in tools or []:
            fn = tool.get("function", tool)
            converted.append(
                {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters")
                    or {"type": "object", "properties": {}},
                }
            )
        return converted

    # -- calls --------------------------------------------------------------

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        if self._client is None:
            raise RuntimeError("AnthropicLLM.start() was not awaited")

        system, turns = self._split(messages)
        if not turns:
            return Completion(text="")

        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens or self._max_tokens,
            "messages": turns,
            **self._tuning(),
        }
        if system:
            request["system"] = system
        if converted := self._tools(tools):
            request["tools"] = converted

        response = await self._client.messages.create(**request)

        # Safety classifiers can decline a request: HTTP 200, empty or partial
        # content. Reading content[0] unconditionally would raise here.
        if response.stop_reason == "refusal":
            category = getattr(getattr(response, "stop_details", None), "category", None)
            log.warning("claude declined the request", extra={"category": category})
            return Completion(
                text="I am not able to help with that. Let me connect you to a colleague.",
                finish_reason="refusal",
            )

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=dict(block.input) if isinstance(block.input, dict) else {},
                    )
                )

        return Completion(
            text=" ".join(p.strip() for p in text_parts if p.strip()).strip(),
            tool_calls=calls,
            finish_reason=response.stop_reason or "stop",
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        if self._client is None:
            raise RuntimeError("AnthropicLLM.start() was not awaited")

        system, turns = self._split(messages)
        if not turns:
            return

        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens or self._max_tokens,
            "messages": turns,
            **self._tuning(),
        }
        if system:
            request["system"] = system

        async with self._client.messages.stream(**request) as stream:
            async for chunk in stream.text_stream:
                if chunk:
                    yield chunk

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
