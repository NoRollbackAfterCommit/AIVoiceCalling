"""The agent runtime: conversation memory plus the tool-calling loop.

One `ConversationAgent` per call. It owns the message history, decides when to
call tools, bounds how many times it may do so, and returns the sentence to
speak — plus any control action (transfer, hang up) that a tool requested.

The loop is deliberately bounded and deliberately boring. On a phone call there
is no room for a model to think for fifteen seconds; every iteration is another
second of silence the caller is listening to.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from vaani.agent.prompt import AgentProfile, render_system_prompt
from vaani.agent.tools.base import ToolContext, ToolRegistry
from vaani.core.logging import get_logger
from vaani.providers.base import LLMProvider, Message

log = get_logger(__name__)


@dataclass(slots=True)
class AgentTurn:
    text: str
    control: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ConversationAgent:
    def __init__(
        self,
        profile: AgentProfile,
        llm: LLMProvider,
        tools: ToolRegistry,
        ctx: ToolContext,
        *,
        history_turns: int = 12,
    ) -> None:
        self._profile = profile
        self._llm = llm
        self._tools = tools
        self._ctx = ctx
        self._history_turns = history_turns
        self._system = Message(role="system", content=render_system_prompt(profile))
        self._history: list[Message] = []
        self._schemas = tools.schemas(profile.tools)

    # -- memory -------------------------------------------------------------

    @property
    def transcript(self) -> list[dict[str, str]]:
        """Human-readable history for the call record and the summariser."""
        return [
            {"role": m.role, "content": m.content}
            for m in self._history
            if m.role in ("user", "assistant") and m.content
        ]

    def _window(self) -> list[Message]:
        """System prompt plus a sliding window of recent history.

        Trimming is by message, but never in the middle of a tool exchange — an
        orphaned `tool` message with no matching `tool_calls` before it is a hard
        400 from most inference servers.
        """
        limit = self._history_turns * 2
        if len(self._history) <= limit:
            return [self._system, *self._history]
        window = self._history[-limit:]
        while window and window[0].role in ("tool", "assistant") and not (
            window[0].role == "assistant" and window[0].content
        ):
            window.pop(0)
        return [self._system, *window]

    def note_user(self, text: str) -> None:
        self._history.append(Message(role="user", content=text))

    def note_assistant(self, text: str) -> None:
        self._history.append(Message(role="assistant", content=text))

    # -- the loop -----------------------------------------------------------

    async def respond(self, user_text: str) -> AgentTurn:
        started = time.monotonic()
        self.note_user(user_text)

        control: dict[str, Any] = {}
        invoked: list[dict[str, Any]] = []
        prompt_tokens = completion_tokens = 0

        for iteration in range(self._profile.max_tool_iterations):
            completion = await self._llm.complete(
                self._window(),
                tools=self._schemas or None,
            )
            prompt_tokens += completion.prompt_tokens
            completion_tokens += completion.completion_tokens

            if not completion.tool_calls:
                text = _clean_for_speech(completion.text)
                self._history.append(Message(role="assistant", content=text))
                return AgentTurn(
                    text=text,
                    control=control,
                    tool_calls=invoked,
                    latency_ms=int((time.monotonic() - started) * 1000),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

            # Record the assistant's tool-call message before the results, or the
            # next request is malformed.
            self._history.append(
                Message(
                    role="assistant",
                    content=completion.text,
                    tool_calls=completion.tool_calls,
                )
            )

            for call in completion.tool_calls:
                log.info(
                    "tool call",
                    extra={
                        "tool": call.name,
                        "tool_args": call.arguments,
                        "iteration": iteration,
                    },
                )
                result = await self._tools.invoke(call.name, call.arguments, self._ctx)
                invoked.append(
                    {
                        "name": call.name,
                        "arguments": call.arguments,
                        "ok": result.ok,
                        "data": result.data,
                    }
                )
                if result.control:
                    control.update(result.control)
                self._history.append(
                    Message(
                        role="tool",
                        content=result.content,
                        tool_call_id=call.id,
                        name=call.name,
                    )
                )

        # Iteration budget exhausted: the model is looping on tools. Say
        # something rather than leaving the caller in silence.
        log.warning(
            "tool iteration limit reached",
            extra={"limit": self._profile.max_tool_iterations},
        )
        fallback = (
            "I am sorry, I am having trouble completing that. "
            "Let me connect you to a colleague who can help."
        )
        self.note_assistant(fallback)
        control.setdefault("action", "transfer")
        control.setdefault("reason", "tool_iteration_limit")
        return AgentTurn(
            text=fallback,
            control=control,
            tool_calls=invoked,
            latency_ms=int((time.monotonic() - started) * 1000),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    async def summarise(self) -> str:
        """A one-paragraph wrap-up for the CRM record. Best effort — a failure
        here must never affect the call."""
        if not self.transcript:
            return ""
        lines = "\n".join(f"{t['role']}: {t['content']}" for t in self.transcript)
        try:
            completion = await self._llm.complete(
                [
                    Message(
                        role="system",
                        content=(
                            "Summarise this call in three sentences: what the caller "
                            "wanted, what was done, and what still needs doing. Plain "
                            "text, no lists."
                        ),
                    ),
                    Message(role="user", content=lines[:6000]),
                ],
                max_tokens=200,
            )
            return completion.text
        except Exception:
            log.exception("summary failed")
            return ""


_MARKDOWN = str.maketrans({"*": None, "_": None, "#": None, "`": None})


def _clean_for_speech(text: str) -> str:
    """Last line of defence against markdown reaching the TTS engine.

    The system prompt forbids it, but small quantised models forget under load,
    and 'asterisk asterisk important asterisk asterisk' down a phone line is
    exactly the kind of thing that ends a pilot.
    """
    import re

    cleaned = text.strip().translate(_MARKDOWN)
    cleaned = re.sub(r"^\s*[-•]\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)  # markdown links
    cleaned = re.sub(r"\s*\n+\s*", " ", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip()
