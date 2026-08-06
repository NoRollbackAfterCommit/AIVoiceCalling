"""Rule-based stand-in for the LLM.

Deliberately dumb, but it exercises the parts of the agent runtime that break in
production: it emits real tool calls, respects the tool results it gets back, and
streams its answer token by token.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from typing import Any, AsyncIterator

from vaani.providers.base import Completion, Message, ToolCall

_RULES: list[tuple[str, str]] = [
    (r"\b(hello|hi|namaste|good (morning|afternoon|evening))\b",
     "Namaste. You have reached the citizen helpline. How may I help you today?"),
    (r"\b(bye|thank you|thanks|that is all|that's all)\b",
     "Thank you for calling. Have a good day."),
    (r"\b(complaint|complain|issue|problem|fault)\b",
     "I can register that complaint for you. May I have your registered mobile number?"),
    (r"\b(agent|human|executive|supervisor|officer)\b",
     "Certainly, I am transferring you to a human executive. Please stay on the line."),
]


class MockLLM:
    name = "mock"

    def __init__(self, model: str = "mock") -> None:
        self._model = model

    async def start(self) -> None:
        return None

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        await asyncio.sleep(0.05)
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        ).lower()

        # If knowledge search is available and the caller asked something factual,
        # call the tool exactly once — the runtime then feeds the result back and
        # we fall through to the text branch on the second pass.
        already_searched = any(m.role == "tool" for m in messages)
        tool_names = {t.get("function", {}).get("name") for t in (tools or [])}
        if (
            not already_searched
            and "search_knowledge" in tool_names
            and re.search(r"\b(bill|due|date|fee|rate|how|what|when|where|policy|document)\b", last_user)
        ):
            return Completion(
                text="",
                tool_calls=[
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        name="search_knowledge",
                        arguments={"query": last_user[:200]},
                    )
                ],
                finish_reason="tool_calls",
            )

        if already_searched:
            evidence = next((m.content for m in reversed(messages) if m.role == "tool"), "")
            if evidence.strip() and "no relevant" not in evidence.lower():
                return Completion(text=_speakable(evidence))
            return Completion(
                text="I could not find that in my knowledge base. Shall I connect you to an officer?"
            )

        for pattern, reply in _RULES:
            if re.search(pattern, last_user):
                return Completion(text=reply)
        return Completion(
            text="I understand. Could you please tell me a little more about what you need?"
        )

    async def stream(  # type: ignore[override]
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        completion = await self.complete(messages)
        for word in completion.text.split(" "):
            await asyncio.sleep(0.01)
            yield word + " "

    async def close(self) -> None:
        return None


def _speakable(evidence: str) -> str:
    """Turn a retrieved passage into something that survives being read aloud.

    A real LLM paraphrases the evidence. The mock cannot, so it does the minimum
    that keeps the demo honest: drop the citation scaffolding a caller must never
    hear, and stop at a sentence boundary rather than mid-word.
    """
    body = re.sub(r"\[\d+\]\s*\(source:[^)]*\)\s*", "", evidence)
    body = " ".join(body.split())
    sentences = re.split(r"(?<=[.!?])\s+", body)
    answer = " ".join(sentences[:2]).strip()
    return answer or body[:200]
