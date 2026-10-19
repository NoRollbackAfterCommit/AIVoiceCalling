"""The OpenAI-wire LLM under the failures production actually saw."""

from __future__ import annotations

import httpx
import pytest

from vaani.providers.base import Message
from vaani.providers.llm.openai_compat import OpenAICompatLLM

_OK = {
    "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
}


def _llm(handler) -> OpenAICompatLLM:
    llm = OpenAICompatLLM(base_url="https://api.openai.com/v1", model="m", retry_backoff_s=0)
    llm._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://api.openai.com/v1"
    )
    return llm


async def test_a_passing_rate_limit_is_ridden_out():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, json={"error": {"code": "rate_limit_exceeded"}})
        return httpx.Response(200, json=_OK)

    completion = await _llm(handler).complete([Message(role="user", content="hi")])
    assert completion.text == "Hello"
    assert len(calls) == 2


async def test_an_exhausted_quota_is_not_retried_and_says_why(caplog):
    """A quota refusal fails identically every time; retrying it only spends
    the caller's silence. The reason must reach the log, or a 429 reads as
    load when it is billing."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            429, json={"error": {"code": "insufficient_quota", "message": "You exceeded"}}
        )

    with pytest.raises(httpx.HTTPStatusError):
        await _llm(handler).complete([Message(role="user", content="hi")])
    assert len(calls) == 1
    assert any(getattr(r, "code", None) == "insufficient_quota" for r in caplog.records)


async def test_a_server_error_that_persists_gives_up_after_bounded_attempts():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(503, json={"error": {"code": "overloaded"}})

    with pytest.raises(httpx.HTTPStatusError):
        await _llm(handler).complete([Message(role="user", content="hi")])
    assert len(calls) == 3


async def test_a_limit_asking_for_longer_than_a_caller_can_wait_is_not_retried():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(429, headers={"retry-after": "30"}, json={"error": {}})

    with pytest.raises(httpx.HTTPStatusError):
        await _llm(handler).complete([Message(role="user", content="hi")])
    assert len(calls) == 1


async def test_a_bad_request_is_not_retried():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(400, json={"error": {"code": "invalid_request_error"}})

    with pytest.raises(httpx.HTTPStatusError):
        await _llm(handler).complete([Message(role="user", content="hi")])
    assert len(calls) == 1
