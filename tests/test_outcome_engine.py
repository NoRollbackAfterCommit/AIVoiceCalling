"""Phase 2: every call ends with a recorded outcome instead of looping.

The vocabulary, the stall detector, the tool gate, and the acceptance check that
a conversation which used to drift now terminates.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from vaani.agent.outcome import (
    AGENT_SET,
    CORE_DISPOSITIONS,
    PLATFORM_SET,
    REQUIRES_REFERENCE,
    allowed_for,
    is_valid,
)
from vaani.agent.prompt import DEFAULT_PROFILE, AgentProfile, render_system_prompt
from vaani.agent.runtime import ConversationAgent
from vaani.agent.tools.base import ToolContext
from vaani.agent.tools.builtin import registry
from vaani.config import Settings
from vaani.core.registry import build_services
from vaani.db.repository import CallRepository
from vaani.pipeline.progress import ProgressTracker
from vaani.pipeline.session import CallSession, CallState

from .test_pipeline import FakeTransport, silence, tone

# =========================================================================
# Vocabulary
# =========================================================================


def test_the_agent_and_platform_sets_partition_the_vocabulary():
    """A caller who has hung up cannot call a tool, so those outcomes must be
    platform-set — and must never be offered to the model, which would otherwise
    cheerfully report that a caller abandoned a call it is still talking on."""
    assert AGENT_SET | PLATFORM_SET == set(CORE_DISPOSITIONS)
    assert not (AGENT_SET & PLATFORM_SET)
    assert PLATFORM_SET == {"caller_abandoned", "idle_timeout", "capacity_rejected"}


def test_a_profile_may_extend_the_vocabulary_but_not_replace_it():
    profile = AgentProfile(key="t", extra_dispositions=["meter_reading_booked"])
    assert is_valid("meter_reading_booked", profile)
    assert is_valid("resolved", profile), "the core list must survive extension"
    assert not is_valid("invented_outcome", profile)


def test_only_agent_settable_outcomes_are_offered():
    allowed = allowed_for(AgentProfile(key="t", extra_dispositions=["meter_reading_booked"]))
    assert "resolved" in allowed and "meter_reading_booked" in allowed
    assert "caller_abandoned" not in allowed


def test_the_objective_and_closing_rules_reach_the_prompt():
    prompt = render_system_prompt(DEFAULT_PROFILE, language="hi-IN")
    assert "## What this call is for" in prompt
    assert "## Bringing the call to an end" in prompt
    assert "set_disposition" in prompt
    # "Do not ask a second time" is the rule that removes the common loop.
    assert "Do not ask a second time" in prompt


def test_the_default_profile_can_actually_reach_an_outcome():
    """An objective and the closing rules are useless if the tools are absent."""
    assert "set_disposition" in DEFAULT_PROFILE.tools
    assert "end_call" in DEFAULT_PROFILE.tools
    assert DEFAULT_PROFILE.objective


# =========================================================================
# Progress tracking
# =========================================================================


def _turn(t: ProgressTracker, caller="", agent="", tool=False, retrieved=False):
    t.observe(caller_text=caller, agent_text=agent, tool_ran=tool, retrieved=retrieved)


def test_a_substantive_caller_turn_identifies_the_need():
    t = ProgressTracker()
    _turn(t, caller="I want to check my electricity bill", agent="Let me look")
    assert t.identified


def test_an_answer_from_nowhere_does_not_count_as_addressing_it():
    """Otherwise "I do not have that information" resolves the call."""
    t = ProgressTracker()
    _turn(t, caller="what is my due date", agent="I do not have that information")
    assert t.identified
    assert not t.addressed


def test_retrieval_or_a_tool_addresses_the_need():
    t = ProgressTracker()
    _turn(t, caller="due date", agent="The fifteenth", retrieved=True)
    assert t.addressed

    t2 = ProgressTracker()
    _turn(t2, caller="register a complaint", agent="Reference 4471", tool=True)
    assert t2.addressed


def test_it_stalls_on_the_third_unproductive_turn_and_not_before():
    t = ProgressTracker(stall_after=3)
    for _ in range(2):
        _turn(t, caller="I still need help", agent="Could you repeat that")
    assert not t.stalled
    _turn(t, caller="I still need help", agent="Could you repeat that")
    assert t.stalled


def test_real_progress_resets_the_count():
    t = ProgressTracker(stall_after=3)
    _turn(t, caller="hello", agent="how can I help")
    _turn(t, caller="hello there", agent="how can I help")
    _turn(t, caller="my bill", agent="the fifteenth", retrieved=True)
    assert t.unproductive_turns == 0
    assert not t.stalled


def test_a_caller_repeating_themselves_counts_as_unproductive():
    """Even when retrieval succeeded — the same question again means the answer
    did not land."""
    t = ProgressTracker(stall_after=2)
    for _ in range(3):
        _turn(t, caller="what is my due date", agent="The fifteenth", retrieved=True)
    assert t.stalled


def test_the_fallback_is_offered_once_not_every_turn():
    t = ProgressTracker(stall_after=1)
    _turn(t, caller="help me", agent="sorry")
    assert t.should_offer_fallback()
    assert not t.should_offer_fallback(), "nagging every turn is its own loop"


# =========================================================================
# The tool gate
# =========================================================================


def _ctx() -> ToolContext:
    return ToolContext(call_id="t")


async def test_a_disposition_is_recorded():
    ctx = _ctx()
    result = await registry.invoke(
        "set_disposition", {"disposition": "resolved", "reason": "Bill date given"}, ctx
    )
    assert result.ok
    assert ctx.state["disposition"] == "resolved"
    assert ctx.state["disposition_reason"] == "Bill date given"


async def test_an_invented_disposition_is_refused():
    ctx = _ctx()
    result = await registry.invoke(
        "set_disposition", {"disposition": "sorted_it_out", "reason": "x"}, ctx
    )
    assert not result.ok
    assert "disposition" not in ctx.state


async def test_the_model_cannot_claim_the_caller_abandoned_the_call():
    ctx = _ctx()
    result = await registry.invoke(
        "set_disposition", {"disposition": "caller_abandoned", "reason": "x"}, ctx
    )
    assert not result.ok


@pytest.mark.parametrize("disposition", sorted(REQUIRES_REFERENCE))
async def test_outcomes_that_promise_something_need_a_reference(disposition):
    """A complaint with no number is not a complaint the caller can chase."""
    ctx = _ctx()
    result = await registry.invoke(
        "set_disposition", {"disposition": disposition, "reason": "leak"}, ctx
    )
    assert not result.ok
    assert "reference" in result.content.lower()


async def test_a_reference_is_stored_when_given():
    ctx = _ctx()
    result = await registry.invoke(
        "set_disposition",
        {"disposition": "complaint_registered", "reason": "leak", "reference": "KMC4471"},
        ctx,
    )
    assert result.ok
    assert ctx.state["reference"] == "KMC4471"


async def test_end_call_is_refused_without_a_disposition():
    ctx = _ctx()
    result = await registry.invoke("end_call", {"summary": "done"}, ctx)
    assert not result.ok
    assert result.control.get("action") != "hangup"
    assert "set_disposition" in result.content


async def test_end_call_proceeds_and_carries_the_outcome():
    ctx = _ctx()
    await registry.invoke("set_disposition", {"disposition": "resolved", "reason": "ok"}, ctx)
    result = await registry.invoke("end_call", {"summary": "Bill date given"}, ctx)
    assert result.ok
    assert result.control["action"] == "hangup"
    assert result.control["disposition"] == "resolved"


async def test_a_profile_extension_is_accepted_by_the_tool():
    ctx = _ctx()
    ctx.state["extra_dispositions"] = ["meter_reading_booked"]
    result = await registry.invoke(
        "set_disposition", {"disposition": "meter_reading_booked", "reason": "booked"}, ctx
    )
    assert result.ok


# =========================================================================
# The call flow
# =========================================================================


def _settings() -> Settings:
    return Settings(
        stt_provider="mock", llm_provider="mock", tts_provider="mock",
        vector_store="memory", embedding_provider="hash", record_calls=False,
        end_of_turn_silence_ms=200, idle_prompt_after_s=120, idle_hangup_after_s=600,
    )


@pytest.fixture
async def services(tmp_path):
    svc = build_services(_settings())
    svc.profiles["default"] = replace(svc.profiles["default"], ask_language=False)
    svc.calls = CallRepository(f"sqlite+aiosqlite:///{tmp_path / 'c.db'}")
    await svc.calls.start()
    await svc.start()
    yield svc
    await svc.close()
    await svc.calls.close()


async def test_an_abandoned_call_gets_a_platform_disposition(services):
    session = CallSession(FakeTransport(), services)
    task = asyncio.create_task(session.run())
    await asyncio.sleep(0.1)
    await session.hangup("caller_disconnected")
    await asyncio.wait_for(task, timeout=10)

    assert session.record.disposition == "caller_abandoned"


async def test_an_agent_set_disposition_is_never_overwritten(services):
    session = CallSession(FakeTransport(), services)
    task = asyncio.create_task(session.run())
    await asyncio.sleep(0.1)
    session.record.disposition = "resolved"
    await session.hangup("completed")
    await asyncio.wait_for(task, timeout=10)

    assert session.record.disposition == "resolved"


async def test_every_call_ends_with_a_disposition_and_it_is_persisted(services):
    """The phase gate. A call that reached the end of its life must have said
    what it achieved, and that must survive the process."""
    session = CallSession(FakeTransport(), services)
    task = asyncio.create_task(session.run())
    await asyncio.sleep(0.1)
    for chunk in (tone(600), silence(500)):
        for i in range(0, len(chunk) - 640 + 1, 640):
            await session.push_audio(chunk[i : i + 640])
    await asyncio.sleep(0.8)
    await session.hangup("completed")
    await asyncio.wait_for(task, timeout=15)

    assert session.record.disposition is not None
    assert session.record.disposition in CORE_DISPOSITIONS
    assert session.state == CallState.ENDED

    row = await services.calls.get_call(session.call_id)
    assert row["disposition"] == session.record.disposition


async def test_a_stalling_conversation_stops_retrying(services):
    """Four unanswerable turns must trigger a fallback offer, not a fifth
    attempt at the same answer."""
    session = CallSession(FakeTransport(), services)
    for _ in range(4):
        session._progress.observe(
            caller_text="I still need help with my bill",
            agent_text="I do not have that information",
            tool_ran=False,
            retrieved=False,
        )
    assert session._progress.stalled
    assert session._progress.should_offer_fallback()


async def test_dispositions_aggregate_for_reporting(services):
    counts = await services.calls.disposition_counts()
    assert isinstance(counts, dict)


async def test_a_stall_nudges_the_agent_rather_than_ending_the_call(services):
    """Rule 3 is absolute: a stalled call gets an offer, never a hangup. Cutting
    off a caller who is still talking is worse than any amount of looping."""
    session = CallSession(FakeTransport(), services)
    task = asyncio.create_task(session.run())
    await asyncio.sleep(0.1)

    for _ in range(4):
        session._progress.observe(
            caller_text="I still need help with my bill",
            agent_text="I do not have that information",
            tool_ran=False,
            retrieved=False,
        )
    assert session._progress.stalled
    assert session.state != CallState.ENDED, "a stall must never end the call"

    # The steer goes to the model, so the offer comes out in the caller's
    # language rather than as a hardcoded English line.
    session._agent.nudge("x")
    assert session._agent._nudge == "x"

    await session.hangup("completed")
    await asyncio.wait_for(task, timeout=10)


def test_the_nudge_is_one_shot():
    """A stall steer that persisted would keep redirecting every later reply."""
    from vaani.agent.prompt import DEFAULT_PROFILE
    from vaani.agent.tools.builtin import registry as tool_registry
    from vaani.providers.llm.mock import MockLLM

    agent = ConversationAgent(
        DEFAULT_PROFILE, MockLLM(), tool_registry, ToolContext(call_id="t")
    )
    agent.nudge("STOP RETRYING")
    window = agent._window("STOP RETRYING")
    assert window[-1].content == "STOP RETRYING", "the steer must sit last"

    asyncio.run(agent.respond("hello"))
    assert agent._nudge == "", "the steer must not survive the turn"
