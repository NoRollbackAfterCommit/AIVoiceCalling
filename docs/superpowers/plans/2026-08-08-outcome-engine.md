# Phase 2 — Outcome Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every call ends in a recorded disposition rather than looping, and carries a commitment the caller can hold the organisation to.

**Architecture:** A closed disposition vocabulary plus an objective on `AgentProfile`; a `ProgressTracker` holding need-identified / addressed / confirmed outside the model; a `set_disposition` tool that `end_call` refuses to run without; and platform-set dispositions for the cases where the caller is already gone.

**Tech Stack:** Python 3.11+, existing tool registry, SQLAlchemy 2.0 async, pytest.

## Global Constraints

- All Phase 1 constraints still apply: `from __future__ import annotations`, ruff at line-length 100 / py311, lazy provider imports, PCM16 mono at `SAMPLE_RATE`, nothing blocking on the event loop, `.venv/Scripts/python.exe`.
- The default pytest suite stays offline, mock-only and fast. No new network dependency.
- **The agent may never hang up on a talking caller.** Only confirmation, idle timeout, or an explicit caller request may end a call. This is absolute and is tested.
- Stall detection must not add an LLM round trip — it uses signals already present in the turn.
- Decisions from the spec, now fixed: fixed core vocabulary with per-profile extensions; stall threshold **3**, per profile; the agent **may** close after confirmation.

---

## Task 1: Disposition vocabulary and call objective

**Files:**
- Create: `vaani/agent/outcome.py`
- Modify: `vaani/agent/prompt.py` (`AgentProfile.objective`, `extra_dispositions`, `stall_after`; render the objective)
- Test: `tests/test_outcome_vocabulary.py`

**Interfaces:**
- Produces: `CORE_DISPOSITIONS: tuple[str, ...]`, `AGENT_SET: frozenset[str]`, `PLATFORM_SET: frozenset[str]`, `is_valid(disposition, profile) -> bool`, `allowed_for(profile) -> tuple[str, ...]`.

- [ ] **Step 1: Write the failing test**

```python
"""The vocabulary is closed on purpose: a government deployment compares
complaint volumes across departments, which only works if the words mean the
same thing everywhere."""

from __future__ import annotations

from vaani.agent.outcome import (
    AGENT_SET,
    CORE_DISPOSITIONS,
    PLATFORM_SET,
    allowed_for,
    is_valid,
)
from vaani.agent.prompt import AgentProfile, render_system_prompt


def test_core_vocabulary_is_complete():
    assert set(CORE_DISPOSITIONS) == {
        "resolved", "complaint_registered", "callback_scheduled", "transferred",
        "out_of_scope", "unresolved", "caller_abandoned", "idle_timeout",
        "capacity_rejected",
    }


def test_agent_and_platform_sets_partition_the_vocabulary():
    """A caller who has hung up cannot call a tool, so those dispositions must
    be platform-set and must not be offered to the model."""
    assert AGENT_SET | PLATFORM_SET == set(CORE_DISPOSITIONS)
    assert not (AGENT_SET & PLATFORM_SET)
    assert PLATFORM_SET == {"caller_abandoned", "idle_timeout", "capacity_rejected"}


def test_a_profile_may_extend_but_not_replace():
    profile = AgentProfile(key="t", extra_dispositions=["meter_reading_booked"])
    assert is_valid("meter_reading_booked", profile)
    assert is_valid("resolved", profile), "core vocabulary must survive extension"
    assert not is_valid("invented_outcome", profile)


def test_allowed_for_offers_agent_settable_only():
    profile = AgentProfile(key="t", extra_dispositions=["meter_reading_booked"])
    allowed = allowed_for(profile)
    assert "resolved" in allowed
    assert "meter_reading_booked" in allowed
    assert "caller_abandoned" not in allowed


def test_the_objective_reaches_the_system_prompt():
    profile = AgentProfile(
        key="t",
        objective="Resolve the billing question or register a complaint with a reference.",
    )
    prompt = render_system_prompt(profile)
    assert "register a complaint with a reference" in prompt


def test_a_profile_without_an_objective_still_renders():
    assert render_system_prompt(AgentProfile(key="t"))


def test_stall_threshold_defaults_to_three():
    assert AgentProfile(key="t").stall_after == 3
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_outcome_vocabulary.py -v`
Expected: `ModuleNotFoundError: No module named 'vaani.agent.outcome'`

- [ ] **Step 3: Write `vaani/agent/outcome.py`**

```python
"""How a call ended, as a closed vocabulary.

Free-text outcomes are unusable in aggregate: three departments will write
"complaint raised", "Complaint Registered" and "cmplnt" for the same thing, and
the first question a government buyer asks is how many complaints came in last
month. So the core list is fixed and a profile may only extend it.

The split matters as much as the list. A caller who has hung up cannot call a
tool, so those dispositions are set by the platform and are never offered to the
model — otherwise it will helpfully claim the caller abandoned a call they are
still on.
"""

from __future__ import annotations

from typing import Any

CORE_DISPOSITIONS: tuple[str, ...] = (
    "resolved",
    "complaint_registered",
    "callback_scheduled",
    "transferred",
    "out_of_scope",
    "unresolved",
    "caller_abandoned",
    "idle_timeout",
    "capacity_rejected",
)

# Only the platform can know these: the caller is already gone.
PLATFORM_SET: frozenset[str] = frozenset(
    {"caller_abandoned", "idle_timeout", "capacity_rejected"}
)
AGENT_SET: frozenset[str] = frozenset(CORE_DISPOSITIONS) - PLATFORM_SET

# Dispositions that must carry a reference the caller can quote back.
REQUIRES_REFERENCE: frozenset[str] = frozenset(
    {"complaint_registered", "callback_scheduled"}
)


def allowed_for(profile: Any) -> tuple[str, ...]:
    """What the model may choose from."""
    extra = tuple(getattr(profile, "extra_dispositions", ()) or ())
    return tuple(sorted(AGENT_SET)) + extra


def is_valid(disposition: str, profile: Any) -> bool:
    extra = set(getattr(profile, "extra_dispositions", ()) or ())
    return disposition in set(CORE_DISPOSITIONS) | extra
```

- [ ] **Step 4: Extend `AgentProfile`**

In `vaani/agent/prompt.py`, add to the dataclass:

```python
    # What a successful call achieves. Frames every other instruction, so it
    # renders above the policies.
    objective: str = ""
    # Domain outcomes this deployment needs on top of the core vocabulary.
    extra_dispositions: list[str] = field(default_factory=list)
    # Unproductive turns before the agent stops re-asking and offers a fallback.
    stall_after: int = 3
```

- [ ] **Step 5: Render the objective**

In `render_system_prompt`, insert immediately after the role line:

```python
    if profile.objective:
        sections += ["", "## What this call is for", profile.objective]
```

- [ ] **Step 6: Run the test and commit**

Run: `.venv/Scripts/python.exe -m pytest tests/test_outcome_vocabulary.py -v && .venv/Scripts/python.exe -m ruff check .`

```bash
git add vaani/agent/outcome.py vaani/agent/prompt.py tests/test_outcome_vocabulary.py
git commit -m "feat: closed disposition vocabulary and call objective"
```

---

## Task 2: Progress tracker

**Files:**
- Create: `vaani/pipeline/progress.py`
- Test: `tests/test_progress_tracker.py`

**Interfaces:**
- Produces: `ProgressTracker(stall_after: int = 3)` with `observe(caller_text: str, agent_text: str, tool_ran: bool, retrieved: bool) -> None`, `confirm_closing() -> None`, properties `identified`, `addressed`, `confirmed`, `stalled`, `unproductive_turns`, and `should_offer_fallback() -> bool`.

**Pure logic, no dependencies.** That is the point — the closing decision must be inspectable and testable without a model.

- [ ] **Step 1: Write the failing test**

```python
"""Closing policy, held outside the model.

The model decides what to say. Whether the call is going anywhere is tracked
here, from signals already present in the turn, so no extra round trip lands in
the latency budget."""

from __future__ import annotations

from vaani.pipeline.progress import ProgressTracker


def _turn(t: ProgressTracker, caller="", agent="", tool=False, retrieved=False):
    t.observe(caller_text=caller, agent_text=agent, tool_ran=tool, retrieved=retrieved)


def test_starts_with_nothing_established():
    t = ProgressTracker()
    assert not t.identified and not t.addressed and not t.confirmed


def test_a_substantive_caller_turn_identifies_the_need():
    t = ProgressTracker()
    _turn(t, caller="I want to check my electricity bill", agent="Let me look")
    assert t.identified


def test_retrieval_that_cleared_the_threshold_addresses_it():
    t = ProgressTracker()
    _turn(t, caller="what is my due date", agent="The fifteenth", retrieved=True)
    assert t.addressed


def test_a_tool_running_addresses_it():
    t = ProgressTracker()
    _turn(t, caller="register a complaint", agent="Done, reference 4471", tool=True)
    assert t.addressed


def test_an_answer_with_no_retrieval_and_no_tool_does_not_address():
    """Otherwise "I do not have that information" counts as resolving the call."""
    t = ProgressTracker()
    _turn(t, caller="what is my due date", agent="I do not have that information")
    assert t.identified
    assert not t.addressed


def test_stalls_after_three_unproductive_turns():
    t = ProgressTracker(stall_after=3)
    for _ in range(2):
        _turn(t, caller="I still need help", agent="Could you repeat that")
    assert not t.stalled
    _turn(t, caller="I still need help", agent="Could you repeat that")
    assert t.stalled


def test_progress_resets_the_unproductive_count():
    t = ProgressTracker(stall_after=3)
    _turn(t, caller="hello", agent="how can I help")
    _turn(t, caller="hello", agent="how can I help")
    _turn(t, caller="my bill", agent="the fifteenth", retrieved=True)
    assert t.unproductive_turns == 0
    assert not t.stalled


def test_a_repeated_caller_utterance_counts_as_unproductive():
    """The strongest signal the last answer did not land."""
    t = ProgressTracker(stall_after=2)
    _turn(t, caller="what is my due date", agent="The fifteenth", retrieved=True)
    _turn(t, caller="what is my due date", agent="The fifteenth", retrieved=True)
    _turn(t, caller="what is my due date", agent="The fifteenth", retrieved=True)
    assert t.stalled


def test_fallback_is_offered_once_when_stalled():
    t = ProgressTracker(stall_after=1)
    _turn(t, caller="help", agent="sorry")
    assert t.should_offer_fallback()
    assert not t.should_offer_fallback(), "must not nag on every subsequent turn"


def test_confirmation_is_explicit():
    t = ProgressTracker()
    _turn(t, caller="my bill", agent="the fifteenth", retrieved=True)
    assert not t.confirmed
    t.confirm_closing()
    assert t.confirmed
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_progress_tracker.py -v`
Expected: `ModuleNotFoundError: No module named 'vaani.pipeline.progress'`

- [ ] **Step 3: Write `vaani/pipeline/progress.py`**

```python
"""Whether the call is going anywhere.

The model decides what to say; this decides whether that is working. Keeping the
judgement out of the model means it is inspectable, deterministic and testable
without a network call — and it costs nothing in the latency budget, because
every signal it uses is already present in the turn.
"""

from __future__ import annotations

import re

from vaani.core.logging import get_logger

log = get_logger(__name__)

# Below this a caller utterance is an acknowledgement, not a request.
_MIN_SUBSTANTIVE_CHARS = 8


class ProgressTracker:
    def __init__(self, stall_after: int = 3) -> None:
        self._stall_after = max(1, stall_after)
        self.identified = False
        self.addressed = False
        self.confirmed = False
        self.unproductive_turns = 0
        self._last_caller = ""
        self._fallback_offered = False

    def observe(
        self, *, caller_text: str, agent_text: str, tool_ran: bool, retrieved: bool
    ) -> None:
        caller = caller_text.strip()

        if not self.identified and len(caller) >= _MIN_SUBSTANTIVE_CHARS:
            self.identified = True

        # An answer only counts when it came from somewhere: retrieval that
        # cleared the relevance threshold, or a tool that did something. Without
        # this, "I do not have that information" would resolve the call.
        progressed = tool_ran or retrieved
        if progressed:
            self.addressed = True

        repeated = bool(caller) and _normalise(caller) == _normalise(self._last_caller)
        self._last_caller = caller

        if progressed and not repeated:
            self.unproductive_turns = 0
        else:
            self.unproductive_turns += 1

    def confirm_closing(self) -> None:
        self.confirmed = True

    @property
    def stalled(self) -> bool:
        return self.unproductive_turns >= self._stall_after

    def should_offer_fallback(self) -> bool:
        """True once, the first time the call stalls. Offering a fallback on
        every subsequent turn is its own kind of loop."""
        if self.stalled and not self._fallback_offered:
            self._fallback_offered = True
            log.info("call stalled, offering fallback",
                     extra={"unproductive_turns": self.unproductive_turns})
            return True
        return False


def _normalise(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.lower()).strip()
```

- [ ] **Step 4: Run the test and commit**

Run: `.venv/Scripts/python.exe -m pytest tests/test_progress_tracker.py -v && .venv/Scripts/python.exe -m ruff check .`

```bash
git add vaani/pipeline/progress.py tests/test_progress_tracker.py
git commit -m "feat: track conversation progress and stall detection"
```

---

## Task 3: set_disposition tool and end_call gating

**Files:**
- Modify: `vaani/agent/tools/builtin.py`
- Test: `tests/test_disposition_tool.py`

**Interfaces:**
- Consumes: `AGENT_SET`, `REQUIRES_REFERENCE`, `is_valid` from Task 1.
- Produces: a `set_disposition` tool storing `ctx.state["disposition"]`, `["disposition_reason"]`, `["reference"]`; `end_call` refusing without one.

- [ ] **Step 1: Write the failing test**

```python
"""end_call refuses without a disposition, so a complete audit trail follows by
construction rather than by anyone remembering."""

from __future__ import annotations

import pytest

from vaani.agent.tools.base import ToolContext
from vaani.agent.tools.builtin import registry


def _ctx() -> ToolContext:
    return ToolContext(call_id="t")


async def test_set_disposition_records_it():
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


async def test_a_platform_only_disposition_is_refused():
    """The model must not be able to claim a caller abandoned a call it is
    still talking on."""
    ctx = _ctx()
    result = await registry.invoke(
        "set_disposition", {"disposition": "caller_abandoned", "reason": "x"}, ctx
    )
    assert not result.ok


async def test_a_disposition_needing_a_reference_is_refused_without_one():
    ctx = _ctx()
    result = await registry.invoke(
        "set_disposition", {"disposition": "complaint_registered", "reason": "leak"}, ctx
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
    assert "disposition" in result.content.lower()


async def test_end_call_proceeds_once_a_disposition_is_set():
    ctx = _ctx()
    await registry.invoke("set_disposition", {"disposition": "resolved", "reason": "ok"}, ctx)
    result = await registry.invoke("end_call", {"summary": "Bill date given"}, ctx)
    assert result.ok
    assert result.control["action"] == "hangup"
    assert result.control["disposition"] == "resolved"
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_disposition_tool.py -v`
Expected: failures — the tool does not exist and `end_call` hangs up unconditionally.

- [ ] **Step 3: Add the tool**

In `vaani/agent/tools/builtin.py`, above `end_call`:

```python
@registry.tool(
    name="set_disposition",
    description=(
        "Record how this call ended, before ending it. Required: end_call will "
        "not run without it. Choose the outcome that actually happened."
    ),
    parameters={
        "type": "object",
        "properties": {
            "disposition": {
                "type": "string",
                "description": (
                    "One of: resolved, complaint_registered, callback_scheduled, "
                    "transferred, out_of_scope, unresolved."
                ),
            },
            "reason": {"type": "string", "description": "One line of justification."},
            "reference": {
                "type": "string",
                "description": "The reference number, for a complaint or callback.",
            },
        },
        "required": ["disposition", "reason"],
    },
)
async def set_disposition(
    disposition: str, reason: str, ctx: ToolContext, reference: str = ""
) -> ToolResult:
    from vaani.agent.outcome import AGENT_SET, REQUIRES_REFERENCE

    profile_extra = set(ctx.state.get("extra_dispositions") or ())
    if disposition not in (AGENT_SET | profile_extra):
        return ToolResult(
            content=(
                f"'{disposition}' is not a valid outcome. Choose one of: "
                f"{', '.join(sorted(AGENT_SET | profile_extra))}."
            ),
            ok=False,
        )

    if disposition in REQUIRES_REFERENCE and not reference.strip():
        return ToolResult(
            content=(
                f"A {disposition} outcome needs the reference number you gave the "
                "caller. Call the tool again with it."
            ),
            ok=False,
        )

    ctx.state["disposition"] = disposition
    ctx.state["disposition_reason"] = reason
    if reference.strip():
        ctx.state["reference"] = reference.strip()

    return ToolResult(
        content="Outcome recorded. You may now end the call.",
        data={"disposition": disposition, "reference": reference},
    )
```

- [ ] **Step 4: Gate `end_call`**

Replace the body of `end_call`:

```python
async def end_call(summary: str, ctx: ToolContext) -> ToolResult:
    disposition = ctx.state.get("disposition")
    if not disposition:
        # Refusing here is what makes the audit trail complete by construction.
        return ToolResult(
            content=(
                "Record the outcome first with set_disposition, then end the call."
            ),
            ok=False,
        )
    return ToolResult(
        content="Say your closing line, then stop.",
        control={
            "action": "hangup",
            "summary": summary,
            "disposition": disposition,
            "reference": ctx.state.get("reference"),
        },
        data={"summary": summary, "disposition": disposition},
    )
```

- [ ] **Step 5: Offer `set_disposition` to every profile**

In `vaani/agent/prompt.py`, add `"set_disposition"` to `AgentProfile.tools`'s default list, next to `search_knowledge` and `transfer_to_human`.

- [ ] **Step 6: Run the tests and commit**

Run: `.venv/Scripts/python.exe -m pytest -q && .venv/Scripts/python.exe -m ruff check .`

```bash
git add vaani/agent/tools/builtin.py vaani/agent/prompt.py tests/test_disposition_tool.py
git commit -m "feat: require a disposition before a call may end"
```

---

## Task 4: Closing rules in the session

**Files:**
- Modify: `vaani/pipeline/session.py`
- Modify: `vaani/agent/prompt.py` (closing rules in the prompt)
- Test: `tests/test_closing_policy.py`

**Interfaces:**
- Consumes: `ProgressTracker` (Task 2), `PLATFORM_SET` (Task 1).
- Produces: `CallRecord.disposition`, `.disposition_reason`, `.reference`; the session sets platform dispositions.

- [ ] **Step 1: Write the failing test**

```python
"""The rules that stop a call looping, and the one that stops it being rude."""

from __future__ import annotations

import asyncio

import pytest

from vaani.config import Settings
from vaani.core.registry import build_services
from vaani.pipeline.session import CallSession, CallState

from .test_pipeline import FakeTransport, silence, tone


def _settings(**over) -> Settings:
    base = dict(
        stt_provider="mock", llm_provider="mock", tts_provider="mock",
        vector_store="memory", embedding_provider="hash", record_calls=False,
        end_of_turn_silence_ms=200, idle_prompt_after_s=120, idle_hangup_after_s=600,
    )
    base.update(over)
    return Settings(**base)


@pytest.fixture
async def services():
    svc = build_services(_settings())
    await svc.start()
    yield svc
    await svc.close()


async def test_an_abandoned_call_gets_a_platform_disposition(services):
    """The caller is gone, so the model cannot record anything."""
    session = CallSession(transport=FakeTransport(), services=services)
    task = asyncio.create_task(session.run())
    await asyncio.sleep(0.1)
    await session.hangup("caller_disconnected")
    await asyncio.wait_for(task, timeout=10)

    assert session.record.disposition == "caller_abandoned"


async def test_an_idle_timeout_gets_its_own_disposition(services):
    session = CallSession(
        transport=FakeTransport(), services=services,
        settings=_settings(idle_prompt_after_s=2.0, idle_hangup_after_s=5.0),
    )
    task = asyncio.create_task(session.run())
    await asyncio.wait_for(task, timeout=20)
    assert session.record.disposition == "idle_timeout"


async def test_an_agent_set_disposition_is_not_overwritten(services):
    """A completed call must keep the outcome the agent recorded."""
    session = CallSession(transport=FakeTransport(), services=services)
    session.record.disposition = "resolved"
    task = asyncio.create_task(session.run())
    await asyncio.sleep(0.1)
    await session.hangup("completed")
    await asyncio.wait_for(task, timeout=10)
    assert session.record.disposition == "resolved"


async def test_the_session_never_hangs_up_while_the_caller_is_speaking(services):
    """Absolute rule: cutting someone off is worse than any amount of looping."""
    session = CallSession(transport=FakeTransport(), services=services)
    task = asyncio.create_task(session.run())
    await asyncio.sleep(0.1)
    for chunk in (tone(400),):
        for i in range(0, len(chunk) - 640 + 1, 640):
            await session.push_audio(chunk[i : i + 640])
    session._progress.unproductive_turns = 99  # force a stall
    await asyncio.sleep(0.3)
    assert session.state != CallState.ENDED
    await session.hangup("completed")
    await asyncio.wait_for(task, timeout=10)


async def test_closing_rules_reach_the_prompt():
    from vaani.agent.prompt import AgentProfile, render_system_prompt

    prompt = render_system_prompt(AgentProfile(key="t")).lower()
    assert "once" in prompt
    assert "set_disposition" in prompt
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_closing_policy.py -v`
Expected: `AttributeError` on `record.disposition` and `session._progress`.

- [ ] **Step 3: Extend `CallRecord`**

In `vaani/pipeline/session.py`, next to `outcome`:

```python
    # The business result, as opposed to `outcome`, which is the transport
    # result. A call can end cleanly and achieve nothing.
    disposition: str | None = None
    disposition_reason: str | None = None
    reference: str | None = None
```

Add all three to `to_dict()`.

- [ ] **Step 4: Hold a tracker on the session**

In `__init__`, next to `self._language`:

```python
        self._progress = ProgressTracker(stall_after=self._profile.stall_after)
```

Import: `from vaani.pipeline.progress import ProgressTracker`.

Make the profile's extensions reachable by the tool, in the same place `ToolContext` is built:

```python
        self._tool_ctx.state["extra_dispositions"] = list(self._profile.extra_dispositions)
```

- [ ] **Step 5: Feed the tracker each turn**

In `_process_turn`, after the turn is appended to `self.record.turns`:

```python
        retrieved = any(t["name"] == "search_knowledge" for t in turn.tool_calls)
        self._progress.observe(
            caller_text=transcript.text,
            agent_text=turn.text,
            tool_ran=bool(turn.tool_calls),
            retrieved=retrieved,
        )
```

- [ ] **Step 6: Carry the disposition off the hangup control**

Where `action == "hangup"` is handled, copy the tool's values onto the record:

```python
            self.record.disposition = turn.control.get("disposition")
            self.record.reference = turn.control.get("reference")
            self.record.disposition_reason = self._tool_ctx.state.get("disposition_reason")
```

- [ ] **Step 7: Set platform dispositions in `_finish`**

Immediately before the repository write:

```python
        if self.record.disposition is None:
            # The caller is gone, so nothing could have recorded an outcome.
            self.record.disposition = {
                "caller_disconnected": "caller_abandoned",
                "idle_timeout": "idle_timeout",
                "rejected": "capacity_rejected",
            }.get(self.record.outcome, "unresolved")
```

- [ ] **Step 8: Add the closing rules to the prompt**

In `vaani/agent/prompt.py`, add a constant and append it to `sections`:

```python
CLOSING_RULES = """\
Drive the call to a conclusion rather than letting it drift:
- Establish what the caller needs within the first two exchanges.
- Once you have answered, confirm once — "is there anything else?" — and if the
  caller says no, record the outcome and end the call. Do not ask twice.
- If you have been unable to help after three exchanges, stop retrying. Offer to
  register a complaint, schedule a callback, or transfer to an officer.
- Before ending any call, record what happened with set_disposition.
- Never hang up while the caller is still speaking or has more to say."""
```

- [ ] **Step 9: Run everything and commit**

Run: `.venv/Scripts/python.exe -m pytest -q && .venv/Scripts/python.exe -m ruff check .`

```bash
git add vaani/pipeline/session.py vaani/agent/prompt.py tests/test_closing_policy.py
git commit -m "feat: closing policy and platform-set dispositions"
```

---

## Task 5: Persist the disposition

**Files:**
- Modify: `vaani/db/models.py`, `vaani/db/repository.py`
- Modify: `vaani/api/routes.py` (analytics by disposition)
- Test: `tests/test_disposition_persistence.py`

**Interfaces:**
- Consumes: `CallRecord.disposition` (Task 4).
- Produces: `calls.disposition`, `.disposition_reason`, `.reference`; `CallRepository.disposition_counts()`.

- [ ] **Step 1: Write the failing test**

```python
"""A disposition that is not persisted is not an audit trail."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from vaani.db.repository import CallRepository


@dataclass
class Rec:
    call_id: str = "c1"
    agent_key: str = "default"
    direction: str = "inbound"
    caller_number: str | None = None
    started_at: float = 1000.0
    ended_at: float | None = 1100.0
    outcome: str = "completed"
    summary: str | None = None
    language: str | None = "hi-IN"
    duration_s: float = 100.0
    disposition: str | None = "complaint_registered"
    disposition_reason: str | None = "Water leak on Park Street"
    reference: str | None = "KMC4471"


@pytest.fixture
async def repo(tmp_path):
    r = CallRepository(f"sqlite+aiosqlite:///{tmp_path / 'c.db'}")
    await r.start()
    yield r
    await r.close()


async def test_disposition_round_trips(repo):
    await repo.create_call(Rec())
    await repo.finish_call(Rec())
    row = await repo.get_call("c1")
    assert row["disposition"] == "complaint_registered"
    assert row["reference"] == "KMC4471"
    assert row["disposition_reason"] == "Water leak on Park Street"


async def test_disposition_counts_aggregate(repo):
    for i, d in enumerate(["resolved", "resolved", "unresolved"]):
        rec = Rec(call_id=f"c{i}", disposition=d, reference=None)
        await repo.create_call(rec)
        await repo.finish_call(rec)
    counts = await repo.disposition_counts()
    assert counts["resolved"] == 2
    assert counts["unresolved"] == 1
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_disposition_persistence.py -v`
Expected: `KeyError: 'disposition'`.

- [ ] **Step 3: Add the columns**

In `vaani/db/models.py`, on `CallRow`:

```python
    disposition: Mapped[str | None] = mapped_column(String(48), nullable=True, index=True)
    disposition_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    reference: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
```

- [ ] **Step 4: Write them in `finish_call`**

In `vaani/db/repository.py`, inside `finish_call`, alongside the existing assignments:

```python
            row.disposition = getattr(record, "disposition", None)
            row.disposition_reason = getattr(record, "disposition_reason", None)
            row.reference = getattr(record, "reference", None)
```

- [ ] **Step 5: Add the aggregate**

```python
    async def disposition_counts(self) -> dict[str, int]:
        """What actually happened across calls — the first question a government
        buyer asks, and the reason the vocabulary is closed."""
        from sqlalchemy import func

        async with self._sessions() as session:
            result = await session.execute(
                select(CallRow.disposition, func.count())
                .where(CallRow.disposition.is_not(None))
                .group_by(CallRow.disposition)
            )
            return {row[0]: row[1] for row in result}
```

- [ ] **Step 6: Expose it**

In `vaani/api/routes.py`, extend the analytics endpoint to include `dispositions` when a repository is configured:

```python
    repository = request.app.state.services.calls
    if repository is not None:
        payload["dispositions"] = await repository.disposition_counts()
```

Adapt to the existing function's variable name for the response dict.

- [ ] **Step 7: Note the schema change**

Phase 1 has no migrations, so an existing `data/vaani.db` will lack the new columns. Add to `README.md` under a "Upgrading" note:

```markdown
Phase 2 adds columns to `calls`. There are no migrations before phase 3, so
delete `data/vaani.db` (or `docker volume rm euphoria-voice_vaani-data`) once
after upgrading. Pilot call records are not yet production data.
```

- [ ] **Step 8: Run and commit**

Run: `.venv/Scripts/python.exe -m pytest -q && .venv/Scripts/python.exe -m ruff check .`

```bash
git add vaani/db/ vaani/api/routes.py README.md tests/test_disposition_persistence.py
git commit -m "feat: persist and aggregate call dispositions"
```

---

## Task 6: Acceptance — a call that used to loop now ends

**Files:**
- Test: `tests/test_conclusive_calls.py`

**This is the phase gate.** Everything above is machinery; this is the behaviour that was asked for.

- [ ] **Step 1: Write the test**

```python
"""The phase gate: a conversation that would previously loop now terminates
with a recorded outcome."""

from __future__ import annotations

import asyncio

import pytest

from vaani.agent.outcome import CORE_DISPOSITIONS
from vaani.config import Settings
from vaani.core.registry import build_services
from vaani.db.repository import CallRepository
from vaani.pipeline.session import CallSession, CallState

from .test_pipeline import FakeTransport, silence, tone


def _settings() -> Settings:
    return Settings(
        stt_provider="mock", llm_provider="mock", tts_provider="mock",
        vector_store="memory", embedding_provider="hash", record_calls=False,
        end_of_turn_silence_ms=200, idle_prompt_after_s=120, idle_hangup_after_s=600,
    )


@pytest.fixture
async def services(tmp_path):
    svc = build_services(_settings())
    svc.calls = CallRepository(f"sqlite+aiosqlite:///{tmp_path / 'c.db'}")
    await svc.calls.start()
    await svc.start()
    yield svc
    await svc.close()
    await svc.calls.close()


async def test_every_call_ends_with_a_disposition(services):
    session = CallSession(transport=FakeTransport(), services=services)
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

    row = await services.calls.get_call(session.call_id)
    assert row["disposition"] is not None


async def test_a_stalling_conversation_stops_retrying(services):
    """The caller repeats the same thing four times. The agent must stop
    re-answering and offer a fallback rather than loop."""
    session = CallSession(transport=FakeTransport(), services=services)
    for _ in range(4):
        session._progress.observe(
            caller_text="I still need help with my bill",
            agent_text="I do not have that information",
            tool_ran=False,
            retrieved=False,
        )
    assert session._progress.stalled
    assert session._progress.should_offer_fallback()


async def test_the_call_reaches_a_terminal_state(services):
    session = CallSession(transport=FakeTransport(), services=services)
    task = asyncio.create_task(session.run())
    await asyncio.sleep(0.1)
    await session.hangup("completed")
    await asyncio.wait_for(task, timeout=10)
    assert session.state == CallState.ENDED
```

- [ ] **Step 2: Run it**

Run: `.venv/Scripts/python.exe -m pytest tests/test_conclusive_calls.py -v`
Expected: PASS. If not, the gap is in Tasks 4–5, not here.

- [ ] **Step 3: Verify against the running container**

```bash
docker compose build vaani && docker compose up -d
# place a call, then:
curl -s http://127.0.0.1:8090/api/calls | python -m json.tool | grep disposition
```

Every returned call must carry a non-null disposition.

- [ ] **Step 4: Commit**

```bash
.venv/Scripts/python.exe -m pytest -q && .venv/Scripts/python.exe -m ruff check .
git add tests/test_conclusive_calls.py
git commit -m "test: every call reaches a recorded outcome"
```

---

## Acceptance

- [ ] Every call in the database has a non-null `disposition`.
- [ ] `end_call` cannot run without one — verified by test, not by inspection.
- [ ] A stalling conversation offers a fallback on the third unproductive turn, once.
- [ ] The agent never hangs up while the caller is speaking.
- [ ] `/api/analytics/summary` reports counts by disposition.
- [ ] `pytest -q` and `ruff check .` clean; the default suite stays offline.

## Deferred

Sentiment, quality scoring and follow-up automation are reporting features built on the disposition, not conversation logic. Retention enforcement and migrations remain phase 3.
