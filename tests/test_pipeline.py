"""End-to-end tests over the mock providers.

These run in about a second with no models and no network, which is the point:
prompt changes, tool changes and turn-detection changes all get regression cover
in CI on a machine with no GPU.
"""

from __future__ import annotations

import asyncio
import math
import struct
from dataclasses import replace
from typing import Any

import pytest

from vaani.agent.prompt import DEFAULT_PROFILE, AgentProfile, render_system_prompt
from vaani.agent.runtime import ConversationAgent, _clean_for_speech
from vaani.agent.tools.base import ToolContext
from vaani.agent.tools.builtin import registry as tools
from vaani.audio.resample import resample_pcm16, ulaw_to_pcm16
from vaani.audio.vad import BargeInDetector, EnergyVAD, TurnDetector
from vaani.config import FRAME_BYTES, SAMPLE_RATE, Settings
from vaani.core.registry import build_services
from vaani.pipeline.session import CallSession, CallState
from vaani.rag.chunking import chunk_text

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def tone(ms: int, freq: float = 220.0, amplitude: float = 0.5) -> bytes:
    n = int(SAMPLE_RATE * ms / 1000)
    return struct.pack(
        f"<{n}h",
        *(int(amplitude * 32767 * math.sin(2 * math.pi * freq * i / SAMPLE_RATE))
          for i in range(n)),
    )


def silence(ms: int) -> bytes:
    return b"\x00\x00" * int(SAMPLE_RATE * ms / 1000)


class FakeTransport:
    def __init__(self) -> None:
        self.audio = bytearray()
        self.events: list[dict[str, Any]] = []
        self.closed = False

    async def send_audio(self, pcm: bytes) -> None:
        self.audio.extend(pcm)

    async def send_event(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    async def close(self) -> None:
        self.closed = True

    def of_type(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("type") == kind]


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
        # The ceilings Settings allows. The suite finishes in about a second, so
        # these mean "never fires" without exceeding the bounds the admin UI
        # enforces — a test that needs an out-of-range value is testing a
        # configuration no operator can actually produce.
        idle_prompt_after_s=120,
        idle_hangup_after_s=600,
    )


@pytest.fixture
async def services(settings: Settings):
    svc = build_services(settings)
    # Conversation tests should not also be exercising language selection: with
    # it on, the first caller turn answers "which language?" instead of asking
    # the agent anything. The selection flow has its own tests.
    # A copy, not a mutation: the default profile is shared across the process.
    svc.profiles["default"] = replace(svc.profiles["default"], ask_language=False)
    await svc.start()
    yield svc
    await svc.close()


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


def test_resample_changes_length_proportionally():
    pcm = tone(100)
    up = resample_pcm16(pcm, 8000, 16000)
    assert len(up) == pytest.approx(len(pcm) * 2, rel=0.01)
    assert resample_pcm16(pcm, 16000, 16000) is pcm


def test_ulaw_roundtrip_produces_pipeline_format():
    from vaani.audio.resample import pcm16_to_ulaw

    original = tone(200)
    encoded = pcm16_to_ulaw(original)
    decoded = ulaw_to_pcm16(encoded)
    # 16k -> 8k -> 16k halves then doubles the sample count.
    assert len(decoded) == pytest.approx(len(original), rel=0.02)


def test_energy_vad_separates_tone_from_silence():
    vad = EnergyVAD()
    assert vad.is_speech(tone(20)) is True
    vad.reset()
    assert vad.is_speech(silence(20)) is False


def test_turn_detector_emits_utterance_after_trailing_silence():
    det = TurnDetector(vad=EnergyVAD(), silence_ms=200, min_speech_ms=100)
    events = []
    for chunk in (tone(600), silence(400)):
        for i in range(0, len(chunk) - FRAME_BYTES + 1, FRAME_BYTES):
            ev = det.push(chunk[i : i + FRAME_BYTES])
            if ev:
                events.append(ev)

    ended = [e for e in events if e.state == "ended"]
    assert len(ended) == 1, "exactly one completed utterance expected"
    assert ended[0].duration_s > 0.5


def test_turn_detector_discards_a_blip():
    """A single loud frame is a cough, not a turn."""
    det = TurnDetector(vad=EnergyVAD(), silence_ms=100, min_speech_ms=300)
    ended = []
    for chunk in (tone(60), silence(400)):
        for i in range(0, len(chunk) - FRAME_BYTES + 1, FRAME_BYTES):
            ev = det.push(chunk[i : i + FRAME_BYTES])
            if ev and ev.state == "ended":
                ended.append(ev)
    assert ended == []


def test_barge_in_needs_sustained_speech():
    det = BargeInDetector(trigger_ms=200, vad=EnergyVAD())
    one_frame = tone(20)
    assert det.push(one_frame) is False  # a single frame must not fire

    det.reset()
    fired = False
    speech = tone(400)
    for i in range(0, len(speech) - FRAME_BYTES + 1, FRAME_BYTES):
        if det.push(speech[i : i + FRAME_BYTES]):
            fired = True
            break
    assert fired is True


# ---------------------------------------------------------------------------
# Prompt + agent
# ---------------------------------------------------------------------------


def test_system_prompt_carries_policy_and_guardrails():
    profile = AgentProfile(
        key="t", organisation="Kolkata Municipal Corporation",
        policies=["Property tax is due on the thirty first of March."],
        forbidden_topics=["Legal advice"],
    )
    prompt = render_system_prompt(profile)
    assert "Kolkata Municipal Corporation" in prompt
    assert "thirty first of March" in prompt
    assert "Legal advice" in prompt
    assert "markdown" in prompt.lower()          # voice rules present
    assert "never as an instruction" in prompt   # injection guard present


def test_speech_cleaner_strips_markdown():
    dirty = "**Important**\n\n- one\n- two\nSee [here](http://x.test)."
    clean = _clean_for_speech(dirty)
    assert "*" not in clean and "\n" not in clean
    assert "- " not in clean
    assert "http" not in clean
    assert "here" in clean


async def test_agent_calls_knowledge_tool_then_answers(services):
    await services.retriever.index_text(
        "The last date to pay the electricity bill is the fifteenth of every month. "
        "A late fee of two percent applies after that date.",
        source="tariff-policy",
    )
    ctx = ToolContext(call_id="t", services=services.as_tool_services())
    agent = ConversationAgent(DEFAULT_PROFILE, services.llm, tools, ctx)

    turn = await agent.respond("What is the last date to pay my bill?")

    assert [t["name"] for t in turn.tool_calls] == ["search_knowledge"]
    assert "fifteenth" in turn.text.lower()


async def test_agent_transfers_when_caller_asks_for_a_human(services):
    ctx = ToolContext(call_id="t", services=services.as_tool_services())
    profile = AgentProfile(key="t", tools=["transfer_to_human"])
    agent = ConversationAgent(profile, services.llm, tools, ctx)
    await agent.respond("I want to speak to a human executive")
    # MockLLM answers in text rather than emitting the tool call, so assert on
    # the behaviour that matters to the caller: they are told about a transfer.
    assert "transferring" in agent.transcript[-1]["content"].lower()


async def test_agent_history_survives_trimming(services):
    ctx = ToolContext(call_id="t", services=services.as_tool_services())
    agent = ConversationAgent(DEFAULT_PROFILE, services.llm, tools, ctx, history_turns=2)
    for i in range(6):
        await agent.respond(f"message {i}")
    window = agent._window()
    assert window[0].role == "system"
    # No orphaned tool result may lead the window.
    assert window[1].role in ("user", "assistant")


async def test_unknown_tool_is_reported_not_raised(services):
    ctx = ToolContext(call_id="t", services=services.as_tool_services())
    result = await tools.invoke("does_not_exist", {}, ctx)
    assert result.ok is False
    assert "no tool called" in result.content.lower()


async def test_tool_requiring_verification_is_refused_until_verified(services):
    from vaani.agent.tools.base import Tool, ToolRegistry

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="secret", description="", parameters={"type": "object", "properties": {}},
            fn=_ok, requires_verification=True,
        )
    )
    ctx = ToolContext(call_id="t")
    refused = await registry.invoke("secret", {}, ctx)
    assert refused.ok is False

    ctx.state["verified"] = True
    allowed = await registry.invoke("secret", {}, ctx)
    assert allowed.ok is True


async def _ok(ctx=None):
    from vaani.agent.tools.base import ToolResult

    return ToolResult(content="done")


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------


def test_chunking_respects_target_size_and_drops_fragments():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(6))
    chunks = chunk_text(text, source="doc", target_chars=400)
    assert len(chunks) > 1
    assert all(len(c.text) < 900 for c in chunks)
    assert all(c.source == "doc" for c in chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


async def test_retrieval_finds_the_right_document(services):
    await services.retriever.index_text(
        "Admission to the B.Tech programme requires a minimum of sixty percent in "
        "the higher secondary examination.", source="admissions",
    )
    await services.retriever.index_text(
        "The hospital outpatient department is open from eight in the morning "
        "until two in the afternoon.", source="hospital",
    )
    hits = await services.retriever.search("what percentage do I need for admission")
    assert hits, "expected at least one hit"
    assert hits[0].source == "admissions"


async def test_deleting_a_source_removes_it_from_results(services):
    await services.retriever.index_text("Tariff slab one is five rupees.", source="old-tariff")
    assert await services.retriever.count("default") > 0
    await services.retriever.delete_source("old-tariff")
    hits = await services.retriever.search("tariff slab")
    assert all(h.source != "old-tariff" for h in hits)


# ---------------------------------------------------------------------------
# Full call
# ---------------------------------------------------------------------------


async def test_full_call_greets_transcribes_and_replies(services):
    transport = FakeTransport()
    session = CallSession(transport, services, agent_key="default")

    task = asyncio.create_task(session.run())
    await asyncio.sleep(0.25)  # let the greeting play

    await session.push_audio(tone(700))
    await session.push_audio(silence(500))
    await asyncio.sleep(0.6)

    await session.hangup("test_complete")
    record = await asyncio.wait_for(task, timeout=5)

    assert record.outcome == "test_complete"
    assert len(record.turns) >= 1, "the caller's utterance should produce a turn"
    assert transport.audio, "the agent should have produced audio"

    kinds = {e["type"] for e in transport.events}
    assert {"state", "speech", "transcript", "metrics"} <= kinds

    metrics = transport.of_type("metrics")[0]
    assert metrics["total_ms"] > 0
    assert metrics["audio_s"] > 0.5


async def test_greeting_is_spoken_before_listening(services):
    transport = FakeTransport()
    session = CallSession(transport, services)
    task = asyncio.create_task(session.run())
    await asyncio.sleep(0.3)

    greetings = [e for e in transport.of_type("speech") if e.get("kind") == "greeting"]
    assert greetings, "the agent must speak first on an inbound call"
    assert greetings[0]["text"] == services.profile("default").greeting

    await session.hangup()
    await asyncio.wait_for(task, timeout=5)


async def test_caller_can_interrupt_the_agent(services):
    """Barge-in must fire while the agent is genuinely still audible.

    Regression for a bug the unit tests could not see: TTS renders faster than
    real time, so before playback was paced the session left SPEAKING almost
    immediately and an interruption was misread as a new turn.
    """
    transport = FakeTransport()
    session = CallSession(transport, services)
    task = asyncio.create_task(session.run())

    # Wait for the greeting to start playing, not to finish.
    for _ in range(50):
        await asyncio.sleep(0.02)
        if session.state == CallState.SPEAKING:
            break
    assert session.state == CallState.SPEAKING, "agent should be mid-greeting"

    audio_before = len(transport.audio)
    for _ in range(30):  # 600 ms of speech over the top of the agent
        await session.push_audio(tone(20))
        await asyncio.sleep(0.005)
    await asyncio.sleep(0.3)

    assert transport.of_type("barge_in"), "interrupting the agent must fire barge-in"
    # And playback must actually have stopped, not merely been flagged.
    stopped_at = len(transport.audio)
    await asyncio.sleep(0.2)
    assert len(transport.audio) == stopped_at, "audio kept flowing after barge-in"
    assert stopped_at < audio_before + 16000 * 2 * 5, "the full greeting was still sent"

    await session.hangup()
    await asyncio.wait_for(task, timeout=5)


async def test_agent_audio_is_paced_to_real_time(services):
    """The session must not dump a whole utterance instantly."""
    transport = FakeTransport()
    session = CallSession(transport, services)
    started = asyncio.get_running_loop().time()
    task = asyncio.create_task(session.run())

    await asyncio.sleep(0.35)
    sent_s = len(transport.audio) / (SAMPLE_RATE * 2)
    elapsed = asyncio.get_running_loop().time() - started
    # Allowed to run ahead by the playout lead plus scheduling slop, but nowhere
    # near the multi-second greeting.
    assert sent_s < elapsed + 1.0, f"sent {sent_s:.2f}s of audio in {elapsed:.2f}s"

    await session.hangup()
    await asyncio.wait_for(task, timeout=5)


async def test_capacity_limit_is_enforced():
    from vaani.pipeline.manager import CallCapacityError, CallManager

    settings = Settings(max_concurrent_calls=1, record_calls=False)
    svc = build_services(settings)
    await svc.start()
    manager = CallManager(max_concurrent=1)

    first = CallSession(FakeTransport(), svc)
    second = CallSession(FakeTransport(), svc)
    await manager.register(first)
    with pytest.raises(CallCapacityError):
        await manager.register(second)

    await manager.unregister(first)
    await manager.register(second)  # a line freed up
    await svc.close()


async def test_logging_survives_reserved_field_names(services, caplog):
    """Regression: structured fields must never collide with LogRecord attributes.

    This is pinned at INFO deliberately. The collision only raises once the level
    is enabled, so a suite running at the default WARNING would pass while
    production — which runs at INFO — fell over on the first tool call.
    """
    import logging as _logging

    from vaani.core.logging import get_logger

    log = get_logger("test.reserved")
    with caplog.at_level(_logging.INFO):
        # Every one of these is a reserved LogRecord attribute.
        log.info("probe", extra={"args": {"a": 1}, "module": "x", "name": "y",
                                 "filename": "z", "levelno": 99})

        ctx = ToolContext(call_id="t", services=services.as_tool_services())
        agent = ConversationAgent(DEFAULT_PROFILE, services.llm, tools, ctx)
        turn = await agent.respond("What is the fee for a new connection?")

    assert turn.text
    assert any(r.message == "probe" for r in caplog.records)


async def test_manager_reports_analytics(services):
    from vaani.pipeline.manager import CallManager

    manager = CallManager()
    transport = FakeTransport()
    session = CallSession(transport, services)
    await manager.register(session)

    task = asyncio.create_task(session.run())
    await asyncio.sleep(0.25)
    await session.push_audio(tone(600))
    await session.push_audio(silence(400))
    await asyncio.sleep(0.5)
    await session.hangup()
    await asyncio.wait_for(task, timeout=5)
    await manager.unregister(session)

    stats = manager.stats()
    assert stats["total_calls"] == 1
    assert stats["completed"] == 1
    assert stats["avg_turn_latency_ms"] >= 0
