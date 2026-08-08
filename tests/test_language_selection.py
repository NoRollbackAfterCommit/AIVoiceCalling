"""Opening in English, asking for a language, then holding to it.

The point of locking is that a caller who chose Bengali and then borrows an
English word still gets Bengali back. Following that drift is the defect.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from vaani.agent.prompt import AgentProfile, render_system_prompt
from vaani.config import Settings
from vaani.core.registry import build_services
from vaani.pipeline.language import LanguageTracker, detect_choice
from vaani.pipeline.session import CONFIRMATIONS, CallSession

from .test_pipeline import FakeTransport

VOICES = {
    "en-IN": "en-IN:priya",
    "hi-IN": "hi-IN:priya",
    "bn-IN": "bn-IN:ritu",
}


@pytest.fixture
def settings() -> Settings:
    return Settings(
        stt_provider="mock", llm_provider="mock", tts_provider="mock",
        vector_store="memory", embedding_provider="hash", record_calls=False,
        end_of_turn_silence_ms=200, idle_prompt_after_s=120, idle_hangup_after_s=600,
    )


@pytest.fixture
async def services(settings: Settings):
    svc = build_services(settings)
    svc.profiles["default"] = replace(
        svc.profiles["default"], ask_language=True, voices=VOICES
    )
    await svc.start()
    yield svc
    await svc.close()


# -- choice detection -----------------------------------------------------


def test_a_named_language_beats_the_language_it_was_named_in():
    """Someone saying "Bengali" in English wants Bengali, not English."""
    assert detect_choice("Bengali please", detected="en-IN") == "bn-IN"


def test_a_native_name_is_recognised():
    assert detect_choice("বাংলা", detected="bn-IN") == "bn-IN"
    assert detect_choice("हिंदी", detected="hi-IN") == "hi-IN"


def test_answering_in_a_language_counts_as_choosing_it():
    """The caller who ignores the menu and just starts talking."""
    assert detect_choice("मुझे बिल के बारे में पूछना है", detected="hi-IN") == "hi-IN"


def test_punctuation_and_case_do_not_matter():
    assert detect_choice("English, please.", detected=None) == "en-IN"


def test_an_unrecognisable_answer_yields_nothing():
    assert detect_choice("mmm", detected=None) is None


def test_a_language_with_no_voice_configured_is_not_offered():
    assert detect_choice("Tamil", detected=None, allowed=set(VOICES)) is None


# -- locking --------------------------------------------------------------


def test_locking_ignores_every_later_detection():
    t = LanguageTracker(default="en-IN", voices=VOICES)
    t.lock("bn-IN")
    for _ in range(5):
        assert t.observe("hi-IN") == "bn-IN"
    assert t.voice() == "bn-IN:ritu"
    assert t.locked is True


def test_the_prompt_tells_the_model_the_choice_is_binding():
    prompt = render_system_prompt(AgentProfile(key="t"), language="bn-IN")
    assert "Bengali" in prompt
    assert "own script" in prompt
    # Code-mixing must be named explicitly as *not* a language change.
    assert "mix in English" in prompt


def test_without_a_choice_the_prompt_still_allows_following_the_caller():
    prompt = render_system_prompt(AgentProfile(key="t"))
    assert "switch with them" in prompt


def test_answer_framing_rules_are_always_present():
    """"Always read the rules" — the checklist ships in every prompt, and last,
    where it carries most weight as the model composes."""
    for language in (None, "hi-IN"):
        prompt = render_system_prompt(AgentProfile(key="t"), language=language)
        assert "Check every answer against this" in prompt
        assert prompt.rstrip().endswith("fix the reply before you speak it.")


# -- the call flow --------------------------------------------------------


async def test_the_call_opens_by_asking_for_a_language(services):
    transport = FakeTransport()
    session = CallSession(transport, services)
    task = asyncio.create_task(session.run())
    await asyncio.sleep(0.3)

    spoken = [e for e in transport.of_type("speech") if e.get("kind") == "greeting"]
    assert spoken, "the agent must speak first"
    assert spoken[0]["text"] == services.profile("default").language_prompt
    assert session._awaiting_language is True

    await session.hangup()
    await asyncio.wait_for(task, timeout=5)


async def test_a_clear_reply_locks_the_language_and_is_confirmed(services):
    from vaani.providers.base import Transcript

    transport = FakeTransport()
    session = CallSession(transport, services)
    task = asyncio.create_task(session.run())
    await asyncio.sleep(0.3)
    session._awaiting_language = True

    await session._settle_language(
        Transcript(text="Bengali please", is_final=True, language="en-IN")
    )

    assert session._language.locked is True
    assert session._language.current == "bn-IN", "a named language must win"
    assert session._awaiting_language is False
    assert session.record.language == "bn-IN"

    langs = transport.of_type("language")
    assert langs and langs[0]["language"] == "bn-IN"
    assert CONFIRMATIONS["bn-IN"] in [e["text"] for e in transport.of_type("speech")]

    await session.hangup()
    await asyncio.wait_for(task, timeout=10)


async def test_an_unclear_reply_asks_again_instead_of_locking(services):
    """The first turn of a call is the one most likely to be a door slamming or
    a colleague talking behind the caller. Locking English on that would strand
    them in a language they never chose."""
    from vaani.providers.base import Transcript

    transport = FakeTransport()
    session = CallSession(transport, services)
    task = asyncio.create_task(session.run())
    await asyncio.sleep(0.3)
    session._awaiting_language = True

    await session._settle_language(Transcript(text="mmm hmm", is_final=True, language=None))

    assert session._language.locked is False, "an unclear answer must not lock"
    assert session._awaiting_language is True, "it must still be waiting"
    prompts = [e["text"] for e in transport.of_type("speech")]
    assert prompts.count(services.profile("default").language_prompt) == 2

    await session.hangup()
    await asyncio.wait_for(task, timeout=10)


async def test_it_settles_on_english_once_it_has_asked_enough(services):
    from vaani.providers.base import Transcript

    session = CallSession(FakeTransport(), services)
    session._awaiting_language = True
    unclear = Transcript(text="mmm hmm", is_final=True, language=None)

    for _ in range(3):
        await session._settle_language(unclear)

    assert session._language.locked is True
    assert session._language.current == "en-IN"


async def test_every_offered_language_has_a_confirmation_line(services):
    """A locked language with no confirmation would fall back to an English
    greeting, which is exactly the confusion this feature removes."""
    for code in VOICES:
        assert code in CONFIRMATIONS, f"no confirmation line for {code}"


async def test_skipping_selection_greets_normally(services):
    services.profiles["default"] = replace(
        services.profiles["default"], ask_language=False
    )
    transport = FakeTransport()
    session = CallSession(transport, services)
    task = asyncio.create_task(session.run())
    await asyncio.sleep(0.3)

    spoken = [e for e in transport.of_type("speech") if e.get("kind") == "greeting"]
    assert spoken[0]["text"] == services.profile("default").greeting
    assert session._awaiting_language is False

    await session.hangup()
    await asyncio.wait_for(task, timeout=5)


def test_the_default_profile_can_speak_every_language_it_offers():
    """The bug this guards: an empty voices map made `allowed` unrestricted, so
    any language the recogniser detected got locked in — and with no voice and
    no confirmation line the caller then heard English back."""
    from vaani.agent.prompt import DEFAULT_PROFILE

    assert DEFAULT_PROFILE.voices, "an empty map accepts any detected language"
    for code, voice in DEFAULT_PROFILE.voices.items():
        assert code in CONFIRMATIONS, f"{code} is offered but has no confirmation line"
        assert ":" in voice, f"{voice} must be language:speaker"


def test_offered_speakers_are_ones_bulbul_accepts():
    """The API rejects an unknown speaker with a 400, so a typo here breaks
    every call in that language."""
    from vaani.agent.prompt import DEFAULT_PROFILE

    valid = {
        "aditya", "ritu", "ashutosh", "priya", "neha", "rahul", "pooja", "rohan",
        "simran", "kavya", "amit", "dev", "ishita", "shreya", "ratan", "varun",
        "manan", "sumit", "roopa", "kabir", "aayan", "shubh", "advait", "anand",
        "tanya", "tarun", "sunny", "mani", "gokul", "vijay", "shruti", "suhani",
        "mohit", "kavitha", "rehan", "soham", "rupali", "niharika",
    }
    for code, voice in DEFAULT_PROFILE.voices.items():
        speaker = voice.split(":", 1)[1]
        assert speaker in valid, f"{speaker!r} for {code} is not a bulbul:v3 speaker"


async def test_an_unsupported_language_eventually_falls_back_to_english(services):
    """A caller answering in Malayalam on a deployment offering seven languages
    must end up in English, never locked to a language with no voice."""
    from vaani.providers.base import Transcript

    session = CallSession(FakeTransport(), services)
    session._awaiting_language = True
    malayalam = Transcript(text="എനിക്ക്", is_final=True, language="ml-IN")

    for _ in range(3):
        await session._settle_language(malayalam)

    assert session._language.current == "en-IN"
    assert session._language.locked is True


def test_the_idle_prompt_exists_in_every_offered_language():
    """Hearing an English "are you still there?" part-way through a Bengali
    call reads as the agent having lost track of the conversation."""
    from vaani.agent.prompt import DEFAULT_PROFILE
    from vaani.pipeline.session import IDLE_PROMPTS

    for code in DEFAULT_PROFILE.voices:
        assert code in IDLE_PROMPTS, f"no idle prompt for {code}"


def test_the_hangup_window_is_longer_than_the_prompt_window():
    """Otherwise the call ends before the caller is ever asked anything."""
    from vaani.config import Settings

    s = Settings()
    assert s.idle_hangup_after_s > s.idle_prompt_after_s


def test_silence_is_measured_generously_enough_for_a_thinking_caller():
    """Below about twenty seconds this interrupts someone reading a bill number
    off a piece of paper, which is what made it so irritating."""
    from vaani.config import Settings

    assert Settings().idle_prompt_after_s >= 20.0
