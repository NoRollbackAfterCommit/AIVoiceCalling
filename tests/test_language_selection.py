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
from vaani.pipeline.language import (
    LanguageTracker,
    detect_choice,
    detect_switch_request,
    drift_qualifies,
)
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
        stt_provider="mock",
        llm_provider="mock",
        tts_provider="mock",
        vector_store="memory",
        embedding_provider="hash",
        record_calls=False,
        end_of_turn_silence_ms=200,
        idle_prompt_after_s=120,
        idle_hangup_after_s=600,
    )


@pytest.fixture(autouse=True)
def brisk_speech(monkeypatch):
    """Stop the mock voice from talking in real time.

    Playback is paced to the audio's own duration, and the mock sizes its audio
    to the text, so a test that re-asks the language question three times spends
    twenty-four seconds of wall clock listening to it. Nothing in this module
    tests pacing — barge-in timing is covered in test_pipeline, where it belongs.
    """
    monkeypatch.setattr("vaani.providers.tts.mock._CHARS_PER_SECOND", 4000.0)


@pytest.fixture
async def services(settings: Settings):
    svc = build_services(settings)
    svc.profiles["default"] = replace(svc.profiles["default"], ask_language=True, voices=VOICES)
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


def test_a_native_name_is_recognised_without_help_from_detection():
    """`\\w` does not match Indic vowel signs, so normalising with it shredded
    every native-script name — "বাংলা" became "ব ল" and only ever matched
    because the detected language happened to agree."""
    assert detect_choice("বাংলা", detected=None) == "bn-IN"
    assert detect_choice("हिंदी।", detected=None) == "hi-IN"


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
    """ "Always read the rules" — the checklist ships in every prompt, and last,
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
    services.profiles["default"] = replace(services.profiles["default"], ask_language=False)
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
        "aditya",
        "ritu",
        "ashutosh",
        "priya",
        "neha",
        "rahul",
        "pooja",
        "rohan",
        "simran",
        "kavya",
        "amit",
        "dev",
        "ishita",
        "shreya",
        "ratan",
        "varun",
        "manan",
        "sumit",
        "roopa",
        "kabir",
        "aayan",
        "shubh",
        "advait",
        "anand",
        "tanya",
        "tarun",
        "sunny",
        "mani",
        "gokul",
        "vijay",
        "shruti",
        "suhani",
        "mohit",
        "kavitha",
        "rehan",
        "soham",
        "rupali",
        "niharika",
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


# -- changing language mid-call -------------------------------------------


def test_a_bare_language_name_is_a_switch_request():
    request = detect_switch_request("Bengali", allowed=set(VOICES))
    assert request is not None and request.language == "bn-IN"
    assert request.carries_question is False


def test_a_polite_request_is_a_switch_request():
    request = detect_switch_request("can you speak in Bengali please", allowed=set(VOICES))
    assert request is not None and request.language == "bn-IN"
    assert request.carries_question is False, "nothing was asked but the switch itself"


def test_a_request_in_the_callers_own_words_is_recognised():
    request = detect_switch_request("हिंदी में बात कीजिए", allowed=set(VOICES))
    assert request is not None and request.language == "hi-IN"


def test_an_incidental_mention_of_a_language_is_not_a_switch_request():
    """The reason mid-call matching cannot reuse `detect_choice`: a Hindi caller
    asking about an English-medium school must not be flipped to English."""
    assert detect_switch_request("English medium school ke baare mein", allowed=set(VOICES)) is None


def test_a_request_carrying_a_question_is_flagged_as_such():
    """The caller asked something real. Confirming and dropping it would make
    them say it twice."""
    request = detect_switch_request("बंगाली में बोलिए मेरा बिल कितना आया है", allowed=set(VOICES))
    assert request is not None and request.language == "bn-IN"
    assert request.carries_question is True


def test_a_language_with_no_voice_is_never_a_switch_request():
    assert detect_switch_request("Tamil please", allowed=set(VOICES)) is None


def test_a_short_utterance_is_too_thin_to_count_as_drift():
    """A one-word "haan" gets mis-tagged often enough that acting on it would
    move the call at random."""
    assert drift_qualifies("হ্যাঁ", None) is False


def test_a_low_confidence_detection_does_not_count_as_drift():
    assert drift_qualifies("আমার বিলের ব্যাপারে জানতে চাই", 0.2) is False
    assert drift_qualifies("আমার বিলের ব্যাপারে জানতে চাই", 0.9) is True


def test_two_turns_in_another_language_move_a_locked_call():
    t = LanguageTracker(default="en-IN", voices=VOICES)
    t.lock("en-IN")
    assert t.observe_drift("bn-IN") is None, "one turn is not enough"
    assert t.observe_drift("bn-IN") == "bn-IN"
    assert t.current == "bn-IN"


def test_a_single_stray_detection_does_not_move_a_locked_call():
    t = LanguageTracker(default="en-IN", voices=VOICES)
    t.lock("bn-IN")
    assert t.observe_drift("hi-IN") is None
    assert t.observe_drift("bn-IN") is None
    assert t.observe_drift("hi-IN") is None, "the streak must have been broken"
    assert t.current == "bn-IN"


def test_a_provisional_lock_gives_way_on_the_very_first_turn():
    """Nobody chose this language — it was settled on after the caller could not
    be heard. One clear turn in another language is enough to overturn it."""
    t = LanguageTracker(default="en-IN", voices=VOICES)
    t.lock("en-IN", provisional=True)
    assert t.observe_drift("bn-IN") == "bn-IN"
    assert t.provisional is False


def test_drift_stops_moving_the_call_after_a_few_switches():
    """A genuinely bilingual line must not be able to oscillate all call."""
    t = LanguageTracker(default="en-IN", voices=VOICES, max_drifts=2)
    for target in ("bn-IN", "hi-IN"):
        t.lock(t.current)
        assert t.observe_drift(target) is None
        assert t.observe_drift(target) == target
    t.lock(t.current)
    assert t.observe_drift("en-IN") is None
    assert t.observe_drift("en-IN") is None, "the cap must hold"


def test_drift_never_moves_to_a_language_with_no_voice():
    t = LanguageTracker(default="en-IN", voices=VOICES)
    t.lock("en-IN")
    for _ in range(4):
        assert t.observe_drift("ml-IN") is None
    assert t.current == "en-IN"


async def test_a_mid_call_request_switches_and_is_confirmed(services):
    from vaani.providers.base import Transcript

    transport = FakeTransport()
    session = CallSession(transport, services)
    session._language.lock("hi-IN")
    session._agent.set_language("hi-IN")

    consumed = await session._maybe_switch_language(
        Transcript(text="Bengali please", is_final=True, language="hi-IN")
    )

    assert consumed is True, "a bare request is answered here, not by the model"
    assert session._language.current == "bn-IN"
    assert session._language.voice() == "bn-IN:ritu"
    assert [e["language"] for e in transport.of_type("language")] == ["bn-IN"]
    assert CONFIRMATIONS["bn-IN"] in [e["text"] for e in transport.of_type("speech")]
    assert "Bengali" in session._agent._system.content


async def test_a_request_that_also_asks_something_is_answered_not_confirmed(services):
    from vaani.providers.base import Transcript

    transport = FakeTransport()
    session = CallSession(transport, services)
    session._language.lock("hi-IN")

    consumed = await session._maybe_switch_language(
        Transcript(text="बंगाली में बोलिए मेरा बिल कितना आया है", is_final=True, language="hi-IN")
    )

    assert consumed is False, "the turn must go on to the model"
    assert session._language.current == "bn-IN"
    assert CONFIRMATIONS["bn-IN"] not in [e["text"] for e in transport.of_type("speech")]


async def test_a_caller_who_simply_keeps_speaking_bengali_is_followed(services):
    """The caller never learns the magic word. They just carry on in their own
    language, and the agent has to notice."""
    from vaani.providers.base import Transcript

    transport = FakeTransport()
    session = CallSession(transport, services)
    session._language.lock("en-IN")
    bengali = Transcript(
        text="আমার বিলের ব্যাপারে জানতে চাই", is_final=True, language="bn-IN", confidence=0.9
    )

    assert await session._maybe_switch_language(bengali) is False
    assert session._language.current == "en-IN", "one turn is not enough"
    assert await session._maybe_switch_language(bengali) is False
    assert session._language.current == "bn-IN"
    assert CONFIRMATIONS["bn-IN"] not in [e["text"] for e in transport.of_type("speech")]


async def test_a_language_settled_on_by_default_is_rescued_by_the_next_turn(services):
    """The fault this feature exists for: noise ate both attempts, English was
    settled on, and the caller was stuck there for the rest of the call."""
    from vaani.providers.base import Transcript

    session = CallSession(FakeTransport(), services)
    session._awaiting_language = True
    unclear = Transcript(text="mmm hmm", is_final=True, language=None)
    for _ in range(3):
        await session._settle_language(unclear)
    assert session._language.current == "en-IN"
    assert session._language.provisional is True

    await session._maybe_switch_language(
        Transcript(
            text="আমার বিলের ব্যাপারে জানতে চাই",
            is_final=True,
            language="bn-IN",
            confidence=0.9,
        )
    )
    assert session._language.current == "bn-IN"


async def test_a_deliberate_choice_is_not_provisional(services):
    from vaani.providers.base import Transcript

    session = CallSession(FakeTransport(), services)
    session._awaiting_language = True
    await session._settle_language(
        Transcript(text="Bengali please", is_final=True, language="en-IN")
    )
    assert session._language.provisional is False


async def test_a_turn_carrying_a_switch_request_reaches_the_switch(services):
    """Wiring: the check has to sit in the turn path, not just be callable."""
    from vaani.providers.stt.mock import MockSTT

    transport = FakeTransport()
    session = CallSession(transport, services)
    session._language.lock("hi-IN")
    services.stt = MockSTT(script=["बांग्ला में बात कीजिए"])

    turn = asyncio.create_task(session._process_turn(b"\x00" * 32000, 1.0))
    await asyncio.sleep(0.4)

    assert session._language.current == "bn-IN"
    assert [e["language"] for e in transport.of_type("language")] == ["bn-IN"]

    turn.cancel()
    await asyncio.gather(turn, return_exceptions=True)


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
