"""Language switching needs hysteresis.

A single mis-detected utterance must not flip the agent into another language
mid-call: answering one sentence in Bengali to a Hindi speaker is far worse than
taking one extra turn to follow a genuine switch.
"""

from __future__ import annotations

from vaani.pipeline.language import LanguageTracker

VOICES = {"hi-IN": "hi-IN:anushka", "en-IN": "en-IN:meera", "bn-IN": "bn-IN:anushka"}


def _tracker() -> LanguageTracker:
    return LanguageTracker(default="hi-IN", voices=VOICES)


def test_starts_on_the_profile_default():
    t = _tracker()
    assert t.current == "hi-IN"
    assert t.voice() == "hi-IN:anushka"


def test_a_single_foreign_utterance_does_not_switch():
    t = _tracker()
    assert t.observe("bn-IN") == "hi-IN"
    assert t.voice() == "hi-IN:anushka"


def test_two_consecutive_utterances_do_switch():
    t = _tracker()
    t.observe("bn-IN")
    assert t.observe("bn-IN") == "bn-IN"
    assert t.voice() == "bn-IN:anushka"


def test_returning_to_the_current_language_resets_the_streak():
    """Alternating detections are noise, not a language change."""
    t = _tracker()
    t.observe("bn-IN")
    t.observe("hi-IN")
    assert t.observe("bn-IN") == "hi-IN"


def test_unknown_language_is_ignored():
    """A language with no configured voice cannot be spoken, so never switch
    to it — the caller would get silence."""
    t = _tracker()
    t.observe("ta-IN")
    assert t.observe("ta-IN") == "hi-IN"


def test_none_is_ignored():
    t = _tracker()
    assert t.observe(None) == "hi-IN"
    assert t.observe(None) == "hi-IN"


def test_hinglish_reported_as_hindi_stays_hindi():
    """Saarika returns hi-IN for code-mixed Hindi-English. That must be a
    no-op, not a switch — which is exactly why no Hinglish tag exists."""
    t = _tracker()
    assert t.observe("hi-IN") == "hi-IN"
    assert t.observe("hi-IN") == "hi-IN"


def test_switch_threshold_is_configurable():
    t = LanguageTracker(default="hi-IN", voices=VOICES, switch_after=1)
    assert t.observe("en-IN") == "en-IN"


def test_no_voice_configured_returns_none_rather_than_guessing():
    t = LanguageTracker(default="hi-IN", voices={})
    assert t.voice() is None


def test_switching_back_needs_the_same_hysteresis():
    t = _tracker()
    t.observe("bn-IN")
    t.observe("bn-IN")
    assert t.current == "bn-IN"
    assert t.observe("hi-IN") == "bn-IN"
    assert t.observe("hi-IN") == "hi-IN"


def test_session_defaults_to_the_first_configured_voice():
    """AgentProfile.languages is prose for the prompt; the tracker needs codes
    that match what STT reports, so the voices mapping is the source of truth."""
    from vaani.agent.prompt import AgentProfile
    from vaani.config import Settings
    from vaani.core.registry import build_services
    from vaani.pipeline.session import CallSession

    profile = AgentProfile(key="t", languages=["Hindi", "English"], voices=VOICES)
    services = build_services(Settings())
    services.profiles["t"] = profile

    class _T:
        async def send_audio(self, pcm): ...
        async def send_event(self, event): ...
        async def close(self): ...

    session = CallSession(transport=_T(), services=services, agent_key="t")
    assert session._language.current == "hi-IN"
    assert session._language.voice() == "hi-IN:anushka"
