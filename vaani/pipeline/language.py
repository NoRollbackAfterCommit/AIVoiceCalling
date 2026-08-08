"""Which language the agent is currently speaking.

Detection is per utterance and imperfect, so acting on every result makes an
agent that stutters between languages. This holds a current language and only
moves after the same new language is seen on consecutive turns.

There is deliberately no "Hinglish" state. Saarika reports code-mixed
Hindi-English as hi-IN and Bulbul's Hindi voices pronounce embedded English
words correctly, so code-mixing works precisely because nothing here treats it
as a decision point.
"""

from __future__ import annotations

import re

from vaani.core.logging import get_logger

log = get_logger(__name__)


class LanguageTracker:
    def __init__(self, default: str, voices: dict[str, str], switch_after: int = 2) -> None:
        self._voices = dict(voices)
        self._switch_after = max(1, switch_after)
        self.current = default
        self.locked = False
        self._candidate: str | None = None
        self._streak = 0

    def lock(self, language: str) -> None:
        """Fix the language for the rest of the call.

        Once the caller has chosen, drift is a defect rather than a feature: a
        caller who picked Bengali and then drops in an English word wants a
        Bengali answer, and switching on them mid-call reads as the agent
        losing track of the conversation.
        """
        self.current = language
        self.locked = True
        self._candidate = None
        self._streak = 0
        log.info("language locked", extra={"language": language})

    def observe(self, language: str | None) -> str:
        """Feed one utterance's detected language. Returns the language to use."""
        if self.locked:
            return self.current
        if not language or language == self.current:
            self._candidate = None
            self._streak = 0
            return self.current

        # A language with no voice configured cannot be spoken. Switching to it
        # would leave the caller listening to silence, so ignore it entirely.
        if language not in self._voices:
            return self.current

        if language == self._candidate:
            self._streak += 1
        else:
            self._candidate = language
            self._streak = 1

        if self._streak >= self._switch_after:
            log.info(
                "caller language changed",
                extra={"previous": self.current, "current": language},
            )
            self.current = language
            self._candidate = None
            self._streak = 0
        return self.current

    def voice(self) -> str | None:
        return self._voices.get(self.current)


# Spoken names a caller might use, per language. Native script first because
# that is what STT returns when they answer in their own language.
LANGUAGE_NAMES: dict[str, tuple[str, ...]] = {
    "en-IN": ("english", "इंग्लिश", "अंग्रेजी", "angrezi", "ইংরেজি"),
    "hi-IN": ("hindi", "हिंदी", "हिन्दी", "hindustani"),
    "bn-IN": ("bengali", "bangla", "বাংলা", "বেঙ্গলি", "बंगाली"),
    "mr-IN": ("marathi", "मराठी"),
    "gu-IN": ("gujarati", "ગુજરાતી", "गुजराती"),
    "pa-IN": ("punjabi", "panjabi", "ਪੰਜਾਬੀ", "पंजाबी"),
    "od-IN": ("odia", "oriya", "ଓଡ଼ିଆ", "उड़िया"),
    "ta-IN": ("tamil", "தமிழ்", "तमिल"),
    "te-IN": ("telugu", "తెలుగు", "तेलुगु"),
    "kn-IN": ("kannada", "ಕನ್ನಡ", "कन्नड़"),
    "ml-IN": ("malayalam", "മലയാളം", "मलयालम"),
}


def detect_choice(
    text: str, detected: str | None = None, allowed: set[str] | None = None
) -> str | None:
    """Work out which language the caller asked for.

    Two signals, in order. A named language wins outright — someone who says
    "Bengali" in English wants Bengali, and trusting the audio's detected
    language there would give them English. Only if no name is recognised does
    the language they spoke in count as an implicit choice, which covers the
    caller who simply answers in their own language.
    """
    haystack = _normalise(text)
    if haystack:
        for code, names in LANGUAGE_NAMES.items():
            if allowed is not None and code not in allowed:
                continue
            if any(name in haystack for name in names):
                return code

    if detected and (allowed is None or detected in allowed):
        return detected
    return None


def _normalise(text: str) -> str:
    """Lowercase and strip punctuation so "Bengali, please." matches "bengali".

    Indic scripts are unaffected: they have no case, and the word-character
    class keeps their letters.
    """
    return re.sub(r"[^\w\s]+", " ", text.lower()).strip()
