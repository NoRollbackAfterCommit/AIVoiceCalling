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

from vaani.core.logging import get_logger

log = get_logger(__name__)


class LanguageTracker:
    def __init__(self, default: str, voices: dict[str, str], switch_after: int = 2) -> None:
        self._voices = dict(voices)
        self._switch_after = max(1, switch_after)
        self.current = default
        self._candidate: str | None = None
        self._streak = 0

    def observe(self, language: str | None) -> str:
        """Feed one utterance's detected language. Returns the language to use."""
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
