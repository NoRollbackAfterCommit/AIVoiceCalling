"""Which language the agent is currently speaking.

Detection is per utterance and imperfect, so acting on every result makes an
agent that stutters between languages. This holds a current language and only
moves after the same new language is seen on consecutive turns.

There is deliberately no "Hinglish" state. Saarika reports code-mixed
Hindi-English as hi-IN and Bulbul's Hindi voices pronounce embedded English
words correctly, so code-mixing works precisely because nothing here treats it
as a decision point.

Locking is not the end of the story. A caller can change their mind, and more
importantly the choice can have been recorded wrongly in the first place — the
opening question is asked into exactly the noise that makes it unanswerable, and
a caller who was never heard would otherwise spend the whole call in a language
they did not pick. Two escape hatches therefore survive the lock: an explicit
spoken request (`detect_switch_request`), and the caller simply carrying on in
another language for consecutive turns (`observe_drift`). Both are deliberately
harder to trigger than the opening choice, because a false switch mid-call is
worse than a slow one.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from vaani.core.logging import get_logger

log = get_logger(__name__)


class LanguageTracker:
    def __init__(
        self,
        default: str,
        voices: dict[str, str],
        switch_after: int = 2,
        drift_after: int = 2,
        max_drifts: int = 3,
    ) -> None:
        self._voices = dict(voices)
        self._switch_after = max(1, switch_after)
        self._drift_after = max(1, drift_after)
        self._max_drifts = max_drifts
        self.current = default
        self.locked = False
        # True when the language was settled on rather than chosen — the caller
        # could not be understood and something had to be picked.
        self.provisional = False
        self._candidate: str | None = None
        self._streak = 0
        self._drift_candidate: str | None = None
        self._drift_streak = 0
        self._drifts = 0

    def lock(self, language: str, *, provisional: bool = False) -> None:
        """Fix the language for the rest of the call, or until the caller moves.

        Once the caller has chosen, following per-utterance detection is a defect
        rather than a feature: a caller who picked Bengali and then drops in an
        English word wants a Bengali answer, and switching on them mid-answer
        reads as the agent losing track of the conversation. Moving away from a
        locked language takes the deliberate signals below, not a stray
        detection.
        """
        self.current = language
        self.locked = True
        self.provisional = provisional
        self._candidate = None
        self._streak = 0
        # The drift *count* deliberately survives, so a call cannot dodge the cap
        # by alternating.
        self._drift_candidate = None
        self._drift_streak = 0
        log.info("language locked", extra={"language": language, "provisional": provisional})

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

    def observe_drift(self, language: str | None) -> str | None:
        """Feed one utterance from a locked call. Returns a new language, or None.

        The caller who was never heard in the first place does not know to say
        "Bengali" — they simply keep speaking Bengali. This is what notices, and
        it is deliberately slower than `observe`: a locked language is a promise,
        so overturning it takes consecutive turns rather than one.
        """
        if not self.locked:
            return None
        if not language or language == self.current:
            self._drift_candidate = None
            self._drift_streak = 0
            return None
        # Unspeakable languages are ignored here for the same reason as above:
        # switching to one leaves the caller listening to silence.
        if language not in self._voices:
            return None
        if self._drifts >= self._max_drifts:
            # A bilingual line can otherwise oscillate all call. Explicit
            # requests are not capped; this only bounds the guessing.
            return None

        if language == self._drift_candidate:
            self._drift_streak += 1
        else:
            self._drift_candidate = language
            self._drift_streak = 1

        # Nobody chose a provisional language, so one clear turn is enough to
        # overturn it. A real choice has to be outvoted.
        needed = 1 if self.provisional else self._drift_after
        if self._drift_streak < needed:
            return None

        log.info(
            "caller drifted to another language",
            extra={"previous": self.current, "current": language, "provisional": self.provisional},
        )
        self._drifts += 1
        self.current = language
        self.provisional = False
        self._drift_candidate = None
        self._drift_streak = 0
        return language

    def voice(self) -> str | None:
        return self._voices.get(self.current)


# Spoken names a caller might use, per language. Native script first because
# that is what STT returns when they answer in their own language.
LANGUAGE_NAMES: dict[str, tuple[str, ...]] = {
    "en-IN": ("english", "इंग्लिश", "अंग्रेजी", "angrezi", "ইংরেজি"),
    "hi-IN": ("hindi", "हिंदी", "हिन्दी", "hindustani"),
    "bn-IN": ("bengali", "bangla", "বাংলা", "বেঙ্গলি", "बंगाली", "बांग्ला"),
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

    Punctuation is removed by Unicode category rather than by `[^\\w\\s]`, which
    looks equivalent and is not: Python's `\\w` does not match combining marks, so
    that pattern shredded every Indic word it touched — "বাংলা" came out as
    "ব ল" and matched nothing. Categories P (punctuation, including the danda),
    S (symbols) and C (control) go; letters, marks and digits stay.
    """
    cleaned = "".join(" " if unicodedata.category(ch)[0] in "PSCZ" else ch for ch in text.lower())
    return " ".join(cleaned.split())


# Words that ask for a language rather than mention one. Matched as substrings so
# one entry covers a verb's inflections — "बोल" covers बोलिए/बोलो/बोलेंगे, "বল"
# covers বলুন/বলবেন. Kept to verbs of *speaking*: broader stems collide with
# ordinary words ("कह" is inside "कहाँ") and turn questions into switch requests.
_REQUEST_MARKERS: tuple[str, ...] = (
    "speak",
    "talk",
    "switch",
    "change",
    "continue",
    "prefer",
    "convert",
    "bol",
    "baat",
    "kaho",
    "बोल",
    "बात",  # Hindi, Marathi
    "বল",
    "কথা",  # Bengali
    "બોલ",
    "વાત",  # Gujarati
    "ਬੋਲ",
    "ਗੱਲ",  # Punjabi
    "କୁହ",
    "କଥା",  # Odia
)

# Words carrying no request of their own. Whatever is left after these, the
# language name and the request verb are removed is the caller's actual question.
_FILLER: frozenset[str] = frozenset(
    """
    can could would will you your i me my we us please kindly in to the a an is it
    do let want ok okay yes no sir madam only just now and language
    kripya mein mai main mujhe aap kya hai hain ho kar karo kijiye sakte sakta
    sakti thoda zara plz pls ke ki ka
    में मे कृपया आप क्या मुझे मैं है हैं हो कर करो कीजिए कीजिये सकते सकता सकती ज़रा जरा
    थोड़ा और ही से को का के की हम आप नहीं अब
    দয়া করে করুন কি কী পারেন একটু আপনি আমি আর না এখন তে এবং
    કૃપા કરીને તમે હું મને શું છે માં અને
    ਕਿਰਪਾ ਕਰਕੇ ਤੁਸੀਂ ਮੈਂ ਮੈਨੂੰ ਕੀ ਹੈ ਵਿੱਚ ਅਤੇ
    ଦୟାକରି ଆପଣ ମୁଁ ମୋତେ କଣ ଅଛି ରେ ଏବଂ
    """.split()
)

_ALL_NAMES: tuple[str, ...] = tuple(name for names in LANGUAGE_NAMES.values() for name in names)

# Below this, an utterance is not a question — it is the request and nothing else.
_QUESTION_WORDS = 3

# Saarika reports how sure it is. A hesitant guess is exactly the signal that
# should not be allowed to move a call that has already settled.
_MIN_DRIFT_CONFIDENCE = 0.6


@dataclass(slots=True)
class SwitchRequest:
    language: str
    # True when the same breath also asked something real. Confirming and
    # returning to listening would make the caller say it a second time.
    carries_question: bool


def detect_switch_request(text: str, allowed: set[str] | None = None) -> SwitchRequest | None:
    """Work out whether a mid-call utterance asks to change language.

    Deliberately stricter than `detect_choice`, which may treat any mention of a
    language as an answer because it is only ever asked right after the question.
    Mid-call, every utterance is arbitrary: a Hindi caller asking about an
    English-medium school must not be flipped into English. So a name only counts
    when a speaking verb sits beside it, or when the utterance is the name and
    nothing else.
    """
    tokens = _normalise(text).split()
    if not tokens:
        return None

    named: str | None = None
    for code, names in LANGUAGE_NAMES.items():
        if allowed is not None and code not in allowed:
            continue
        if any(any(name in token for name in names) for token in tokens):
            named = code
            break
    if named is None:
        return None

    has_marker = any(marker in token for token in tokens for marker in _REQUEST_MARKERS)
    content = [
        token
        for token in tokens
        if token not in _FILLER
        and not any(name in token for name in _ALL_NAMES)
        and not any(marker in token for marker in _REQUEST_MARKERS)
    ]
    if not has_marker and content:
        return None
    return SwitchRequest(language=named, carries_question=len(content) >= _QUESTION_WORDS)


def drift_qualifies(text: str, confidence: float | None = None) -> bool:
    """Whether an utterance is solid enough to count towards a language drift.

    Short utterances are where per-utterance detection is least reliable — a
    one-word "হ্যাঁ" gets tagged as half the languages in the country — and acting
    on them would move settled calls at random.
    """
    if confidence is not None and confidence < _MIN_DRIFT_CONFIDENCE:
        return False
    compact = _normalise(text)
    return len(compact.split()) >= _QUESTION_WORDS or len(compact) >= 12
