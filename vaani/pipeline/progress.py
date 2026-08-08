"""Whether the call is going anywhere.

The model decides what to say; this decides whether that is working. Keeping the
judgement out of the model means it is deterministic, inspectable and testable
without a network call — and it costs nothing in the latency budget, because
every signal it reads is already present in the turn.
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
            log.info(
                "call stalled, offering a fallback",
                extra={"unproductive_turns": self.unproductive_turns},
            )
            return True
        return False


def _normalise(text: str) -> str:
    """A caller repeating themselves verbatim is the strongest signal that the
    last answer did not land, so punctuation and case must not hide it."""
    return re.sub(r"[^\w]+", " ", text.lower()).strip()
