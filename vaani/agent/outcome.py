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

# Outcomes that must carry a reference the caller can quote back. A complaint
# with no number is not a complaint the caller can chase.
REQUIRES_REFERENCE: frozenset[str] = frozenset(
    {"complaint_registered", "callback_scheduled"}
)

# What the platform records when the caller is gone, keyed by transport outcome.
PLATFORM_FOR_OUTCOME: dict[str, str] = {
    "caller_disconnected": "caller_abandoned",
    "caller_ended": "caller_abandoned",
    "idle_timeout": "idle_timeout",
    "rejected": "capacity_rejected",
}


def allowed_for(profile: Any) -> tuple[str, ...]:
    """What the model may choose from."""
    extra = tuple(getattr(profile, "extra_dispositions", ()) or ())
    return tuple(sorted(AGENT_SET)) + extra


def is_valid(disposition: str, profile: Any) -> bool:
    extra = set(getattr(profile, "extra_dispositions", ()) or ())
    return disposition in set(CORE_DISPOSITIONS) | extra
