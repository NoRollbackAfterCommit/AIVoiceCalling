"""Which organisation and which agent answer a given number.

One function, shared by every carrier transport, because the rule has to be the
same whichever way a call arrives: look the dialled number up, and if that fails
for any reason at all, answer anyway with the configured fallback.

Answering anyway is the important half. A number nobody has mapped yet, a
withdrawn line, a suspended organisation and a database that is not there are
four different operator problems and one identical caller experience — the phone
rings and a person expects a voice. Refusing would turn a configuration gap into
a dropped citizen, which is the trade the handshake already declines to make
when a call arrives without an id.
"""

from __future__ import annotations

from typing import Any

from vaani.core.logging import get_logger

log = get_logger(__name__)


async def route_call(services: Any, called: str | None) -> tuple[int | None, str]:
    """`(organisation_id, agent_key)` for a dialled number.

    A None organisation means the call is unattributed: served, recorded, and
    visible in reporting as needing a mapping rather than counted against
    whichever organisation happens to be first.
    """
    fallback = getattr(services.settings, "smartflo_agent", "default") or "default"
    tenancy = getattr(services, "tenancy", None)
    if tenancy is None:
        # A bare install with no database. Every call is unattributed, which is
        # the honest answer when there is nowhere to record an organisation.
        return None, fallback

    try:
        found = await tenancy.resolve_did(called)
    except Exception:
        # Reporting attribution is not worth a dropped call. Log it and answer.
        log.exception("did lookup failed; falling back", extra={"did": str(called)[:24]})
        return None, fallback

    if found is None:
        log.warning(
            "no organisation is mapped to this number; using the fallback agent",
            extra={"did": str(called)[:24], "agent": fallback},
        )
        return None, fallback

    organisation_id, agent_key = found
    return organisation_id, agent_key
