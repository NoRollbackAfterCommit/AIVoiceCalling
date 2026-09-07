"""Who is calling, delivered before the audio is.

AudioSocket sends a UUID and then audio. Nothing in the protocol carries the
caller's number or the number they dialled, and both matter: without the first
the call record is anonymous and a promised callback has nowhere to go; without
the second one Asterisk box fronting several helplines cannot pick an agent
profile. The dialplan has all of it, so it posts them to
/api/telephony/announce under the same UUID it then hands to AudioSocket(), and
the bridge claims that entry when the socket connects.

Entries expire on their own. A dialplan that announces and then fails before
AudioSocket() would otherwise leave a caller number in memory for good.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

# Asterisk connects within milliseconds of the announce. A minute covers a box
# that is swapping, or a dialplan that plays a prompt in between.
DEFAULT_TTL_S = 60.0


@dataclass(slots=True)
class CallAnnouncement:
    caller_number: str | None = None
    agent_key: str | None = None
    dialled_number: str | None = None
    expires_at: float = 0.0


class CallAnnouncements:
    def __init__(
        self,
        ttl_s: float = DEFAULT_TTL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl_s
        self._clock = clock
        self._entries: dict[str, CallAnnouncement] = {}

    @staticmethod
    def canonical(call_uuid: str) -> str:
        """One spelling on both sides: the dialplan's ${UUID()} text and the 16
        bytes the socket sends must land on the same key."""
        return str(uuid.UUID(call_uuid))

    def announce(
        self,
        call_uuid: str,
        *,
        caller_number: str | None = None,
        agent_key: str | None = None,
        dialled_number: str | None = None,
    ) -> None:
        now = self._clock()
        self._purge(now)
        self._entries[self.canonical(call_uuid)] = CallAnnouncement(
            caller_number=caller_number,
            agent_key=agent_key,
            dialled_number=dialled_number,
            expires_at=now + self._ttl,
        )

    def claim(self, call_uuid: str) -> CallAnnouncement | None:
        """Take the announcement for a UUID. One-shot: a replayed UUID gets nothing."""
        try:
            key = self.canonical(call_uuid)
        except ValueError:
            return None
        entry = self._entries.pop(key, None)
        if entry is None or entry.expires_at <= self._clock():
            return None
        return entry

    def _purge(self, now: float) -> None:
        # Swept on announce rather than by a timer: the table only grows when a
        # call arrives, so that is the only moment it can need trimming.
        for key in [k for k, e in self._entries.items() if e.expires_at <= now]:
            del self._entries[key]
