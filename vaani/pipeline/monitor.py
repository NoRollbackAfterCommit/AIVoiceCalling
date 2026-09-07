"""Fan-out of live call events to supervisors.

A call's events go to its own transport, which is the caller. A supervisor
watching twenty lines at once is not party to any of them, so they need a
separate feed — this is it.

The one rule that matters: watching must never slow down talking. A supervisor
on a slow link, or a browser tab that has been backgrounded, must not be able to
add latency to a live conversation. So publishing is synchronous and
non-blocking, every subscriber has a bounded queue, and a subscriber that cannot
keep up loses events rather than applying back-pressure to the call.
"""

from __future__ import annotations

import asyncio
from typing import Any

from vaani.core.logging import get_logger

log = get_logger(__name__)

# Roughly a few seconds of a busy call. Past this a subscriber is not watching
# in any useful sense and is better off dropping to the next snapshot.
_QUEUE_MAX = 256


class MonitorHub:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._dropped = 0

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_QUEUE_MAX)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)

    def publish(self, call_id: str, event: dict[str, Any]) -> None:
        """Called from the call path. Synchronous and never raises.

        Nothing is awaited here: an await would let a stalled supervisor socket
        suspend the turn loop of the call being watched.
        """
        if not self._subscribers:
            return
        message = {"call_id": call_id, **event}
        for queue in self._subscribers:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                self._dropped += 1
                if self._dropped % 100 == 1:
                    log.warning(
                        "monitor subscriber falling behind",
                        extra={"dropped": self._dropped},
                    )
