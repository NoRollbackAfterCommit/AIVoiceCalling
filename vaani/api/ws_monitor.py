"""Supervisor feed: every live call's events on one socket.

A supervisor is not party to any call, so they cannot use the call transport.
This streams the same events across every line at once, tagged with the call
they belong to.

Read-only by design. Barging into a live conversation from a dashboard is a
different feature with different consequences, and hanging up is already
available as an explicit POST rather than something a stray click can do over a
socket.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from fastapi import APIRouter, WebSocket

from vaani.core.logging import get_logger

log = get_logger(__name__)
router = APIRouter()

# How often the roster of live calls is resent. Events carry the detail; this
# is what makes a dashboard opened mid-call show the calls already in progress,
# and what corrects any state a dropped event would have left stale.
_SNAPSHOT_S = 2.0


@router.websocket("/ws/monitor")
async def monitor(ws: WebSocket) -> None:
    services = ws.app.state.services
    manager = ws.app.state.calls
    hub = getattr(services, "monitor", None)

    await ws.accept()
    if hub is None:
        await ws.send_text(json.dumps({"type": "error", "message": "monitoring is disabled"}))
        await ws.close()
        return

    queue = hub.subscribe()
    lock = asyncio.Lock()

    async def send(payload: dict[str, Any]) -> None:
        # The snapshot loop and the event pump both write; FastAPI's WebSocket
        # is not safe for concurrent sends.
        async with lock:
            await ws.send_text(json.dumps(payload, default=str))

    async def snapshots() -> None:
        while True:
            await send({"type": "snapshot", "calls": manager.live(), "capacity": manager.capacity})
            await asyncio.sleep(_SNAPSHOT_S)

    async def events() -> None:
        while True:
            await send({"type": "event", "event": await queue.get()})

    # A supervisor who opens the page mid-shift must see the calls already
    # running, not wait for the next thing that happens on one.
    await send({"type": "snapshot", "calls": manager.live(), "capacity": manager.capacity})

    tasks = [asyncio.create_task(snapshots()), asyncio.create_task(events())]
    try:
        # Either task ending means the socket is gone.
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    except Exception:
        log.info("monitor socket closed")
    finally:
        for task in tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        hub.unsubscribe(queue)
        with contextlib.suppress(Exception):
            await ws.close()
