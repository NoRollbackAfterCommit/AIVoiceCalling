"""Live call registry and admission control.

Two jobs. First, a hard cap on concurrency: GPU inference degrades
catastrophically rather than gracefully, so the 51st caller must get a polite
"all lines busy" instead of everyone getting three-second pauses. Second, a
handle on every live call so a supervisor can watch, whisper, or force-transfer
one from the admin portal.
"""

from __future__ import annotations

import asyncio
from typing import Any

from vaani.core.logging import get_logger
from vaani.pipeline.session import CallRecord, CallSession

log = get_logger(__name__)


class CallCapacityError(RuntimeError):
    pass


class CallManager:
    def __init__(self, max_concurrent: int = 50, history_size: int = 500) -> None:
        self._max = max_concurrent
        self._live: dict[str, CallSession] = {}
        self._history: list[CallRecord] = []
        self._history_size = history_size
        self._lock = asyncio.Lock()
        self.total_calls = 0

    @property
    def live_count(self) -> int:
        return len(self._live)

    @property
    def at_capacity(self) -> bool:
        return len(self._live) >= self._max

    async def register(self, session: CallSession) -> None:
        async with self._lock:
            if len(self._live) >= self._max:
                raise CallCapacityError(
                    f"all {self._max} lines are busy"
                )
            self._live[session.call_id] = session
            self.total_calls += 1

    async def unregister(self, session: CallSession) -> None:
        async with self._lock:
            self._live.pop(session.call_id, None)
            self._history.append(session.record)
            # In-memory ring; the durable copy is in the database.
            if len(self._history) > self._history_size:
                del self._history[: len(self._history) - self._history_size]

    def get(self, call_id: str) -> CallSession | None:
        return self._live.get(call_id)

    def live(self) -> list[dict[str, Any]]:
        return [
            {
                "call_id": s.call_id,
                "state": s.state,
                "agent_key": s.record.agent_key,
                "direction": s.record.direction,
                "caller_number": s.record.caller_number,
                "duration_s": round(s.record.duration_s, 1),
                "turns": len(s.record.turns),
            }
            for s in self._live.values()
        ]

    def history(self, limit: int = 50) -> list[dict[str, Any]]:
        return [r.to_dict() for r in reversed(self._history[-limit:])]

    async def hangup(self, call_id: str, outcome: str = "operator_ended") -> bool:
        session = self._live.get(call_id)
        if session is None:
            return False
        await session.hangup(outcome)
        return True

    async def drain(self, timeout: float = 30.0) -> None:
        """Let live calls finish before shutdown; hang up whatever is left."""
        deadline = asyncio.get_running_loop().time() + timeout
        while self._live and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.5)
        for session in list(self._live.values()):
            await session.hangup("shutdown")

    def stats(self) -> dict[str, Any]:
        completed = [r for r in self._history if r.turns]
        latencies = [
            t["metrics"]["total_ms"] for r in completed for t in r.turns if t.get("metrics")
        ]
        outcomes: dict[str, int] = {}
        for record in self._history:
            outcomes[record.outcome] = outcomes.get(record.outcome, 0) + 1
        return {
            "live": len(self._live),
            "capacity": self._max,
            "total_calls": self.total_calls,
            "completed": len(self._history),
            "outcomes": outcomes,
            "avg_turn_latency_ms": round(sum(latencies) / len(latencies)) if latencies else 0,
            "p95_turn_latency_ms": _percentile(latencies, 0.95),
            "avg_turns_per_call": (
                round(sum(len(r.turns) for r in completed) / len(completed), 1)
                if completed
                else 0
            ),
            "escalation_rate": (
                round(outcomes.get("transferred", 0) / max(len(self._history), 1), 3)
            ),
        }


def _percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(q * len(ordered)))
    return ordered[idx]
