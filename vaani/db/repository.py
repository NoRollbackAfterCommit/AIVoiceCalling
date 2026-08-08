"""Call persistence, kept off the call path.

One uvicorn worker carries every concurrent call on a single event loop, and
SQLite serialises writers. A synchronous insert inside the turn loop would
therefore stall every other live call for the duration of the write, so turn
records are queued and drained by one background task.

The queue is bounded and drops on overflow. Losing a turn record is bad; adding
latency to a live conversation because the disk is busy is worse.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from vaani.core.logging import get_logger
from vaani.db.migrate import upgrade
from vaani.db.models import CallRow, TurnRow

log = get_logger(__name__)

_QUEUE_MAX = 2000
# Rows per transaction while draining.
_BATCH_MAX = 64


class CallRepository:
    def __init__(self, database_url: str) -> None:
        self._url = database_url
        self._engine: Any = None
        self._sessions: Any = None
        self._queue: asyncio.Queue[TurnRow] = asyncio.Queue(maxsize=_QUEUE_MAX)
        self._writer: asyncio.Task[None] | None = None
        self._dropped = 0

    async def start(self) -> None:
        _ensure_parent_dir(self._url)
        # Migrations, not create_all: the latter adds missing tables but never a
        # missing column, so an upgraded deployment boots healthy on the old
        # schema and fails at the first write of a new field.
        await asyncio.to_thread(upgrade, self._url)
        self._engine = create_async_engine(self._url, future=True)
        self._sessions = async_sessionmaker(self._engine, expire_on_commit=False)
        self._writer = asyncio.create_task(self._drain(), name="call-writer")
        log.info("call repository ready")

    # -- write ---------------------------------------------------------------

    async def create_call(self, record: Any) -> None:
        async with self._sessions() as session, session.begin():
            session.add(
                CallRow(
                    call_id=record.call_id,
                    agent_key=record.agent_key,
                    direction=record.direction,
                    caller_number=record.caller_number,
                    started_at=record.started_at,
                    outcome=record.outcome,
                    language=getattr(record, "language", None),
                )
            )

    def append_turn(
        self,
        call_id: str,
        seq: int,
        role: str,
        text: str,
        language: str | None,
        metrics: dict[str, Any],
    ) -> None:
        """Synchronous by design: called from the turn loop, must not await."""
        row = TurnRow(
            call_id=call_id,
            seq=seq,
            role=role,
            text=text,
            language=language,
            stt_ms=int(metrics.get("stt_ms", 0)),
            agent_ms=int(metrics.get("agent_ms", 0)),
            tts_first_chunk_ms=int(metrics.get("tts_first_chunk_ms", 0)),
            total_ms=int(metrics.get("total_ms", 0)),
            barged_in=bool(metrics.get("barged_in", False)),
        )
        try:
            self._queue.put_nowait(row)
        except asyncio.QueueFull:
            self._dropped += 1
            log.warning("turn write queue full", extra={"dropped": self._dropped})

    async def finish_call(self, record: Any) -> None:
        await self.flush()
        async with self._sessions() as session, session.begin():
            row = await session.get(CallRow, record.call_id)
            if row is None:
                return
            row.ended_at = record.ended_at
            row.outcome = record.outcome
            row.summary = record.summary
            row.duration_s = getattr(record, "duration_s", 0.0)
            row.language = getattr(record, "language", None)
            row.recording_path = getattr(record, "recording_path", None)
            row.disposition = getattr(record, "disposition", None)
            row.disposition_reason = getattr(record, "disposition_reason", None)
            row.reference = getattr(record, "reference", None)

    async def flush(self) -> None:
        """Wait for queued turns to land. Used at call end and in tests."""
        await self._queue.join()

    # -- read ----------------------------------------------------------------

    async def get_call(self, call_id: str) -> dict[str, Any] | None:
        async with self._sessions() as session:
            row = await session.get(CallRow, call_id)
            return _as_dict(row) if row else None

    async def turns_for(self, call_id: str) -> list[dict[str, Any]]:
        async with self._sessions() as session:
            result = await session.execute(
                select(TurnRow).where(TurnRow.call_id == call_id).order_by(TurnRow.seq)
            )
            return [_as_dict(row) for row in result.scalars()]

    async def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        async with self._sessions() as session:
            result = await session.execute(
                select(CallRow).order_by(CallRow.started_at.desc()).limit(limit)
            )
            return [_as_dict(row) for row in result.scalars()]

    async def disposition_counts(self) -> dict[str, int]:
        """What calls actually achieved, in aggregate. This is the reason the
        vocabulary is closed rather than free text."""
        from sqlalchemy import func

        async with self._sessions() as session:
            result = await session.execute(
                select(CallRow.disposition, func.count())
                .where(CallRow.disposition.is_not(None))
                .group_by(CallRow.disposition)
            )
            return {row[0]: row[1] for row in result}

    async def purge_older_than(self, days: int) -> int:
        """Delete calls, their turns and their recordings past the retention window.

        A retention setting that nothing enforces is worse than none at all: it
        is the answer a government buyer is given about how long citizen
        recordings are kept, and it has to be true.

        Turns go first — the foreign key means the reverse order fails on any
        backend that enforces it.
        """
        if days <= 0:
            return 0
        cutoff = time.time() - days * 86400

        async with self._sessions() as session, session.begin():
            stale = (
                await session.execute(
                    select(CallRow.call_id, CallRow.recording_path).where(
                        CallRow.started_at < cutoff
                    )
                )
            ).all()
            if not stale:
                return 0
            ids = [row[0] for row in stale]
            await session.execute(delete(TurnRow).where(TurnRow.call_id.in_(ids)))
            await session.execute(delete(CallRow).where(CallRow.call_id.in_(ids)))

        # The audio is the bulk of it and the most sensitive part, so a row
        # deleted without its recording is not actually deleted.
        removed = 0
        for _, path in stale:
            if not path:
                continue
            with contextlib.suppress(OSError):
                Path(path).unlink(missing_ok=True)
                removed += 1

        log.info(
            "purged records past retention",
            extra={"calls": len(ids), "recordings": removed, "days": days},
        )
        return len(ids)

    # -- lifecycle -----------------------------------------------------------

    async def close(self) -> None:
        await self.flush()
        if self._writer is not None:
            self._writer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._writer
            self._writer = None
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None

    async def _drain(self) -> None:
        while True:
            batch = [await self._queue.get()]
            # Opportunistic batching. A transaction per row is what makes SQLite
            # slow, and turns arrive in bursts — both sides of a turn at once,
            # and every live call landing in the same window.
            while len(batch) < _BATCH_MAX:
                try:
                    batch.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            try:
                async with self._sessions() as session, session.begin():
                    session.add_all(batch)
            except Exception:
                log.exception("failed to persist turns", extra={"count": len(batch)})
            finally:
                for _ in batch:
                    self._queue.task_done()


def _ensure_parent_dir(url: str) -> None:
    """SQLite will not create a missing directory, and the default database_url
    points at ./data — which does not exist in a fresh checkout, so the server
    would refuse to boot with "unable to open database file"."""
    if not url.startswith("sqlite"):
        return
    _, _, path = url.partition(":///")
    if not path or path.startswith(":memory:"):
        return
    parent = Path(path).expanduser().parent
    if parent.name:
        parent.mkdir(parents=True, exist_ok=True)


def _as_dict(row: Any) -> dict[str, Any]:
    return {c.name: getattr(row, c.name) for c in row.__table__.columns}
