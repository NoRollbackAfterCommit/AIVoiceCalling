"""Reporting, computed over the stored calls.

Replaces figures derived from `CallManager`'s in-memory ring of the last two
hundred calls. That ring is lost whenever the container is recreated — which a
deploy does every time — and it cannot be filtered, so it could never answer the
question this platform now has to answer: how is *this* organisation's *this*
line doing, over *these* dates.

Everything here is a read. Aggregates are computed in SQL rather than by pulling
rows into Python, because "how many calls last quarter" must not become a
hundred thousand row objects on the event loop the live calls are sharing.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Float, Integer, and_, case, func, select

from vaani.core.logging import get_logger
from vaani.db.models import CallRow, TurnRow
from vaani.telephony.numbers import to_e164

log = get_logger(__name__)

# Outcomes that mean a person took over. Everything else the bot finished on its
# own, which is the figure a buyer asks about first.
_HANDED_OVER = ("transferred",)

# The agent admitting it cannot answer. Lowercased substrings, matched in
# SQL so a quarter's worth of turns never reaches Python.
_NO_ANSWER = ("could not find", "do not have that", "don't have that", "not sure", "দুঃখিত", "क्षमा")


class AnalyticsRepository:
    def __init__(self, sessions: Any) -> None:
        self._sessions = sessions

    async def summary(
        self,
        *,
        organisation_id: int | None = None,
        did: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> dict[str, Any]:
        """Everything the dashboard shows, in one round trip's worth of queries.

        `organisation_id=None` is the platform view and includes unattributed
        calls — those belong to nobody, and letting a customer inherit whatever
        arrived unrouted would flatter their figures with somebody else's calls.
        """
        where = self._filters(organisation_id, did, since, until)

        async with self._sessions() as session:
            totals = (
                await session.execute(
                    select(
                        func.count().label("calls"),
                        func.avg(CallRow.duration_s).label("avg_duration"),
                        func.sum(case((CallRow.outcome.in_(_HANDED_OVER), 1), else_=0)).label(
                            "handed_over"
                        ),
                    ).where(and_(*where))
                )
            ).one()

            calls = int(totals.calls or 0)
            handed_over = int(totals.handed_over or 0)

            outcomes = {
                row[0]: row[1]
                for row in await session.execute(
                    select(CallRow.outcome, func.count())
                    .where(and_(*where))
                    .group_by(CallRow.outcome)
                )
            }
            dispositions = {
                row[0]: row[1]
                for row in await session.execute(
                    select(CallRow.disposition, func.count())
                    .where(and_(*where, CallRow.disposition.is_not(None)))
                    .group_by(CallRow.disposition)
                )
            }
            languages = {
                row[0]: row[1]
                for row in await session.execute(
                    select(CallRow.language, func.count())
                    .where(and_(*where, CallRow.language.is_not(None)))
                    .group_by(CallRow.language)
                )
            }
            per_did = [
                {"did": row[0], "calls": row[1], "avg_duration_s": round(row[2] or 0.0, 1)}
                for row in await session.execute(
                    select(CallRow.did, func.count(), func.avg(CallRow.duration_s))
                    .where(and_(*where, CallRow.did.is_not(None)))
                    .group_by(CallRow.did)
                    .order_by(func.count().desc())
                )
            ]

            # SQLite has no date_trunc; the epoch seconds divide cleanly.
            day = (func.cast(CallRow.started_at, Integer) / 86400) * 86400
            per_day = [
                {"day": _iso_day(row[0]), "calls": row[1]}
                for row in await session.execute(
                    select(day.label("day"), func.count())
                    .where(and_(*where))
                    .group_by("day")
                    .order_by("day")
                )
            ]

            latency = await self._latency(session, where)

        return {
            "calls": calls,
            "transferred": handed_over,
            # What share the bot finished without a person. Zero calls is zero
            # containment rather than a division by nothing.
            "containment": round((calls - handed_over) / calls, 3) if calls else 0,
            "avg_duration_s": round(float(totals.avg_duration or 0.0), 1),
            "outcomes": outcomes,
            "dispositions": dispositions,
            "languages": languages,
            "per_did": per_did,
            "per_day": per_day,
            "latency": latency,
        }

    async def _latency(self, session: Any, where: list[Any]) -> dict[str, Any]:
        """What the caller waited, kept apart from how long the agent spoke.

        `total_ms` wraps the whole turn including the reply playing out at real
        time, so reporting it as latency overstates it two- or threefold. The
        wait is the three measured stages; the rest is speech.
        """
        wait = TurnRow.stt_ms + TurnRow.agent_ms + TurnRow.tts_first_chunk_ms
        row = (
            await session.execute(
                select(
                    func.count().label("turns"),
                    func.avg(func.cast(wait, Float)).label("wait_ms"),
                    func.avg(func.cast(TurnRow.total_ms - wait, Float)).label("spoken_ms"),
                    func.avg(func.cast(TurnRow.agent_ms, Float)).label("agent_ms"),
                    func.avg(func.cast(TurnRow.stt_ms, Float)).label("stt_ms"),
                    func.avg(func.cast(TurnRow.tts_first_chunk_ms, Float)).label("tts_ms"),
                    func.sum(case((TurnRow.barged_in.is_(True), 1), else_=0)).label("barged_in"),
                )
                .select_from(TurnRow)
                .join(CallRow, CallRow.call_id == TurnRow.call_id)
                .where(and_(*where, TurnRow.role == "agent", TurnRow.total_ms > 0))
            )
        ).one()

        return {
            "turns": int(row.turns or 0),
            "wait_ms": round(float(row.wait_ms or 0.0)),
            "spoken_ms": round(float(row.spoken_ms or 0.0)),
            "stt_ms": round(float(row.stt_ms or 0.0)),
            "agent_ms": round(float(row.agent_ms or 0.0)),
            "tts_first_chunk_ms": round(float(row.tts_ms or 0.0)),
            "barged_in": int(row.barged_in or 0),
        }

    async def knowledge_gaps(
        self,
        *,
        organisation_id: int | None = None,
        did: str | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = 25,
    ) -> list[str]:
        """What callers asked that the agent could not answer.

        The highest-value input to the next round of content authoring: every
        one of these is a question somebody rang up with and did not get an
        answer to. Matched on the agent's own admissions of ignorance, which is
        crude but needs no extra bookkeeping on the call path — and the call
        path is the one place not to add bookkeeping.
        """
        where = self._filters(organisation_id, did, since, until)
        admission = None
        for phrase in _NO_ANSWER:
            clause = func.lower(TurnRow.text).like(f"%{phrase}%")
            admission = clause if admission is None else (admission | clause)

        async with self._sessions() as session:
            failed = (
                await session.execute(
                    select(TurnRow.call_id, TurnRow.seq)
                    .select_from(TurnRow)
                    .join(CallRow, CallRow.call_id == TurnRow.call_id)
                    .where(and_(*where, TurnRow.role == "agent", admission))
                    .order_by(TurnRow.call_id, TurnRow.seq)
                    .limit(limit * 4)
                )
            ).all()
            if not failed:
                return []

            # The question is the caller turn immediately before the admission.
            wanted = {(call_id, seq - 1) for call_id, seq in failed}
            rows = (
                await session.execute(
                    select(TurnRow.call_id, TurnRow.seq, TurnRow.text).where(
                        TurnRow.role == "caller",
                        TurnRow.call_id.in_({c for c, _ in wanted}),
                    )
                )
            ).all()

        asked = [text for call_id, seq, text in rows if (call_id, seq) in wanted and text]
        # Deduplicated, because one unanswerable question asked by forty callers
        # is one gap to fill, not forty.
        seen: dict[str, None] = {}
        for question in asked:
            seen.setdefault(question.strip(), None)
        return list(seen)[:limit]

    def _filters(
        self,
        organisation_id: int | None,
        did: str | None,
        since: float | None,
        until: float | None,
    ) -> list[Any]:
        where: list[Any] = [CallRow.started_at.is_not(None)]
        if organisation_id is not None:
            where.append(CallRow.organisation_id == organisation_id)
        if did:
            # Normalised both sides, or a number typed one way in the filter box
            # silently reports zero against rows stored the other way.
            where.append(CallRow.did == (to_e164(did) or did))
        if since is not None:
            where.append(CallRow.started_at >= since)
        if until is not None:
            where.append(CallRow.started_at <= until)
        return where


def _iso_day(epoch: float | None) -> str:
    if not epoch:
        return ""
    return datetime.fromtimestamp(float(epoch), tz=UTC).date().isoformat()


def default_range(days: int = 30) -> tuple[float, float]:
    """The window a dashboard opens on."""
    now = time.time()
    return now - days * 86400, now
