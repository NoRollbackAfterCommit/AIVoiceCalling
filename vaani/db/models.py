"""Durable call records.

For a government deployment the transcript and its timings are the audit trail,
so they outlive the process. Deliberately two flat tables and no migrations:
phase 1 runs on SQLite at pilot scale, and Alembic arrives with Postgres.
"""

from __future__ import annotations

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class CallRow(Base):
    __tablename__ = "calls"

    call_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    agent_key: Mapped[str] = mapped_column(String(64), index=True)
    direction: Mapped[str] = mapped_column(String(16), default="inbound")
    caller_number: Mapped[str | None] = mapped_column(String(32), nullable=True)
    started_at: Mapped[float] = mapped_column(Float, index=True)
    ended_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    outcome: Mapped[str] = mapped_column(String(32), default="in_progress", index=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    recording_path: Mapped[str | None] = mapped_column(Text, nullable=True)


class TurnRow(Base):
    __tablename__ = "turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_id: Mapped[str] = mapped_column(String(32), ForeignKey("calls.call_id"), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(16))
    text: Mapped[str] = mapped_column(Text, default="")
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    stt_ms: Mapped[int] = mapped_column(Integer, default=0)
    agent_ms: Mapped[int] = mapped_column(Integer, default=0)
    tts_first_chunk_ms: Mapped[int] = mapped_column(Integer, default=0)
    total_ms: Mapped[int] = mapped_column(Integer, default=0)
    barged_in: Mapped[bool] = mapped_column(Boolean, default=False)
