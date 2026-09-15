"""Durable call records.

For a government deployment the transcript and its timings are the audit trail,
so they outlive the process. Flat tables, upgraded by Alembic at boot (see
migrate.py). Agent profiles live here too, so one backup carries both the calls
and the behaviour they ran under.
"""

from __future__ import annotations

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class OrganisationRow(Base):
    """One customer's call centre inside a shared deployment."""

    __tablename__ = "organisations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Stable and URL-safe: it names the organisation in exports and report
    # filenames, where a display name that gets corrected would break the trail.
    slug: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(160))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[float] = mapped_column(Float)


class DidRow(Base):
    """A dialled number, and the agent that answers it.

    The number is the primary key because one number answers for exactly one
    organisation at a time. Two live mappings would make which agent picks up
    depend on row order, which is not a thing to discover from a caller.
    """

    __tablename__ = "dids"

    number: Mapped[str] = mapped_column(String(24), primary_key=True)
    organisation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("organisations.id"), index=True
    )
    # Which profile answers. An organisation may run several lines — admissions
    # and examinations — each with its own corpus, so this is per number rather
    # than per organisation.
    agent_key: Mapped[str] = mapped_column(String(64))
    label: Mapped[str | None] = mapped_column(String(120), nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[float] = mapped_column(Float)


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
    # The business result, indexed because "how many complaints last month" is
    # the first question a government buyer asks.
    disposition: Mapped[str | None] = mapped_column(String(48), nullable=True, index=True)
    disposition_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    reference: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)


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


class AgentProfileRow(Base):
    __tablename__ = "agent_profiles"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    # The whole profile as JSON rather than a column per field: the profile
    # gains a field most releases, it is read whole and never queried by field,
    # and a column each would mean a migration each.
    payload: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[float] = mapped_column(Float)
