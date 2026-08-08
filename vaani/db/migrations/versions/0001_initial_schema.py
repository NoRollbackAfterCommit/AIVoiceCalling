"""Calls and turns.

The baseline. Everything phases 1 and 2 built, captured as one revision so a
fresh database and an existing one converge on the same schema.

Revision ID: 0001
Revises:
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    existing = set(sa.inspect(bind).get_table_names())

    if "calls" not in existing:
        op.create_table(
            "calls",
            sa.Column("call_id", sa.String(32), primary_key=True),
            sa.Column("agent_key", sa.String(64), nullable=False),
            sa.Column("direction", sa.String(16), nullable=False, server_default="inbound"),
            sa.Column("caller_number", sa.String(32), nullable=True),
            sa.Column("started_at", sa.Float(), nullable=False),
            sa.Column("ended_at", sa.Float(), nullable=True),
            sa.Column("outcome", sa.String(32), nullable=False, server_default="in_progress"),
            sa.Column("summary", sa.Text(), nullable=True),
            sa.Column("language", sa.String(16), nullable=True),
            sa.Column("duration_s", sa.Float(), nullable=False, server_default="0"),
            sa.Column("recording_path", sa.Text(), nullable=True),
            sa.Column("disposition", sa.String(48), nullable=True),
            sa.Column("disposition_reason", sa.Text(), nullable=True),
            sa.Column("reference", sa.String(64), nullable=True),
        )
        op.create_index("ix_calls_agent_key", "calls", ["agent_key"])
        op.create_index("ix_calls_started_at", "calls", ["started_at"])
        op.create_index("ix_calls_outcome", "calls", ["outcome"])
        op.create_index("ix_calls_disposition", "calls", ["disposition"])
        op.create_index("ix_calls_reference", "calls", ["reference"])
    else:
        # An existing pilot database predates the phase 2 columns, and
        # create_all never adds a column to a table that already exists — which
        # is exactly how a deployment ends up healthy but unable to write an
        # outcome.
        present = {c["name"] for c in sa.inspect(bind).get_columns("calls")}
        for name, column in (
            ("disposition", sa.Column("disposition", sa.String(48), nullable=True)),
            ("disposition_reason", sa.Column("disposition_reason", sa.Text(), nullable=True)),
            ("reference", sa.Column("reference", sa.String(64), nullable=True)),
        ):
            if name not in present:
                op.add_column("calls", column)

    if "turns" not in existing:
        op.create_table(
            "turns",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column(
                "call_id",
                sa.String(32),
                sa.ForeignKey("calls.call_id"),
                nullable=False,
            ),
            sa.Column("seq", sa.Integer(), nullable=False),
            sa.Column("role", sa.String(16), nullable=False),
            sa.Column("text", sa.Text(), nullable=False, server_default=""),
            sa.Column("language", sa.String(16), nullable=True),
            sa.Column("stt_ms", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("agent_ms", sa.Integer(), nullable=False, server_default="0"),
            sa.Column(
                "tts_first_chunk_ms", sa.Integer(), nullable=False, server_default="0"
            ),
            sa.Column("total_ms", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("barged_in", sa.Boolean(), nullable=False, server_default=sa.false()),
        )
        op.create_index("ix_turns_call_id", "turns", ["call_id"])


def downgrade() -> None:
    op.drop_table("turns")
    op.drop_table("calls")
