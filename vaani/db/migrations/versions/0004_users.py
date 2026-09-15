"""People who sign in to the portal.

No user is seeded. A migration that creates an administrator with a known
password is a backdoor in every deployment that runs it — the first account is
created deliberately, by `python -m vaani.admin create-admin`, and only while
the table is empty.

Revision ID: 0004
Revises: 0003
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("email", sa.String(254), nullable=False, unique=True),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("name", sa.String(120), nullable=False, server_default=""),
        sa.Column("role", sa.String(24), nullable=False),
        sa.Column(
            "organisation_id",
            sa.Integer(),
            sa.ForeignKey("organisations.id"),
            nullable=True,
        ),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("last_login_at", sa.Float(), nullable=True),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_role", "users", ["role"])
    op.create_index("ix_users_organisation_id", "users", ["organisation_id"])


def downgrade() -> None:
    op.drop_index("ix_users_organisation_id", "users")
    op.drop_index("ix_users_role", "users")
    op.drop_index("ix_users_email", "users")
    op.drop_table("users")
