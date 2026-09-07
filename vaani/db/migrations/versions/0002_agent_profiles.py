"""Agent profiles.

One JSON payload per profile rather than a column per field: the profile gains
a field most releases, it is read whole and never queried by field, and a
column each would mean a migration each.

Revision ID: 0002
Revises: 0001
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_profiles",
        sa.Column("key", sa.String(64), primary_key=True),
        sa.Column("payload", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Float(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("agent_profiles")
