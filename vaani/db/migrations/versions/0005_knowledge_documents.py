"""Training documents, kept beside the calls.

The vector index becomes a cache of this table rather than the only copy of
what an operator uploaded.

Revision ID: 0005
Revises: 0004
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "knowledge_documents",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("namespace", sa.String(80), nullable=False),
        sa.Column("source", sa.String(512), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("metadata_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("doc_hash", sa.String(32), nullable=False, server_default=""),
        sa.Column("deleted", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("updated_at", sa.Float(), nullable=False),
        sa.UniqueConstraint("namespace", "source", name="uq_knowledge_namespace_source"),
    )
    op.create_index("ix_knowledge_documents_namespace", "knowledge_documents", ["namespace"])


def downgrade() -> None:
    op.drop_index("ix_knowledge_documents_namespace", table_name="knowledge_documents")
    op.drop_table("knowledge_documents")
