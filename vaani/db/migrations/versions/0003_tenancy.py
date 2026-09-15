"""Organisations and the numbers that reach them.

One deployment runs several customers' call centres, and which one a caller
reaches is decided by the number they dialled. Existing rows are seeded into a
single organisation so the live deployment keeps answering through the upgrade:
a migration that leaves production unable to route calls is not a migration.

`calls.organisation_id` is nullable on purpose. A call to a number nobody has
mapped yet is still served — the alternative is dropping a real caller over a
configuration gap — and it must then be visible in reporting as unattributed
rather than quietly counted against whichever organisation happens to be first.

Revision ID: 0003
Revises: 0002
"""

from __future__ import annotations

import time

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

_SEED_SLUG = "default"
_SEED_NAME = "Default organisation"


def upgrade() -> None:
    organisations = op.create_table(
        "organisations",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("slug", sa.String(64), nullable=False, unique=True),
        sa.Column("name", sa.String(160), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.Float(), nullable=False),
    )
    op.create_index("ix_organisations_slug", "organisations", ["slug"])

    op.create_table(
        "dids",
        sa.Column("number", sa.String(24), primary_key=True),
        sa.Column(
            "organisation_id",
            sa.Integer(),
            sa.ForeignKey("organisations.id"),
            nullable=False,
        ),
        sa.Column("agent_key", sa.String(64), nullable=False),
        sa.Column("label", sa.String(120), nullable=True),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.Float(), nullable=False),
    )
    op.create_index("ix_dids_organisation_id", "dids", ["organisation_id"])

    with op.batch_alter_table("agent_profiles") as batch:
        batch.add_column(sa.Column("organisation_id", sa.Integer(), nullable=True))

    with op.batch_alter_table("calls") as batch:
        batch.add_column(sa.Column("organisation_id", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("did", sa.String(24), nullable=True))

    # Reporting always filters on one of these two, over a time range.
    op.create_index("ix_calls_org_started", "calls", ["organisation_id", "started_at"])
    op.create_index("ix_calls_did_started", "calls", ["did", "started_at"])

    # Everything that exists today belongs to one organisation. Without this the
    # first upgraded deployment has agents owned by nobody, which no role can
    # then administer.
    seeded = op.get_bind().execute(
        sa.insert(organisations).values(
            slug=_SEED_SLUG, name=_SEED_NAME, active=True, created_at=time.time()
        )
    )
    organisation_id = seeded.inserted_primary_key[0]
    op.get_bind().execute(
        sa.text("UPDATE agent_profiles SET organisation_id = :oid").bindparams(oid=organisation_id)
    )
    op.get_bind().execute(
        sa.text("UPDATE calls SET organisation_id = :oid").bindparams(oid=organisation_id)
    )


def downgrade() -> None:
    op.drop_index("ix_calls_did_started", "calls")
    op.drop_index("ix_calls_org_started", "calls")
    with op.batch_alter_table("calls") as batch:
        batch.drop_column("did")
        batch.drop_column("organisation_id")
    with op.batch_alter_table("agent_profiles") as batch:
        batch.drop_column("organisation_id")
    op.drop_index("ix_dids_organisation_id", "dids")
    op.drop_table("dids")
    op.drop_index("ix_organisations_slug", "organisations")
    op.drop_table("organisations")
