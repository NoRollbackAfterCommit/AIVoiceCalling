"""Alembic environment, driven from Settings rather than alembic.ini.

The database URL lives in one place — the settings store — so the migration
runner reads it from there instead of a second copy in a config file that
drifts. Migrations run online only; there is no offline SQL-generation mode
because a government deployment applies them through this same path.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool

from vaani.db.models import Base

config = context.config
target_metadata = Base.metadata


def run_migrations_online() -> None:
    connectable = config.attributes.get("connection", None)
    if connectable is not None:
        _run(connectable)
        return

    engine = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with engine.connect() as connection:
        _run(connection)


def _run(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # SQLite cannot ALTER a column in place; batch mode rewrites the table
        # instead, which is what makes the same migration work on both backends.
        render_as_batch=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


run_migrations_online()
