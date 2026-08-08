"""Bringing the database up to date, on both SQLite and Postgres.

Replaces `create_all`, which only ever creates missing *tables* — it will not
add a column to a table that already exists. That is how a deployment ends up
booting healthy on last release's schema and only failing when it tries to write
the new field, which is exactly what happened when phase 2 added the disposition
columns to a live pilot database.

Migrations run inside the application at startup rather than as a separate
deploy step, because an on-premise operator runs `docker compose up` and nothing
else. That is safe for the single-worker pilot this targets; a multi-worker
deployment should run `alembic upgrade head` once in an init container instead,
since concurrent upgrades race.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config

from vaani.core.logging import get_logger

log = get_logger(__name__)

_ROOT = Path(__file__).resolve().parents[2]


def _config(sync_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(Path(__file__).parent / "migrations"))
    config.set_main_option("sqlalchemy.url", sync_url.replace("%", "%%"))
    return config


def to_sync_url(url: str) -> str:
    """Alembic runs on a synchronous driver; the app runs on an async one."""
    return (
        url.replace("+aiosqlite", "")
        .replace("+asyncpg", "+psycopg2")
        .replace("+aiomysql", "+pymysql")
    )


def upgrade(url: str) -> None:
    """Apply every outstanding migration. Idempotent."""
    sync_url = to_sync_url(url)
    command.upgrade(_config(sync_url), "head")
    log.info("database schema up to date")


def current_revision(connection: Any) -> str | None:
    from alembic.migration import MigrationContext

    return MigrationContext.configure(connection).get_current_revision()
