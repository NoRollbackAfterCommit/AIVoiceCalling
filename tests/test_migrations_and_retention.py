"""Phase 3: the schema upgrades itself, and retention is actually enforced.

Both of these exist because of concrete failures. A live pilot database booted
healthy on the old schema after phase 2 added columns, because create_all never
alters an existing table. And retention_days was a number in the settings page
that nothing acted on — which is the wrong answer to give a government buyer
about how long citizen recordings are kept.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

import pytest

from vaani.db.migrate import to_sync_url, upgrade
from vaani.db.repository import CallRepository

NEW_COLUMNS = ("disposition", "disposition_reason", "reference")


@dataclass
class Rec:
    call_id: str = "c1"
    agent_key: str = "default"
    direction: str = "inbound"
    caller_number: str | None = None
    started_at: float = 0.0
    ended_at: float | None = None
    outcome: str = "completed"
    summary: str | None = None
    language: str | None = "hi-IN"
    duration_s: float = 10.0
    disposition: str | None = "resolved"
    disposition_reason: str | None = "answered"
    reference: str | None = None
    recording_path: str | None = None


def _columns(path) -> set[str]:
    return {c[1] for c in sqlite3.connect(str(path)).execute("PRAGMA table_info(calls)")}


# -- url translation ------------------------------------------------------


def test_async_urls_become_sync_ones_for_alembic():
    """Alembic runs on a sync driver while the app runs on an async one."""
    assert to_sync_url("sqlite+aiosqlite:///./x.db") == "sqlite:///./x.db"
    assert (
        to_sync_url("postgresql+asyncpg://u:p@h/db") == "postgresql+psycopg2://u:p@h/db"
    )


def test_a_plain_url_is_left_alone():
    assert to_sync_url("sqlite:///./x.db") == "sqlite:///./x.db"


# -- migrations -----------------------------------------------------------


async def test_a_fresh_database_gets_the_whole_schema(tmp_path):
    db = tmp_path / "fresh.db"
    repo = CallRepository(f"sqlite+aiosqlite:///{db}")
    await repo.start()
    try:
        assert NEW_COLUMNS[0] in _columns(db)
        assert "turns" in {
            r[0] for r in sqlite3.connect(str(db)).execute(
                "select name from sqlite_master where type='table'"
            )
        }
    finally:
        await repo.close()


def test_a_pre_phase_2_database_gains_the_missing_columns(tmp_path):
    """The exact failure that shipped: a pilot database created before the
    disposition columns existed. create_all leaves it alone and the deployment
    looks healthy right up until it tries to record an outcome."""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE calls (
            call_id VARCHAR(32) PRIMARY KEY,
            agent_key VARCHAR(64) NOT NULL,
            direction VARCHAR(16) DEFAULT 'inbound',
            caller_number VARCHAR(32),
            started_at FLOAT NOT NULL,
            ended_at FLOAT,
            outcome VARCHAR(32) DEFAULT 'in_progress',
            summary TEXT,
            language VARCHAR(16),
            duration_s FLOAT DEFAULT 0,
            recording_path TEXT
        );
        CREATE TABLE turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            call_id VARCHAR(32) REFERENCES calls(call_id),
            seq INTEGER, role VARCHAR(16), text TEXT, language VARCHAR(16),
            stt_ms INTEGER DEFAULT 0, agent_ms INTEGER DEFAULT 0,
            tts_first_chunk_ms INTEGER DEFAULT 0, total_ms INTEGER DEFAULT 0,
            barged_in BOOLEAN DEFAULT 0
        );
        INSERT INTO calls (call_id, agent_key, started_at) VALUES ('old-1', 'default', 1.0);
        """
    )
    conn.commit()
    conn.close()

    assert not (set(NEW_COLUMNS) & _columns(db)), "fixture must start without them"

    upgrade(f"sqlite+aiosqlite:///{db}")

    assert set(NEW_COLUMNS) <= _columns(db)
    # And the pilot's existing rows survive the upgrade.
    rows = sqlite3.connect(str(db)).execute("select call_id from calls").fetchall()
    assert rows == [("old-1",)]


def test_migrating_twice_is_a_no_op(tmp_path):
    db = tmp_path / "twice.db"
    url = f"sqlite+aiosqlite:///{db}"
    upgrade(url)
    upgrade(url)
    assert set(NEW_COLUMNS) <= _columns(db)


async def test_an_upgraded_database_can_actually_be_written_to(tmp_path):
    """The point of the migration, not just the shape of the table."""
    db = tmp_path / "write.db"
    repo = CallRepository(f"sqlite+aiosqlite:///{db}")
    await repo.start()
    try:
        rec = Rec(started_at=time.time())
        await repo.create_call(rec)
        await repo.finish_call(rec)
        assert (await repo.get_call("c1"))["disposition"] == "resolved"
    finally:
        await repo.close()


# -- retention ------------------------------------------------------------


@pytest.fixture
async def repo(tmp_path):
    r = CallRepository(f"sqlite+aiosqlite:///{tmp_path / 'r.db'}")
    await r.start()
    yield r
    await r.close()


async def test_records_past_the_window_are_deleted(repo):
    old = Rec(call_id="old", started_at=time.time() - 400 * 86400)
    recent = Rec(call_id="recent", started_at=time.time() - 10 * 86400)
    for rec in (old, recent):
        await repo.create_call(rec)
        repo.append_turn(rec.call_id, 0, "caller", "hello", "hi-IN", {})
    await repo.flush()

    assert await repo.purge_older_than(365) == 1
    assert await repo.get_call("old") is None
    assert await repo.get_call("recent") is not None
    # The turns must go too, or the transcript outlives its retention window.
    assert await repo.turns_for("old") == []
    assert await repo.turns_for("recent")


async def test_recordings_are_deleted_with_their_records(repo, tmp_path):
    """The audio is the bulk of it and the most sensitive part. A row removed
    while its recording stays on disk has not been deleted in any real sense."""
    audio = tmp_path / "old-call.wav"
    audio.write_bytes(b"RIFF")
    rec = Rec(
        call_id="old", started_at=time.time() - 400 * 86400, recording_path=str(audio)
    )
    await repo.create_call(rec)
    await repo.finish_call(rec)

    await repo.purge_older_than(365)
    assert not audio.exists()


async def test_a_missing_recording_file_does_not_break_the_sweep(repo):
    rec = Rec(
        call_id="old",
        started_at=time.time() - 400 * 86400,
        recording_path="/no/such/file.wav",
    )
    await repo.create_call(rec)
    await repo.finish_call(rec)
    assert await repo.purge_older_than(365) == 1


async def test_nothing_is_deleted_when_nothing_is_old_enough(repo):
    await repo.create_call(Rec(call_id="new", started_at=time.time()))
    assert await repo.purge_older_than(365) == 0
    assert await repo.get_call("new") is not None


async def test_a_non_positive_window_disables_the_sweep(repo):
    """A misconfigured zero must not be read as "delete everything"."""
    await repo.create_call(Rec(call_id="old", started_at=0.0))
    assert await repo.purge_older_than(0) == 0
    assert await repo.get_call("old") is not None
