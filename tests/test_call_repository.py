"""Persistence must survive a restart and must never block the call path."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from vaani.db.repository import CallRepository


@dataclass
class FakeRecord:
    call_id: str = "c1"
    agent_key: str = "default"
    direction: str = "inbound"
    caller_number: str | None = "+919876543210"
    started_at: float = 1000.0
    ended_at: float | None = None
    outcome: str = "in_progress"
    summary: str | None = None
    language: str | None = "hi-IN"
    duration_s: float = 0.0


@pytest.fixture
async def repo(tmp_path):
    r = CallRepository(f"sqlite+aiosqlite:///{tmp_path / 'calls.db'}")
    await r.start()
    yield r
    await r.close()


async def test_creates_and_reads_back_a_call(repo):
    await repo.create_call(FakeRecord())
    row = await repo.get_call("c1")
    assert row["caller_number"] == "+919876543210"
    assert row["outcome"] == "in_progress"


async def test_append_turn_does_not_block(repo):
    """It is called from the turn loop, so it must be synchronous and return
    immediately — the write happens on a background task."""
    await repo.create_call(FakeRecord())
    result = repo.append_turn(
        "c1",
        0,
        "caller",
        "बिजली का बिल",
        "hi-IN",
        {"stt_ms": 300, "agent_ms": 400, "tts_first_chunk_ms": 200, "total_ms": 900},
    )
    assert result is None
    await repo.flush()
    turns = await repo.turns_for("c1")
    assert len(turns) == 1
    assert turns[0]["text"] == "बिजली का बिल"
    assert turns[0]["stt_ms"] == 300


async def test_turns_come_back_in_order(repo):
    await repo.create_call(FakeRecord())
    for i in range(5):
        repo.append_turn("c1", i, "caller", f"turn {i}", "hi-IN", {})
    await repo.flush()
    turns = await repo.turns_for("c1")
    assert [t["seq"] for t in turns] == [0, 1, 2, 3, 4]


async def test_finish_call_records_the_outcome(repo):
    await repo.create_call(FakeRecord())
    await repo.finish_call(
        FakeRecord(ended_at=1090.0, outcome="completed", summary="Bill query", duration_s=90.0)
    )
    row = await repo.get_call("c1")
    assert row["outcome"] == "completed"
    assert row["duration_s"] == 90.0
    assert row["summary"] == "Bill query"


async def test_data_survives_a_restart(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'calls.db'}"
    first = CallRepository(url)
    await first.start()
    await first.create_call(FakeRecord())
    first.append_turn("c1", 0, "caller", "नमस्ते", "hi-IN", {})
    await first.flush()
    await first.close()

    second = CallRepository(url)
    await second.start()
    assert (await second.get_call("c1"))["call_id"] == "c1"
    assert len(await second.turns_for("c1")) == 1
    await second.close()


async def test_a_full_queue_drops_rather_than_blocking(repo):
    """Losing a turn record is bad. Stalling every concurrent call because the
    disk is slow is worse."""
    await repo.create_call(FakeRecord())
    # Comfortably past the 2000-slot queue: the point is that put_nowait never
    # blocks, not how many rows land.
    for i in range(2500):
        repo.append_turn("c1", i, "caller", "x", "hi-IN", {})
    await repo.flush()
    assert len(await repo.turns_for("c1")) > 0


async def test_recent_returns_newest_first(repo):
    await repo.create_call(FakeRecord(call_id="old", started_at=100.0))
    await repo.create_call(FakeRecord(call_id="new", started_at=200.0))
    assert [r["call_id"] for r in await repo.recent()] == ["new", "old"]


async def test_finishing_an_unknown_call_is_a_no_op(repo):
    """A call rejected at capacity never got a row. Finishing it must not raise."""
    await repo.finish_call(FakeRecord(call_id="never-created", outcome="rejected"))
    assert await repo.get_call("never-created") is None
