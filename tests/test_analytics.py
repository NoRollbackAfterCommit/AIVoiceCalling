"""Reporting that survives a restart, and answers per organisation and per DID.

The figures here are the ones a government buyer asks for in the first meeting:
how many calls, what happened to them, how many the bot handled without a human,
and how long the caller waited. They are computed in SQL over the stored calls
rather than from the in-memory ring of the last two hundred, which is lost on
every restart and cannot be filtered.
"""

from __future__ import annotations

import time

import pytest

from vaani.db.analytics import AnalyticsRepository
from vaani.db.repository import CallRepository
from vaani.db.tenancy import TenancyRepository

DAY = 86400


class _Record:
    """The shape create_call reads. Simpler than driving a whole session."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


async def _call(repo, **kw):
    defaults = dict(
        call_id=kw.pop("call_id"),
        agent_key="a",
        direction="inbound",
        caller_number="+919876543210",
        organisation_id=None,
        did=None,
        started_at=time.time(),
        outcome="completed",
        language="en-IN",
    )
    defaults.update(kw)
    record = _Record(**defaults)
    await repo.create_call(record)
    return record


@pytest.fixture
async def world(tmp_path):
    repo = CallRepository(f"sqlite+aiosqlite:///{tmp_path / 'mis.db'}")
    await repo.start()
    tenancy = TenancyRepository(repo.sessions)
    alpha = await tenancy.create_organisation(slug="alpha", name="Alpha")
    beta = await tenancy.create_organisation(slug="beta", name="Beta")
    yield repo, AnalyticsRepository(repo.sessions), alpha, beta
    await repo.close()


# -- the figures -----------------------------------------------------------


async def test_volume_and_outcomes_are_counted_per_organisation(world):
    repo, mis, alpha, beta = world
    for i in range(3):
        await _call(repo, call_id=f"a{i}", organisation_id=alpha.id, did="+918065605871")
    await _call(repo, call_id="a3", organisation_id=alpha.id, outcome="transferred")
    await _call(repo, call_id="b0", organisation_id=beta.id, outcome="caller_disconnected")

    summary = await mis.summary(organisation_id=alpha.id)
    assert summary["calls"] == 4
    assert summary["outcomes"] == {"completed": 3, "transferred": 1}

    other = await mis.summary(organisation_id=beta.id)
    assert other["calls"] == 1
    assert other["outcomes"] == {"caller_disconnected": 1}


async def test_everything_is_counted_when_no_organisation_is_named(world):
    """The platform view. Unattributed calls are in it and nowhere else."""
    repo, mis, alpha, _beta = world
    await _call(repo, call_id="a0", organisation_id=alpha.id)
    await _call(repo, call_id="orphan", organisation_id=None)

    assert (await mis.summary())["calls"] == 2
    assert (await mis.summary(organisation_id=alpha.id))["calls"] == 1


async def test_figures_can_be_narrowed_to_one_number(world):
    """Two lines for one organisation answer differently, and the point of
    per-DID reporting is being able to see which one is failing callers."""
    repo, mis, alpha, _beta = world
    for i in range(3):
        await _call(repo, call_id=f"x{i}", organisation_id=alpha.id, did="+918065605871")
    await _call(
        repo,
        call_id="y0",
        organisation_id=alpha.id,
        did="+918065605872",
        outcome="caller_disconnected",
    )

    one = await mis.summary(organisation_id=alpha.id, did="+918065605872")
    assert one["calls"] == 1
    assert one["outcomes"] == {"caller_disconnected": 1}


async def test_a_number_is_matched_however_it_is_written(world):
    repo, mis, alpha, _beta = world
    await _call(repo, call_id="x", organisation_id=alpha.id, did="+918065605871")
    assert (await mis.summary(did="08065605871"))["calls"] == 1


async def test_a_date_range_excludes_what_falls_outside_it(world):
    repo, mis, alpha, _beta = world
    now = time.time()
    await _call(repo, call_id="old", organisation_id=alpha.id, started_at=now - 10 * DAY)
    await _call(repo, call_id="new", organisation_id=alpha.id, started_at=now - 1 * DAY)

    recent = await mis.summary(organisation_id=alpha.id, since=now - 3 * DAY)
    assert recent["calls"] == 1

    assert (await mis.summary(organisation_id=alpha.id, until=now - 5 * DAY))["calls"] == 1


async def test_containment_is_the_share_the_bot_finished_alone(world):
    """The number a buyer actually cares about: how many callers got what they
    needed without a person."""
    repo, mis, alpha, _beta = world
    for i in range(7):
        await _call(repo, call_id=f"c{i}", organisation_id=alpha.id, outcome="completed")
    for i in range(3):
        await _call(repo, call_id=f"t{i}", organisation_id=alpha.id, outcome="transferred")

    summary = await mis.summary(organisation_id=alpha.id)
    assert summary["transferred"] == 3
    assert summary["containment"] == pytest.approx(0.7)


async def test_dispositions_and_languages_are_broken_down(world):
    repo, mis, alpha, _beta = world
    await _call(repo, call_id="d0", organisation_id=alpha.id, language="bn-IN")
    await _call(repo, call_id="d1", organisation_id=alpha.id, language="bn-IN")
    await _call(repo, call_id="d2", organisation_id=alpha.id, language="hi-IN")

    summary = await mis.summary(organisation_id=alpha.id)
    assert summary["languages"] == {"bn-IN": 2, "hi-IN": 1}


async def test_an_empty_range_reports_zero_rather_than_failing(world):
    """A new organisation opens its dashboard before taking a single call."""
    _repo, mis, _alpha, beta = world
    summary = await mis.summary(organisation_id=beta.id)
    assert summary["calls"] == 0
    assert summary["containment"] == 0
    assert summary["outcomes"] == {}
    assert summary["per_day"] == []


# -- what the caller waited for --------------------------------------------


async def test_the_wait_is_reported_apart_from_the_time_spent_speaking(world):
    """total_ms is turn wall-clock — it includes the reply playing out at real
    time. Reporting it as latency is what sent a whole afternoon hunting for
    seconds that were never lost."""
    repo, mis, alpha, _beta = world
    await _call(repo, call_id="m0", organisation_id=alpha.id)
    for seq, (stt, agent, ttfa, total) in enumerate(
        [(300, 1800, 300, 6000), (500, 2200, 300, 8000)]
    ):
        repo.append_turn(
            "m0",
            seq,
            "agent",
            "hello",
            "en-IN",
            {"stt_ms": stt, "agent_ms": agent, "tts_first_chunk_ms": ttfa, "total_ms": total},
        )
    await repo.flush()

    latency = (await mis.summary(organisation_id=alpha.id))["latency"]
    # 300+1800+300 = 2400 and 500+2200+300 = 3000; the median of two is the mean.
    assert latency["wait_ms"] == pytest.approx(2700, abs=1)
    # 6000-2400 = 3600 and 8000-3000 = 5000.
    assert latency["spoken_ms"] == pytest.approx(4300, abs=1)
    assert latency["turns"] == 2


async def test_latency_ignores_turns_from_another_organisation(world):
    repo, mis, alpha, beta = world
    await _call(repo, call_id="mine", organisation_id=alpha.id)
    await _call(repo, call_id="theirs", organisation_id=beta.id)
    repo.append_turn(
        "mine",
        0,
        "agent",
        "x",
        "en-IN",
        {"stt_ms": 100, "agent_ms": 100, "tts_first_chunk_ms": 100, "total_ms": 1000},
    )
    repo.append_turn(
        "theirs",
        0,
        "agent",
        "x",
        "en-IN",
        {"stt_ms": 9000, "agent_ms": 9000, "tts_first_chunk_ms": 9000, "total_ms": 90000},
    )
    await repo.flush()

    latency = (await mis.summary(organisation_id=alpha.id))["latency"]
    assert latency["turns"] == 1
    assert latency["wait_ms"] == pytest.approx(300, abs=1)


# -- the trend -------------------------------------------------------------


async def test_calls_are_counted_per_day_for_the_trend(world):
    repo, mis, alpha, _beta = world
    now = time.time()
    for i in range(2):
        await _call(repo, call_id=f"t{i}", organisation_id=alpha.id, started_at=now - 1 * DAY)
    await _call(repo, call_id="t2", organisation_id=alpha.id, started_at=now)

    per_day = (await mis.summary(organisation_id=alpha.id))["per_day"]
    assert [row["calls"] for row in per_day] == [2, 1]
    assert per_day[0]["day"] < per_day[1]["day"], "the trend must read left to right"


# -- durability ------------------------------------------------------------


async def test_the_figures_survive_a_restart(tmp_path):
    """The defect in the implementation this replaces: an in-memory ring of the
    last two hundred calls, gone the moment the container is recreated — which
    a deploy does every time."""
    url = f"sqlite+aiosqlite:///{tmp_path / 'restart.db'}"
    first = CallRepository(url)
    await first.start()
    org = await TenancyRepository(first.sessions).create_organisation(slug="o", name="O")
    await _call(first, call_id="before", organisation_id=org.id)
    await first.close()

    second = CallRepository(url)
    await second.start()
    try:
        summary = await AnalyticsRepository(second.sessions).summary(organisation_id=org.id)
        assert summary["calls"] == 1
    finally:
        await second.close()
