"""Agent profiles edited through the API survive a restart.

The profile is behaviour-as-data: greeting, policies, voices, objective. An
operator who tunes it over a week of pilot calls and loses it to a reboot will
not trust the platform with anything else.
"""

from __future__ import annotations

import json
from dataclasses import fields, replace

import httpx
import pytest
from alembic import command
from sqlalchemy import create_engine, inspect, text

from vaani.agent.prompt import AgentProfile, profile_from_dict, profile_to_dict
from vaani.db.migrate import _config, to_sync_url, upgrade
from vaani.db.profiles import ProfileRepository
from vaani.db.repository import CallRepository

TUNED = AgentProfile(
    key="pension",
    name="Sahayak",
    organisation="Pension Directorate",
    greeting="Namaste, pension helpline.",
    policies=["Never quote a disbursement date."],
    voices={"hi-IN": "hi-IN:priya", "bn-IN": "bn-IN:shreya"},
    objective="Register the grievance and give a reference.",
    extra_dispositions=["grievance_registered"],
    stall_after=2,
    ask_language=False,
)


@pytest.fixture
async def repo(tmp_path):
    r = CallRepository(f"sqlite+aiosqlite:///{tmp_path / 'vaani.db'}")
    await r.start()
    yield r
    await r.close()


@pytest.fixture
def store(repo):
    return ProfileRepository(repo.sessions)


# -- codec -------------------------------------------------------------------


def test_codec_round_trips_every_field():
    assert profile_from_dict(profile_to_dict(TUNED)) == TUNED


def test_codec_tolerates_payloads_from_other_releases():
    """A payload from an older release lacks the fields added since; one from
    a newer release carries fields this one does not know. Both must load."""
    profile = profile_from_dict({"key": "old", "name": "Old", "field_from_the_future": 1})
    assert profile.name == "Old"
    assert profile.ask_language is True, "missing fields take the dataclass default"
    assert profile.tools == AgentProfile(key="x").tools


# -- store -------------------------------------------------------------------


async def test_a_saved_profile_survives_a_restart(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'vaani.db'}"
    first = CallRepository(url)
    await first.start()
    await ProfileRepository(first.sessions).save(TUNED)
    await first.close()

    second = CallRepository(url)
    await second.start()
    try:
        loaded = await ProfileRepository(second.sessions).load()
    finally:
        await second.close()
    assert loaded == {"pension": TUNED}


async def test_saving_again_replaces_rather_than_duplicates(store):
    await store.save(TUNED)
    await store.save(replace(TUNED, greeting="Updated."))
    loaded = await store.load()
    assert list(loaded) == ["pension"]
    assert loaded["pension"].greeting == "Updated."


async def test_delete_removes_the_profile(store):
    await store.save(TUNED)
    await store.delete("pension")
    assert await store.load() == {}


async def test_an_unreadable_row_is_skipped_not_fatal(store, repo):
    """A corrupt row must not stop the phone being answered."""
    await store.save(TUNED)
    async with repo.sessions() as session, session.begin():
        await session.execute(
            text(
                "INSERT INTO agent_profiles (key, payload, updated_at) "
                "VALUES ('broken', 'not json', 0)"
            )
        )
    loaded = await store.load()
    assert set(loaded) == {"pension"}


async def test_the_row_key_wins_over_the_payload_key(store, repo):
    async with repo.sessions() as session, session.begin():
        await session.execute(
            text("INSERT INTO agent_profiles (key, payload, updated_at) VALUES ('rowkey', :p, 0)"),
            {"p": json.dumps({"key": "payloadkey", "name": "X"})},
        )
    loaded = await store.load()
    assert loaded["rowkey"].key == "rowkey"


# -- migration ---------------------------------------------------------------


def test_a_phase_3_database_gains_the_profiles_table(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'old.db'}"
    command.upgrade(_config(to_sync_url(url)), "0001")
    assert "agent_profiles" not in inspect(create_engine(to_sync_url(url))).get_table_names()

    upgrade(url)
    assert "agent_profiles" in inspect(create_engine(to_sync_url(url))).get_table_names()


# -- API ---------------------------------------------------------------------


# Everything a caller may set. `organisation_id` is deliberately absent: it
# decides who owns the profile, and a field the body could set is one an org
# admin could use to hand their agent — and its calls and corpus — to another
# organisation, or to claim one of theirs. The endpoint fills it from the
# authenticated user instead, which is why it is excluded here rather than
# added to the model.
_SERVER_OWNED = {"organisation_id"}


def test_the_api_input_model_covers_every_profile_field():
    """The request model is a hand-written mirror of the dataclass. When they
    drift, a save silently resets the fields the model forgot — which is how
    voices and the outcome tools were being wiped."""
    from vaani.api.routes import AgentProfileIn

    expected = {f.name for f in fields(AgentProfile)} - _SERVER_OWNED
    assert set(AgentProfileIn.model_fields) == expected


async def test_an_agents_owner_cannot_be_set_from_the_request_body(services, store):
    """The guard against an org admin re-homing an agent by hand."""
    from vaani.api.routes import AgentProfileIn

    assert "organisation_id" not in AgentProfileIn.model_fields
    body = AgentProfileIn(key="x", organisation_id=999)  # type: ignore[call-arg]
    assert not hasattr(body, "organisation_id")


def _client(services):
    from vaani.main import create_app

    app = create_app()
    app.state.services = services
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_put_persists_the_profile(services, store):
    services.profile_store = store
    async with _client(services) as client:
        resp = await client.put(
            "/api/agents/pension",
            json={
                "key": "pension",
                "name": "Sahayak",
                "voices": {"hi-IN": "hi-IN:priya"},
                "objective": "Register the grievance.",
                "ask_language": False,
            },
        )
    assert resp.status_code == 200, resp.text
    loaded = await store.load()
    assert loaded["pension"].voices == {"hi-IN": "hi-IN:priya"}
    assert loaded["pension"].objective == "Register the grievance."
    assert loaded["pension"].ask_language is False


async def test_put_without_tools_keeps_the_outcome_tools(services, store):
    """end_call refuses to run without set_disposition; a save that dropped
    both would turn every call for that agent into one that cannot end."""
    services.profile_store = store
    async with _client(services) as client:
        resp = await client.put("/api/agents/pension", json={"key": "pension"})
        assert resp.status_code == 200, resp.text
        shown = (await client.get("/api/agents/pension")).json()
    assert {"set_disposition", "end_call"} <= set(shown["tools"])


async def test_delete_removes_the_persisted_profile(services, store):
    services.profile_store = store
    await store.save(TUNED)
    services.profiles["pension"] = TUNED
    async with _client(services) as client:
        resp = await client.delete("/api/agents/pension")
    assert resp.status_code == 200
    assert await store.load() == {}
    assert "pension" not in services.profiles


async def test_saving_works_with_no_store_attached(services):
    """A bare install has no database wired to the services; the editor must
    still work in memory, as it always has."""
    async with _client(services) as client:
        resp = await client.put("/api/agents/demo", json={"key": "demo", "name": "Demo"})
    assert resp.status_code == 200, resp.text
    assert services.profiles["demo"].name == "Demo"


async def test_reading_a_profile_and_writing_it_straight_back_changes_nothing(services, store):
    """What the agent editor in the console does on every save.

    PUT replaces the whole profile, so the page cannot send only the fields it
    draws — it merges the form over the profile the server gave it. That is only
    safe while a GET body is accepted verbatim by PUT. If the two shapes ever
    drift, the editor starts quietly resetting whatever GET returns and PUT
    ignores, which is precisely how voices and the outcome tools were wiped once
    before.
    """
    services.profile_store = store
    services.profiles["pension"] = TUNED
    async with _client(services) as client:
        before = (await client.get("/api/agents/pension")).json()
        sent = {k: v for k, v in before.items() if k != "system_prompt"}
        assert (await client.put("/api/agents/pension", json=sent)).status_code == 200
        after = (await client.get("/api/agents/pension")).json()

    drifted = {k: (before[k], after.get(k)) for k in before if before[k] != after.get(k)}
    assert not drifted, f"a round trip through the editor would change {drifted}"


async def test_the_api_says_what_a_new_agent_starts_as(services):
    """The console draws a blank form for a new agent and then posts it. Without
    somewhere to read the starting values from, every field left alone is posted
    empty and turns its default off — which produced an agent with `tools: []`:
    one that could not search its own documents, could not transfer, and could
    not hang up.

    The endpoint sits outside the /api/agents prefix deliberately. Under it, the
    path would have to be registered ahead of /agents/{key} to avoid being
    swallowed as a key, and that is an ordering dependency nothing would catch
    when somebody reorders the file.
    """
    async with _client(services) as client:
        resp = await client.get("/api/agent-defaults")

    assert resp.status_code == 200, resp.text
    starting = resp.json()
    assert starting["key"] == "", "a new agent has no key until the operator names one"
    assert {"set_disposition", "end_call"} <= set(starting["tools"]), (
        "end_call will not run without set_disposition; an agent starting "
        "without both is one that can never end a call"
    )
    assert starting["closing"], "a call that just stops is a call that sounds dropped"

    # Every field the form has to draw must be here, or it falls back to a blank.
    assert set(starting) >= set(AgentProfileIn_fields()), "the form would draw blanks"


def AgentProfileIn_fields():
    from vaani.api.routes import AgentProfileIn

    return set(AgentProfileIn.model_fields)
