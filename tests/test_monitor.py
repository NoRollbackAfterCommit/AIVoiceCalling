"""The supervisor feed: hub fan-out, the transport tap, and what a turn records
about retrieval.

The contract that matters is the one that keeps monitoring harmless: watching a
call must never slow it down. A supervisor who cannot keep up loses events; the
call never waits.
"""

from __future__ import annotations

import asyncio

from tests.test_pipeline import FakeTransport, silence, tone
from vaani.pipeline.manager import CallManager
from vaani.pipeline.monitor import _QUEUE_MAX, MonitorHub
from vaani.pipeline.session import CallSession, _MonitoredTransport

# ---------------------------------------------------------------------------
# MonitorHub
# ---------------------------------------------------------------------------


async def test_publish_reaches_every_subscriber_tagged_with_call():
    hub = MonitorHub()
    first, second = hub.subscribe(), hub.subscribe()
    hub.publish("c1", {"type": "transcript", "text": "hello"})
    for queue in (first, second):
        assert queue.get_nowait() == {"call_id": "c1", "type": "transcript", "text": "hello"}


async def test_publish_with_no_subscribers_is_a_noop():
    MonitorHub().publish("c1", {"type": "state"})  # must not raise


async def test_unsubscribed_queue_stops_receiving():
    hub = MonitorHub()
    queue = hub.subscribe()
    hub.unsubscribe(queue)
    hub.publish("c1", {"type": "state"})
    assert queue.empty()


async def test_slow_subscriber_loses_events_instead_of_blocking():
    hub = MonitorHub()
    queue = hub.subscribe()
    for i in range(_QUEUE_MAX + 10):
        hub.publish("c1", {"type": "state", "n": i})
    assert queue.qsize() == _QUEUE_MAX, "overflow must drop, never wait"


# ---------------------------------------------------------------------------
# The transport tap
# ---------------------------------------------------------------------------


async def test_monitored_transport_mirrors_events_but_not_audio():
    hub = MonitorHub()
    queue = hub.subscribe()
    inner = FakeTransport()
    wrapped = _MonitoredTransport(inner, hub, "c9")

    await wrapped.send_audio(b"\x00\x00" * 160)
    await wrapped.send_event({"type": "state", "state": "LISTENING"})
    await wrapped.close()

    assert inner.audio and inner.events and inner.closed, "everything passes through"
    assert queue.get_nowait() == {"call_id": "c9", "type": "state", "state": "LISTENING"}
    assert queue.empty(), "audio frames must never reach supervisors"


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_monitor_endpoint_is_routed():
    from vaani.main import create_app

    # url_path_for raises NoMatchFound when the router was never included, and
    # sees through the lazy router wrappers that app.routes hides paths behind.
    assert create_app().url_path_for("monitor") == "/ws/monitor"


def test_manager_exposes_capacity():
    assert CallManager(max_concurrent=7).capacity == 7


# ---------------------------------------------------------------------------
# Retrieval observability
# ---------------------------------------------------------------------------


def test_failed_searches_are_not_recorded():
    from vaani.pipeline.session import _knowledge_searches

    calls = [
        {
            "name": "search_knowledge",
            "arguments": {"query": "bill due date"},
            "ok": True,
            "data": {"sources": ["faq"], "scores": [0.8]},
        },
        {"name": "search_knowledge", "arguments": {"query": "nothing"}, "ok": False, "data": None},
        {"name": "end_call", "arguments": {}, "ok": True, "data": {}},
    ]
    assert _knowledge_searches(calls) == [
        {"query": "bill due date", "sources": ["faq"], "scores": [0.8]}
    ]


async def test_turn_records_what_was_retrieved(services):
    await services.retriever.index_text(
        "You can check your electricity bill online. Electricity bills are due "
        "on the fifteenth of every month.",
        source="billing-faq",
    )
    transport = FakeTransport()
    session = CallSession(transport, services)
    task = asyncio.create_task(session.run())
    await asyncio.sleep(0.25)

    # MockSTT's first scripted line asks about an electricity bill, which makes
    # MockLLM call search_knowledge — the same path a real call takes.
    await session.push_audio(tone(700))
    await session.push_audio(silence(500))
    await asyncio.sleep(0.8)

    await session.hangup("test_complete")
    record = await asyncio.wait_for(task, timeout=5)

    turn = record.turns[0]
    assert turn["retrieval"], "a turn that searched must record what came back"
    assert "billing-faq" in turn["retrieval"][0]["sources"]

    events = [e for e in transport.events if e.get("type") == "retrieval"]
    assert events and "billing-faq" in events[0]["sources"]


async def test_supervisor_sees_call_events(services):
    hub = MonitorHub()
    services.monitor = hub
    queue = hub.subscribe()

    transport = FakeTransport()
    session = CallSession(transport, services)
    task = asyncio.create_task(session.run())
    await asyncio.sleep(0.3)
    await session.hangup()
    await asyncio.wait_for(task, timeout=5)

    seen = []
    while not queue.empty():
        seen.append(queue.get_nowait())
    assert seen, "a watched call must produce a feed"
    assert all(m["call_id"] == session.call_id for m in seen)
    assert {"state", "speech"} <= {m["type"] for m in seen}
