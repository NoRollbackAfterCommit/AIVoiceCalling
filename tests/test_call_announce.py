"""Asterisk tells Vaani who is calling before the audio socket opens.

AudioSocket carries a UUID and nothing else, so the caller's number and the
helpline they dialled have to arrive by a side channel. The dialplan posts them
under the same UUID it then passes to AudioSocket(); the bridge claims that
announcement when the socket connects.
"""

from __future__ import annotations

import uuid

import httpx

from vaani.telephony.announce import CallAnnouncements


def test_claim_returns_what_was_announced_exactly_once():
    reg = CallAnnouncements()
    call_uuid = str(uuid.uuid4())
    reg.announce(call_uuid, caller_number="+919876543210", agent_key="pension")

    first = reg.claim(call_uuid)
    assert first is not None
    assert first.caller_number == "+919876543210"
    assert first.agent_key == "pension"
    assert reg.claim(call_uuid) is None, "a claim is one-shot; a replayed UUID gets nothing"


def test_unknown_uuid_claims_nothing():
    assert CallAnnouncements().claim(str(uuid.uuid4())) is None


def test_expired_announcement_is_not_claimable():
    now = [1000.0]
    reg = CallAnnouncements(ttl_s=30, clock=lambda: now[0])
    call_uuid = str(uuid.uuid4())
    reg.announce(call_uuid, caller_number="+911234567890")

    now[0] += 31
    assert reg.claim(call_uuid) is None, "a dialplan that never connected must not leak an entry"


def test_uuid_spelling_does_not_matter():
    reg = CallAnnouncements()
    canonical = str(uuid.uuid4())
    reg.announce(canonical.upper().replace("-", ""), caller_number="+911111111111")

    claimed = reg.claim(canonical)
    assert claimed is not None and claimed.caller_number == "+911111111111"


def _app_with_registry():
    from vaani.main import create_app

    app = create_app()
    # The lifespan is what normally installs this; ASGITransport does not run
    # lifespans, so the test wires the one object the route needs.
    app.state.announcements = CallAnnouncements()
    return app


async def test_announce_endpoint_accepts_asterisks_form_post():
    app = _app_with_registry()
    call_uuid = str(uuid.uuid4())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Exactly what CURL() in the dialplan sends: url-encoded form fields.
        resp = await client.post(
            "/api/telephony/announce",
            data={
                "uuid": call_uuid,
                "caller": "+919876543210",
                "did": "1800123",
                "agent": "pension",
            },
        )
    assert resp.status_code == 200, resp.text

    claimed = app.state.announcements.claim(call_uuid)
    assert claimed is not None
    assert claimed.caller_number == "+919876543210"
    assert claimed.agent_key == "pension"
    assert claimed.dialled_number == "1800123"


async def test_announce_endpoint_restores_a_plus_lost_to_form_decoding():
    # A dialplan that forgets URIENCODE() sends "+91..." raw; form decoding turns
    # the plus into a space. The number is still recoverable, so recover it.
    app = _app_with_registry()
    call_uuid = str(uuid.uuid4())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/api/telephony/announce",
            content=f"uuid={call_uuid}&caller=+919876543210",
            headers={"content-type": "application/x-www-form-urlencoded"},
        )
    assert resp.status_code == 200, resp.text
    claimed = app.state.announcements.claim(call_uuid)
    assert claimed is not None and claimed.caller_number == "+919876543210"


async def test_announce_endpoint_treats_withheld_caller_as_unknown():
    app = _app_with_registry()
    call_uuid = str(uuid.uuid4())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/api/telephony/announce", data={"uuid": call_uuid, "caller": ""})
    assert resp.status_code == 200
    claimed = app.state.announcements.claim(call_uuid)
    assert claimed is not None and claimed.caller_number is None


async def test_an_announce_body_over_the_ceiling_is_refused_before_it_is_read(asgi_post):
    """This path is open by design — a token here would sit in the dialplan file
    in cleartext — so anything that can reach the LAN can post to it. Reading the
    body before deciding anything turned that into an unauthenticated way to make
    the process buffer whatever a stranger cared to send, on a host that also
    serves a government website.
    """
    oversized = 8 * 1024 * 1024  # far over the ceiling; small enough to survive a regression
    status, read = await asgi_post(
        _app_with_registry(),
        "/api/telephony/announce",
        headers={
            "content-type": "application/x-www-form-urlencoded",
            "content-length": str(oversized),
        },
        body_bytes=oversized,
    )
    assert status == 413
    assert read == 0, "the body was buffered before its declared size was checked"


async def test_announce_endpoint_rejects_a_malformed_uuid():
    app = _app_with_registry()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/api/telephony/announce", data={"uuid": "not-a-uuid"})
    assert resp.status_code == 422


async def test_announce_endpoint_refuses_a_malformed_json_body_with_the_same_422():
    """The dialplan reads this response. A body that is not the JSON it claims
    to be used to escape `request.json()` as a 500 and a logged traceback on an
    open path; it must land on the route's existing refusal instead."""
    app = _app_with_registry()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/api/telephony/announce",
            content=b'{"uuid": ',
            headers={"content-type": "application/json"},
        )
    assert resp.status_code == 422, "an unreadable body must refuse, not crash"


async def test_announce_endpoint_refuses_a_malformed_form_body_with_the_same_422():
    """python-multipart raises on a body that is not the form it claims to be.
    Same shape as the JSON branch, same refusal the dialplan already knows."""
    app = _app_with_registry()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/api/telephony/announce",
            content=b"--zz\r\ngarbage\r\n",
            headers={"content-type": "multipart/form-data; boundary=zz"},
        )
    assert resp.status_code == 422, "an unparseable form must refuse, not crash"


async def test_a_negative_content_length_does_not_sail_past_the_ceiling(asgi_post):
    """`int("-1")` parses, and `-1 > 65536` is False, so a negative declared
    length skipped the size check entirely and the unbounded read went ahead.
    Every server in front rejects one today; the guard must not rely on that.
    """
    oversized = 8 * 1024 * 1024
    status, read = await asgi_post(
        _app_with_registry(),
        "/api/telephony/announce",
        headers={
            "content-type": "application/x-www-form-urlencoded",
            "content-length": "-1",
        },
        body_bytes=oversized,
    )
    assert read == 0, "a body behind a negative Content-Length was read anyway"
    assert status == 411
