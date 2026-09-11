"""The dialplan's side channel: who is calling, before the audio socket opens."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from vaani.api.limits import enforce_body_limit
from vaani.core.logging import get_logger

log = get_logger(__name__)

router = APIRouter()


@router.post("/telephony/announce", tags=["telephony"])
async def announce_call(request: Request) -> dict[str, Any]:
    """Register caller and agent for a call AudioSocket is about to hand over.

    Asterisk's CURL() sends a url-encoded form; anything else integrating here
    sends JSON. Both are accepted so the dialplan stays a one-liner.
    """
    # Before the body is read, not after: this path is open by design, so an
    # unbounded read here is an unauthenticated way to exhaust the host's memory.
    enforce_body_limit(request)
    fields = await _fields(request)
    try:
        canonical = str(uuid.UUID(str(fields.get("uuid") or "").strip()))
    except ValueError:
        raise HTTPException(422, "uuid must be the UUID later passed to AudioSocket()") from None

    request.app.state.announcements.announce(
        canonical,
        caller_number=_number(fields.get("caller")),
        agent_key=_text(fields.get("agent")),
        dialled_number=_number(fields.get("did")),
    )
    return {"status": "announced", "uuid": canonical}


async def _fields(request: Request) -> dict[str, Any]:
    """Caller beware: this reads the body, so `enforce_body_limit` comes first.

    Neither branch may raise. This path is open by design, so a body that is not
    what its content-type claims escaped as a 500 and a logged traceback for any
    stranger who asked. A body we cannot read simply carries no fields, which
    leaves the missing uuid on the route's existing 422 — the refusal the
    dialplan already reads.
    """
    if "application/json" in request.headers.get("content-type", ""):
        try:
            body = await request.json()
        except Exception:
            body = None
        return body if isinstance(body, dict) else {}
    try:
        return dict(await request.form())
    except Exception:
        # python-multipart raises on a body that is not the form it claims to be.
        log.warning("an announce body could not be parsed as a form")
        return {}


def _text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _number(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    # A raw "+91…" that skipped URIENCODE() decodes with a space where the plus
    # was. Nothing else legitimately starts a number with a space.
    if text.startswith(" "):
        text = "+" + text[1:]
    text = text.strip()
    # Carriers flag withheld numbers in prose ("anonymous", "restricted"). The
    # record's caller column means "a number we could ring back", or nothing.
    if not any(ch.isdigit() for ch in text):
        return None
    # The column is 32 wide; anything longer is not a phone number.
    return text[:32]
