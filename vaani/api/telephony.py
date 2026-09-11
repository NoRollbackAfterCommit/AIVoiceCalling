"""The dialplan's side channel: who is calling, before the audio socket opens."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from vaani.api.limits import enforce_body_limit

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
    if "application/json" in request.headers.get("content-type", ""):
        body = await request.json()
        return body if isinstance(body, dict) else {}
    return dict(await request.form())


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
