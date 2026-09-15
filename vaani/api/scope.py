"""Which organisation's data the caller is allowed to touch.

One helper, used by every endpoint that reads or writes something belonging to
an organisation. The rule it enforces is short and is the whole security
property of the tenancy model:

    Scope comes from who you are, never from what you asked for.

An `organisation_id` in a query string is a *request*, and for anyone below a
platform administrator it is ignored entirely. Endpoints that accept one are
letting the platform administrator narrow a view, not letting a supervisor
widen one.

Refusals are 404, not 403. Telling a supervisor at Alpha that Beta's call
exists but is forbidden is itself a disclosure — that Beta is a customer, how
many calls they take, and which ids are real. Not found is the truthful answer
to "is there such a call *for you*".
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request


def caller(request: Request) -> Any:
    """The signed-in user, or None for the shared token and open deployments."""
    return getattr(getattr(request, "state", None), "user", None)


def is_unrestricted(request: Request) -> bool:
    """True for a platform administrator and for the shared machine token.

    The token has no account and therefore no organisation; it is how the deploy
    script, the health probe and the carrier harness reach the API, and they
    have no narrower identity to be given.
    """
    user = caller(request)
    return user is None or user.is_platform_admin


def visible_organisation(request: Request, requested: int | None = None) -> int | None:
    """The organisation to filter by, or None for "all of them".

    `requested` is honoured only for an unrestricted caller. For everyone else
    the answer is their own organisation whatever they asked for, which is what
    makes a crafted query parameter useless rather than dangerous.
    """
    if is_unrestricted(request):
        return requested
    return caller(request).organisation_id


def ensure_visible(request: Request, organisation_id: int | None, what: str = "record") -> None:
    """Refuse a record belonging to somebody else.

    Unattributed records — a call to a number nobody has mapped — belong to the
    platform rather than to a customer, so only an unrestricted caller sees
    them. A customer inheriting whatever arrived unrouted would be the same bug
    in a friendlier costume.
    """
    if is_unrestricted(request):
        return
    if organisation_id is None or organisation_id != caller(request).organisation_id:
        raise HTTPException(404, f"No such {what}")
