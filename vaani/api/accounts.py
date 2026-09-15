"""Signing in, and who the caller is.

The shared bearer token stays: the deploy script, the health probe and the
carrier harness have no browser to log in with, and they are treated as a
platform administrator. What changes is that a *person* gets an account with a
role and an organisation instead of the master key.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from vaani.core.logging import get_logger
from vaani.db.accounts import Role, User

log = get_logger(__name__)
router = APIRouter()

COOKIE = "vaani_session"

# One wording for every failure. "No such account" and "wrong password" as
# separate messages turn the login form into a list of who works here.
_REFUSED = "Email or password is incorrect"


class LoginIn(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=512)


def _accounts(request: Request) -> Any:
    store = getattr(request.app.state.services, "accounts", None)
    if store is None:
        raise HTTPException(503, "No database is configured, so accounts cannot be stored")
    return store


@router.post("/auth/login", tags=["auth"])
async def login(body: LoginIn, request: Request, response: Response) -> dict[str, Any]:
    user = await _accounts(request).authenticate(body.email, body.password)
    if user is None:
        log.warning("failed sign-in", extra={"email": body.email[:64]})
        raise HTTPException(401, _REFUSED)

    signer = request.app.state.session_signer
    response.set_cookie(
        COOKIE,
        signer.issue(user.id),
        httponly=True,  # an XSS in the console must not be able to lift it
        samesite="lax",
        secure=_https_only(request),
        max_age=signer_ttl(signer),
        path="/",
    )
    log.info("signed in", extra={"email": user.email, "role": user.role})
    return _public(user)


@router.post("/auth/logout", tags=["auth"])
async def logout(response: Response) -> dict[str, str]:
    response.delete_cookie(COOKIE, path="/")
    return {"status": "signed out"}


@router.get("/auth/me", tags=["auth"])
async def me(request: Request) -> dict[str, Any]:
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(401, "Not signed in")
    return _public(user)


def signer_ttl(signer: Any) -> int:
    return int(getattr(signer, "_ttl", 12 * 3600))


def _https_only(request: Request) -> bool:
    """Marking the cookie Secure over plain HTTP would stop it being stored at
    all, which breaks the laptop demo. Behind Caddy the scheme is https and the
    flag is set."""
    forwarded = request.headers.get("x-forwarded-proto", "")
    return request.url.scheme == "https" or forwarded.split(",")[0].strip() == "https"


def _public(user: User) -> dict[str, Any]:
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "role": user.role,
        "organisation_id": user.organisation_id,
    }


# ---------------------------------------------------------------------------
# What each role may do
# ---------------------------------------------------------------------------

# Deliberately a table rather than a decorator per endpoint: a permission model
# spread across fifty call sites is one nobody can audit, and the question an
# auditor asks is "what can a supervisor do", not "what does this endpoint
# allow". Ordered most privileged first.
_MAY_ADMINISTER_DEPLOYMENT = (Role.PLATFORM_ADMIN,)
_MAY_ADMINISTER_TENANTS = (Role.PLATFORM_ADMIN,)
_MAY_CONFIGURE_ORGANISATION = (Role.PLATFORM_ADMIN, Role.ORG_ADMIN)
_MAY_OPERATE_CALLS = (Role.PLATFORM_ADMIN, Role.ORG_ADMIN, Role.SUPERVISOR)
_MAY_READ = Role.ALL


def required_roles(method: str, path: str) -> tuple[str, ...]:
    """Which roles may make this request.

    Matched on the route path rather than declared per handler so that a new
    endpoint is covered by the nearest prefix instead of defaulting to open.
    """
    write = method not in ("GET", "HEAD", "OPTIONS")

    if path.startswith("/api/settings"):
        # Provider keys, the concurrency cap, the webhook secret.
        return _MAY_ADMINISTER_DEPLOYMENT if write else _MAY_CONFIGURE_ORGANISATION
    if path.startswith("/api/organisations"):
        # Creating a customer, or suspending one, is the platform's business.
        return _MAY_ADMINISTER_TENANTS if write else _MAY_READ
    if path.startswith("/api/dids"):
        return _MAY_ADMINISTER_TENANTS if write else _MAY_READ
    if path.startswith("/api/users"):
        return _MAY_CONFIGURE_ORGANISATION
    if path.startswith("/api/agents") or path.startswith("/api/knowledge"):
        return _MAY_CONFIGURE_ORGANISATION if write else _MAY_READ
    if path.startswith("/api/calls"):
        # Hanging up a live call is an operator action, not a configuration one.
        return _MAY_OPERATE_CALLS if write else _MAY_READ
    return _MAY_READ
