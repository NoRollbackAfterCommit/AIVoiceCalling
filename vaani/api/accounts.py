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
# Managing accounts
# ---------------------------------------------------------------------------


class UserIn(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=512)
    role: str
    name: str = ""
    # Honoured only for a platform administrator. An org admin naming somebody
    # else's organisation gets their own, which is what makes this field useless
    # as a way to plant a user in another customer's account.
    organisation_id: int | None = None


class UserPatch(BaseModel):
    active: bool | None = None
    role: str | None = None
    password: str | None = Field(default=None, min_length=1, max_length=512)
    name: str | None = Field(default=None, max_length=120)


def _unrestricted(request: Request) -> bool:
    user = getattr(request.state, "user", None)
    return user is None or user.is_platform_admin


def _own_organisation(request: Request) -> int | None:
    user = getattr(request.state, "user", None)
    return None if user is None else user.organisation_id


async def _visible_user(request: Request, user_id: int) -> Any:
    """The account, or 404 when it belongs to another organisation.

    404 rather than 403 for the same reason as everywhere else: a 403 confirms
    the account exists, which is half of what an attacker wanted.
    """
    target = await _accounts(request).get(user_id)
    if target is None:
        raise HTTPException(404, f"No user {user_id}")
    if not _unrestricted(request) and target.organisation_id != _own_organisation(request):
        raise HTTPException(404, f"No user {user_id}")
    return target


@router.get("/users", tags=["users"])
async def list_users(request: Request) -> list[dict[str, Any]]:
    mine = None if _unrestricted(request) else _own_organisation(request)
    users = await _accounts(request).users(mine)
    return [_public(u) | {"active": u.active} for u in users]


@router.post("/users", tags=["users"], status_code=201)
async def create_user(body: UserIn, request: Request) -> dict[str, Any]:
    accounts = _accounts(request)

    if _unrestricted(request):
        organisation_id = body.organisation_id
        role = body.role
    else:
        # An organisation administrator is trusted inside their organisation and
        # nowhere else. Both of these are the escalation this endpoint exists to
        # refuse: awarding the whole deployment, and planting a user elsewhere.
        if body.role == Role.PLATFORM_ADMIN:
            raise HTTPException(403, "Only a platform administrator can create another")
        organisation_id = _own_organisation(request)
        role = body.role

    try:
        user = await accounts.create_user(
            email=body.email,
            password=body.password,
            role=role,
            organisation_id=organisation_id,
            name=body.name,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _public(user) | {"active": user.active}


@router.patch("/users/{user_id}", tags=["users"])
async def update_user(user_id: int, body: UserPatch, request: Request) -> dict[str, Any]:
    accounts = _accounts(request)
    target = await _visible_user(request, user_id)

    if body.role is not None and body.role != target.role:
        if not _unrestricted(request):
            # Including promoting yourself, which is the same request.
            raise HTTPException(403, "Only a platform administrator can change a role")
        if body.role not in Role.ALL:
            raise HTTPException(400, f"Unknown role {body.role!r}")
        await accounts.set_role(user_id, body.role)

    if body.active is not None and body.active != target.active:
        if not body.active and await _is_last_platform_admin(accounts, target):
            # There would be no way back in without shell access to the box.
            raise HTTPException(400, "This is the last platform administrator")
        await accounts.set_active(user_id, body.active)

    if body.password is not None:
        try:
            await accounts.set_password(user_id, body.password)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        log.info("password changed", extra={"email": target.email})

    if body.name is not None:
        await accounts.set_name(user_id, body.name)

    updated = await accounts.get(user_id)
    return _public(updated) | {"active": updated.active}


async def _is_last_platform_admin(accounts: Any, target: Any) -> bool:
    if target.role != Role.PLATFORM_ADMIN:
        return False
    others = [
        u
        for u in await accounts.users()
        if u.role == Role.PLATFORM_ADMIN and u.active and u.id != target.id
    ]
    return not others


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
