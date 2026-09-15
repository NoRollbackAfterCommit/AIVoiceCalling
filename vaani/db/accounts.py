"""Who may sign in, and what they are allowed to see.

One deployment carries several customers' call centres, so "authenticated" is
not a permission. Every account names a role and — unless it is a platform
administrator — exactly one organisation, and that pair is the only thing any
scope filter is ever allowed to read from.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select, update

from vaani.core.logging import get_logger
from vaani.db.models import UserRow
from vaani.security.passwords import MIN_LENGTH, hash_password, needs_rehash, verify_password

log = get_logger(__name__)


class Role:
    """Four roles, ordered by what they may change rather than what they may see."""

    PLATFORM_ADMIN = "platform_admin"  # every organisation, plus deployment settings
    ORG_ADMIN = "org_admin"  # their organisation: agents, knowledge, numbers, users
    SUPERVISOR = "supervisor"  # their organisation's live calls and reporting
    VIEWER = "viewer"  # their organisation's reporting, read only

    ALL = (PLATFORM_ADMIN, ORG_ADMIN, SUPERVISOR, VIEWER)


@dataclass(slots=True)
class User:
    id: int
    email: str
    name: str
    role: str
    organisation_id: int | None
    active: bool

    @property
    def is_platform_admin(self) -> bool:
        return self.role == Role.PLATFORM_ADMIN


class AccountRepository:
    def __init__(self, sessions: Any) -> None:
        self._sessions = sessions

    # -- creation ------------------------------------------------------------

    async def create_user(
        self,
        *,
        email: str,
        password: str,
        role: str,
        organisation_id: int | None,
        name: str = "",
    ) -> User:
        email = _canonical(email)
        if not email or "@" not in email:
            raise ValueError("a user needs an email address")
        if role not in Role.ALL:
            raise ValueError(f"unknown role {role!r}")
        if len(password) < MIN_LENGTH:
            raise ValueError(f"the password must be at least {MIN_LENGTH} characters")
        if role != Role.PLATFORM_ADMIN and organisation_id is None:
            # Otherwise the scope filter matches nothing, or everything, and
            # which one depends on how each endpoint happened to write it.
            raise ValueError(f"the {role} role must belong to an organisation")
        if role == Role.PLATFORM_ADMIN and organisation_id is not None:
            raise ValueError("a platform administrator belongs to no single organisation")

        # Hashing is deliberately expensive, and this runs on the event loop the
        # live calls are sharing.
        digest = await asyncio.to_thread(hash_password, password)

        async with self._sessions() as session, session.begin():
            clash = await session.scalar(select(UserRow).where(UserRow.email == email))
            if clash is not None:
                raise ValueError(f"a user with the email {email} already exists")
            row = UserRow(
                email=email,
                password_hash=digest,
                name=name.strip(),
                role=role,
                organisation_id=organisation_id,
                active=True,
                created_at=time.time(),
            )
            session.add(row)
            await session.flush()
            user = _user(row)
        log.info("user created", extra={"email": email, "role": role})
        return user

    async def ensure_first_admin(self, email: str, password: str) -> User | None:
        """Create the bootstrap administrator, but only on an empty table.

        Returns None when accounts already exist. Bootstrapping must not double
        as a way to mint an administrator on a live deployment.
        """
        async with self._sessions() as session:
            existing = await session.scalar(select(func.count()).select_from(UserRow))
        if existing:
            return None
        return await self.create_user(
            email=email,
            password=password,
            role=Role.PLATFORM_ADMIN,
            organisation_id=None,
            name="Platform administrator",
        )

    # -- authentication ------------------------------------------------------

    async def authenticate(self, email: str, password: str) -> User | None:
        """The user, or None for every kind of failure.

        One answer for "no such account", "wrong password" and "suspended": a
        login form that distinguishes them tells an attacker which addresses are
        worth attacking.
        """
        email = _canonical(email)
        async with self._sessions() as session:
            row = await session.scalar(select(UserRow).where(UserRow.email == email))

        if row is None:
            # Still spend the time: returning instantly for an unknown address
            # turns the login form into an account-enumeration oracle.
            await asyncio.to_thread(verify_password, password, _DUMMY_HASH)
            return None

        stored = row.password_hash
        if not await asyncio.to_thread(verify_password, password, stored):
            return None
        if not row.active:
            return None

        await self._note_login(row.id)
        if needs_rehash(stored):
            # The only moment the plaintext is available to upgrade the row.
            await self.set_password(row.id, password)
        return _user(row)

    async def _note_login(self, user_id: int) -> None:
        async with self._sessions() as session, session.begin():
            await session.execute(
                update(UserRow).where(UserRow.id == user_id).values(last_login_at=time.time())
            )

    # -- administration ------------------------------------------------------

    async def set_password(self, user_id: int, password: str) -> None:
        if len(password) < MIN_LENGTH:
            raise ValueError(f"the password must be at least {MIN_LENGTH} characters")
        digest = await asyncio.to_thread(hash_password, password)
        async with self._sessions() as session, session.begin():
            await session.execute(
                update(UserRow).where(UserRow.id == user_id).values(password_hash=digest)
            )

    async def set_active(self, user_id: int, active: bool) -> None:
        """Revoking access without deleting the person: the changes and calls
        attributed to them keep their owner."""
        async with self._sessions() as session, session.begin():
            await session.execute(
                update(UserRow).where(UserRow.id == user_id).values(active=active)
            )

    async def set_role(self, user_id: int, role: str) -> None:
        if role not in Role.ALL:
            raise ValueError(f"unknown role {role!r}")
        async with self._sessions() as session, session.begin():
            await session.execute(update(UserRow).where(UserRow.id == user_id).values(role=role))

    async def set_name(self, user_id: int, name: str) -> None:
        async with self._sessions() as session, session.begin():
            await session.execute(
                update(UserRow).where(UserRow.id == user_id).values(name=name.strip())
            )

    async def get(self, user_id: int) -> User | None:
        async with self._sessions() as session:
            row = await session.get(UserRow, user_id)
            return _user(row) if row else None

    async def users(self, organisation_id: int | None = None) -> list[User]:
        async with self._sessions() as session:
            stmt = select(UserRow).order_by(UserRow.email)
            if organisation_id is not None:
                stmt = stmt.where(UserRow.organisation_id == organisation_id)
            return [_user(r) for r in (await session.scalars(stmt)).all()]

    async def count(self) -> int:
        async with self._sessions() as session:
            return int(await session.scalar(select(func.count()).select_from(UserRow)) or 0)


def _canonical(email: str) -> str:
    return (email or "").strip().lower()


def _user(row: UserRow) -> User:
    return User(
        id=row.id,
        email=row.email,
        name=row.name or "",
        role=row.role,
        organisation_id=row.organisation_id,
        active=bool(row.active),
    )


# Verified against when the address is unknown, so the response takes the same
# time either way. Its plaintext is never a valid password for any account.
_DUMMY_HASH = hash_password("::no-such-account::")
