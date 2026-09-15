"""Organisations and the numbers that reach them.

One deployment runs several customers' call centres. Which one a caller reaches
is decided entirely by the number they dialled, so `resolve_did` is the hinge the
whole tenancy model turns on — a wrong answer here puts a citizen through to
another organisation's agent, trained on the wrong corpus.

Numbers are normalised to E.164 on the way in *and* on the way out, because the
two sides never agree otherwise: the carrier sends `" 918065605873"` and an
operator types `+91 80 6560 5873`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete, select, update

from vaani.core.logging import get_logger
from vaani.db.models import DidRow, OrganisationRow
from vaani.telephony.numbers import to_e164

log = get_logger(__name__)


@dataclass(slots=True)
class Organisation:
    id: int
    slug: str
    name: str
    active: bool


@dataclass(slots=True)
class Did:
    number: str
    organisation_id: int
    agent_key: str
    label: str | None
    active: bool


class TenancyRepository:
    def __init__(self, sessions: Any) -> None:
        # Shares the call repository's engine: one database, one connection
        # pool, one set of migrations.
        self._sessions = sessions

    # -- organisations -------------------------------------------------------

    async def create_organisation(self, *, slug: str, name: str) -> Organisation:
        slug = slug.strip().lower()
        if not slug or not name.strip():
            raise ValueError("an organisation needs a slug and a name")
        async with self._sessions() as session, session.begin():
            existing = await session.scalar(
                select(OrganisationRow).where(OrganisationRow.slug == slug)
            )
            if existing is not None:
                # Checked rather than left to the unique index, so the caller
                # gets a message naming the clash instead of a driver error.
                raise ValueError(f"an organisation with slug {slug!r} already exists")
            row = OrganisationRow(slug=slug, name=name.strip(), active=True, created_at=time.time())
            session.add(row)
            await session.flush()
            return _organisation(row)

    async def organisations(self, *, include_inactive: bool = True) -> list[Organisation]:
        async with self._sessions() as session:
            stmt = select(OrganisationRow).order_by(OrganisationRow.name)
            if not include_inactive:
                stmt = stmt.where(OrganisationRow.active.is_(True))
            return [_organisation(r) for r in (await session.scalars(stmt)).all()]

    async def organisation(self, organisation_id: int) -> Organisation | None:
        async with self._sessions() as session:
            row = await session.get(OrganisationRow, organisation_id)
            return _organisation(row) if row else None

    async def set_organisation_active(self, organisation_id: int, active: bool) -> None:
        async with self._sessions() as session, session.begin():
            await session.execute(
                update(OrganisationRow)
                .where(OrganisationRow.id == organisation_id)
                .values(active=active)
            )

    async def rename_organisation(self, organisation_id: int, name: str) -> None:
        if not name.strip():
            raise ValueError("an organisation needs a name")
        async with self._sessions() as session, session.begin():
            await session.execute(
                update(OrganisationRow)
                .where(OrganisationRow.id == organisation_id)
                .values(name=name.strip())
            )

    # -- numbers -------------------------------------------------------------

    async def upsert_did(
        self,
        *,
        number: str,
        organisation_id: int,
        agent_key: str,
        label: str | None = None,
        active: bool = True,
    ) -> Did:
        canonical = to_e164(number)
        if canonical is None:
            raise ValueError(f"{number!r} is not a usable phone number")
        async with self._sessions() as session, session.begin():
            row = DidRow(
                number=canonical,
                organisation_id=organisation_id,
                agent_key=agent_key,
                label=label,
                active=active,
                created_at=time.time(),
            )
            # merge, so moving a number between organisations replaces the
            # mapping rather than adding a second one.
            row = await session.merge(row)
            await session.flush()
            return _did(row)

    async def dids_for(self, organisation_id: int) -> list[Did]:
        async with self._sessions() as session:
            rows = await session.scalars(
                select(DidRow)
                .where(DidRow.organisation_id == organisation_id)
                .order_by(DidRow.number)
            )
            return [_did(r) for r in rows.all()]

    async def dids(self) -> list[Did]:
        async with self._sessions() as session:
            rows = await session.scalars(select(DidRow).order_by(DidRow.number))
            return [_did(r) for r in rows.all()]

    async def delete_did(self, number: str) -> None:
        canonical = to_e164(number)
        if canonical is None:
            return
        async with self._sessions() as session, session.begin():
            await session.execute(delete(DidRow).where(DidRow.number == canonical))

    async def resolve_did(self, number: str | None) -> tuple[int, str] | None:
        """`(organisation_id, agent_key)` for a dialled number, or None.

        None covers every reason a number might not route — unknown, withdrawn
        from service, or belonging to a suspended organisation — because the
        caller-facing behaviour is identical in all three: the fallback agent
        answers and the call is recorded unattributed. Distinguishing them here
        would only give the routing code three ways to do the same thing.
        """
        canonical = to_e164(number)
        if canonical is None:
            return None
        async with self._sessions() as session:
            row = await session.scalar(
                select(DidRow, OrganisationRow)
                .join(OrganisationRow, OrganisationRow.id == DidRow.organisation_id)
                .where(
                    DidRow.number == canonical,
                    DidRow.active.is_(True),
                    OrganisationRow.active.is_(True),
                )
            )
            if row is None:
                return None
            return row.organisation_id, row.agent_key


def _organisation(row: OrganisationRow) -> Organisation:
    return Organisation(id=row.id, slug=row.slug, name=row.name, active=bool(row.active))


def _did(row: DidRow) -> Did:
    return Did(
        number=row.number,
        organisation_id=row.organisation_id,
        agent_key=row.agent_key,
        label=row.label,
        active=bool(row.active),
    )
