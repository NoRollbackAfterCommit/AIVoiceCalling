"""Agent profiles that outlive the process.

A profile is behaviour as data — greeting, policies, voices, objective — and an
operator tunes it over days of pilot calls. Until now an edit through the API
lived in `services.profiles` and vanished on restart. Saved profiles are loaded
over the built-in default at boot, the same way persisted admin settings are
layered over the environment.
"""

from __future__ import annotations

import json
import time
from typing import Any

from sqlalchemy import delete, select

from vaani.agent.prompt import AgentProfile, profile_from_dict, profile_to_dict
from vaani.core.logging import get_logger
from vaani.db.models import AgentProfileRow

log = get_logger(__name__)


class ProfileRepository:
    def __init__(self, sessions: Any) -> None:
        # Shares the call repository's engine: one database, one connection
        # pool, one set of migrations.
        self._sessions = sessions

    async def save(self, profile: AgentProfile) -> None:
        async with self._sessions() as session, session.begin():
            await session.merge(
                AgentProfileRow(
                    key=profile.key,
                    payload=json.dumps(profile_to_dict(profile)),
                    updated_at=time.time(),
                )
            )

    async def delete(self, key: str) -> None:
        async with self._sessions() as session, session.begin():
            await session.execute(delete(AgentProfileRow).where(AgentProfileRow.key == key))

    async def load(self) -> dict[str, AgentProfile]:
        async with self._sessions() as session:
            rows = (await session.execute(select(AgentProfileRow))).scalars().all()
        profiles: dict[str, AgentProfile] = {}
        for row in rows:
            try:
                # The row key is the identity; the payload's copy is whatever
                # the writer thought it was.
                profiles[row.key] = profile_from_dict({**json.loads(row.payload), "key": row.key})
            except Exception:
                # One corrupt row must not stop the phone being answered.
                log.exception("skipping unreadable agent profile", extra={"agent": row.key})
        return profiles
