"""Training documents that outlive the vector index.

The index is derived data: vectors depend on the embedder and live in a store
that can be lost — the hosted Qdrant cluster this pilot used was suspended and
took every document with it. The text an operator uploaded lives here, in the
same database as the calls, so the index can always be rebuilt from it.

A delete leaves a tombstone rather than removing the row. An index can hold a
stale copy of a document — the store was down at delete time, or an older
collection is read again after an embedder is switched back — and without the
tombstone the next reconciliation would adopt that copy as a document nobody
had ever seen, bringing a withdrawn circular back into callers' answers.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select

from vaani.db.models import KnowledgeDocumentRow


def doc_hash(text: str) -> str:
    """A short version stamp, so a sync can tell a stale index from a current one."""
    return hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


@dataclass(slots=True)
class KnowledgeDocument:
    namespace: str
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class KnowledgeRepository:
    def __init__(self, sessions: Any) -> None:
        # Shares the call repository's engine: one database, one backup.
        self._sessions = sessions
        # Serialises every change to a document against reconciliation. Lives
        # here, not on the retriever, because a settings save replaces the
        # retriever while the old one may still be finishing an upload.
        self.lock = asyncio.Lock()

    async def _row(self, session: Any, namespace: str, source: str) -> Any:
        return (
            await session.execute(
                select(KnowledgeDocumentRow).where(
                    KnowledgeDocumentRow.namespace == namespace,
                    KnowledgeDocumentRow.source == source,
                )
            )
        ).scalar_one_or_none()

    async def save(self, doc: KnowledgeDocument) -> None:
        """Insert or replace by (namespace, source) — uploading a document
        again means it was corrected, and it lifts any earlier delete."""
        async with self._sessions() as session, session.begin():
            row = await self._row(session, doc.namespace, doc.source)
            if row is None:
                row = KnowledgeDocumentRow(namespace=doc.namespace, source=doc.source)
                session.add(row)
            row.text = doc.text
            row.doc_hash = doc_hash(doc.text)
            row.metadata_json = json.dumps(doc.metadata or {})
            row.deleted = False
            row.updated_at = time.time()

    async def adopt(self, doc: KnowledgeDocument) -> bool:
        """Save a document recovered from the index, unless one by that name
        is already known — live or deleted. Never overwrites."""
        async with self._sessions() as session, session.begin():
            if await self._row(session, doc.namespace, doc.source) is not None:
                return False
            session.add(
                KnowledgeDocumentRow(
                    namespace=doc.namespace,
                    source=doc.source,
                    text=doc.text,
                    doc_hash=doc_hash(doc.text),
                    metadata_json=json.dumps(doc.metadata or {}),
                    deleted=False,
                    updated_at=time.time(),
                )
            )
        return True

    async def delete(self, namespace: str, source: str) -> bool:
        async with self._sessions() as session, session.begin():
            row = await self._row(session, namespace, source)
            if row is None or row.deleted:
                return False
            # The text goes; the name stays, as the tombstone.
            row.deleted = True
            row.text = ""
            row.doc_hash = ""
            row.updated_at = time.time()
        return True

    async def get(self, namespace: str, source: str) -> KnowledgeDocument | None:
        async with self._sessions() as session:
            row = await self._row(session, namespace, source)
        if row is None or row.deleted:
            return None
        return KnowledgeDocument(
            namespace=row.namespace,
            source=row.source,
            text=row.text,
            metadata=json.loads(row.metadata_json or "{}"),
        )

    async def versions(self) -> dict[tuple[str, str], str | None]:
        """Every known (namespace, source) → its version, None if deleted.

        Names and hashes only: this runs every few minutes on the event loop
        that is carrying live calls, so it must not load the documents' text.
        """
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    select(
                        KnowledgeDocumentRow.namespace,
                        KnowledgeDocumentRow.source,
                        KnowledgeDocumentRow.doc_hash,
                        KnowledgeDocumentRow.deleted,
                    )
                )
            ).all()
        return {(ns, src): (None if deleted else h) for ns, src, h, deleted in rows}

    async def all(self) -> list[KnowledgeDocument]:
        async with self._sessions() as session:
            rows = (
                (
                    await session.execute(
                        select(KnowledgeDocumentRow).where(KnowledgeDocumentRow.deleted.is_(False))
                    )
                )
                .scalars()
                .all()
            )
        return [
            KnowledgeDocument(
                namespace=r.namespace,
                source=r.source,
                text=r.text,
                metadata=json.loads(r.metadata_json or "{}"),
            )
            for r in rows
        ]
