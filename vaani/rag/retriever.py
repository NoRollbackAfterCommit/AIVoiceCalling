"""Retrieval and ingestion, the two halves of the RAG pipeline.

Retrieval here is hybrid: dense vectors find semantic matches, a lexical pass
rescues the exact-token queries dense retrieval is famously bad at — scheme
names, form numbers, section references, the things citizens actually ask about
by name. Scores are blended, then deduplicated by source so one verbose circular
cannot crowd out every other document.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from pathlib import Path
from typing import Any

from vaani.core.logging import get_logger
from vaani.db.knowledge import doc_hash
from vaani.providers.base import EmbeddingProvider
from vaani.rag.chunking import Chunk, chunk_text, load_file
from vaani.rag.store import DimensionMismatch, SearchHit, VectorStore

log = get_logger(__name__)

_TOKEN = re.compile(r"\w+", re.UNICODE)
_STOP = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "of",
    "to",
    "for",
    "in",
    "on",
    "and",
    "or",
    "my",
    "i",
    "you",
    "it",
    "what",
    "how",
    "when",
    "where",
    "can",
    "do",
    "does",
    "please",
    "tell",
    "me",
    "about",
}


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingProvider,
        *,
        top_k: int = 5,
        min_score: float = 0.25,
        lexical_weight: float = 0.3,
        max_per_source: int = 2,
        documents: Any = None,
    ) -> None:
        self._store = store
        # The system of record for what was uploaded (a KnowledgeRepository).
        # Optional so a bare install still answers from an in-memory index; with
        # it, the index is a cache that `sync()` can always rebuild.
        self._documents = documents
        self._embedder = embedder
        self._top_k = top_k
        # The configured threshold is an upper bound; a provider whose scores
        # live on a lower scale pulls it down to its own floor, so switching
        # embedders never silently empties the result set.
        self._min_score = min(min_score, getattr(embedder, "similarity_floor", min_score))
        self._lexical_weight = lexical_weight
        self._max_per_source = max_per_source
        self._ready = False
        self._embedder_started = False

    async def start(self) -> None:
        # Once only: start() is retried by the healer while the store is down,
        # and an embedder that opens a client would open one per attempt.
        if not self._embedder_started:
            await self._embedder.start()
            self._embedder_started = True
        dim = getattr(self._embedder, "dim", 0) or 384
        try:
            await self._store.ensure(dim)
        except DimensionMismatch:
            if self._documents is None:
                raise
            await self._rebuild(dim)
        # A provider that only knows its dimension after loading (any real
        # transformer) also settles its floor here.
        self._min_score = min(
            self._min_score, getattr(self._embedder, "similarity_floor", self._min_score)
        )
        self._ready = True
        log.info(
            "retriever ready",
            extra={"embedder": self._embedder.name, "dim": dim, "min_score": self._min_score},
        )
        if self._documents is not None:
            await self.sync()

    def use_documents(self, documents: Any) -> None:
        """Attach the database's document store. Set before start(), which is
        when the index is first reconciled with it."""
        self._documents = documents

    @property
    def documents(self) -> Any:
        return self._documents

    @property
    def ready(self) -> bool:
        return self._ready

    # -- durability ---------------------------------------------------------

    def _guard(self) -> Any:
        """The lock every document change and reconciliation step takes."""
        return self._documents.lock if self._documents is not None else contextlib.nullcontext()

    async def sync(self) -> dict[str, int]:
        """Make the index match the documents in the database.

        Run at boot, after the store or embedder changes, and periodically, so
        an index that was lost — a hosted cluster suspended, a fresh empty one
        put in its place — fills itself back in without anyone re-uploading.
        Anything indexed that the database has never seen (an index from before
        the database kept documents) is adopted first rather than dropped, and
        anything the database says was deleted is removed.

        Each step re-reads its document under the lock, because a pass over a
        lost index re-embeds everything and can take minutes: a delete or an
        upload landing mid-pass must win over the snapshot this pass started
        from, or a withdrawn document gets indexed again.
        """
        if self._documents is None:
            return {"adopted": 0, "indexed": 0, "removed": 0}
        adopted = await self.backfill()
        indexed = removed = 0
        present = await self._store.inventory()
        for (ns, source), version in (await self._documents.versions()).items():
            found = present.get((ns, source))
            if version is not None and found == {version}:
                continue
            if version is None and not found:
                continue
            async with self._guard():
                doc = await self._documents.get(ns, source)
                if doc is None:
                    if found:
                        await self._store.delete_source(source, namespace=ns)
                        removed += 1
                    continue
                chunks = chunk_text(doc.text, source=doc.source, metadata=doc.metadata)
                indexed += await self._index(chunks, ns, doc_hash(doc.text))
        if adopted or indexed or removed:
            log.info(
                "knowledge synced",
                extra={"adopted": adopted, "indexed": indexed, "removed": removed},
            )
        return {"adopted": adopted, "indexed": indexed, "removed": removed}

    async def backfill(self) -> int:
        """Copy into the database any document that exists only in the index.

        Reassembled from its chunks in order. Chunks overlap by a sentence or
        so, so the text repeats a little at each seam; that costs nothing in
        retrieval and is far better than the document being lost. Insert-only:
        a name the database already knows — live, or deleted — is never
        overwritten from an index copy.
        """
        if self._documents is None:
            return 0
        from vaani.db.knowledge import KnowledgeDocument

        known = await self._documents.versions()
        adopted = 0
        for namespace, source in await self._store.inventory():
            if (namespace, source) in known:
                continue
            chunks = await self._store.export(namespace, source)
            if not chunks:
                continue
            meta = {k: v for k, v in chunks[0].metadata.items() if k != "doc_hash"}
            async with self._guard():
                saved = await self._documents.adopt(
                    KnowledgeDocument(
                        namespace=namespace,
                        source=source,
                        text="\n\n".join(c.text for c in chunks),
                        metadata={**meta, "recovered_from_index": True},
                    )
                )
            if saved:
                adopted += 1
                log.warning(
                    "adopted a document found only in the index",
                    extra={"source": source, "scope": namespace, "chunks": len(chunks)},
                )
        return adopted

    async def _rebuild(self, dim: int) -> None:
        """Move the index to a collection of the new width, filled from the
        database by the sync that follows.

        The embedder changed, so no vector can be reused. Anything only the old
        collection holds is copied into the database first. The old collection
        is left exactly as it was: the retriever being replaced may still be
        serving calls from it.
        """
        adopted = await self.backfill()
        log.warning(
            "embedder changed; building a knowledge index at the new width",
            extra={"dim": dim, "adopted": adopted},
        )
        await self._store.switch_width(dim)

    # -- scopes -------------------------------------------------------------

    @staticmethod
    def namespace(agent_key: str = "default", organisation_id: int | None = None) -> str:
        """Which partition a document lives in.

        Two kinds, and a caller reads from both. An organisation's own set is
        shared by every line it runs — the fee schedule uploaded once for a
        university — and a line's set is its alone, so an admissions caller is
        never read the examination timetable.

        `organisation_id` wins when given, because passing one is how a caller
        says "the shared set" rather than a line's. The prefix cannot collide
        with a line: an agent key is validated to letters, digits, hyphen and
        underscore, so none of them can contain a colon.
        """
        return f"org:{organisation_id}" if organisation_id is not None else agent_key

    # -- write --------------------------------------------------------------

    async def index_chunks(
        self,
        chunks: list[Chunk],
        *,
        agent_key: str = "default",
        organisation_id: int | None = None,
    ) -> int:
        if not chunks:
            return 0
        return await self._index(chunks, self.namespace(agent_key, organisation_id))

    async def _index(self, chunks: list[Chunk], ns: str, doc_hash: str | None = None) -> int:
        if doc_hash is not None:
            chunks = [
                Chunk(c.text, c.source, c.ordinal, {**c.metadata, "doc_hash": doc_hash})
                for c in chunks
            ]
        # A source name is the document, so indexing it again means it was
        # corrected. Neither store replaces on its own — the memory one extends
        # its bucket and Qdrant mints a fresh id per chunk — so without this the
        # superseded text stays searchable beside its replacement and the bot
        # can quote last year's circular. Cleared once up front rather than per
        # batch, or the second batch would delete the first.
        replaced = 0
        for source in dict.fromkeys(c.source for c in chunks):
            replaced += await self._store.delete_source(source, namespace=ns)

        # Embed in batches: a 500-page circular in one call will OOM the GPU.
        total = 0
        for start in range(0, len(chunks), 64):
            batch = chunks[start : start + 64]
            vectors = await self._embedder.embed([c.text for c in batch])
            total += await self._store.upsert(batch, vectors, namespace=ns)
        log.info("indexed", extra={"chunks": total, "replaced": replaced, "scope": ns})
        return total

    async def index_text(
        self,
        text: str,
        source: str,
        *,
        agent_key: str = "default",
        organisation_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Keep the document, then index it.

        Saved first, so a store that is down loses nothing: the upload is
        recorded and the next sync indexes it. The error still reaches the
        caller, who should hear that it is not searchable yet.
        """
        ns = self.namespace(agent_key, organisation_id)
        async with self._guard():
            if self._documents is not None:
                from vaani.db.knowledge import KnowledgeDocument

                await self._documents.save(
                    KnowledgeDocument(
                        namespace=ns, source=source, text=text, metadata=metadata or {}
                    )
                )
            return await self._index(
                chunk_text(text, source=source, metadata=metadata), ns, doc_hash(text)
            )

    async def index_paths(
        self,
        paths: list[Path],
        *,
        agent_key: str = "default",
        organisation_id: int | None = None,
    ) -> int:
        total = 0
        for path in paths:
            try:
                text = await asyncio.to_thread(load_file, path)
            except Exception:
                log.exception("failed to ingest", extra={"path": str(path)})
                continue
            total += await self.index_text(
                text, path.name, agent_key=agent_key, organisation_id=organisation_id
            )
        return total

    async def delete_source(
        self,
        source: str,
        *,
        agent_key: str = "default",
        organisation_id: int | None = None,
    ) -> int:
        ns = self.namespace(agent_key, organisation_id)
        # The index first: if the store is down this raises and the database
        # still holds the document, so nothing comes back from the dead. The
        # other order would let the next sync re-adopt it from the index.
        async with self._guard():
            removed = await self._store.delete_source(source, namespace=ns)
            if self._documents is not None:
                await self._documents.delete(ns, source)
        return removed

    async def count(self, agent_key: str | None = None, organisation_id: int | None = None) -> int:
        if agent_key is None and organisation_id is None:
            return await self._store.count(None)
        return await self._store.count(self.namespace(agent_key or "default", organisation_id))

    async def sources(
        self, *, agent_key: str = "default", organisation_id: int | None = None
    ) -> list[tuple[str, int]]:
        return await self._store.sources(self.namespace(agent_key, organisation_id))

    # -- read ---------------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        agent_key: str = "default",
        organisation_id: int | None = None,
        top_k: int | None = None,
    ) -> list[SearchHit]:
        """What this caller's line is allowed to answer from.

        Two partitions, searched together and ranked as one: the line's own
        documents and the organisation's shared set. A line with no
        organisation — the fallback agent on an unmapped number — reads only
        its own, because an unconfigured number belongs to no customer and must
        not fall into one's documents.

        Two queries rather than one because the store partitions by a single
        namespace. The cost is one extra round trip on a call that already
        waits on an LLM, and the alternative is a scan across every customer's
        corpus with a filter afterwards.
        """
        if not self._ready or not query.strip():
            return []
        k = top_k or self._top_k
        scopes = [agent_key]
        if organisation_id is not None:
            scopes.append(self.namespace(organisation_id=organisation_id))

        # Over-fetch, then rerank and diversify down to k.
        vector = (await self._embedder.embed([query]))[0]
        hits: list[SearchHit] = []
        for scope in scopes:
            hits += await self._store.search(
                vector, namespace=scope, top_k=k * 3, min_score=self._min_score * 0.6
            )
        if not hits:
            return []
        reranked = self._rerank(query, hits)
        return self._diversify(reranked, k)

    def _rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        terms = {t for t in _TOKEN.findall(query.lower()) if t not in _STOP and len(t) > 2}
        if not terms:
            return sorted(hits, key=lambda h: h.score, reverse=True)
        for hit in hits:
            body = set(_TOKEN.findall(hit.text.lower()))
            overlap = len(terms & body) / len(terms)
            hit.score = round(
                (1 - self._lexical_weight) * hit.score + self._lexical_weight * overlap, 4
            )
        return sorted(hits, key=lambda h: h.score, reverse=True)

    def _diversify(self, hits: list[SearchHit], k: int) -> list[SearchHit]:
        seen: dict[str, int] = {}
        chosen: list[SearchHit] = []
        for hit in hits:
            if hit.score < self._min_score:
                continue
            count = seen.get(hit.source, 0)
            if count >= self._max_per_source:
                continue
            seen[hit.source] = count + 1
            chosen.append(hit)
            if len(chosen) >= k:
                break
        return chosen
