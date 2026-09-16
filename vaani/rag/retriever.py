"""Retrieval and ingestion, the two halves of the RAG pipeline.

Retrieval here is hybrid: dense vectors find semantic matches, a lexical pass
rescues the exact-token queries dense retrieval is famously bad at — scheme
names, form numbers, section references, the things citizens actually ask about
by name. Scores are blended, then deduplicated by source so one verbose circular
cannot crowd out every other document.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from vaani.core.logging import get_logger
from vaani.providers.base import EmbeddingProvider
from vaani.rag.chunking import Chunk, chunk_files, chunk_text
from vaani.rag.store import SearchHit, VectorStore

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
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._top_k = top_k
        # The configured threshold is an upper bound; a provider whose scores
        # live on a lower scale pulls it down to its own floor, so switching
        # embedders never silently empties the result set.
        self._min_score = min(min_score, getattr(embedder, "similarity_floor", min_score))
        self._lexical_weight = lexical_weight
        self._max_per_source = max_per_source
        self._ready = False

    async def start(self) -> None:
        await self._embedder.start()
        dim = getattr(self._embedder, "dim", 0) or 384
        await self._store.ensure(dim)
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
        ns = self.namespace(agent_key, organisation_id)

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
        return await self.index_chunks(
            chunk_text(text, source=source, metadata=metadata),
            agent_key=agent_key,
            organisation_id=organisation_id,
        )

    async def index_paths(
        self,
        paths: list[Path],
        *,
        agent_key: str = "default",
        organisation_id: int | None = None,
    ) -> int:
        chunks = await asyncio.to_thread(chunk_files, paths)
        return await self.index_chunks(chunks, agent_key=agent_key, organisation_id=organisation_id)

    async def delete_source(
        self,
        source: str,
        *,
        agent_key: str = "default",
        organisation_id: int | None = None,
    ) -> int:
        return await self._store.delete_source(
            source, namespace=self.namespace(agent_key, organisation_id)
        )

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
