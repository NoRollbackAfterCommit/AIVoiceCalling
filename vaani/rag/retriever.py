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
    "the", "a", "an", "is", "are", "was", "of", "to", "for", "in", "on", "and",
    "or", "my", "i", "you", "it", "what", "how", "when", "where", "can", "do",
    "does", "please", "tell", "me", "about",
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

    # -- write --------------------------------------------------------------

    async def index_chunks(self, chunks: list[Chunk], *, agent_key: str = "default") -> int:
        if not chunks:
            return 0
        # Embed in batches: a 500-page circular in one call will OOM the GPU.
        total = 0
        for start in range(0, len(chunks), 64):
            batch = chunks[start : start + 64]
            vectors = await self._embedder.embed([c.text for c in batch])
            total += await self._store.upsert(batch, vectors, namespace=agent_key)
        log.info("indexed", extra={"chunks": total, "agent": agent_key})
        return total

    async def index_text(
        self, text: str, source: str, *, agent_key: str = "default",
        metadata: dict[str, Any] | None = None
    ) -> int:
        return await self.index_chunks(
            chunk_text(text, source=source, metadata=metadata), agent_key=agent_key
        )

    async def index_paths(self, paths: list[Path], *, agent_key: str = "default") -> int:
        chunks = await asyncio.to_thread(chunk_files, paths)
        return await self.index_chunks(chunks, agent_key=agent_key)

    async def delete_source(self, source: str, *, agent_key: str = "default") -> int:
        return await self._store.delete_source(source, namespace=agent_key)

    async def count(self, agent_key: str | None = None) -> int:
        return await self._store.count(agent_key)

    # -- read ---------------------------------------------------------------

    async def search(
        self, query: str, *, agent_key: str = "default", top_k: int | None = None
    ) -> list[SearchHit]:
        if not self._ready or not query.strip():
            return []
        k = top_k or self._top_k
        # Over-fetch, then rerank and diversify down to k.
        vector = (await self._embedder.embed([query]))[0]
        hits = await self._store.search(
            vector, namespace=agent_key, top_k=k * 3, min_score=self._min_score * 0.6
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
