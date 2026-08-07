"""Vector stores.

`MemoryVectorStore` is a brute-force cosine scan. That sounds naive, and for a
few thousand chunks — which is what one department's policy corpus actually is —
it is faster than the network round-trip to a real vector database. It keeps the
default install to zero infrastructure. `QdrantVectorStore` is the production
path: persistent, filterable per tenant, and clusterable.

Both partition by `namespace`, which is the agent key. One deployment can serve
the electricity board and the municipal corporation without either seeing the
other's documents.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from vaani.core.logging import get_logger
from vaani.rag.chunking import Chunk

log = get_logger(__name__)


@dataclass(slots=True)
class SearchHit:
    text: str
    source: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(Protocol):
    async def ensure(self, dim: int) -> None: ...
    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]],
                     namespace: str) -> int: ...
    async def search(self, vector: list[float], namespace: str, top_k: int,
                     min_score: float) -> list[SearchHit]: ...
    async def delete_source(self, source: str, namespace: str) -> int: ...
    async def count(self, namespace: str | None = None) -> int: ...


# ---------------------------------------------------------------------------


class MemoryVectorStore:
    name = "memory"

    def __init__(self) -> None:
        # namespace -> list of (vector, chunk)
        self._data: dict[str, list[tuple[list[float], Chunk]]] = {}

    async def ensure(self, dim: int) -> None:
        return None

    async def upsert(
        self, chunks: list[Chunk], vectors: list[list[float]], namespace: str
    ) -> int:
        bucket = self._data.setdefault(namespace, [])
        bucket.extend(zip(vectors, chunks, strict=True))
        return len(chunks)

    async def search(
        self, vector: list[float], namespace: str, top_k: int, min_score: float
    ) -> list[SearchHit]:
        bucket = self._data.get(namespace, [])
        scored = [
            (_cosine(vector, vec), chunk)
            for vec, chunk in bucket
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            SearchHit(text=c.text, source=c.source, score=round(s, 4), metadata=c.metadata)
            for s, c in scored[:top_k]
            if s >= min_score
        ]

    async def delete_source(self, source: str, namespace: str) -> int:
        bucket = self._data.get(namespace, [])
        keep = [(v, c) for v, c in bucket if c.source != source]
        removed = len(bucket) - len(keep)
        self._data[namespace] = keep
        return removed

    async def count(self, namespace: str | None = None) -> int:
        if namespace is not None:
            return len(self._data.get(namespace, []))
        return sum(len(b) for b in self._data.values())


def _cosine(a: list[float], b: list[float]) -> float:
    dot = na = nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


# ---------------------------------------------------------------------------


class QdrantVectorStore:
    name = "qdrant"

    def __init__(
        self, url: str = "http://localhost:6333", api_key: str | None = None,
        collection: str = "vaani_knowledge"
    ) -> None:
        self._url = url
        self._api_key = api_key
        self._collection = collection
        self._client: Any = None

    async def ensure(self, dim: int) -> None:
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.models import Distance, VectorParams

        if self._client is None:
            self._client = AsyncQdrantClient(url=self._url, api_key=self._api_key)

        existing = await self._client.get_collections()
        if self._collection in {c.name for c in existing.collections}:
            return
        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        # Without this index, every namespace filter is a full scan.
        from qdrant_client.models import PayloadSchemaType

        await self._client.create_payload_index(
            collection_name=self._collection,
            field_name="namespace",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        await self._client.create_payload_index(
            collection_name=self._collection,
            field_name="source",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        log.info("created qdrant collection", extra={"collection": self._collection, "dim": dim})

    async def upsert(
        self, chunks: list[Chunk], vectors: list[list[float]], namespace: str
    ) -> int:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=uuid.uuid4().hex,
                vector=vec,
                payload={
                    "text": chunk.text,
                    "source": chunk.source,
                    "ordinal": chunk.ordinal,
                    "namespace": namespace,
                    **chunk.metadata,
                },
            )
            for chunk, vec in zip(chunks, vectors, strict=True)
        ]
        # Batched: a large circular sent as one request will time out.
        for start in range(0, len(points), 256):
            await self._client.upsert(
                collection_name=self._collection, points=points[start : start + 256]
            )
        return len(points)

    async def search(
        self, vector: list[float], namespace: str, top_k: int, min_score: float
    ) -> list[SearchHit]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        results = await self._client.search(
            collection_name=self._collection,
            query_vector=vector,
            query_filter=Filter(
                must=[FieldCondition(key="namespace", match=MatchValue(value=namespace))]
            ),
            limit=top_k,
            score_threshold=min_score,
        )
        return [
            SearchHit(
                text=r.payload.get("text", ""),
                source=r.payload.get("source", "unknown"),
                score=round(r.score, 4),
                metadata={k: v for k, v in r.payload.items()
                          if k not in ("text", "source", "namespace")},
            )
            for r in results
        ]

    async def delete_source(self, source: str, namespace: str) -> int:
        from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

        await self._client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(key="namespace", match=MatchValue(value=namespace)),
                        FieldCondition(key="source", match=MatchValue(value=source)),
                    ]
                )
            ),
        )
        return -1  # Qdrant does not report a delete count

    async def count(self, namespace: str | None = None) -> int:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        flt = None
        if namespace:
            flt = Filter(
                must=[FieldCondition(key="namespace", match=MatchValue(value=namespace))]
            )
        result = await self._client.count(collection_name=self._collection, count_filter=flt)
        return result.count
