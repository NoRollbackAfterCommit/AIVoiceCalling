"""Qdrant lookups, against a fake client.

This exists because the store called AsyncQdrantClient.search(), which newer
clients removed. Every Qdrant-backed search was a 500, and from the outside that
looked exactly like an empty knowledge base — the agent simply said it did not
have the information.
"""

from __future__ import annotations

from types import SimpleNamespace

from vaani.rag.store import QdrantVectorStore


class FakeAsyncQdrant:
    """Deliberately exposes only the current API. A call to the removed
    `search()` raises AttributeError here, exactly as it does in production."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    score=0.61,
                    payload={
                        "text": "The last date to apply is the thirtieth of June.",
                        "source": "wb-admission-rules",
                        "namespace": "default",
                        "ordinal": 0,
                    },
                )
            ]
        )


async def test_search_uses_the_current_client_api():
    store = QdrantVectorStore()
    fake = FakeAsyncQdrant()
    store._client = fake

    hits = await store.search([0.1] * 8, namespace="default", top_k=5, min_score=0.25)

    assert fake.calls, "query_points must be the method used"
    assert hits[0].source == "wb-admission-rules"
    assert hits[0].score == 0.61
    assert "thirtieth of June" in hits[0].text


async def test_the_namespace_and_threshold_reach_the_query():
    store = QdrantVectorStore()
    fake = FakeAsyncQdrant()
    store._client = fake

    await store.search([0.1] * 8, namespace="tenant-a", top_k=3, min_score=0.4)

    call = fake.calls[0]
    assert call["limit"] == 3
    assert call["score_threshold"] == 0.4
    assert call["query_filter"] is not None, "namespaces must not leak across agents"
