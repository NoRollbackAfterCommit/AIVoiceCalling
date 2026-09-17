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


# -- the collection's dimensions must match the embedder ---------------------


class _CollectionsFake:
    """A client whose collection already exists, at a fixed vector size."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.created = False

    async def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name="vaani")])

    async def get_collection(self, collection_name):
        params = SimpleNamespace(size=self.dim, distance="Cosine")
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=params)))

    async def create_collection(self, **kwargs):
        self.created = True


async def test_an_existing_collection_of_the_right_size_is_left_alone():
    store = QdrantVectorStore(collection="vaani")
    store._client = _CollectionsFake(dim=384)

    await store.ensure(384)

    assert store._client.created is False, "the corpus must not be recreated on every boot"


async def test_changing_the_embedder_under_an_existing_collection_is_refused():
    """Switching embeddings changes the vector width — 384 for the hash
    embedder, 1536 for OpenAI's small model. Qdrant fixes that width when the
    collection is made and `ensure` returned early whenever the collection
    existed, so the change was accepted in the settings page and then every
    upsert and every search failed at the Qdrant API.

    From the caller's side that is indistinguishable from an empty knowledge
    base: the agent just says it does not have the information.
    """
    store = QdrantVectorStore(collection="vaani")
    store._client = _CollectionsFake(dim=384)

    try:
        await store.ensure(1536)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("a mismatched collection must not be used")

    assert "384" in message and "1536" in message, message
    assert "vaani" in message, "say which collection, so it can be dealt with"
