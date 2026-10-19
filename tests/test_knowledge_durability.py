"""Training documents survive losing the vector index.

Production, autumn 2026: every uploaded document lived only in a hosted Qdrant
cluster. The cluster was suspended, and the training went with it; earlier, an
embedder change had been "fixed" by deleting the collection, which wiped it
twice more. The database now keeps each document's text and the index is a
cache rebuilt from it.
"""

from __future__ import annotations

import pytest

from vaani.config import Settings
from vaani.core.registry import build_services
from vaani.db.knowledge import KnowledgeRepository
from vaani.db.repository import CallRepository
from vaani.providers.embeddings.providers import HashEmbedding
from vaani.rag.chunking import chunk_text
from vaani.rag.retriever import Retriever
from vaani.rag.store import MemoryVectorStore, QdrantVectorStore

FEES = (
    "The hostel fee for the academic year is twelve thousand rupees, payable in two "
    "instalments before the first of July and the first of December. Late payment "
    "attracts a fine of five hundred rupees per month."
)


@pytest.fixture
async def documents(tmp_path):
    repo = CallRepository(f"sqlite+aiosqlite:///{tmp_path / 'vaani.db'}")
    await repo.start()
    yield KnowledgeRepository(repo.sessions)
    await repo.close()


def _qdrant() -> QdrantVectorStore:
    from qdrant_client import AsyncQdrantClient

    store = QdrantVectorStore(collection="test_knowledge")
    store._client = AsyncQdrantClient(location=":memory:")
    return store


async def _retriever(store, documents, dim: int = 384) -> Retriever:
    r = Retriever(store=store, embedder=HashEmbedding(dim=dim), min_score=0.0, documents=documents)
    await r.start()
    return r


async def test_a_document_survives_the_index_being_lost(documents):
    first = await _retriever(MemoryVectorStore(), documents)
    await first.index_text(FEES, "fees.txt", agent_key="admissions")

    # A brand-new, empty index — a restarted in-memory store, or a hosted
    # cluster replaced by a fresh one.
    second = await _retriever(MemoryVectorStore(), documents)
    hits = await second.search("hostel fee", agent_key="admissions")
    assert hits and hits[0].source == "fees.txt"


async def test_changing_the_embedder_rebuilds_the_index_instead_of_refusing(documents):
    store = _qdrant()
    first = await _retriever(store, documents, dim=384)
    await first.index_text(FEES, "fees.txt", organisation_id=3)

    # Same collection, an embedder of another width. This used to refuse and
    # tell the operator to delete the collection, which is how it was wiped.
    second = await _retriever(store, documents, dim=64)
    assert await store._dimension() == 64
    hits = await second.search("hostel fee", agent_key="admissions", organisation_id=3)
    assert hits and hits[0].source == "fees.txt"


async def test_an_embedder_change_never_drops_the_collection_live_calls_read(documents):
    """The retriever being replaced keeps serving calls from the old collection
    while the new one fills, and a collection is never ours to delete."""
    store = _qdrant()
    await (await _retriever(store, documents, dim=384)).index_text(FEES, "fees.txt")

    await _retriever(_qdrant_sharing(store), documents, dim=64)

    names = {c.name for c in (await store._client.get_collections()).collections}
    assert {"test_knowledge", "test_knowledge_64d"} <= names
    assert await store.count() > 0, "the old collection's points were removed"


async def test_a_deleted_document_stays_deleted_when_an_old_index_is_read_again(documents):
    """Switch embedder, delete a document, switch back: the old collection still
    holds the document. Without a tombstone the sync adopts it back."""
    store = _qdrant()
    await (await _retriever(store, documents, dim=384)).index_text(FEES, "fees.txt")
    wide = await _retriever(_qdrant_sharing(store), documents, dim=64)
    await wide.delete_source("fees.txt")

    back = await _retriever(_qdrant_sharing(store), documents, dim=384)

    assert await documents.all() == []
    assert await back.search("hostel fee") == []


def _qdrant_sharing(store: QdrantVectorStore) -> QdrantVectorStore:
    """A second store over the same Qdrant, as a settings save or restart builds."""
    other = QdrantVectorStore(collection="test_knowledge")
    other._client = store._client
    return other


async def test_an_index_from_before_the_database_kept_documents_is_adopted(documents):
    store = _qdrant()
    legacy = Retriever(store=store, embedder=HashEmbedding(), min_score=0.0)
    await legacy.start()
    await legacy.index_chunks(chunk_text(FEES, source="fees.txt"), organisation_id=3)

    await _retriever(store, documents)

    saved = {(d.namespace, d.source): d for d in await documents.all()}
    assert ("org:3", "fees.txt") in saved
    doc = saved[("org:3", "fees.txt")]
    assert "twelve thousand rupees" in doc.text
    assert doc.metadata.get("recovered_from_index") is True


class _DownStore(MemoryVectorStore):
    """A store that has gone away, as the hosted cluster did."""

    async def upsert(self, chunks, vectors, namespace):
        raise ConnectionError("connection reset by peer")

    async def delete_source(self, source, namespace):
        raise ConnectionError("connection reset by peer")


async def test_an_upload_while_the_index_is_down_is_kept_and_indexed_later(documents):
    down = await _retriever(_DownStore(), documents)
    with pytest.raises(ConnectionError):
        await down.index_text(FEES, "fees.txt", agent_key="admissions")
    assert [d.source for d in await documents.all()] == ["fees.txt"]

    back = await _retriever(MemoryVectorStore(), documents)
    assert await back.search("hostel fee", agent_key="admissions")


async def test_a_delete_while_the_index_is_down_does_not_half_happen(documents):
    """If the database forgot it while the index still held it, the next sync
    would adopt it back from the index and the deleted document would return."""
    up = await _retriever(MemoryVectorStore(), documents)
    await up.index_text(FEES, "fees.txt", agent_key="admissions")

    # Not started: against a dead store the boot-time sync fails too, which is
    # Services.start's business, not this test's.
    down = Retriever(store=_DownStore(), embedder=HashEmbedding(), documents=documents)
    with pytest.raises(ConnectionError):
        await down.delete_source("fees.txt", agent_key="admissions")
    assert [d.source for d in await documents.all()] == ["fees.txt"]


async def test_a_corrected_document_replaces_a_stale_index_copy(documents):
    store = MemoryVectorStore()
    r = await _retriever(store, documents)
    await r.index_text(FEES, "fees.txt", agent_key="admissions")
    # Corrected in the database while the index kept the old version (an upload
    # that was saved but could not be indexed).
    from vaani.db.knowledge import KnowledgeDocument

    await documents.save(
        KnowledgeDocument("admissions", "fees.txt", FEES.replace("twelve", "fifteen"))
    )
    await r.sync()
    hits = await r.search("hostel fee", agent_key="admissions")
    assert all("fifteen" in h.text for h in hits if h.source == "fees.txt")
    assert len(await store.export("admissions", "fees.txt")) == len(
        chunk_text(FEES, source="fees.txt")
    )


async def test_the_phone_is_still_answered_when_the_knowledge_store_is_down():
    """A Qdrant outage at boot used to fail Services.start(), which takes the
    whole line down — not just the knowledge."""
    services = build_services(
        Settings(stt_provider="mock", llm_provider="mock", tts_provider="mock")
    )

    async def unreachable(dim: int) -> None:
        raise ConnectionError("connection reset by peer")

    services.retriever._store.ensure = unreachable
    await services.start()
    assert not services.retriever.ready
    assert await services.retriever.search("hostel fee") == []
    await services.close()
