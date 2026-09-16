"""Correcting a training document must replace it, not add a second copy.

A source name is the document: `fees-2026.pdf` is one circular, and uploading it
again means it was corrected. Both stores appended instead — the memory store
extends its bucket, and the Qdrant store mints a fresh uuid per chunk — so the
superseded text stayed searchable next to its replacement, and the bot could
quote last year's fee to a caller with no way for anyone to tell which copy it
had used.

Deleting is the other half: an operator who withdraws a circular needs it gone
from what the bot can say, in the scope they withdrew it from and no other.
"""

from __future__ import annotations

import pytest

from vaani.providers.embeddings.providers import HashEmbedding
from vaani.rag.retriever import Retriever
from vaani.rag.store import MemoryVectorStore


@pytest.fixture
async def retriever():
    r = Retriever(store=MemoryVectorStore(), embedder=HashEmbedding())
    await r.start()
    return r


async def test_uploading_a_corrected_document_replaces_the_one_before_it(retriever):
    await retriever.index_text("The fee is 40,000 rupees.", source="fees.pdf", agent_key="a")
    await retriever.index_text("The fee is 45,000 rupees.", source="fees.pdf", agent_key="a")

    assert await retriever.count("a") == 1, "the superseded circular is still searchable"
    assert await retriever.sources(agent_key="a") == [("fees.pdf", 1)]


async def test_a_correction_does_not_disturb_the_other_documents(retriever):
    await retriever.index_text("Office hours are 10 to 5.", source="hours.pdf", agent_key="a")
    await retriever.index_text("The fee is 40,000 rupees.", source="fees.pdf", agent_key="a")
    await retriever.index_text("The fee is 45,000 rupees.", source="fees.pdf", agent_key="a")

    assert dict(await retriever.sources(agent_key="a")) == {"hours.pdf": 1, "fees.pdf": 1}


async def test_the_same_document_in_another_scope_is_left_alone(retriever):
    """Two call centres may each hold their own `fees.pdf`. Correcting one must
    not silently empty the other's."""
    await retriever.index_text("Health fee is 45,000.", source="fees.pdf", agent_key="health")
    await retriever.index_text("Portal fee is nil.", source="fees.pdf", agent_key="portal")
    await retriever.index_text("Health fee is 50,000.", source="fees.pdf", agent_key="health")

    assert await retriever.count("health") == 1
    assert await retriever.count("portal") == 1
    hits = await retriever.search("portal fee", agent_key="portal")
    assert hits and "nil" in hits[0].text


async def test_a_replaced_document_is_no_longer_quotable(retriever):
    """The point of all this: the old wording must not come back in an answer."""
    await retriever.index_text(
        "Admission fees for 2025 are 40,000 rupees.", source="fees.pdf", agent_key="a"
    )
    await retriever.index_text(
        "Admission fees for 2026 are 45,000 rupees.", source="fees.pdf", agent_key="a"
    )

    hits = await retriever.search("what are the admission fees", agent_key="a")
    assert hits, "the corrected circular must still be findable"
    assert all("40,000" not in hit.text for hit in hits), "the superseded fee is still quotable"


async def test_deleting_a_document_removes_it_from_that_scope_only(retriever):
    await retriever.index_text("Health fee is 45,000.", source="fees.pdf", agent_key="health")
    await retriever.index_text("Portal fee is nil.", source="fees.pdf", agent_key="portal")

    removed = await retriever.delete_source("fees.pdf", agent_key="health")

    assert removed == 1
    assert await retriever.count("health") == 0
    assert await retriever.count("portal") == 1


async def test_deleting_something_that_was_never_there_is_not_an_error(retriever):
    """An operator clicking delete twice, or on a document another admin has
    just removed, should see nothing removed rather than a failure."""
    assert await retriever.delete_source("never-uploaded.pdf", agent_key="a") == 0
