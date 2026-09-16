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


# -- the organisation's own set, shared by all of its lines -------------------


async def test_a_line_answers_from_its_organisations_set_as_well_as_its_own(retriever):
    """The fee schedule is uploaded once for the whole university; the exam
    timetable belongs to the exams line alone. A caller on the exams line can
    be told either."""
    await retriever.index_text(
        "Admission fees for 2026 are 45,000 rupees.", source="fees.pdf", organisation_id=1
    )
    await retriever.index_text(
        "Examinations begin on 12 November.", source="timetable.pdf", agent_key="health-exams"
    )

    # Both are in scope for this line, so both may come back; the question is
    # which one it leads with. Asserting on the top hit rather than the whole
    # list, because the mock embedder scores loosely and the ranking is what
    # decides what the bot actually says.
    found = await retriever.search(
        "when do examinations begin", agent_key="health-exams", organisation_id=1
    )
    assert found and found[0].source == "timetable.pdf"

    found = await retriever.search(
        "what are the admission fees", agent_key="health-exams", organisation_id=1
    )
    assert found and found[0].source == "fees.pdf"


async def test_every_line_of_the_organisation_gets_the_shared_set(retriever):
    """Uploaded once, answered on all of them. That is the whole point."""
    await retriever.index_text("Office hours are 10 to 5.", source="hours.pdf", organisation_id=1)

    for line in ("health-admissions", "health-exams", "health-grievances"):
        found = await retriever.search(
            "what are the office hours", agent_key=line, organisation_id=1
        )
        assert [h.source for h in found] == ["hours.pdf"], f"{line} could not see the shared set"


async def test_one_lines_own_documents_stay_off_the_other_lines(retriever):
    """The reason per-line sets still exist: an admissions caller must not be
    read the examination timetable just because both are the same university."""
    await retriever.index_text(
        "Examinations begin on 12 November.", source="timetable.pdf", agent_key="health-exams"
    )

    found = await retriever.search(
        "when do examinations begin", agent_key="health-admissions", organisation_id=1
    )
    assert found == []


async def test_another_organisation_sees_neither(retriever):
    await retriever.index_text("Admission fees are 45,000.", source="fees.pdf", organisation_id=1)
    await retriever.index_text(
        "Examinations begin on 12 November.", source="timetable.pdf", agent_key="health-exams"
    )

    assert await retriever.search("admission fees", agent_key="wb-general", organisation_id=2) == []
    assert (
        await retriever.search(
            "when do examinations begin", agent_key="wb-general", organisation_id=2
        )
        == []
    )


async def test_an_agent_with_no_organisation_searches_only_its_own(retriever):
    """The fallback agent answering an unmapped number. It must not fall into
    somebody's shared set — an unconfigured number is nobody's customer."""
    await retriever.index_text("Admission fees are 45,000.", source="fees.pdf", organisation_id=1)
    await retriever.index_text("We are the switchboard.", source="switch.txt", agent_key="fallback")

    found = await retriever.search("admission fees", agent_key="fallback", organisation_id=None)
    assert [h.source for h in found] == []


async def test_a_shared_document_is_editable_and_deletable_in_its_own_scope(retriever):
    """Edit and delete have to reach the shared set too, or a wrong circular
    uploaded organisation-wide could never be corrected."""
    await retriever.index_text("The fee is 40,000.", source="fees.pdf", organisation_id=1)
    await retriever.index_text("The fee is 45,000.", source="fees.pdf", organisation_id=1)

    assert await retriever.sources(organisation_id=1) == [("fees.pdf", 1)]
    hits = await retriever.search("fee", agent_key="health-exams", organisation_id=1)
    assert hits and "45,000" in hits[0].text

    assert await retriever.delete_source("fees.pdf", organisation_id=1) == 1
    assert await retriever.search("fee", agent_key="health-exams", organisation_id=1) == []


async def test_deleting_the_shared_set_leaves_the_lines_own_documents(retriever):
    await retriever.index_text("Office hours are 10 to 5.", source="hours.pdf", organisation_id=1)
    await retriever.index_text(
        "Examinations begin on 12 November.", source="timetable.pdf", agent_key="health-exams"
    )

    await retriever.delete_source("hours.pdf", organisation_id=1)

    found = await retriever.search(
        "when do examinations begin", agent_key="health-exams", organisation_id=1
    )
    assert [h.source for h in found] == ["timetable.pdf"]


# -- the API side of the two scopes ------------------------------------------


def _client(services):
    import httpx

    from vaani.main import create_app

    app = create_app()
    app.state.services = services
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_an_agent_key_cannot_impersonate_an_organisation_scope(services):
    """The shared set lives at `org:<id>`, so an agent allowed to be called
    `org:1` would write straight into a customer's shared documents. Keys are
    restricted to the shape the console already tells operators to use."""
    async with _client(services) as client:
        refused = await client.put("/api/agents/org:1", json={"key": "org:1"})
        assert refused.status_code == 422, refused.text

        allowed = await client.put(
            "/api/agents/health-admissions", json={"key": "health-admissions"}
        )
        assert allowed.status_code == 200, allowed.text


async def test_uploading_to_the_organisation_scope_needs_an_agent_that_has_one(services, tmp_path):
    """The fallback agent belongs to no organisation, so there is no shared set
    to put a document in. Saying so beats writing it somewhere invisible."""
    async with _client(services) as client:
        response = await client.post(
            "/api/knowledge/text",
            json={"text": "Office hours are 10 to 5.", "source": "h.pdf", "scope": "organisation"},
        )
    assert response.status_code == 400
    assert "organisation" in response.json()["detail"].lower()
