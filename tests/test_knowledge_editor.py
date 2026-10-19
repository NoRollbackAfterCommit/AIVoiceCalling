"""The Knowledge page's editor: see, read, correct and rename every document.

Read from the database copy, so the list survives the index being down — the
state production was in for weeks, when the page showed a 500 and an operator
could reasonably conclude their training had been deleted.
"""

from __future__ import annotations

import httpx
import pytest

from vaani.core.registry import build_services
from vaani.db.knowledge import KnowledgeRepository
from vaani.db.repository import CallRepository

FEES = "The hostel fee is twelve thousand rupees, payable before the first of July."


@pytest.fixture
async def services(settings, tmp_path):
    repo = CallRepository(f"sqlite+aiosqlite:///{tmp_path / 'vaani.db'}")
    await repo.start()
    svc = build_services(settings)
    svc.retriever.use_documents(KnowledgeRepository(repo.sessions))
    await svc.start()
    yield svc
    await svc.close()
    await repo.close()


def _client(services) -> httpx.AsyncClient:
    from vaani.main import create_app

    app = create_app()
    app.state.services = services
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def _add(client, source="fees.txt", text=FEES):
    r = await client.post("/api/knowledge/text", json={"text": text, "source": source})
    assert r.status_code == 200, r.text


async def test_the_full_text_of_a_document_can_be_read_back(services):
    async with _client(services) as client:
        await _add(client)
        r = await client.get("/api/knowledge/document", params={"source": "fees.txt"})
    assert r.status_code == 200, r.text
    assert r.json()["text"] == FEES


async def test_editing_a_document_replaces_what_callers_are_told(services):
    async with _client(services) as client:
        await _add(client)
        r = await client.put(
            "/api/knowledge/document",
            json={"source": "fees.txt", "text": FEES.replace("twelve", "fifteen")},
        )
        assert r.status_code == 200, r.text
        hits = (await client.get("/api/knowledge/search", params={"q": "hostel fee"})).json()
    texts = [h["text"] for h in hits["hits"]]
    assert texts and all("twelve" not in t for t in texts)
    assert any("fifteen" in t for t in texts)


async def test_renaming_a_document_leaves_one_copy_under_the_new_name(services):
    async with _client(services) as client:
        await _add(client)
        r = await client.put(
            "/api/knowledge/document",
            json={"source": "fees.txt", "new_source": "hostel-fees-2026", "text": FEES},
        )
        assert r.status_code == 200, r.text
        rows = (await client.get("/api/knowledge/sources")).json()
    assert [row["source"] for row in rows] == ["hostel-fees-2026"]


async def test_documents_are_still_listed_while_the_index_is_down(services):
    async with _client(services) as client:
        await _add(client)

        async def unreachable(namespace):
            raise ConnectionError("connection reset by peer")

        services.retriever._store.sources = unreachable
        r = await client.get("/api/knowledge/sources")
    assert r.status_code == 200, r.text
    assert r.json() == [{"source": "fees.txt", "chunks": None, "indexed": False, "scope": "agent"}]


async def test_a_missing_document_is_a_404(services):
    async with _client(services) as client:
        r = await client.get("/api/knowledge/document", params={"source": "nope.txt"})
    assert r.status_code == 404


async def test_the_organisation_scope_cannot_be_edited_through_a_line_without_one(services):
    """Ownership comes from the agent, as for uploads: the fallback agent has no
    organisation, so it cannot reach anyone's shared set."""
    async with _client(services) as client:
        r = await client.put(
            "/api/knowledge/document",
            json={"source": "fees.txt", "text": FEES, "scope": "organisation"},
        )
    assert r.status_code == 400
