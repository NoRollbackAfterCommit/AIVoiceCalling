"""Operator-set retrieval guidelines, and the business-logic round trip.

The guidelines are behaviour-as-data: a per-agent instruction block telling the
model how to use what it retrieves, edited in the admin portal, never in code.
The metadata test pins the other half — structured business facts survive the
trip through chunking, the store, and search.
"""

from __future__ import annotations

from dataclasses import replace

from vaani.agent.prompt import DEFAULT_PROFILE, render_system_prompt


def test_guidelines_render_into_the_prompt():
    profile = replace(
        DEFAULT_PROFILE,
        knowledge_guidelines="When two circulars conflict, prefer the newer one.",
    )
    assert "prefer the newer one" in render_system_prompt(profile)


def test_prompt_has_no_guidelines_section_when_unset():
    assert "How to use retrieved knowledge" not in render_system_prompt(DEFAULT_PROFILE)


def test_agent_api_accepts_guidelines():
    from vaani.api.routes import AgentProfileIn

    body = AgentProfileIn(key="desk", knowledge_guidelines="Quote fees exactly as written.")
    assert body.knowledge_guidelines == "Quote fees exactly as written."


async def test_metadata_survives_indexing_and_search(services):
    await services.retriever.index_text(
        "GST registration takes seven working days after document verification.",
        source="gst-guide",
        metadata={"subject": "tax", "version": "2026-04"},
    )
    hits = await services.retriever.search("how many days for gst registration documents")
    assert hits, "the indexed passage should be retrievable"
    assert hits[0].metadata["subject"] == "tax"
