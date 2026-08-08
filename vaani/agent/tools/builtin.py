"""The tools every deployment gets.

Domain packs (electricity board, admissions, hospital) add their own on top by
importing `registry` and decorating their own functions — see docs/tools.md.
"""

from __future__ import annotations

import random
import re
from datetime import UTC, datetime, timedelta

from vaani.agent.tools.base import ToolContext, ToolRegistry, ToolResult

registry = ToolRegistry()


# ---------------------------------------------------------------------------
# Knowledge
# ---------------------------------------------------------------------------


@registry.tool(
    name="search_knowledge",
    description=(
        "Search the organisation's knowledge base — policies, circulars, FAQs, "
        "procedures, fees and timelines. Use this before answering any factual "
        "question about the organisation. Search in the caller's own words."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to look for, phrased as a question or keywords.",
            }
        },
        "required": ["query"],
    },
    timeout_s=6.0,
)
async def search_knowledge(query: str, ctx: ToolContext) -> ToolResult:
    retriever = ctx.services.get("retriever")
    if retriever is None:
        return ToolResult(content="No knowledge base is configured.", ok=False)

    hits = await retriever.search(query, agent_key=ctx.agent_key)
    if not hits:
        return ToolResult(
            content="No relevant information found in the knowledge base.", ok=False
        )

    # Numbered passages with sources: the model can cite, and a reviewer can
    # trace any spoken claim back to the document it came from.
    lines = []
    for i, hit in enumerate(hits, 1):
        lines.append(f"[{i}] (source: {hit.source}) {hit.text.strip()}")
    return ToolResult(
        content="\n\n".join(lines),
        data={"sources": [h.source for h in hits], "scores": [h.score for h in hits]},
    )


# ---------------------------------------------------------------------------
# Call control
# ---------------------------------------------------------------------------


@registry.tool(
    name="transfer_to_human",
    description=(
        "Hand the call to a human executive. Use when the caller asks for a "
        "person, is distressed or abusive, reports an emergency, or when you "
        "cannot resolve their issue."
    ),
    parameters={
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Why the transfer is needed, for the human agent.",
            },
            "department": {
                "type": "string",
                "description": "Which desk to route to, if known.",
            },
        },
        "required": ["reason"],
    },
)
async def transfer_to_human(
    reason: str, ctx: ToolContext, department: str = "general"
) -> ToolResult:
    return ToolResult(
        content=(
            "Transfer accepted. Tell the caller you are connecting them now and "
            "ask them to stay on the line. Then stop."
        ),
        control={"action": "transfer", "department": department, "reason": reason},
        data={"reason": reason, "department": department},
    )


@registry.tool(
    name="set_disposition",
    description=(
        "Record how this call ended, before ending it. Required: end_call will "
        "not run without it. Choose the outcome that actually happened."
    ),
    parameters={
        "type": "object",
        "properties": {
            "disposition": {
                "type": "string",
                "description": (
                    "One of: resolved, complaint_registered, callback_scheduled, "
                    "transferred, out_of_scope, unresolved."
                ),
            },
            "reason": {"type": "string", "description": "One line of justification."},
            "reference": {
                "type": "string",
                "description": "The reference number, for a complaint or callback.",
            },
        },
        "required": ["disposition", "reason"],
    },
)
async def set_disposition(
    disposition: str, reason: str, ctx: ToolContext, reference: str = ""
) -> ToolResult:
    from vaani.agent.outcome import AGENT_SET, REQUIRES_REFERENCE

    allowed = AGENT_SET | set(ctx.state.get("extra_dispositions") or ())
    if disposition not in allowed:
        # Refusing an invented outcome is what keeps the vocabulary closed, and
        # therefore what makes the numbers comparable across departments.
        return ToolResult(
            content=(
                f"'{disposition}' is not a valid outcome. Choose one of: "
                f"{', '.join(sorted(allowed))}."
            ),
            ok=False,
        )

    if disposition in REQUIRES_REFERENCE and not reference.strip():
        return ToolResult(
            content=(
                f"A {disposition} outcome needs the reference number you gave the "
                "caller. Call the tool again with it."
            ),
            ok=False,
        )

    ctx.state["disposition"] = disposition
    ctx.state["disposition_reason"] = reason
    if reference.strip():
        ctx.state["reference"] = reference.strip()

    return ToolResult(
        content="Outcome recorded. You may now end the call.",
        data={"disposition": disposition, "reference": reference},
    )


@registry.tool(
    name="end_call",
    description=(
        "End the call. Use only after the caller has confirmed they need nothing "
        "further, or after they say goodbye."
    ),
    parameters={
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "One line on what was handled."}
        },
        "required": ["summary"],
    },
)
async def end_call(summary: str, ctx: ToolContext) -> ToolResult:
    disposition = ctx.state.get("disposition")
    if not disposition:
        # Refusing here is what makes the audit trail complete by construction
        # rather than by anyone remembering to fill it in.
        return ToolResult(
            content="Record the outcome first with set_disposition, then end the call.",
            ok=False,
        )
    return ToolResult(
        content="Say your closing line, then stop.",
        control={
            "action": "hangup",
            "summary": summary,
            "disposition": disposition,
            "reference": ctx.state.get("reference"),
        },
        data={"summary": summary, "disposition": disposition},
    )


@registry.tool(
    name="schedule_callback",
    description=(
        "Arrange for someone to call the caller back at a later time, when the "
        "matter cannot be resolved now and no human is available."
    ),
    parameters={
        "type": "object",
        "properties": {
            "when": {
                "type": "string",
                "description": "Requested time, e.g. 'tomorrow morning', 'after 5 pm'.",
            },
            "topic": {"type": "string", "description": "What the callback is about."},
        },
        "required": ["when", "topic"],
    },
)
async def schedule_callback(when: str, topic: str, ctx: ToolContext) -> ToolResult:
    ref = f"CB{random.randint(100000, 999999)}"
    ctx.state["callback_ref"] = ref
    return ToolResult(
        content=(
            f"Callback booked, reference {_spoken_digits(ref[2:])}. Read the "
            f"reference back to the caller slowly."
        ),
        data={"reference": ref, "when": when, "topic": topic},
    )


# ---------------------------------------------------------------------------
# Caller identity
# ---------------------------------------------------------------------------


@registry.tool(
    name="verify_caller",
    description=(
        "Verify the caller's identity from their registered mobile number. This "
        "must succeed before any tool that changes or reveals account data."
    ),
    parameters={
        "type": "object",
        "properties": {
            "mobile_number": {
                "type": "string",
                "description": "The number the caller gave, digits only.",
            }
        },
        "required": ["mobile_number"],
    },
)
async def verify_caller(mobile_number: str, ctx: ToolContext) -> ToolResult:
    digits = re.sub(r"\D", "", mobile_number)
    if len(digits) < 10:
        return ToolResult(
            content="That number is incomplete. Ask the caller to repeat all ten digits.",
            ok=False,
        )
    # Replace with the real CRM lookup for a given deployment.
    ctx.state["verified"] = True
    ctx.state["mobile_number"] = digits
    return ToolResult(
        content=f"Verified. The caller's number ends in {digits[-4:]}.",
        data={"masked": f"******{digits[-4:]}"},
    )


# ---------------------------------------------------------------------------
# Reference domain pack: utility billing. Swap the bodies for real integrations.
# ---------------------------------------------------------------------------


@registry.tool(
    name="check_bill",
    description="Look up the current outstanding bill for a consumer number.",
    parameters={
        "type": "object",
        "properties": {
            "consumer_number": {"type": "string", "description": "Consumer or account number."}
        },
        "required": ["consumer_number"],
    },
    requires_verification=False,
)
async def check_bill(consumer_number: str, ctx: ToolContext) -> ToolResult:
    consumer = re.sub(r"\D", "", consumer_number)
    if not consumer:
        return ToolResult(content="No consumer number was given. Ask for it.", ok=False)
    # Deterministic from the number so a demo repeats consistently.
    seed = int(consumer[-6:] or 0)
    amount = 500 + (seed % 9500)
    due = datetime.now(UTC) + timedelta(days=7 + seed % 14)
    return ToolResult(
        content=(
            f"Consumer {consumer}: amount due {amount} rupees, "
            f"due on {due.strftime('%d %B %Y')}. No disconnection notice is active."
        ),
        data={"amount": amount, "due_date": due.date().isoformat(), "consumer": consumer},
    )


@registry.tool(
    name="register_complaint",
    description=(
        "Raise a formal complaint and return its reference number. Confirm the "
        "category and a one-line description with the caller before calling this."
    ),
    parameters={
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "e.g. billing, outage, voltage, meter, staff conduct",
            },
            "description": {"type": "string", "description": "What the caller reported."},
            "address": {"type": "string", "description": "Location, if relevant."},
        },
        "required": ["category", "description"],
    },
)
async def register_complaint(
    category: str, description: str, ctx: ToolContext, address: str = ""
) -> ToolResult:
    ref = f"CMP{random.randint(100000, 999999)}"
    ctx.state["last_complaint_ref"] = ref
    return ToolResult(
        content=(
            f"Complaint registered under {category}. Reference number "
            f"{_spoken_digits(ref[3:])}. Read it back slowly, digit by digit, and "
            f"tell the caller it will be actioned within three working days."
        ),
        data={"reference": ref, "category": category, "description": description,
              "address": address},
    )


@registry.tool(
    name="check_application_status",
    description="Check the status of an application, request or complaint by reference number.",
    parameters={
        "type": "object",
        "properties": {
            "reference": {"type": "string", "description": "The reference number given earlier."}
        },
        "required": ["reference"],
    },
)
async def check_application_status(reference: str, ctx: ToolContext) -> ToolResult:
    ref = reference.strip().upper()
    stages = ["received", "under review", "pending document verification", "approved"]
    stage = stages[sum(ord(c) for c in ref) % len(stages)]
    return ToolResult(
        content=f"Reference {ref} is currently: {stage}.",
        data={"reference": ref, "status": stage},
    )


def _spoken_digits(digits: str) -> str:
    """TTS reads '482913' as 'four hundred eighty two thousand...'. Spacing the
    digits forces it to read them individually, which is what a caller writing a
    reference number down actually needs."""
    return " ".join(digits)
