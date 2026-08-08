# Phase 2 — Outcome engine

**Date:** 2026-08-08
**Status:** Approved. All three decisions resolved 2026-08-08.
**Phase:** 2 of 6

## The problem

An LLM voice agent left to itself does not end calls. It loops — "is there
anything else I can help with?" answered by "no thanks" answered by "please let
me know if you need anything" — or it wanders without ever establishing what the
caller wanted, and the call ends with nothing recorded and nothing resolved.

Phase 1 gives the agent tools and a transcript. It has no notion of *what this
call was for* or *whether that was achieved*, which is exactly what "drive the
discussion to a conclusive end" requires.

## Goal

Every call ends in a recorded disposition, and where applicable a commitment the
caller can hold the organisation to — a complaint reference, a scheduled
callback, or a transfer to a named human.

## Non-goals

Sentiment analysis, quality scoring, agent-performance analytics, and automated
follow-up. Those are reporting features built on top of the disposition once it
exists; none belong in the conversation loop.

## Design

### 1. Objective — what the call is for

`AgentProfile` gains an `objective`: one or two sentences stating what a
successful call achieves. It renders into the system prompt above the policies,
because it frames every other instruction.

> "Establish which service the caller is asking about, answer from the knowledge
> base, and either resolve the question or register a complaint with a reference
> number. Do not end the call without one of those outcomes."

Behaviour stays data, consistent with the rest of the platform.

### 2. Disposition — how the call ended

A closed vocabulary, recorded on every call:

| Disposition | Meaning |
|---|---|
| `resolved` | The caller's question was answered from knowledge |
| `complaint_registered` | A complaint was filed; carries a reference |
| `callback_scheduled` | A callback was booked; carries a time |
| `transferred` | Handed to a human, with the reason |
| `out_of_scope` | A real request this agent is not for |
| `unresolved` | The agent could not help and no fallback applied |
| `caller_abandoned` | The caller hung up mid-conversation |
| `idle_timeout` | Silence exceeded the hangup threshold |
| `capacity_rejected` | All lines busy; the call was never served |

The first six are set by the agent through a tool. The last three are set by the
platform, because a caller who has hung up cannot call a tool.

**Decision 1 — resolved: fixed core, with an optional per-profile extension
list.** A
government deployment will want to compare complaint volumes across departments,
and that only works if `complaint_registered` means the same thing everywhere.
Profiles that need a domain-specific outcome add to the list rather than
replacing it.

### 3. The closing policy — what actually drives to an end

Three states tracked across the call, independent of the LLM's own reasoning:

- **Need identified** — the agent has captured what the caller wants
- **Need addressed** — an answer was given, or a tool produced a commitment
- **Confirmed** — the caller has acknowledged

The rules that follow from them:

1. **Confirm once, then close.** After the need is addressed, the agent asks
   "anything else?" exactly once. A negative answer ends the call. This single
   rule removes the most common failure.
2. **Stall detection.** If N consecutive turns pass with the need still
   unidentified, or addressed but unconfirmed, the agent stops re-asking and
   offers a concrete fallback: register a complaint, schedule a callback, or
   transfer.
3. **Never end a call the caller has not finished.** The agent may only hang up
   after confirmation, on idle timeout, or on an explicit caller request.
   Cutting off a talking caller is worse than any amount of looping.
4. **`end_call` requires a disposition.** The tool refuses without one, which is
   what makes the audit trail complete by construction rather than by diligence.

**Decision 2 — resolved: three unproductive turns.** Two is impatient on a line
where callers pause to think; five leaves the caller convinced the agent is
useless. The threshold is per profile, defaulting to 3.

**Decision 3 — resolved: the agent may close after confirmation.** A government
helpline is measured on line availability, and a caller who has said "no, that's
all" and then sits through thirty seconds of silence is a worse experience than
a clean goodbye.

### 4. Stall detection without an LLM call

Progress is inferred from signals already available, so no extra model round trip
lands in the turn budget:

- Whether a tool ran this turn
- Whether the retrieval score cleared the relevance threshold
- Whether the caller's utterance is near-identical to their previous one — the
  strongest signal that the agent's last answer did not land
- Turns elapsed since the last state change

### 5. Persistence

`calls` gains `disposition`, `disposition_reason` and `reference`. The existing
`outcome` column stays as the transport-level result (`completed`,
`caller_ended`); disposition is the *business* result. Conflating them would
lose the distinction between a call that connected cleanly and a call that
achieved something.

## Testing

The default suite stays offline and mock-only. New coverage:

- Each closing rule as a unit test over a scripted transcript
- Stall detection fires at the threshold and not before
- `end_call` without a disposition is refused
- Platform-set dispositions apply when the caller abandons or times out
- A full mock call reaches a terminal disposition rather than looping

Acceptance: a scripted conversation that previously looped now terminates, and
every call in the database has a non-null disposition.

## Risks

| Risk | Response |
|---|---|
| The agent closes too eagerly and cuts callers off | Rule 3 is absolute; only confirmation, idle timeout or explicit request may end a call |
| The LLM ignores the disposition tool | `end_call` refuses without one, so the failure is loud, not silent |
| Stall detection misfires on a thoughtful caller | Threshold is per-profile and the fallback is an offer, never a hangup |
| Dispositions drift per deployment | Decision 1 — fixed core vocabulary |

## Dependencies

Phase 1 Tasks 4–6 (language tracking and persistence) are complete and shipped.
Nothing here depends on telephony, so phase 2 proceeds while Exotel or Ozonetel
access is still being arranged.
