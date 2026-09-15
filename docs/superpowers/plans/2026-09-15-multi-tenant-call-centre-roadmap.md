# Multi-tenant call centre — phase and sprint execution plan

**Spec:** `docs/superpowers/specs/2026-09-15-multi-tenant-call-centre-design.md`
**Date:** 2026-09-15

A sprint here is a unit of deliverable work, not a calendar fortnight. Each one
ends with something deployable and tested; each phase ends with something an
operator can use. Phases ship in order because each depends on the one before —
except Phase 4, which is gated on an external answer and can land whenever that
arrives.

## Global constraints

Binding on every sprint. Copied from the spec and from `CLAUDE.md`.

- **Provider imports stay lazy.** No vendor SDK at module scope.
- **One audio format across every boundary.** PCM16 LE mono 16 kHz; telephony
  codecs convert in `vaani/telephony/` only.
- **Mock providers carry no dependencies.** The suite runs on `pip install -e .`,
  GPU-free, no network.
- **`cfg()` metadata is the single source of truth** for settings; never
  hand-write admin UI for a field.
- **`secret: true` never leaves the server.**
- **The call path is async and latency-critical.** No blocking work on the loop;
  nothing per-call that cannot be cancelled.
- **The existing deployment keeps working through every phase.** Every migration
  seeds existing rows into a default organisation.
- **Organisation scope comes from the authenticated identity, never from a query
  parameter**, for anyone below `platform_admin`.
- ruff, line length 100, py311, `from __future__ import annotations`.
- Tests run against mocks and stay fast enough for CI.

---

## Phase 1 — Tenancy and DID routing

**Delivers:** five DIDs, two organisations, calls routed and attributed
correctly. This is the business ask; it ships without waiting for login screens.

### Sprint 1.1 — Schema and the tenancy repository

- `organisations` and `dids` tables; `organisation_id` on `agent_profiles`;
  `organisation_id` (nullable) and `did` on `calls`.
- One Alembic migration that creates them **and seeds a default organisation
  owning today's agent and DID**, so the live box is unaffected.
- Repository methods: organisation and DID CRUD, and `resolve_did(number)`
  returning `(organisation_id, agent_key)` or `None`.
- E.164 normalisation, with tests for the shapes the carrier actually sends —
  `918065605873`, `+918065605873`, and the leading-space form seen in the live
  handshake log (`" 918065605873"`).

**Done when:** migration applies forward and back on a copy of the production
database, and `resolve_did` is covered including the malformed inputs.

### Sprint 1.2 — Route the call

- `smartflo.py:320` resolves the agent from `start.called` instead of the
  `smartflo_agent` setting.
- Unknown, inactive or unmapped DID → configured fallback agent, a warning log,
  and a call stamped with a NULL organisation so MIS shows it as unattributed.
- The session stamps `organisation_id` and `did` on the call record at creation,
  not at completion, so a crash mid-call still leaves an attributed row.

**Done when:** a simulated carrier call to each of several DIDs reaches the right
agent and lands in the right organisation, and an unknown DID is served rather
than dropped.

### Sprint 1.3 — Portal: organisations and DIDs

- `/api/organisations` and `/api/dids` CRUD.
- A console page to create an organisation, attach DIDs, and pick each one's
  agent. Reuses the existing console conventions rather than new markup.

**Done when:** the Health University and WB Portal split can be configured
entirely from the portal, with no shell access.

### Sprint 1.4 — Ask Tata about transfer

Not code. Send the Phase 4 question — can the Voice Bot transfer a live call, and
what payload does it expect — so the answer is in hand before Phase 4 starts.

---

## Phase 2 — Access control

**Delivers:** people log in, and see only their own organisation.

### Sprint 2.1 — Users and authentication

- `users` table; bcrypt via `passlib`, added to the base install.
- Login, logout, and a signed HTTP-only session cookie.
- A first `platform_admin` created by a CLI command, not a seeded default
  password.

### Sprint 2.2 — The guard

- One dependency resolving either a session cookie or the shared bearer token,
  the latter treated as `platform_admin` for machine-to-machine callers.
- Role enforcement, and the role matrix from the spec asserted as a table-driven
  test rather than one test per endpoint.
- No endpoint may end up guarded by neither mechanism; a test walks the route
  table and fails on any unguarded `/api` or `/ws` path.

### Sprint 2.3 — Organisation scoping

- Every list and read endpoint filters by the caller's organisation.
- **Adversarial tests are the deliverable here**, not an afterthought: a
  supervisor of one organisation attempting another's calls, knowledge, agents,
  users and MIS — by direct id, by query parameter, and by crafted filter.

### Sprint 2.4 — Portal: login and user administration

- Login page; role-aware navigation that hides what the role cannot use.
- User management for `org_admin` within their organisation, and for
  `platform_admin` across all.

---

## Phase 3 — MIS and dashboard

**Delivers:** durable reporting, filterable by organisation, DID and date.

### Sprint 3.1 — Durable aggregates

- Move analytics off `CallManager`'s in-memory ring of 200 into SQL over `calls`
  and `turns`. Indexes on `(organisation_id, started_at)` and `(did, started_at)`.
- A regression test that figures survive a restart — the defect the current
  implementation has.

### Sprint 3.2 — The analytics API

- Volume, duration, outcomes, dispositions, languages, containment, transfer rate,
  and the latency split — caller-perceived wait separate from speech duration.
- Filters: organisation, DID, date range. Knowledge-gap mining becomes per
  organisation.

### Sprint 3.3 — The dashboard

- A console page: headline figures, a volume trend, outcome and disposition
  breakdowns, per-DID comparison, and the knowledge gaps list.
- Judged by whether an operator can answer "which line is failing callers this
  week" without exporting anything.

### Sprint 3.4 — Export

- CSV for call lists and summary figures, honouring the same scoping as the API.

---

## Phase 4 — Live transfer *(gated on Tata's answer)*

**Delivers:** a caller the bot cannot satisfy reaches a human.

### Sprint 4.1 — Destinations

- Transfer destination per DID, with an organisation-level default, in the portal.
- The bot must not promise a transfer it cannot make: with no destination
  configured it says so and offers a callback.

### Sprint 4.2 — The carrier instruction

- Send whatever Tata's answer specifies, and hang up our leg only once the
  carrier has accepted.
- Carrier refuses → the call stays with the bot, which apologises and offers a
  callback. Never a silent drop.

### Sprint 4.3 — Transfer in the record

- Transfer reason, destination and outcome on the call row; transfer rate and
  containment in MIS.

**If Tata cannot transfer a live call**, this phase becomes a single sprint: record
a callback request with the transcript, surface it in the portal, and drop 4.2.

---

## Phase 5 — Documentation

### Sprint 5.1 — Technical writeup and user manual

- **Technical writeup:** architecture, the tenancy model, the call path, the
  provider contracts, deployment, and the operational traps already learned —
  logs are UTC while the host is IST, a redeploy wipes `docker logs`, the DID
  must be registered Dynamic not Static.
- **User manual:** task-shaped, for an operator — add an organisation, attach a
  DID, train an agent, read the MIS, handle a transfer.

### Sprint 5.2 — FAQs

Drawn from what actually went wrong in this deployment, not invented: why a
caller heard silence, why a call rang and disconnected, why the bot talked over
someone, what the latency figures mean.

---

## Sequencing summary

```
Phase 1 ──▶ Phase 2 ──▶ Phase 3 ──▶ Phase 5
   │                                   ▲
   └── ask Tata ──▶ Phase 4 ───────────┘
```

Phase 5 is written last because documentation describing what was planned rather
than what shipped is worse than none.
