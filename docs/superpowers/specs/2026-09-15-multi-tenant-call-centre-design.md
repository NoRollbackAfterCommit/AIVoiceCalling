# Multi-tenant call centre: tenancy, access control, MIS, transfer

**Date:** 2026-09-15
**Status:** Approved for planning
**Supersedes nothing.** Builds on the live Smartflo deployment.

## Context

SamparkAI answers real calls on a single DID today. One `AgentProfile` handles
every one of them, anybody holding the shared bearer token is a full
administrator, analytics live in memory and are lost on restart, and a call the
bot cannot satisfy is hung up on rather than transferred.

The ask is to run several organisations' call centres from one deployment —
worked example: three DIDs for a Health University, two for a West Bengal
centralised portal — each with its own training, its own staff, and its own
reporting.

Much of the groundwork exists and is deliberately reused rather than rebuilt:

- `AgentProfile` already carries `organisation`, `policies`, `knowledge_guidelines`,
  `forbidden_topics`, `escalation_rules` and per-language `voices`. Profiles are
  DB-persisted with full CRUD.
- Retrieval is already namespaced per agent: `index_chunks(agent_key=...)` writes
  to a per-agent vector namespace, and search, upload and delete are all scoped.
- The dialled number already reaches us: `SmartfloEvent.called`, parsed from the
  carrier's `start` frame.
- The transfer *decision* path works end to end — a tool sets
  `control["action"] = "transfer"` and `_transfer()` hands over department, reason
  and the full transcript.
- Alembic migrations are in place (`vaani/db/migrate.py`).

## Goals

1. A call routes to the right organisation's agent based on the number dialled.
2. Staff of one organisation can see and configure only their own organisation.
3. Reporting is durable, and filterable by organisation, by DID and by date.
4. A call the bot cannot satisfy reaches a human instead of being dropped.
5. An operator can learn the system from written documentation.

## Non-goals

Outbound campaigns. Billing or per-tenant metering. Per-organisation branding of
the console. SSO or directory integration — local accounts only, behind an
interface that would admit one later. Supervisors taking call audio in the
browser: transfer goes through the carrier, not through us.

## Architecture

### The tenancy spine

```
organisations   id, slug, name, active, created_at
dids            number (PK, E.164), organisation_id, agent_key, label, active
agent_profiles  + organisation_id                     (table exists)
calls           + organisation_id (nullable), + did   (table exists)
users           id, email, password_hash, name, organisation_id, role,
                active, created_at, last_login_at
```

**Routing.** `start.called` is normalised to E.164 and looked up in `dids`,
yielding `(organisation_id, agent_key)`. `smartflo.py:320` currently hardcodes the
`smartflo_agent` setting for every call; that becomes the fallback, taken only
when the DID is unknown, and it logs at warning level. A real call is never
dropped over a configuration gap — the same judgement the handshake already makes
when `callId` is missing.

**Why the call row stamps both `organisation_id` and `did`** rather than joining
through `dids` at report time: DIDs get reassigned between organisations. A call
must stay attributed to whoever owned the number when it happened, or last
quarter's MIS silently rewrites itself the day a number moves.

**`AgentProfile.organisation` and `agent_profiles.organisation_id` are different
things and both stay.** The existing free-text field is what the bot *says* — "you
are the voice assistant for the Health University" — and an operator edits it as
copy. The new foreign key is the tenancy link that decides who may edit that
profile and whose reports its calls appear in. Conflating them would mean renaming
an organisation in the portal silently re-homed its data, so the column keeps its
name and the key is added beside it.

**Knowledge needs no schema change.** It is already namespaced by `agent_key`, and
an agent belongs to exactly one organisation, so isolation is transitive. An
organisation with several lines — admissions and examinations, say — has several
agents, each with its own corpus, which is why `dids` maps to an agent rather than
only to an organisation.

### Access control

Local accounts: email, a hashed password, a role, and an organisation. A NULL
`organisation_id` marks a platform administrator.

| Role | Scope |
|---|---|
| `platform_admin` | Everything, across all organisations. Owns `/settings`. |
| `org_admin` | Their organisation: agents, knowledge, DIDs, users, MIS. |
| `supervisor` | Their organisation's live calls, hang-up, MIS. No configuration. |
| `viewer` | Their organisation's MIS, read only. |

The shared bearer token is **not** removed. It stays as the machine-to-machine
credential — health checks, the deploy script, the simulated-carrier harness —
and is treated as `platform_admin`. Human sessions ride a signed, HTTP-only
cookie. The two coexist in one dependency so that no endpoint can accidentally
be guarded by neither.

Password hashing is bcrypt via `passlib`, the one new runtime dependency, added to
the base install because authentication is not optional in any tier.

**Every list and read endpoint gains an organisation filter derived from the
caller's identity, never from a query parameter.** A supervisor asking for
`/api/calls?organisation_id=other` must get their own organisation's calls, not
somebody else's — the parameter is ignored for anyone but a platform admin.

### MIS

Today `/api/analytics/summary` reads `CallManager`'s in-memory ring of the last
200 calls, so it is wrong after a restart and cannot be filtered. It moves to the
repository, with aggregates computed in SQL over `calls` and `turns`, indexed on
`(organisation_id, started_at)` and `(did, started_at)`.

Reported, per organisation and per DID, over a date range: call volume, average
and median duration, outcomes, dispositions, languages, containment (calls the bot
resolved without transfer), transfer rate, and the latency split the `/calls`
console already shows — caller-perceived wait separated from speech duration,
because `total_ms` is turn wall-clock and reads as latency it is not.

Knowledge-gap mining survives the move: the questions that produced no useful
retrieval are the highest-value input to the next round of content authoring, and
they become per-organisation.

### Transfer

`_transfer()` currently emits a `transfer` event and hangs up. For Smartflo,
`send_event` only logs it, so **the caller is dropped, not transferred**.

Each DID gains a transfer destination — a number or agent extension — with an
organisation-level default. On transfer the session instructs the carrier to
hand the live call over, and only hangs up our leg once the carrier has accepted.

**External dependency.** The exact instruction Smartflo accepts is not documented
to us. Tata must confirm whether the Voice Bot can transfer a live call and what
it expects — a JSON event on the media socket, or a REST call to a Smartflo API.
Phase 4 cannot be built without that answer, and asking for it is a Phase 1
action so it does not sit on the critical path. If the answer is that no live
transfer exists, Phase 4 degrades to a recorded callback request carrying the
transcript, which is a much smaller piece of work.

## Data flow, one call

```
Smartflo start frame
  └─ called = +918065605873
       └─ dids lookup  → (organisation "Health University", agent "health-admissions")
            ├─ CallSession(agent_key="health-admissions")
            │    └─ retrieval namespace "health-admissions"   (already scoped)
            └─ CallRow(organisation_id=…, did="+918065605873")
                 └─ MIS filters, and every read a logged-in user makes
```

## Error handling

- **Unknown DID:** fall back to the configured default agent, log a warning, stamp
  the call with a NULL organisation so it surfaces in MIS as unattributed rather
  than being silently assigned to somebody.
- **Inactive organisation or DID:** treated as unknown, same fallback.
- **A user whose organisation was deactivated:** login refused, session invalidated
  at the next request.
- **Transfer destination unset:** the bot must not promise a transfer it cannot
  make. It says so and offers a callback instead.
- **Carrier refuses the transfer:** the call stays with the bot, which apologises
  and offers a callback. Never a silent drop.

## Testing

Everything runs against mock providers, GPU-free, in the existing suite.

- Routing: a call to each of several DIDs reaches the right agent; an unknown DID
  falls back and is recorded unattributed.
- Isolation, which is the security surface and gets adversarial tests: a
  supervisor of organisation A cannot read A′'s calls, knowledge, agents, users or
  MIS — by direct id, by query parameter, and by crafted filter.
- Roles: each role's permitted and forbidden actions, asserted as a matrix rather
  than one test per endpoint.
- MIS: aggregates over a seeded corpus of calls match hand-computed figures, and
  survive a restart — the defect the current in-memory implementation has.
- Transfer: destination resolution, the carrier instruction, and the refusal path.

## Rollout

Existing data is migrated into a single organisation owning today's DID and
agent, so the live deployment keeps working through every phase. Each phase is
independently deployable, and phase 1 alone delivers the business ask: five DIDs,
two organisations, correctly separated.

## Roadmap

| Phase | Sprints | Delivers |
|---|---|---|
| 1 · Tenancy & DID routing | 1 | Organisations, DIDs, routing, call attribution, portal CRUD |
| 2 · Access control | 1.5 | Users, roles, login, org scoping on every read |
| 3 · MIS & dashboard | 1.5 | Durable filtered analytics and the dashboard |
| 4 · Live transfer | 1 | Carrier transfer to a supervisor *(blocked on Tata)* |
| 5 · Documentation | 0.5 | Technical writeup, user manual, FAQs |
