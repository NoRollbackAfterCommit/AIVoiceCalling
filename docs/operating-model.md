# Vaani — Build Operating Model (one human + Claude + agents)

**Written:** 2026-09-07, adapted from the TRINETRA BMS operating model (revision of
2026-08-21) to this repository, to the agents and skills it actually has, and to a
session that runs Fable 5.1 rather than Opus.
**Purpose:** the standing playbook for *how* the remaining roadmap gets built when the
team is one human (decisions, review) plus Claude (execution) plus subagents (bounded
parallel work). It answers who does what, on which model, and when to fan out.
**Companions:** [CLAUDE.md](../CLAUDE.md) (the invariants) ·
[specs](superpowers/specs/) (the *what*, per phase) · [plans](superpowers/plans/) (the
*how*, per phase) · the `pending-next-step` memory (where the last session stopped, read
first on "continue").

---

## 1. The core principle

With one human we do not get calendar parallelism. We get Claude's. Subagents multiply
execution; they cannot multiply the two things only the human provides:

1. **Decisions** — scope, phase order, provider and dependency choices, interface shape,
   anything touching credentials, telephony exposure or citizen data.
2. **Review and merge approval** — the final gate.

**The bottleneck is human decision-and-review bandwidth, not coding hands.** Every rule
below spends that bandwidth sparingly and lets agents absorb the rest. The roadmap's
dependency order still governs; parallelism never skips an enabler.

A second budget sits underneath: **tokens**. Two levers, in order of size:

- **Rate** — which model runs a unit of work (§2, §4).
- **Volume** — how much context each turn re-sends (§7). Volume multiplies rate, so it is
  the larger lever.

Nothing here trades quality for spend. Each rule removes work that was redundant or
routes it to the cheapest model that can carry it without a second pass.

---

## 2. The session model is the largest single cost

This is the one place the source document had to be inverted. It assumed an Opus session
and named the two steps that had to *climb* to a stronger model. This session runs
**Fable 5.1**, the premium tier, and every turn re-sends the whole conversation at that
rate. Read the old rule against this session and the entire build, including README
edits and migrations, lands on the most expensive model available. That is what happened
on 2026-09-07: two features, both built inline on Fable, mechanical units included.

The rule is therefore:

> **The model is a property of the work unit, not of the session.** Every step names its
> model. Every `Agent` dispatch carries an explicit `model:`. The session itself runs the
> cheapest model that can hold the dialogue.

| Model | Runs | How it gets there |
|---|---|---|
| **Fable 5.1** | Step 3 plans; step 4 units that define a seam, span several plan tasks, or touch an auth, telephony-exposure or settings-metadata surface in production code | Delegated (`model: "fable"` on the `Plan` agent), or the operator flips `/model fable` for a seam that is too entangled to hand off, and flips back after |
| **Opus** | Step 2 brainstorm and spec; ordinary multi-file feature work with judgment in the execution; step 5 reviews; root-cause debugging that survived one pass | The recommended **session model for build cycles**; also the `implementer` agent's default pin |
| **Sonnet** | Mechanical, well-specified units: a migration with its test, a `cfg()` knob, a doc, `.env.example`, a console script against a written spec; step 6 evidence gathering; `test-writer` | Delegated with `model: "sonnet"` |
| **Haiku** | Locating a file, grepping a symbol, reading one config, summarising one file, confirming the next roadmap item is unblocked | Delegated with `model: "haiku"`, or done inline with `Grep`/`Glob` when it is one call |

**Recommendation for the operator:** run build sessions on Opus (`/model opus`). Reach
Fable by delegation for the plan and for seam units, not by keeping the whole session on
it. The rate difference on one read-only plan is smaller than one review-fix loop it
prevents; the rate difference on two hundred build turns is not.

**Never let a dispatch inherit.** `Explore`, `Plan`, `general-purpose` and `claude`
declare no model of their own; an unpinned dispatch runs on whatever the session is set
to, which on a Fable session is the expensive direction, silently. `subagent_type:
"fork"` ignores `model:` and always runs the parent model with the parent's whole
context, so it is not a build mechanism either.

**Agents in this repository and their pins** (frontmatter, so the pin survives a `/model`
flip):

| Agent | Model | Use |
|---|---|---|
| `implementer` | `opus` (override per unit by the ladder in §4) | Step 4 builds from an approved plan; TDD; returns a summary |
| `test-writer` | `sonnet` | Coverage for a module with a written brief; mocks only |
| `realtime-audit` | `opus` | Step 5 for anything under `vaani/pipeline`, `vaani/audio`, `vaani/api/ws_voice.py`, or a provider's call-path methods. A review never routes down |

**A skill has no model of its own.** `superpowers:*`, `project-conventions`,
`add-provider`, `code-review`, `security-review`, `simplify` all run on the session model
when invoked inline. That is fine for the dialogue steps and wrong for the build step,
which is why step 4 delegates.

---

## 3. The per-feature loop

Every roadmap item goes through the same cycle. The human owns **steps 2 and 7**.

| Step | Who | Model | Skill / tool | Human touch |
|---|---|---|---|---|
| 1. **Pick** the next unblocked item | Claude | Haiku | `pending-next-step` memory, spec roadmap (phase order), `git log` | — |
| 2. **Brainstorm** — classify bounded vs architectural, resolve the one or two decisions that change the work, write the spec for architectural items | Human + Claude | Opus, inline | `superpowers:brainstorming`; spec to `docs/superpowers/specs/` | ✅ **gate** |
| 3. **Plan** — written, reviewable, with the model named per unit | Claude | Fable, delegated | `Plan` agent with `model: "fable"`, or `superpowers:writing-plans` inline on a flipped session; plan to `docs/superpowers/plans/` | 👀 skim |
| 4. **Build via TDD** | Claude + `implementer` | Per unit (§4) | `implementer` with `model:` per unit; `superpowers:test-driven-development`; the ruff hook formats every write | — |
| 5. **Review** | Subagents, one message | Opus | `/code-review` at `high` for auth, telephony, settings, persistence; `medium` otherwise · `/security-review` when the diff touches auth, the settings API, telephony or citizen data · `realtime-audit` when it touches the call path | 👀 batched |
| 6. **Verify against the running system** | Claude | Sonnet for evidence, session model for reading it | §6 | — |
| 7. **Approve, commit, push** | Human | — | Commit on request; push only on request; update `pending-next-step` | ✅ **gate** |

Bounded items (a knob, an endpoint, a one-file fix, the current profile-persistence
and token-guard work) skip the written spec and the written plan: the design is a few
paragraphs in chat, approved with one question, and the plan is the unit list with a
model per unit stated in that same message. Architectural items (a new transport, a new
phase, anything that reshapes `CallSession` or the provider protocols) take the full
spec → plan path.

### Step 4 is delegated only when the unit pays for the cold start

A subagent starts cold. One that re-derives context the parent already holds costs more
than staying inline and produces worse seams on coupled work. Delegate a unit when
**all** hold:

- An approved plan describes it, so the agent does not re-derive intent.
- It is self-contained in files no other in-flight unit touches.
- It returns a **summary** — files changed, tests added, what failed — not a transcript.

Otherwise build inline. The serial spine (§5) is always inline.

### Step 6 is not the test suite again

The suite says the code is right. It cannot say what is running or what an operator
sees. On 2026-09-07 the suite passed while GET on any agent profile had never worked;
the boot smoke found it. Verify the layers the change touches and record which were
**N/A** so a reader can tell "not applicable" from "not checked".

---

## 4. The model ladder for step 4, per unit

| Unit looks like | Model | Vaani examples |
|---|---|---|
| Defines a seam other work hangs off; spans several plan tasks; touches auth, telephony exposure or settings metadata in production code | **Fable** | A new `Transport`; a change to the four provider `Protocol`s or to `registry.py`; the `CallSession` state machine; an auth middleware; the AudioSocket bridge; anything in `config.py` that changes how `cfg()` metadata is interpreted |
| Ordinary multi-file feature work with judgment in the execution | **Opus** | A new provider behind an existing protocol; a repository with its tests; a language-tracking change; an outcome-engine rule |
| Well-specified, self-contained, mechanical | **Sonnet** | An Alembic migration with its test; one `cfg()` knob plus `.env.example`; a console script against a written spec; a doc page; a rename; a fixture sweep; a `--flag` on a script |
| A lookup with a short verifiable answer | **Haiku** | Which file registers a tool; what a setting defaults to; whether a test exists for a module |

Say the reason in the dispatch so the choice is reviewable. The stronger model is not
faster per token; it is chosen so that fewer passes are needed.

---

## 5. When to fan out, and the serial spine

**Fan out** — independent, well-specified, non-overlapping in files:

- Sibling providers once the protocol is frozen: two STT or two TTS providers in
  parallel, each in its own file under `vaani/providers/`, each with its own worktree
  (`isolation: "worktree"`).
- Read-only exploration alongside a build (`Explore`, pinned Haiku or Sonnet).
- The step-5 review set, in one message. They are read-only and never collide.
- The browser half of step 6. This fan-out *saves* tokens: a screenshot costs roughly
  two thousand tokens per call inside whichever context takes it. Run the console check
  inside a Sonnet agent with the Playwright tools and have it return sentences, not
  images.

**Do not fan out** — the serial spine, built inline one at a time:

- `vaani/pipeline/session.py`, `vaani/core/registry.py`, `vaani/config.py` metadata,
  `vaani/providers/base.py`, `vaani/telephony/audiosocket.py`. These define what
  everything else hangs off.
- Scope, security and telephony-exposure decisions.
- Anything two agents would edit the same file for.

Only spawn when the fan-out is real: two or more genuinely independent units. Splitting
one coupled feature across agents costs more and produces worse seams.

**Isolation:** one feature, one branch, merge at step 7. Worktrees only for parallel
siblings; for a single line of work they are overhead. Dependent work never runs in
parallel worktrees; if B needs A's interface, A merges first.

---

## 6. Verification layers for Vaani

| Layer | How | When |
|---|---|---|
| Unit and integration | `.venv/Scripts/python.exe -m pytest` (mocks, no GPU, about 25 s) and `ruff check` | Every unit; the hook already formats on write |
| Boot | Start the app twice on a scratch database with mock providers and assert across the restart (the pattern in the 2026-09-07 profile-persistence work) | Anything touching the lifespan, settings, persistence |
| Media plane | `scripts/smoke_call.py` against a running server | Anything touching the pipeline, VAD, transports |
| Telephony | A softphone registered against Asterisk, dialplan pointed at `vaani-inbound` (docs/telephony.md, "Testing before the trunk exists") | Anything under `vaani/telephony/` or the announce route |
| Hosted providers | One real call with the Sarvam and LLM keys set, latency read from the turn metrics | A provider change; before a pilot |
| Console | A Sonnet agent with the Playwright tools, returning findings as text | Anything under `vaani/web/static/` |

Check both directions: the defect is gone, and the fix does not fire when it should not.
Record N/A layers explicitly in the closing message.

---

## 7. Token discipline

§2 decides the rate. This decides the volume, which is the larger lever.

1. **Batch independent tool calls in one message.** A read, a grep and a test run that
   do not depend on each other go out together. Each separate round trip re-sends the
   context.
2. **Read the slice, not the file.** `Grep` with context lines, `sed -n a,b p`, and
   `Read` with an offset. This repository has no code graph; targeted reads are the
   substitute. Never `cat` a file to find one function.
3. **Do not verify an edit by re-reading it.** A successful `Edit` or `Write` wrote the
   change. The hook reformats it; the test run proves it.
4. **Subagents return conclusions.** Ask for the verdict and the `file:line` anchors.
   Never ask an agent to print a file back. Never let one return a screenshot.
5. **Scope reviewers to the diff.** Give `/code-review` the effort level the surface
   deserves (`low`/`medium` for a knob or a doc, `high` for auth, telephony, settings,
   persistence). "Review the branch" invites a repository read.
6. **Fan the review set out in one message** so the results land in one batched read.
7. **Start each item clean.** `/clear` between items. The hand-off is on disk: the spec,
   the plan, the commit, and the `pending-next-step` memory. Carrying one feature's
   context into the next re-sends it on every remaining turn.
8. **Keep the premium stretches short and deliberate.** Brainstorm and plan are worth
   their rate; the build that follows is not. Routing step 4 down saves more than any
   prompt-level economy, because it applies to every turn of the longest step.
9. **Do not re-derive what memory records.** The `pending-next-step` memory exists so a
   resumed session does not spend its first minutes reconstructing state from `git log`.
   Keep it current at the end of every session.

---

## 8. Realistic cadence

One feature, or one small batch of parallel siblings, per cycle. The roadmap says what is
eligible; the mock suite lets agents move fast without the human reading every line;
human time concentrates on the phase decisions, the telephony and security surfaces, and
merge approval.

---

## 9. Starting a cycle — the checklist

```
[ ] 1. Confirm the next item is unblocked (pending-next-step, spec roadmap).     [Haiku]
[ ] 2. Brainstorm: bounded or architectural; resolve the decisions that change    [Opus, inline]
       the work; spec if architectural. Human approves.
[ ] 3. Plan with a model named per unit. Human skims.                             [Fable, delegated]
[ ] 4. TDD build. Delegate self-contained units to implementer with model: per    [Fable/Opus/Sonnet]
       the ladder; build the seam inline. Worktrees only for independent siblings.
[ ] 5. /code-review at the effort the surface deserves, /security-review when     [Opus]
       auth/telephony/settings/citizen data, realtime-audit when the call path.
       One message, scoped to the diff.
[ ] 6. Verify the running system. Record which layers were N/A.                   [Sonnet evidence]
[ ] 7. Human approves. Commit; push on request. Update pending-next-step. /clear.
```

---

## 10. Applied now: the API token guard

Classified **bounded**: the API, the console and both WebSockets exist; the change adds
one setting and one gate. Decision resolved 2026-09-07: refuse to boot in `env=prod`
without a token; warn in `dev` and `staging`; the empty token keeps the laptop demo
zero-config.

| Unit | Files | Model | Why |
|---|---|---|---|
| A. Gate | `vaani/api/auth.py` (ASGI middleware for `http` and `websocket` scopes), `api_token` in `config.py` (`secret=True`, `restart=False`), boot check in `main.py`, `tests/test_api_auth.py` | **Fable, inline** | An auth surface and a seam every route sits behind; the session is already on Fable, so no flip is needed |
| B. Console | `vaani/web/static/auth.js` (prompt once, keep in localStorage, attach `Authorization` to same-origin `/api` fetches, `?token=` to the call WebSocket, re-prompt on 401), one `<script>` line in each of the three pages | **Sonnet, delegated** | Self-contained files, a written spec, a cheap-to-spot failure |
| C. Operator surface | `.env.example`, README limitations line, `docs/telephony.md` firewall row, `scripts/smoke_call.py --token` | **Sonnet, delegated, same dispatch as B** | Mechanical; one cold start covers both |
| Review | `/code-review high` and `/security-review` on the diff, one message | **Opus** | Auth |
| Verify | pytest and ruff; boot smoke with a token set (401 without, 200 with, health open, announce open); console check by a Sonnet agent with Playwright. Telephony layer N/A: the announce route is exempt by design and its tests already cover it | — | |

---

## Appendix: what changed from the TRINETRA version

- The session runs Fable, not Opus, so §2 is inverted: the recommendation is to run the
  session down and reach Fable by delegation, and every pin below the session is the
  saving rather than the cost.
- No ADR process, `BACKLOG.md`, `.codegraph/`, or Docker verification stack exists here.
  The equivalents are the phase specs and plans under `docs/superpowers/`, the
  `pending-next-step` memory, targeted `Grep`/`sed` reads, and the verification table in
  §6 (pytest, boot smoke, `smoke_call.py`, softphone, Playwright).
- The named agents (`plan-architect`, `code-reviewer`, `security-reviewer`,
  `agents-compliance-reviewer`, `migration-reviewer`, `browser-verifier`) do not exist
  here. The step-5 set is the built-in `/code-review` and `/security-review` skills plus
  `realtime-audit`; the plan is the built-in `Plan` agent pinned to Fable on the
  dispatch; the console check is a general-purpose Sonnet agent with the Playwright
  tools. An `implementer` agent was added for step 4, and `test-writer` and
  `realtime-audit` were given pins, because an unpinned project agent inherited Fable.
- Phase 0 ("build the safety net first") is already done here: the mock suite, the ruff
  hook and migrations at boot exist. It is replaced by the standing rule that the
  serial spine files are never delegated.
