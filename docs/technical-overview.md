# SamparkAI — technical overview

For an engineer taking this on. It describes what is deployed, not what was
planned; where the two differed during the build, the reasoning is in the commit
messages and in `docs/superpowers/specs/`.

## What it is

A self-hosted voice agent platform. A citizen dials a phone number, a carrier
streams the audio to this application over a WebSocket, and the application
listens, transcribes, answers from a document corpus, and speaks back — in the
caller's own language. One deployment serves several organisations, separated by
the number dialled.

Python 3.11+, FastAPI on uvicorn, SQLite (or Postgres) through SQLAlchemy, no
front-end build step. The whole test suite runs against mock providers in about
two minutes with no GPU and no network.

## The call path

```
carrier ──ws──▶ SmartfloTransport ──▶ CallSession ──▶ Transport ──ws──▶ carrier
                (mu-law 8 kHz)        │                (mu-law 8 kHz)
                                      ▼
                    VAD ─▶ STT ─▶ agent (LLM + RAG + tools) ─▶ TTS
```

`CallSession` (`vaani/pipeline/session.py`) is the state machine and the actual
product:

```
GREETING ─▶ LISTENING ─▶ THINKING ─▶ SPEAKING ─┐
                ▲                        │     │
                └────────────────────────┘     ▼
                    (barge-in cancels)       ENDED
```

Two properties are load-bearing and easy to break:

**Barge-in.** Playback runs as a cancellable task. When the caller talks over the
agent for `barge_in_ms` of continuous speech, that task is cancelled and buffered
audio is dropped. Anything coarser and interrupting does nothing, which callers
find infuriating. Anything finer and an echo cuts the agent off mid-word — see
`docs/call-quality.md`, which is the single most useful document here if calls
sound wrong.

**Pacing.** TTS renders far faster than real time, so audio is sent throttled to
its own duration, keeping `playout_lead_ms` of slack. Without it the session
would leave SPEAKING while the caller is still listening to buffered audio, and
anything they said during that window would be treated as a new turn rather than
an interruption.

### Audio format

PCM16 little-endian, mono, 16 kHz everywhere inside the pipeline. 20 ms frames =
320 samples = 640 bytes. Telephony mu-law at 8 kHz converts at the edge, in
`vaani/telephony/`, and nowhere else. A provider emitting another rate resamples
inside itself.

## Tenancy: who answers which number

```
organisations   id, slug, name, active
dids            number (E.164, primary key), organisation_id, agent_key, label, active
agent_profiles  key, payload (JSON), organisation_id
calls           … organisation_id (nullable), did
users           id, email, password_hash, role, organisation_id, active
```

A call arrives, `vaani/telephony/routing.py` normalises the dialled number to
E.164 and looks it up. That yields the organisation and the agent profile that
answers, and both travel with the call as far as `ToolContext`.

Knowledge is partitioned by that pair. A vector-store namespace is either an
agent key or `org:<id>`, and a lookup reads both — the line's own documents and
its organisation's shared set, ranked as one result. Two queries rather than
one because the store partitions by a single namespace; the alternative is
scanning every customer's corpus and filtering afterwards.

    DID ─▶ route_call ─▶ (organisation_id, agent_key) ─▶ ToolContext
                                                            │
                          search_knowledge ─▶ namespaces: agent_key + org:<id>

Isolation is therefore structural rather than a filter somebody has to
remember. An agent with no organisation — the fallback on an unmapped number —
reads only its own namespace, because an unconfigured number belongs to no
customer and must not fall into one's documents. Agent keys are validated to
`[a-z0-9][a-z0-9_-]*` so none can ever spell `org:<id>`.

Three decisions worth understanding before changing any of it:

- **An unmapped number is still answered.** It gets the fallback agent from
  Settings and is recorded with a null organisation. A configuration gap must
  never cost a real caller their call; the unattributed row is how an operator
  finds out.
- **The call row stores the organisation and the number**, rather than joining
  through `dids` at report time. Numbers get reassigned between customers, and a
  call must stay attributed to whoever owned it when it happened.
- **`AgentProfile.organisation` and `organisation_id` are different things.** The
  first is what the bot says out loud and is edited as copy. The second decides
  ownership. Renaming one must not move the other.

## Access control

Two credentials reach the guard (`vaani/api/auth.py`), for two kinds of caller.

A **person** signs in at `/login` and carries a signed session cookie. A
**machine** — the deploy script, a health probe, the simulated carrier — presents
the shared bearer token and is treated as a platform administrator, because it
has no account to carry a narrower role.

| Role | May |
|---|---|
| `platform_admin` | everything, across every organisation, plus deployment settings |
| `org_admin` | their organisation: agents, knowledge, people. Reads its numbers; only a platform admin maps one |
| `supervisor` | their organisation's live calls, hang-up, reports |
| `viewer` | their organisation's reports, read only |

Roles are enforced in the guard against a table of path prefixes
(`required_roles` in `vaani/api/accounts.py`), not with a decorator per endpoint.
A permission model spread over fifty call sites is one nobody can audit, and a
new endpoint inherits the nearest prefix instead of defaulting to open.

Scope comes from **who you are, never from what you asked for**. An
`organisation_id` in a query string is honoured only for a platform
administrator. Refusals are `404`, not `403`: telling a supervisor at one
customer that another's call exists but is forbidden discloses that the customer
exists and which ids are real.

Passwords use `hashlib.scrypt` from the standard library rather than bcrypt
behind passlib — an air-gapped install should not depend on a package being
available, and the stored form carries its own cost parameters so they can be
raised later without invalidating existing hashes. Sessions are stateless and
signed, but the account is reloaded on every request, which is what makes
suspending someone take effect on their next click.

**The console never uses the shared token.** It did once, and a browser holding
one bypassed the sign-in screen entirely. Any token found in browser storage is
discarded on load.

## Providers

Four contracts in `vaani/providers/base.py`: STT, LLM, TTS, Embedding. The only
place a configuration string becomes a class is `build_*` in
`vaani/core/registry.py`, and **every concrete provider is imported inside its
branch**, never at module scope. That is what makes the optional dependency
groups real: an on-premise install that never installed `[cloud]` is
structurally incapable of calling an external API.

Currently deployed: Sarvam `saaras:v3` for recognition, OpenAI `gpt-4o-mini` for
the model, Sarvam `bulbul:v3` for speech. Sarvam is there because Western voices
cannot pronounce Indian languages, and an English-accented model reading
Devanagari is unusable on a citizen helpline.

## Reporting

`vaani/db/analytics.py` computes aggregates in SQL over the stored calls — not in
Python, because "how many calls last quarter" must not become a hundred thousand
row objects on the event loop the live calls share.

One figure is routinely misread and the dashboard states it plainly. **`total_ms`
is turn wall-clock**: it wraps `_speak()`, which returns only once the reply has
played out at real time. So it includes the agent talking. What a caller waits
for is `stt_ms + agent_ms + tts_first_chunk_ms`, and the remainder is speech.
Measured over 86 real turns: 2,647 ms waited, 6,005 ms spoken. Reporting the
total as latency overstates it roughly threefold.

## Deployment

```bash
deploy/samparkai/deploy.sh ubuntu@<address> EuphoriaKey.pem
```

Ships `git archive HEAD` — no virtualenv, no data, no keys — builds the image on
the box, and restarts the container. The database and `.env` survive. Alembic
migrations run at boot, so an upgraded deployment migrates itself.

The app listens on 8080 in the container, published to `127.0.0.1:8091`, fronted
by the host's Caddy which also serves other sites on the same VM. Settings saved
in the portal live in `data/settings.json` and are layered over the environment
at boot, so what boots is what the operator last saved.

## Things that have actually bitten us

- **A redeploy recreates the container, which wipes `docker logs`.** Never deploy
  while someone is mid-test; a call's only evidence was lost that way.
- **The host clock is IST; the application logs in UTC.** `journalctl` and
  `docker inspect` read IST, the app's JSON logs read `+0000`. Reading them side
  by side misleads by five and a half hours.
- **The Smartflo endpoint must be registered Dynamic, not Static.** Static makes
  the carrier open a WebSocket against the HTTP handshake path, which refuses it.
- **`$callId`-style variables substitute only from the dashboard's key/value
  parameter table**, not typed inline into the URL.
- **Sarvam rate-limits.** Thirty simultaneous calls were clean; fifty drew
  twenty-one `429`s, and each one left a caller on a live line in silence. The
  TTS client now retries twice with backoff, and a reply that still produces no
  audio emits `reply_silent`, which the live console shows. Our own box was never
  the constraint — 0.13% CPU throughout.
- **`data/` is gitignored**, including the database and recordings. Backups are
  not automatic.

## Where to look

| Question | File |
|---|---|
| Calls sound wrong — talking over, silence | `docs/call-quality.md` |
| Carrier setup, Smartflo specifics | `docs/telephony.md` |
| How an operator does a thing | `docs/user-manual.md` |
| Why does it do that | `docs/faq.md` |
| Invariants and conventions | `CLAUDE.md`, `.claude/skills/project-conventions/` |
| Why a design is the way it is | `docs/superpowers/specs/` |
