# Phase 1 — Live Indic inbound voice slice

**Date:** 2026-08-08
**Status:** Approved for planning
**Phase:** 1 of 6 (see *Roadmap* at the end)

## Context

Vaani already has the call state machine (`pipeline/session.py`) with barge-in and
per-stage latency accounting, a browser WebSocket transport (`api/ws_voice.py`), an
Asterisk AudioSocket bridge (`telephony/audiosocket.py`), RAG with chunking and a
Qdrant/memory store, `AgentProfile` as editable per-tenant data, and a runtime
settings store with an admin API.

What it does not have: a carrier connection, real Indic speech providers, and any
persistence at all — `vaani/db/` is an empty package.

This phase closes exactly those three gaps and nothing else.

**Deployment context:** government citizen helpline, pilot scale (<10 concurrent),
inbound first. Launch languages are Hindi, English, and Hinglish code-mixing;
Bengali, Marathi, Gujarati, Punjabi, and Odia follow in phase 5.

## Goals

One Indian phone number that answers a real inbound call, converses in Hindi,
English, and code-mixed Hinglish, answers from a real document corpus, closes the
call rather than looping, and writes a durable call record.

## Non-goals

Outbound calling, campaigns, DLT compliance, the admin UI, multi-tenancy, Postgres,
Redis, the remaining five languages, and fine-tuning of any model. Each has its own
phase.

## Architecture

Three transports, one session. Nothing below the `Transport` protocol changes.

```
Caller → PSTN → Exotel (Mumbai PoP — required for IP-PSTN intermixing)
                  ├─ WebSocket stream ──→ ExotelTransport ──┐
                  └─ vSIP → Asterisk → AudioSocket ─────────┤
Browser test console ──→ /ws/call → WSTransport ────────────┼──→ CallSession
                                                             │    (unchanged)
        TurnDetector → Saarika STT → Agent(LLM + RAG) → Bulbul TTS
                                                             │
                                                             └──→ CallRepository
                                                                       │
                                                                    SQLite
```

**Telephony decision.** Exotel offers a WebSocket streaming applet, vSIP SIP trunking,
and a managed StreamKit connector. The pilot uses the WebSocket path because it
removes Asterisk from the deployment entirely, and because `api/ws_voice.py` is
already a working WebSocket transport that `ExotelTransport` can closely follow.
AudioSocket and Asterisk remain in the tree as the on-premise path for when the
deployment moves in-house; both terminate at the same protocol, so neither choice
is load-bearing on the rest of the system.

## Components

### `vaani/providers/stt/sarvam.py` — `SarvamSTT`

Implements `STTProvider`. Wraps Saarika (`saaras:v3`). REST transcription per
completed utterance to start; the WebSocket streaming API is a later optimisation,
not a phase-1 requirement, because `TurnDetector` already segments utterances.

Returns `Transcript.language` populated from the API response. The `sarvam_sdk`
import lives inside `start()`, per the provider laziness invariant.

### `vaani/providers/tts/sarvam.py` — `SarvamTTS`

Implements `TTSProvider`. Wraps Bulbul v3. Uses the streaming endpoint so the first
audio chunk leaves before the full utterance is synthesised — time-to-first-audio is
the metric that matters, not total synthesis time.

Resamples Bulbul output to `SAMPLE_RATE` inside the provider. The pipeline sees one
format only.

### `vaani/api/exotel_ws.py` — `ExotelTransport`

Implements `Transport`. Terminates Exotel's WebSocket media stream, converts its
G.711 payload to PCM16 at `SAMPLE_RATE`, and hands frames to `CallSession`. Codec
conversion happens here and nowhere else, matching the rule already applied in
`telephony/audiosocket.py`.

Carries the caller's number and Exotel call identifier into `CallRecord`.

### `vaani/db/models.py`, `vaani/db/repository.py`

Two tables:

- **`calls`** — `call_id` (PK), `agent_key`, `direction`, `caller_number`,
  `started_at`, `ended_at`, `outcome`, `summary`, `language`, `duration_s`,
  `recording_path`
- **`turns`** — `id`, `call_id` (FK), `seq`, `role`, `text`, `language`, `stt_ms`,
  `agent_ms`, `tts_first_chunk_ms`, `total_ms`, `barged_in`

`CallRepository` exposes `create_call`, `append_turn`, and `finish_call`, and is
reached through `Services`. SQLAlchemy 2.0 async over aiosqlite; both are already
declared dependencies. `create_all` at boot — no Alembic until Postgres arrives in
phase 3.

**Writes stay off the call path.** `append_turn` puts onto a queue with
`put_nowait`; a single background task drains it. SQLite serialises writers, and a
blocked writer inside the turn loop would stall every concurrent call.

## Language design

**No language routing is built.** Saarika handles mixed-language content natively and
covers all seven launch-and-later languages including `od-IN`. Detecting a language
and switching engines would be strictly worse for the code-mixed speech that most
Indian callers actually produce.

- `Transcript.language` already exists in the provider protocol; no change needed.
- `CallSession` holds a current language, seeded from the agent profile default.
- **Voice switching uses hysteresis:** the TTS voice changes only after two
  consecutive utterances in a new language. One mis-detect otherwise flips the agent
  into the wrong language mid-call, which is worse than lagging a real switch by a
  turn.
- **Hinglish is not a distinct language tag.** Saarika returns `hi-IN` for code-mixed
  Hindi-English and Bulbul's Hindi voices render embedded English words correctly.
  This works precisely because no forced choice is made.
- Voice selection becomes data on `AgentProfile`:
  `voices: {"hi-IN": ..., "en-IN": ..., "bn-IN": ...}`, consistent with the existing
  treatment of agent behaviour as data rather than code.

The LLM must reply in the caller's *script*, not merely the right language — a
Bengali answer in romanised Bengali sounds broken through TTS. `GROUNDING_RULES` and
`VOICE_RULES` already instruct this; phase 1 verifies it with real audio rather than
assuming it.

## Configuration

New `cfg()` fields in `config.py`, each with `group`, `label`, `help`, and
`depends_on` so the admin page renders them without hand-written markup:

- `stt_provider` / `tts_provider` gain a `sarvam` option
- `sarvam_api_key` (`secret=True`)
- `exotel_enabled`, `exotel_api_key` (`secret=True`), `exotel_account_sid`
- `retention_days` — declared now because a government deployment will be asked
  about it; enforcement lands in phase 3

New settings must be added to the matching `changed(...)` list in
`Services.reload()`, or the admin portal will appear to save them while the running
provider never rebuilds.

`pyproject.toml` gains a `sarvam` optional-dependency group. Base install stays
model-free and offline.

## Testing

The existing 23 mock-based tests remain the default suite: no GPU, no network,
about a second. That property is preserved.

1. **Provider contract tests** — `SarvamSTT` and `SarvamTTS` satisfy their protocols,
   exercised against recorded fixtures rather than the live API.
2. **Golden audio set** — 30–50 real recorded utterances across Hindi, English, and
   Hinglish with expected transcripts, behind a pytest marker since it needs network
   and an API key. Reports word error rate.
3. **Latency assertion** — p95 time-to-first-audio, computed from the `TurnMetrics`
   the session already records.
4. **Persistence test** — a call written mid-conversation survives process restart.

## Acceptance criteria

Phase 1 is complete when, on a real PSTN call:

- The number answers within two rings.
- p95 time-to-first-audio is ≤1.5s. Budget: ~300ms STT, ~400ms LLM first token,
  ~250ms TTS first chunk, ~150ms network.
- **Barge-in works over the carrier**, not only over a local WebSocket.
- A Hinglish utterance transcribes correctly at the agreed word error rate.
- The LLM replies in the caller's script.
- The agent closes the call via `end_call` rather than looping.
- The call record and its turns survive a process restart.

## Risks

| Risk | Impact | Response |
|---|---|---|
| Exotel Voicebot/Stream applet is documented as beta/alpha | Blocks the pilot path | Confirm availability and SLA before committing; vSIP + Asterisk is the fallback and is already built |
| Barge-in degrades under carrier jitter | Callers cannot interrupt — the defect users hate most | Test over PSTN early, not at the end; `playout_lead_ms` and `barge_in_ms` are already tunable |
| Scanned government PDFs yield poor Devanagari/Bengali/Odia text | RAG answers from garbage | Audit the real corpus before implementation; OCR becomes its own sub-project if needed |
| Azure OpenAI India access approval is slow | Blocks the LLM slot late | Start the access request in parallel with phase 1, or use Sarvam-M |
| LLM replies in the wrong script | Unintelligible speech output | Explicit acceptance test; informs the Sarvam-M vs Azure decision |

## Procurement

Required before a real call can be tested: an Exotel account on the Mumbai instance
with one DID or 1800 number and Voicebot/Stream applet access; a Sarvam API key
covering Saarika and Bulbul; an LLM endpoint (Sarvam-M, or Azure OpenAI in Central
India); a modest cloud VM in an Indian region, no GPU; and the actual document
corpus.

Not yet required: GPU hardware, DLT registration, Postgres, Redis, Asterisk hosting.

## Roadmap

1. **Live Indic inbound slice** — this document
2. **Outcome engine** — call objectives, disposition taxonomy, the policy that drives
   a conversation to a conclusive end
3. **Persistence, audit, multi-tenancy** — Postgres, retention enforcement, migrations
4. **Admin module** — domain configuration, knowledge ingestion, profile editing,
   evaluation dashboard
5. **Language expansion** — the remaining five languages against a measured test set
6. **Outbound and DLT compliance** — campaigns, pacing, DND scrubbing, consent trails
