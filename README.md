# Vaani — AI Voice Calling Agent Platform

Self-hosted, open-source platform for AI agents that answer and place telephone
calls. Speech in, reasoning over your own documents, speech out — with tool
calling into your existing systems and a human hand-off when the agent is out of
its depth.

Built to run **entirely on-premise**: no component calls out to the internet at
runtime, which is what makes it deployable in a government cloud.

**Status:** the conversation core is complete and tested end to end. Phases 1–3
of the roadmap are functional; see [Roadmap](#roadmap) for what is not built yet.

---

## Run it now

No GPU, no models, no infrastructure. The default providers are mocks that
exercise the real pipeline.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
uvicorn vaani.main:app --port 8080
```

Open <http://localhost:8080>, press **Start call**, and talk. The agent greets
you, transcribes, thinks, replies — and you can interrupt it mid-sentence.

Verify the media plane without a microphone:

```bash
python scripts/smoke_call.py
```

```
  PASS  greeting spoken          PASS  agent audio received
  PASS  caller transcribed       PASS  latency reported
  PASS  agent replied            PASS  barge-in fired
```

Talk to an agent in the terminal — the fastest loop for prompt and knowledge
work:

```bash
vaani chat
```

---

## Switching on the real AI

Every model is selected by one environment variable. Nothing else changes.

```bash
pip install -e ".[ai,rag]"
ollama pull qwen2.5:7b-instruct
bash scripts/fetch_voices.sh

export VAANI_STT_PROVIDER=faster_whisper   VAANI_STT_MODEL=small
export VAANI_LLM_PROVIDER=openai_compat    VAANI_LLM_MODEL=qwen2.5:7b-instruct
export VAANI_TTS_PROVIDER=piper            VAANI_TTS_VOICE=hi_IN-pratham-medium
export VAANI_EMBEDDING_PROVIDER=sentence_transformers
export VAANI_VECTOR_STORE=qdrant
```

`VAANI_LLM_BASE_URL` points at anything speaking the OpenAI chat-completions
protocol: **vLLM, Ollama, llama.cpp, TGI, SGLang**. Nothing reaches OpenAI
unless you aim it there.

Full stack in containers:

```bash
docker compose --profile ai up -d
```

---

## Architecture

```
  Phone ─ SIP ─▶ Asterisk ─ AudioSocket ─┐
                                         ├─▶  CallSession  ─▶  CallRecord
  Browser / mobile ─── WebSocket ────────┘        │
                                                  │
        ┌─────────────────────────────────────────┴──────────────────┐
        │                                                            │
    VAD + turn detection          Agent runtime            Streaming TTS
    (barge-in, end of turn)   (memory, tool calling)     (paced to real time)
              │                          │                          │
             STT                    ┌────┴────┐                    out
        (Whisper)                RAG          Tools
                            (Qdrant + BGE)  (CRM, ERP, SQL Server)
```

`CallSession` is transport-agnostic — the same object drives a browser call, a
SIP call and a WhatsApp voice call. Adding a channel means implementing three
methods (`send_audio`, `send_event`, `close`), not touching the pipeline.

### Where the interesting problems are

| File | What it solves |
|---|---|
| [`pipeline/session.py`](vaani/pipeline/session.py) | The state machine. Barge-in, real-time playout pacing, latency accounting, watchdogs. |
| [`audio/vad.py`](vaani/audio/vad.py) | When has the caller stopped talking. Gets this wrong and nothing else matters. |
| [`agent/runtime.py`](vaani/agent/runtime.py) | Bounded tool-calling loop with conversation memory. |
| [`agent/prompt.py`](vaani/agent/prompt.py) | Behaviour as data. Voice rules, grounding, escalation, injection guards. |
| [`rag/retriever.py`](vaani/rag/retriever.py) | Hybrid dense + lexical retrieval, source diversification. |
| [`providers/base.py`](vaani/providers/base.py) | The four protocols every model sits behind. |

### Three design decisions worth knowing

**Agent audio is paced to real time.** TTS renders far faster than speech plays.
Sending chunks as fast as they are produced pushes a whole utterance down the
wire in milliseconds, so the session leaves `SPEAKING` while the caller still has
five seconds of audio buffered — and anything they say lands as a *new turn*
instead of an interruption. Barge-in silently never fires. Playout is throttled
to the audio's own duration with 300 ms of slack.

**Turn processing runs off the consume loop.** A turn takes seconds. Running it
inline blocks the frame reader for that whole time, and a blocked reader is a
deaf agent — you cannot interrupt the reply you are listening to. The greeting
has the same problem and the same fix.

**Relevance thresholds belong to the embedding model.** A strong match scores
~0.75 with BGE and ~0.10 with a lexical hash. A hardcoded floor silently returns
nothing when you switch embedders, so each provider declares its own.

---

## Multi-tenancy

One deployment, many departments. An agent is a profile plus a knowledge
namespace plus a tool allowlist.

```bash
curl -X PUT localhost:8080/api/agents/hospital -H 'Content-Type: application/json' -d '{
  "key": "hospital",
  "name": "Asha",
  "organisation": "District Hospital",
  "languages": ["English", "Hindi", "Bengali"],
  "greeting": "Namaste, District Hospital helpline.",
  "policies": ["OPD registration closes at one in the afternoon."],
  "forbidden_topics": ["Medical diagnosis or treatment advice"],
  "tools": ["search_knowledge", "schedule_callback", "transfer_to_human"]
}'
```

Then `/ws/call?agent=hospital`. Knowledge indexed under one agent key is
invisible to the others.

---

## Tools

The model never *is* the business logic — it only decides which operation to
invoke. Each tool is a typed async function; deterministic code does the work and
talks to your CRM, ERP or SQL Server. Every invocation is logged with arguments
and result, so any call can be reconstructed.

Built in: `search_knowledge`, `transfer_to_human`, `end_call`,
`schedule_callback`, `verify_caller`, `check_bill`, `register_complaint`,
`check_application_status`.

```python
from vaani.agent.tools.builtin import registry
from vaani.agent.tools.base import ToolContext, ToolResult

@registry.tool(
    name="book_appointment",
    description="Book an OPD slot. Confirm the department and date with the caller first.",
    parameters={
        "type": "object",
        "properties": {
            "department": {"type": "string"},
            "date": {"type": "string", "description": "ISO date"},
        },
        "required": ["department", "date"],
    },
    requires_verification=True,   # caller identity must be confirmed first
)
async def book_appointment(department: str, date: str, ctx: ToolContext) -> ToolResult:
    token = await hospital_api.book(ctx.state["mobile_number"], department, date)
    return ToolResult(
        content=f"Booked. Token number {' '.join(token)}. Read it back slowly.",
        data={"token": token},
    )
```

---

## Telephony

Asterisk over AudioSocket — plain TCP, no RTP stack, no media server.

```
exten => 1912,1,Answer()
 same  =>       ,AudioSocket(${UUID},vaani-host:9092)
 same  =>       ,Hangup()
```

Asterisk speaks 8 kHz; the pipeline speaks 16 kHz. Conversion happens at the
edge in [`telephony/audiosocket.py`](vaani/telephony/audiosocket.py) and nowhere
else. For carrier-scale NAT traversal put LiveKit in front and hand its PCM to
the same `CallSession`.

---

## API

| | |
|---|---|
| `GET /api/health` · `/api/ready` | Liveness; readiness refuses traffic at capacity |
| `GET/PUT/DELETE /api/agents/{key}` | Agent profiles, returns the rendered system prompt |
| `POST /api/knowledge/text` · `/upload` | Ingest text or a document |
| `GET /api/knowledge/search?q=` | Test retrieval without placing a call |
| `GET /api/calls/live` · `/api/calls` | Live monitoring and history |
| `POST /api/calls/{id}/hangup` | Supervisor force-end |
| `GET /api/analytics/summary` | Volume, latency percentiles, escalation rate, knowledge gaps |
| `POST /api/simulate/turn` | One agent turn over text — assert on answers in CI |
| `WS /ws/call?agent=` | Media plane |

Interactive docs at `/docs`.

---

## Testing

```bash
pytest -q     # 23 passed in ~5s
```

The suite runs the complete pipeline on mock providers: no models, no network,
no GPU. Prompt changes, tool changes and turn-detection changes all get
regression cover on a laptop. `POST /api/simulate/turn` lets you assert on what
the agent actually says without synthesising a second of audio.

---

## Roadmap

| Phase | Status | |
|---|---|---|
| 1 | **Partial** | Telephony: AudioSocket bridge works; call queues, recording playback, conference and DTMF-driven IVR trees are not built |
| 2 | **Done** | Streaming speech pipeline, VAD, barge-in, multilingual STT/TTS, conversation engine |
| 3 | **Done** | RAG: chunking, embeddings, Qdrant, hybrid retrieval, ingestion API |
| 4 | **Partial** | Tool calling and the workflow contract are done; real CRM/ERP connectors are per-deployment |
| 5 | **Partial** | Analytics API and a web console exist; the Angular admin portal is not built |
| 6 | **Not started** | Security hardening (OAuth/JWT/RBAC, PII masking), HA deployment, load testing |
| 7 | **Not started** | UAT, documentation, production rollout |

Not yet built, and deliberately: outbound campaign engine, speaker diarisation,
voice biometrics, live agent coaching, the multi-agent supervisor topology.
The single-agent-plus-tools design covers the listed use cases at lower latency;
a supervisor topology is worth adding when a domain genuinely needs it, not
before.

### Known limitations

- **Nothing is persisted.** Agent profiles, call records and the in-memory vector
  store all live in process memory and are lost on restart. Only call recordings
  reach disk. `vaani/db/` is an empty placeholder — SQLAlchemy is declared as a
  dependency but no models or migrations exist yet.
- `MemoryVectorStore` is a brute-force scan. Correct and fast to a few thousand
  chunks; use Qdrant beyond that.
- No authentication on the API. Do not expose this to a network before phase 6.
- Whisper is not a streaming model, so STT latency is paid per turn rather than
  continuously. Interim transcripts are available but cost GPU.

---

## Licence

Apache 2.0. All bundled components are permissively licensed and self-hostable.

## Upgrading

The schema migrates itself. `CallRepository.start()` runs `alembic upgrade head`
before opening the pool, so a deployment carrying an older database is brought
forward on boot — including one created before the phase 2 disposition columns
existed, which `create_all` would silently leave behind.

That happens inside the application because an on-premise operator runs
`docker compose up` and nothing else. It is safe for a single worker. A
multi-worker deployment should run `alembic upgrade head` once in an init
container instead, since concurrent upgrades race.

Postgres needs the prod extra, which carries both drivers — asyncpg for the
application and psycopg2 for Alembic:

```bash
docker compose build --build-arg EXTRAS='[prod]' vaani
docker compose --profile prod up -d
```

Call records, transcripts and recordings are deleted once they pass
`retention_days` (default 365). The sweep runs at boot and then daily; the boot
pass matters because a box powered off overnight never reaches a scheduled run.
Setting it to zero disables deletion rather than deleting everything.
