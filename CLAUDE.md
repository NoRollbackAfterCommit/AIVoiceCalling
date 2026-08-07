# Vaani

Open-source, self-hosted AI voice calling agent platform. FastAPI + uvicorn, Python 3.11+.
One binary serves three very different deployments — a laptop demo with no models at all, a
single-workstation GPU pilot, and an air-gapped government cluster — separated only by settings.

## Commands

The venv is at `.venv/` (Windows layout, `Scripts/` not `bin/`).

```bash
.venv/Scripts/python.exe -m uvicorn vaani.main:app --reload --port 8099   # dev server
.venv/Scripts/python.exe -m pytest                                        # tests (~1s, no GPU)
.venv/Scripts/python.exe -m ruff check --fix . && .venv/Scripts/python.exe -m ruff format .
.venv/Scripts/python.exe scripts/smoke_call.py                            # end-to-end call check
```

Install tiers — the base install is deliberately minimal and boots with mocks only:

```bash
pip install -e .              # mocks, no GPU, no network
pip install -e ".[ai,rag]"    # faster-whisper + piper + qdrant
pip install -e ".[cloud]"     # anthropic SDK
pip install -e ".[dev]"       # pytest + ruff
```

Web console: `/` (`vaani/web/static/index.html`), admin settings at `/settings.html`.

## Architecture

Audio in → VAD segments a turn → STT → agent (LLM + RAG + tools) → TTS → audio out.

| Layer | Where | Note |
|---|---|---|
| Provider contracts | `vaani/providers/base.py` | Four `Protocol`s: STT, LLM, TTS, Embedding |
| Provider selection | `vaani/core/registry.py` | `build_stt/llm/tts/embedder/store` — the *only* place a config string becomes a class |
| Settings | `vaani/config.py` | Pydantic, `VAANI_` env prefix; `cfg()` carries UI metadata |
| Persisted overrides | `vaani/settings_store.py` | Admin-portal saves layered over env at boot |
| Call state machine | `vaani/pipeline/session.py` | GREETING → LISTENING → THINKING → SPEAKING → ENDED |
| Concurrency | `vaani/pipeline/manager.py` | Enforces `max_concurrent_calls` |
| Turn/barge-in detection | `vaani/audio/vad.py` | `TurnDetector`, `BargeInDetector` |
| HTTP + WS API | `vaani/api/` | `routes.py`, `settings.py`, `ws_voice.py` (`/ws/call`) |
| Retrieval | `vaani/rag/` | Chunking, embedding, memory or Qdrant store |

## Invariants

These are load-bearing. Breaking one produces a bug that only shows up in a deployment tier
you are not currently running.

1. **Provider imports stay lazy.** Every concrete provider is imported *inside* its `build_*`
   branch in `registry.py`, never at module top level. This is what makes `[ai]`, `[rag]`, and
   `[cloud]` genuinely optional — and it is the hard guarantee that an on-premise install with
   no `anthropic` package installed cannot call out. A top-level `import anthropic` silently
   destroys that property.

2. **One audio format across every boundary.** PCM16 little-endian, mono, 16 kHz
   (`SAMPLE_RATE`). Telephony codecs (8 kHz mu-law) convert at the edge in `vaani/telephony/`,
   never inside the pipeline. Frames are 20 ms / 320 samples / 640 bytes.

3. **Mock providers carry no dependencies.** `MockSTT`/`MockLLM`/`MockTTS`/`HashEmbedding` must
   keep working on a bare `pip install -e .`. They are what the whole test suite runs against.

4. **`cfg()` metadata is the single source of truth.** A settings field's `group`, `label`,
   `help`, `secret`, `options`, `depends_on`, and `restart` drive validation, the
   `/api/settings/schema` response, *and* the rendering of the admin page. Adding a knob means
   editing one line in `config.py` — do not hand-write UI for it.

5. **`secret: true` fields never leave the server.** The settings API returns a masked hint
   only. Do not add a code path that echoes an API key back.

6. **The call path is async and latency-critical.** Anything blocking on the event loop stalls
   *every* concurrent call, not just one. Model loading belongs in `start()` (called once at
   boot), never per call. Wrap unavoidable sync work in `asyncio.to_thread`.

7. **`Services.reload()` is selective on purpose.** It rebuilds only the providers whose
   settings changed, because the in-memory vector store lives inside the retriever — a wholesale
   rebuild to change a TTS voice would wipe every uploaded document. New providers must be
   added to the right `changed(...)` key list.

## Adding a provider

Five places, in this order. The `/add-provider` skill walks it with a stub and a checklist.

1. `vaani/providers/<kind>/<name>.py` — implement the Protocol from `providers/base.py`
2. `vaani/core/registry.py` — a branch in the matching `build_*`, with the import **inside** it
3. `vaani/core/registry.py` — add new setting names to the relevant `changed(...)` call in `reload()`
4. `vaani/config.py` — provider `Literal` + fields, each with `cfg(...)` metadata and `depends_on`
5. `.env.example` + `pyproject.toml` optional-dependency group

## Conventions

- ruff, line length 100, target py311. `from __future__ import annotations` everywhere.
- `@dataclass(slots=True)` for hot-path structures; plain `@dataclass` where mutation is needed.
- Logging via `vaani.core.logging.get_logger`; structured fields go in `extra={...}`.
  `call_id_var` is a ContextVar — log lines inside a call are automatically tagged.
- Tests run against mocks only. Keep them fast and GPU-free; that is why they can run in CI.
- Comments explain *why* a choice was made, not what the line does. Match that when editing.
