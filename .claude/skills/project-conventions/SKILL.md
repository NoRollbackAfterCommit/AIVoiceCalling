---
name: project-conventions
description: Vaani code patterns for the async call path, provider laziness, settings metadata, and logging. Load before editing anything under vaani/pipeline, vaani/audio, vaani/api/ws_voice.py, vaani/core/registry.py, or vaani/config.py.
user-invocable: false
---

# Vaani conventions

CLAUDE.md lists the invariants. This is how to actually satisfy them, and the traps that look
correct in review.

## The call path is shared, not per-request

One uvicorn worker carries up to `max_concurrent_calls` live calls on a single event loop.
There is no per-call thread and no per-call process. Every blocking moment is felt by all of
them at once, and a caller hears it as dead air.

```python
# Wrong — stalls every concurrent call for the duration of the load.
def transcribe(self, pcm): return self._model.transcribe(pcm)

# Right — the sync SDK runs off-loop.
async def transcribe(self, pcm):
    return await asyncio.to_thread(self._model.transcribe, pcm)
```

Things that are blocking and do not look it: model loading, `open()`/`write()` on recordings,
`requests`, `time.sleep`, `sentence_transformers.encode`, `faster_whisper.transcribe`, anything
in `numpy` over a large array, and CPU-bound resampling of a long buffer.

Rules that follow:

- Model loading belongs in `start()`, called once from `Services.start()` at boot.
- Anything spawned per turn must be cancellable. Playback is a task precisely so barge-in can
  cancel it; if you add a long-running per-turn task, make sure the barge-in path kills it too.
- Await points inside `SPEAKING` are where barge-in gets noticed. Coarse-grained awaits make
  interruption feel laggy even when the detector is correct.
- Unbounded queues are a leak under load. Bound them and decide what to drop.

## Provider modules stay importable without their dependency

Every `vaani/providers/**` module is imported lazily from inside a `build_*` branch in
`registry.py`. The vendor SDK import goes inside `__init__` or `start()` — not at module scope.

This is not style. `pip install -e .` with no extras must boot, and an on-premise deployment
that never installed `[cloud]` must be structurally incapable of calling an external API. A
top-level `import anthropic` turns that guarantee into a promise.

Two follow-ons that get missed:

- New settings must be added to the matching `changed(...)` list in `Services.reload()`, or the
  admin portal saves a value that never takes effect on the running provider.
- `reload()` starts the replacement before closing the old one, so a bad API key fails the save
  instead of leaving the platform with no LLM. Preserve that ordering when editing it.

## Settings are declarative

`vaani/config.py` is the only place a knob is defined. The `cfg()` metadata drives validation,
`/api/settings/schema`, and the admin page rendering simultaneously.

- `depends_on={"llm_provider": ["anthropic"]}` hides the field for other providers.
- `secret=True` means the API returns a masked hint and never the value. Do not add a read path.
- `restart=False` marks a setting that takes effect per call rather than rebuilding providers.
  Getting this wrong either wastes a rebuild or leaves a setting apparently ignored.
- Never hand-write markup in `vaani/web/static/settings.html` for a new field.

## Audio

PCM16 little-endian, mono, `SAMPLE_RATE` (16 kHz), 20 ms frames = 320 samples = 640 bytes.
Use `FRAME_BYTES` / `FRAME_SAMPLES` from `vaani.config` rather than recomputing.

Telephony mu-law at 8 kHz converts in `vaani/telephony/`, at the edge. A provider that emits
another rate resamples inside itself. The pipeline sees exactly one format, always.

## Logging

`get_logger(__name__)`, structured fields in `extra={...}`, never f-strings into the message.
`call_id_var` is a ContextVar set per session, so anything logged inside a call is already
tagged — do not thread `call_id` through by hand.

```python
log.info("turn complete", extra={"stt_ms": m.stt_ms, "agent_ms": m.agent_ms})
```

Exceptions in a per-call task must not kill the session silently. `log.exception` and degrade
the call; a caller mid-sentence should hear a recovery line, not silence.

## Tests

`tests/` runs against mock providers only: about a second, no GPU, no network. That is what
makes it runnable in CI. Keep it that way — a test that needs a model belongs behind a marker
and is not the default suite.

`asyncio_mode = "auto"`, so `async def test_...` needs no decorator. Build audio with the
`tone()` / `silence()` helpers already in `tests/test_pipeline.py` rather than new fixtures.

## Style

ruff, line length 100, py311 target. `from __future__ import annotations` at the top of every
module. `@dataclass(slots=True)` on hot-path structures. Modern typing (`str | None`, not
`Optional[str]`).

Comments in this codebase explain why a decision was made and what breaks otherwise — see the
module docstrings in `session.py` and `config.py`. Match that register. Do not add comments that
restate the code.
