---
name: add-provider
description: Add a new STT, LLM, TTS, or embedding provider to Vaani — stub, registry branch, reload keys, config fields, .env.example docs, and optional-dependency group. Use when wiring up a new model backend or hosted API.
disable-model-invocation: true
---

# Add a provider

A provider touches five places. Wiring four of them produces something that imports fine, passes
tests, and silently never gets selected — so work the checklist in order and verify at the end.

Arguments: the kind (`stt` | `llm` | `tts` | `embedding`) and a name, e.g. `stt deepgram`.
If they weren't given, ask before writing anything.

## Checklist

Create one todo per step.

### 1. Implement the Protocol

Copy `templates/provider_stub.py` to `vaani/providers/<kind>/<name>.py` (LLM/STT/TTS live in
their subpackage; a provider that spans several kinds for one vendor may share a single module
at `vaani/providers/<vendor>_hosted.py` — `openai_hosted.py` is the precedent).

Read the matching Protocol in `vaani/providers/base.py` first and implement it exactly.

- Import the vendor SDK **inside** `__init__` or `start()`, not at module top level.
- `start()` loads models and is called once at boot — never per call.
- All audio in and out is PCM16 mono at `vaani.config.SAMPLE_RATE`. Resample at the provider
  boundary if the vendor disagrees; do not push a different rate into the pipeline.
- Nothing in the call path may block the event loop. A synchronous SDK goes in
  `asyncio.to_thread`; prefer an async client where one exists.
- Set `self.name` — it shows up in logs and `/ready`.

### 2. Register it

In `vaani/core/registry.py`, add a branch to the matching `build_*` function:

```python
if s.stt_provider == "deepgram":
    from vaani.providers.stt.deepgram import DeepgramSTT   # import INSIDE the branch

    return DeepgramSTT(api_key=s.deepgram_api_key or "", ...)
```

The import must sit inside the branch. That laziness is what keeps optional extras optional and
what guarantees an air-gapped install cannot reach the network. Mock stays the fallthrough.

### 3. Add the settings keys to `reload()`

Still in `registry.py`, find the `changed(...)` call for this provider kind inside
`Services.reload()` and add every new setting name to it. Miss this and the admin portal
appears to save the setting while the running provider never rebuilds — the most common way
this workflow is gotten wrong.

### 4. Declare the settings

In `vaani/config.py`:

- Add the value to the provider `Literal[...]` **and** to its `_opts(...)` list.
- Add each new field with `cfg(...)`: `group`, `label`, `help`, `depends_on` so the admin UI
  hides it for other providers, and `secret=True` for anything key-shaped.
- Put the field in the group its provider already uses, next to its siblings.

Do not write any HTML — `config.py` metadata renders the settings page on its own.

### 5. Document and package it

- `.env.example` — a commented block under the right tier, matching the surrounding style.
- `pyproject.toml` — the SDK goes in an existing optional group (`ai`, `rag`, `cloud`, `prod`)
  or a new one. Never in base `dependencies`; base must stay installable with no models.
- `README.md` — only if this adds a deployment path a reader would not otherwise find.

## Verify

Do not report done until these have actually been run and their output checked:

```bash
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -c "from vaani.config import Settings; print(Settings().model_dump_json(indent=2))"
curl -s localhost:8099/api/settings/schema | grep -i <name>    # server running
```

Then confirm by inspection:

- Base install untouched — `pip install -e .` with no extras still boots and passes tests.
- The new field appears under the right group on `/settings.html`, and disappears when a
  different provider is selected.
- Switching to the provider in the admin portal and saving actually rebuilds it (step 3) — the
  log line from `Services.reload()` names the rebuilt component.
