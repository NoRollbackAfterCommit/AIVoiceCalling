# Phase 1 — Live Indic Inbound Voice Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One Indian phone number that answers a real inbound call, converses in Hindi, English and code-mixed Hinglish grounded in a document corpus, closes the call, and writes a durable record.

**Architecture:** Two new providers (Sarvam Saarika STT, Bulbul TTS) behind the existing `STTProvider`/`TTSProvider` protocols; a new `ExotelTransport` implementing the existing `Transport` protocol so `CallSession` is untouched by telephony; language state with hysteresis inside the session; and a `CallRepository` that persists off the call path via a queue drained by a background writer.

**Tech Stack:** Python 3.11+, FastAPI, httpx, SQLAlchemy 2.0 async + aiosqlite, pytest + pytest-asyncio, ruff.

## Global Constraints

- Python `>=3.11`; ruff `line-length = 100`, `target-version = "py311"`.
- Every module starts with `from __future__ import annotations`.
- Vendor SDK / HTTP client imports live **inside** `start()` or inside the `build_*` branch — never at module top level. This is what keeps optional extras optional and an on-premise install unable to call out.
- All audio crossing a protocol boundary is **PCM16 little-endian, mono, `vaani.config.SAMPLE_RATE` (16 kHz)**. Codec conversion happens at the transport edge only.
- Frame constants come from `vaani.config`: `FRAME_MS=20`, `FRAME_SAMPLES=320`, `FRAME_BYTES=640`.
- Mock providers stay dependency-free. The default pytest suite must keep running with **no network, no GPU, no API key**, in about a second.
- Nothing blocking runs on the event loop. Sync work goes in `asyncio.to_thread`.
- Every new setting needs `cfg()` metadata **and** an entry in the matching `changed(...)` list in `Services.reload()`.
- `asyncio_mode = "auto"` — `async def test_*` needs no decorator.
- Interpreter is `.venv/Scripts/python.exe` (Windows venv layout). All commands below use it.
- Comments explain *why*, not *what*. Match the density of the surrounding code.

---

## Task 0: Confirm the Exotel wire format (prerequisite, no code)

**Blocking for Tasks 7–8 only.** Tasks 1–6 proceed regardless.

Exotel's public documentation does not publish the WebSocket frame schema. The quick-start specifies "100 ms PCM chunks with 3200 bytes of raw audio" — which is exactly 16 kHz PCM16 mono, matching `SAMPLE_RATE`, so **no resampling is expected on this path**. The event envelope must be confirmed.

- [ ] **Step 1: Obtain the specification**

Contact Exotel (`hello@exotel.com` or your CSM) and request the Voice Streaming Guide. Also read `simple_server.py` in `github.com/exotel/Agent-Stream-echobot`.

Record answers to exactly these questions in `docs/exotel-protocol.md`:

1. Event type names and JSON envelope (expected: `connected`, `start`, `media`, `stop`, and a barge-in/`clear` event).
2. Is the media payload base64-encoded inside JSON, or sent as binary frames?
3. Sample rate and encoding — confirm 16 kHz PCM16, or note if it is 8 kHz μ-law/A-law.
4. The field carrying the caller's number and Exotel's call identifier.
5. Which event the server sends to flush already-queued audio on barge-in.

- [ ] **Step 2: Confirm applet availability**

Confirm the Voicebot/Stream applet is enabled on your account with an SLA acceptable for a government service. **If it is not available, stop and switch Tasks 7–8 to the vSIP + Asterisk path** — `vaani/telephony/audiosocket.py` already implements it and needs only Asterisk dialplan configuration.

- [ ] **Step 3: Commit the findings**

```bash
git add docs/exotel-protocol.md
git commit -m "docs: record Exotel streaming wire format"
```

---

## Task 1: Sarvam STT provider

**Files:**
- Create: `vaani/providers/stt/sarvam.py`
- Test: `tests/test_sarvam_stt.py`

**Interfaces:**
- Consumes: `Transcript` from `vaani.providers.base`; `SAMPLE_RATE` from `vaani.config`.
- Produces: `SarvamSTT(api_key: str, model: str = "saaras:v3", base_url: str = "https://api.sarvam.ai", language: str | None = None, timeout_s: float = 30.0)` satisfying `STTProvider`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sarvam_stt.py`:

```python
"""Sarvam STT contract tests.

Exercised against a stubbed transport rather than the live API: the default
suite must stay offline. Real-audio accuracy is covered by the golden set in
tests/test_golden_audio.py, which is marked and opt-in.
"""

from __future__ import annotations

import httpx
import pytest

from vaani.config import SAMPLE_RATE
from vaani.providers.base import STTProvider
from vaani.providers.stt.sarvam import SarvamSTT


def _pcm(ms: int) -> bytes:
    return b"\x01\x00" * int(SAMPLE_RATE * ms / 1000)


def _stub(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://api.sarvam.ai"
    )


async def test_satisfies_the_protocol():
    assert isinstance(SarvamSTT(api_key="k"), STTProvider)


async def test_transcribes_and_reports_language():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["api-subscription-key"] == "k"
        return httpx.Response(
            200, json={"transcript": "बिजली का बिल", "language_code": "hi-IN"}
        )

    stt = SarvamSTT(api_key="k")
    stt._http = _stub(handler)
    result = await stt.transcribe(_pcm(800))

    assert result.text == "बिजली का बिल"
    assert result.language == "hi-IN"
    assert result.is_final is True
    assert result.duration_s == pytest.approx(0.8, rel=0.01)


async def test_short_audio_is_not_sent_to_the_api():
    """Under 150 ms is a click or a breath. Sending it wastes a round trip
    and returns hallucinated text."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not call the API for sub-threshold audio")

    stt = SarvamSTT(api_key="k")
    stt._http = _stub(handler)
    result = await stt.transcribe(_pcm(80))

    assert result.text == ""
    assert result.is_final is True


async def test_missing_transcript_field_yields_empty_text_not_a_crash():
    stt = SarvamSTT(api_key="k")
    stt._http = _stub(lambda r: httpx.Response(200, json={}))
    result = await stt.transcribe(_pcm(500))
    assert result.text == ""


async def test_stream_buffers_then_yields_one_final():
    """TurnDetector already segments utterances, so streaming reduces to a
    single transcription of the completed turn."""

    stt = SarvamSTT(api_key="k")
    stt._http = _stub(
        lambda r: httpx.Response(200, json={"transcript": "हाँ", "language_code": "hi-IN"})
    )

    async def audio():
        yield _pcm(300)
        yield _pcm(300)

    results = [t async for t in stt.stream(audio())]
    assert len(results) == 1
    assert results[0].is_final is True
    assert results[0].text == "हाँ"


async def test_start_without_api_key_fails_loudly():
    with pytest.raises(ValueError, match="API key"):
        await SarvamSTT(api_key="").start()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_sarvam_stt.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vaani.providers.stt.sarvam'`

- [ ] **Step 3: Write the implementation**

Create `vaani/providers/stt/sarvam.py`:

```python
"""Sarvam Saarika speech to text.

Chosen over Whisper for Indian deployments because it is trained on code-mixed
Indian speech. A caller saying "मेरा electricity bill pending hai" is the normal
case on an Indian helpline, not an edge case, and engines configured for a single
language degrade badly on it. Saarika also covers od-IN, which most vendors do
not.

Transcription is per completed utterance rather than streaming: TurnDetector has
already decided where the turn ended, so a streaming socket would add complexity
without removing latency from the critical path.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from vaani.config import SAMPLE_RATE
from vaani.core.logging import get_logger
from vaani.providers.base import Transcript

log = get_logger(__name__)

# Below this, the buffer is a click, a breath or line noise. Sending it wastes a
# round trip and these models return confident nonsense for near-silence.
_MIN_UTTERANCE_S = 0.15


class SarvamSTT:
    name = "sarvam"

    def __init__(
        self,
        api_key: str,
        model: str = "saaras:v3",
        base_url: str = "https://api.sarvam.ai",
        language: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._language = language
        self._timeout = timeout_s
        self._http: Any = None

    async def start(self) -> None:
        import httpx

        if not self._api_key:
            raise ValueError("A Sarvam API key is required. Set it in Settings.")
        self._http = httpx.AsyncClient(
            base_url=self._base_url.rstrip("/"),
            headers={"api-subscription-key": self._api_key},
            timeout=httpx.Timeout(self._timeout, connect=5.0),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
        )
        log.info("sarvam stt ready", extra={"model": self._model})

    async def transcribe(self, pcm: bytes, *, language: str | None = None) -> Transcript:
        if self._http is None:
            raise RuntimeError("SarvamSTT.start() was not awaited")

        duration = len(pcm) / (SAMPLE_RATE * 2)
        if duration < _MIN_UTTERANCE_S:
            return Transcript(text="", is_final=True, duration_s=duration)

        data: dict[str, Any] = {"model": self._model}
        # Omitting language_code lets Saarika auto-detect, which is what makes
        # a caller switching mid-call work at all.
        if lang := (language or self._language):
            data["language_code"] = lang

        response = await self._http.post(
            "/speech-to-text",
            files={"file": ("audio.wav", _to_wav(pcm), "audio/wav")},
            data=data,
        )
        response.raise_for_status()
        payload = response.json()
        return Transcript(
            text=(payload.get("transcript") or "").strip(),
            is_final=True,
            language=payload.get("language_code") or language or self._language,
            duration_s=duration,
        )

    async def stream(
        self, audio: AsyncIterator[bytes], *, language: str | None = None
    ) -> AsyncIterator[Transcript]:
        buffer = bytearray()
        async for chunk in audio:
            buffer.extend(chunk)
        yield await self.transcribe(bytes(buffer), language=language)

    async def close(self) -> None:
        http, self._http = self._http, None
        if http is not None:
            await http.aclose()


def _to_wav(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes:
    """The API wants a recognised container, not bare samples."""
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return buffer.getvalue()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_sarvam_stt.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Verify field names against the live API**

The response keys `transcript` and `language_code` are the documented shape but must be confirmed. With a real key set:

```bash
curl -s -X POST https://api.sarvam.ai/speech-to-text \
  -H "api-subscription-key: $SARVAM_API_KEY" \
  -F "file=@sample.wav" -F "model=saaras:v3"
```

If the keys differ, change them in `transcribe()` and update the test fixtures to match. **Do not leave the code and the test agreeing on a shape the API does not return** — that is a green suite over a broken integration.

- [ ] **Step 6: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check .
git add vaani/providers/stt/sarvam.py tests/test_sarvam_stt.py
git commit -m "feat: Sarvam Saarika speech-to-text provider"
```

---

## Task 2: Sarvam TTS provider

**Files:**
- Create: `vaani/providers/tts/sarvam.py`
- Test: `tests/test_sarvam_tts.py`

**Interfaces:**
- Consumes: `SAMPLE_RATE`, `FRAME_SAMPLES` from `vaani.config`; `resample_pcm16` from `vaani.audio.resample`.
- Produces: `SarvamTTS(api_key: str, model: str = "bulbul:v3", default_voice: str = "anushka", base_url: str = "https://api.sarvam.ai", speed: float = 1.0, timeout_s: float = 30.0)` satisfying `TTSProvider`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sarvam_tts.py`:

```python
"""Sarvam TTS contract tests, offline."""

from __future__ import annotations

import base64
import io
import struct
import wave

import httpx
import pytest

from vaani.config import SAMPLE_RATE
from vaani.providers.base import TTSProvider
from vaani.providers.tts.sarvam import SarvamTTS


def _wav_b64(seconds: float = 0.5, rate: int = 22050) -> str:
    """A silent WAV at a rate deliberately different from SAMPLE_RATE, so the
    test proves the provider resamples rather than passing bytes through."""
    n = int(rate * seconds)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(struct.pack(f"<{n}h", *([0] * n)))
    return base64.b64encode(buffer.getvalue()).decode()


def _stub(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://api.sarvam.ai"
    )


async def test_satisfies_the_protocol():
    assert isinstance(SarvamTTS(api_key="k"), TTSProvider)


async def test_synthesize_returns_pcm_at_the_pipeline_rate():
    tts = SarvamTTS(api_key="k")
    tts._http = _stub(
        lambda r: httpx.Response(200, json={"audios": [_wav_b64(0.5, 22050)]})
    )
    pcm = await tts.synthesize("नमस्ते")

    # 0.5 s at SAMPLE_RATE, 2 bytes per sample, within resampler rounding.
    assert len(pcm) == pytest.approx(SAMPLE_RATE * 2 * 0.5, rel=0.02)
    assert len(pcm) % 2 == 0


async def test_voice_and_language_reach_the_request():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"audios": [_wav_b64()]})

    tts = SarvamTTS(api_key="k")
    tts._http = _stub(handler)
    await tts.synthesize("नमस्ते", voice="hi-IN:meera")

    assert seen["speaker"] == "meera"
    assert seen["target_language_code"] == "hi-IN"


async def test_stream_yields_frame_aligned_chunks():
    tts = SarvamTTS(api_key="k")
    tts._http = _stub(
        lambda r: httpx.Response(200, json={"audios": [_wav_b64(1.0, 22050)]})
    )
    chunks = [c async for c in tts.stream("एक दो तीन")]

    assert chunks, "expected at least one chunk"
    # Misaligned chunks put a half sample on the wire and click audibly.
    assert all(len(c) % 2 == 0 for c in chunks)
    assert sum(len(c) for c in chunks) == pytest.approx(SAMPLE_RATE * 2, rel=0.02)


async def test_empty_text_produces_no_audio_and_no_request():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not call the API for empty text")

    tts = SarvamTTS(api_key="k")
    tts._http = _stub(handler)
    assert await tts.synthesize("   ") == b""
    assert [c async for c in tts.stream("")] == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_sarvam_tts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vaani.providers.tts.sarvam'`

- [ ] **Step 3: Write the implementation**

Create `vaani/providers/tts/sarvam.py`:

```python
"""Sarvam Bulbul text to speech.

Bulbul is the reason this provider exists: OpenAI's and most Western vendors'
voices cannot pronounce Indian languages, and an English-accented model reading
Devanagari is unusable on a citizen helpline. Bulbul covers all eleven of the
languages this platform targets, including Odia.

Output is chunked to roughly 200 ms and resampled to SAMPLE_RATE here, so the
pipeline continues to see exactly one audio format.
"""

from __future__ import annotations

import base64
import io
import wave
from collections.abc import AsyncIterator
from typing import Any

from vaani.audio.resample import resample_pcm16
from vaani.config import FRAME_SAMPLES, SAMPLE_RATE
from vaani.core.logging import get_logger

log = get_logger(__name__)

_CHUNK_BYTES = FRAME_SAMPLES * 10 * 2  # 200 ms of PCM16


class SarvamTTS:
    name = "sarvam"

    def __init__(
        self,
        api_key: str,
        model: str = "bulbul:v3",
        default_voice: str = "anushka",
        base_url: str = "https://api.sarvam.ai",
        speed: float = 1.0,
        timeout_s: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._default_voice = default_voice
        self._base_url = base_url
        self._speed = speed
        self._timeout = timeout_s
        self._http: Any = None

    async def start(self) -> None:
        import httpx

        if not self._api_key:
            raise ValueError("A Sarvam API key is required. Set it in Settings.")
        self._http = httpx.AsyncClient(
            base_url=self._base_url.rstrip("/"),
            headers={"api-subscription-key": self._api_key},
            timeout=httpx.Timeout(self._timeout, connect=5.0),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
        )
        log.info("sarvam tts ready", extra={"model": self._model})

    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        if not text.strip():
            return b""
        if self._http is None:
            raise RuntimeError("SarvamTTS.start() was not awaited")

        language, speaker = _split_voice(voice or self._default_voice)
        response = await self._http.post(
            "/text-to-speech",
            json={
                "inputs": [text],
                "target_language_code": language,
                "speaker": speaker,
                "model": self._model,
                "pace": self._speed,
            },
        )
        response.raise_for_status()
        audios = response.json().get("audios") or []
        if not audios:
            log.warning("sarvam tts returned no audio", extra={"chars": len(text)})
            return b""
        return _wav_to_pipeline_pcm(base64.b64decode(audios[0]))

    async def stream(self, text: str, *, voice: str | None = None) -> AsyncIterator[bytes]:
        pcm = await self.synthesize(text, voice=voice)
        for start in range(0, len(pcm), _CHUNK_BYTES):
            yield pcm[start : start + _CHUNK_BYTES]

    async def close(self) -> None:
        http, self._http = self._http, None
        if http is not None:
            await http.aclose()


def _split_voice(voice: str) -> tuple[str, str]:
    """Voices are configured as "hi-IN:meera" so one profile field carries both
    the language and the speaker. A bare name keeps the Hindi default."""
    if ":" in voice:
        language, speaker = voice.split(":", 1)
        return language, speaker
    return "hi-IN", voice


def _wav_to_pipeline_pcm(raw: bytes) -> bytes:
    with wave.open(io.BytesIO(raw), "rb") as wav:
        rate = wav.getframerate()
        pcm = wav.readframes(wav.getnframes())
    return resample_pcm16(pcm, rate, SAMPLE_RATE) if rate != SAMPLE_RATE else pcm
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_sarvam_tts.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Verify the request and response shape against the live API**

Confirm the request fields (`inputs`, `target_language_code`, `speaker`, `model`, `pace`) and that the response returns base64 WAV under `audios`. If they differ, fix the implementation **and** the test fixtures together.

- [ ] **Step 6: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check .
git add vaani/providers/tts/sarvam.py tests/test_sarvam_tts.py
git commit -m "feat: Sarvam Bulbul text-to-speech provider"
```

---

## Task 3: Wire Sarvam into config, registry and packaging

**Files:**
- Modify: `vaani/config.py` (provider `Literal`s, `_opts` lists, new fields)
- Modify: `vaani/core/registry.py` (`build_stt`, `build_tts`, `reload` key lists)
- Modify: `.env.example`
- Test: `tests/test_sarvam_wiring.py`

**Note — no optional-dependency group is needed.** The spec anticipated a `sarvam`
extra, but Tasks 1–2 use `httpx`, which is already a base dependency, rather than a
vendor SDK. Nothing new to install, so `pyproject.toml` is untouched here. This is
strictly better than the spec: one less extra to keep in sync, and the providers
work on a bare `pip install -e .`.

**Interfaces:**
- Consumes: `SarvamSTT`, `SarvamTTS` from Tasks 1–2.
- Produces: settings keys `sarvam_api_key`, `sarvam_stt_model`, `sarvam_tts_model`, `sarvam_voice`; `"sarvam"` accepted by `stt_provider` and `tts_provider`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sarvam_wiring.py`:

```python
"""Sarvam must be reachable through config alone — that is the whole point of
the registry. Also guards the reload() key lists, which are silently easy to
forget: a missing key means the admin portal saves a setting that never takes
effect on the running provider."""

from __future__ import annotations

import inspect

import pytest

from vaani.config import Settings
from vaani.core.registry import Services, build_stt, build_tts


def test_sarvam_is_a_selectable_provider():
    s = Settings(stt_provider="sarvam", tts_provider="sarvam", sarvam_api_key="k")
    assert build_stt(s).name == "sarvam"
    assert build_tts(s).name == "sarvam"


def test_defaults_still_select_mocks():
    """A bare install with no keys must keep booting."""
    s = Settings()
    assert build_stt(s).name == "mock"
    assert build_tts(s).name == "mock"


def test_sarvam_api_key_is_secret():
    assert "sarvam_api_key" in Settings.secret_fields()


def test_sarvam_fields_are_hidden_for_other_providers():
    meta = Settings.field_meta("sarvam_api_key")
    assert meta["depends_on"] == {"stt_provider": ["sarvam"], "tts_provider": ["sarvam"]}


@pytest.mark.parametrize(
    "key", ["sarvam_api_key", "sarvam_stt_model", "sarvam_tts_model", "sarvam_voice"]
)
def test_reload_watches_the_new_keys(key):
    source = inspect.getsource(Services.reload)
    assert key in source, f"{key} missing from a changed(...) list in Services.reload"


def test_importing_sarvam_module_needs_no_sdk():
    """Provider modules must import on a bare install — the HTTP client is
    constructed in start(), not at module scope."""
    import vaani.providers.stt.sarvam as stt_mod
    import vaani.providers.tts.sarvam as tts_mod

    assert stt_mod.SarvamSTT is not None
    assert tts_mod.SarvamTTS is not None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_sarvam_wiring.py -v`
Expected: FAIL — `ValidationError: Input should be 'mock', 'faster_whisper' or 'openai'`

- [ ] **Step 3: Add the settings fields**

In `vaani/config.py`, extend the STT provider literal and options:

```python
    stt_provider: Literal["mock", "faster_whisper", "openai", "sarvam"] = cfg(
        "mock", group="Speech to text", label="Provider",
        options=_opts(
            ("mock", "Mock — no model, for development"),
            ("faster_whisper", "Faster-Whisper — self-hosted, offline"),
            ("openai", "OpenAI Whisper API — hosted"),
            ("sarvam", "Sarvam Saarika — Indian languages, hosted in India"),
        ),
    )
```

Extend the TTS provider literal and options:

```python
    tts_provider: Literal["mock", "piper", "openai", "sarvam"] = cfg(
        "mock", group="Text to speech", label="Provider",
        options=_opts(
            ("mock", "Mock — tone generator, for development"),
            ("piper", "Piper — self-hosted, offline"),
            ("openai", "OpenAI speech — hosted"),
            ("sarvam", "Sarvam Bulbul — Indian languages, hosted in India"),
        ),
    )
```

Add these fields immediately after the OpenAI block in the "Language model" group's neighbouring sections — `sarvam_api_key` in "Speech to text" since it is shared by both speech providers:

```python
    sarvam_api_key: str | None = cfg(
        None, group="Speech to text", label="Sarvam API key", secret=True,
        help="One key covers both Saarika speech recognition and Bulbul speech "
             "synthesis. Sarvam hosts in India, which is what makes it usable "
             "for a deployment with data residency obligations.",
        depends_on={"stt_provider": ["sarvam"], "tts_provider": ["sarvam"]},
    )
    sarvam_stt_model: str = cfg(
        "saaras:v3", group="Speech to text", label="Saarika model",
        depends_on={"stt_provider": ["sarvam"]},
    )
    sarvam_tts_model: str = cfg(
        "bulbul:v3", group="Text to speech", label="Bulbul model",
        depends_on={"tts_provider": ["sarvam"]},
    )
    sarvam_voice: str = cfg(
        "hi-IN:anushka", group="Text to speech", label="Sarvam voice",
        help="Written as language:speaker, for example hi-IN:meera or "
             "bn-IN:anushka. The language half selects pronunciation.",
        depends_on={"tts_provider": ["sarvam"]},
    )
```

- [ ] **Step 4: Add the registry branches**

In `vaani/core/registry.py`, inside `build_stt`, **before** the mock fallthrough:

```python
    if s.stt_provider == "sarvam":
        from vaani.providers.stt.sarvam import SarvamSTT

        return SarvamSTT(
            api_key=s.sarvam_api_key or "",
            model=s.sarvam_stt_model,
            language=s.stt_language,
            timeout_s=s.llm_timeout_s,
        )
```

Inside `build_tts`, before the mock fallthrough:

```python
    if s.tts_provider == "sarvam":
        from vaani.providers.tts.sarvam import SarvamTTS

        return SarvamTTS(
            api_key=s.sarvam_api_key or "",
            model=s.sarvam_tts_model,
            default_voice=s.sarvam_voice,
            speed=s.tts_speed,
        )
```

- [ ] **Step 5: Add the keys to `Services.reload()`**

In the STT `changed(...)` call, append `"sarvam_api_key", "sarvam_stt_model"`. In the TTS `changed(...)` call, append `"sarvam_api_key", "sarvam_tts_model", "sarvam_voice"`.

- [ ] **Step 6: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_sarvam_wiring.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 7: Document in `.env.example`**

Add under the Tier 2 section:

```bash
# Sarvam — Indian-language speech, hosted in India. One key covers both.
# VAANI_STT_PROVIDER=sarvam
# VAANI_TTS_PROVIDER=sarvam
# VAANI_SARVAM_API_KEY=
# VAANI_SARVAM_STT_MODEL=saaras:v3
# VAANI_SARVAM_TTS_MODEL=bulbul:v3
# VAANI_SARVAM_VOICE=hi-IN:anushka
```

- [ ] **Step 8: Run the whole suite and commit**

Run: `.venv/Scripts/python.exe -m pytest -q && .venv/Scripts/python.exe -m ruff check .`
Expected: all tests pass, lint clean.

```bash
git add vaani/config.py vaani/core/registry.py .env.example tests/test_sarvam_wiring.py
git commit -m "feat: select Sarvam speech providers from settings"
```

---

## Task 4: Language state with hysteresis

**Files:**
- Modify: `vaani/agent/prompt.py` (add `voices` to `AgentProfile`)
- Create: `vaani/pipeline/language.py`
- Modify: `vaani/pipeline/session.py` (use the tracker when choosing a voice)
- Test: `tests/test_language_tracker.py`

**Interfaces:**
- Consumes: `AgentProfile` from `vaani.agent.prompt`.
- Produces: `LanguageTracker(default: str, voices: dict[str, str], switch_after: int = 2)` with `observe(language: str | None) -> str` returning the current language, and `voice() -> str | None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_language_tracker.py`:

```python
"""Language switching needs hysteresis.

A single mis-detected utterance must not flip the agent into another language
mid-call: answering one sentence in Bengali to a Hindi speaker is far worse than
taking one extra turn to follow a genuine switch."""

from __future__ import annotations

from vaani.pipeline.language import LanguageTracker

VOICES = {"hi-IN": "hi-IN:anushka", "en-IN": "en-IN:meera", "bn-IN": "bn-IN:anushka"}


def _tracker() -> LanguageTracker:
    return LanguageTracker(default="hi-IN", voices=VOICES)


def test_starts_on_the_profile_default():
    t = _tracker()
    assert t.current == "hi-IN"
    assert t.voice() == "hi-IN:anushka"


def test_a_single_foreign_utterance_does_not_switch():
    t = _tracker()
    assert t.observe("bn-IN") == "hi-IN"
    assert t.voice() == "hi-IN:anushka"


def test_two_consecutive_utterances_do_switch():
    t = _tracker()
    t.observe("bn-IN")
    assert t.observe("bn-IN") == "bn-IN"
    assert t.voice() == "bn-IN:anushka"


def test_returning_to_the_current_language_resets_the_streak():
    """Alternating detections are noise, not a language change."""
    t = _tracker()
    t.observe("bn-IN")
    t.observe("hi-IN")
    assert t.observe("bn-IN") == "hi-IN"


def test_unknown_language_is_ignored():
    """A language with no configured voice cannot be spoken, so never switch
    to it — the caller would get silence."""
    t = _tracker()
    t.observe("ta-IN")
    assert t.observe("ta-IN") == "hi-IN"


def test_none_is_ignored():
    t = _tracker()
    assert t.observe(None) == "hi-IN"
    assert t.observe(None) == "hi-IN"


def test_hinglish_reported_as_hindi_stays_hindi():
    """Saarika returns hi-IN for code-mixed Hindi-English. That must be a
    no-op, not a switch — which is exactly why no Hinglish tag exists."""
    t = _tracker()
    assert t.observe("hi-IN") == "hi-IN"
    assert t.observe("hi-IN") == "hi-IN"


def test_switch_threshold_is_configurable():
    t = LanguageTracker(default="hi-IN", voices=VOICES, switch_after=1)
    assert t.observe("en-IN") == "en-IN"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_language_tracker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vaani.pipeline.language'`

- [ ] **Step 3: Write the implementation**

Create `vaani/pipeline/language.py`:

```python
"""Which language the agent is currently speaking.

Detection is per utterance and imperfect, so acting on every result makes an
agent that stutters between languages. This holds a current language and only
moves after the same new language is seen on consecutive turns.

There is deliberately no "Hinglish" state. Saarika reports code-mixed
Hindi-English as hi-IN and Bulbul's Hindi voices pronounce embedded English
words correctly, so code-mixing works precisely because nothing here treats it
as a decision point.
"""

from __future__ import annotations

from vaani.core.logging import get_logger

log = get_logger(__name__)


class LanguageTracker:
    def __init__(
        self, default: str, voices: dict[str, str], switch_after: int = 2
    ) -> None:
        self._voices = dict(voices)
        self._switch_after = max(1, switch_after)
        self.current = default
        self._candidate: str | None = None
        self._streak = 0

    def observe(self, language: str | None) -> str:
        """Feed one utterance's detected language. Returns the language to use."""
        if not language or language == self.current:
            self._candidate = None
            self._streak = 0
            return self.current

        # A language with no voice configured cannot be spoken. Switching to it
        # would leave the caller with silence, so ignore it entirely.
        if language not in self._voices:
            return self.current

        if language == self._candidate:
            self._streak += 1
        else:
            self._candidate = language
            self._streak = 1

        if self._streak >= self._switch_after:
            log.info(
                "caller language changed",
                extra={"from": self.current, "to": language},
            )
            self.current = language
            self._candidate = None
            self._streak = 0
        return self.current

    def voice(self) -> str | None:
        return self._voices.get(self.current)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_language_tracker.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Add `voices` to `AgentProfile`**

In `vaani/agent/prompt.py`, add to the `AgentProfile` dataclass, after the existing `voice` field:

```python
    # Per-language voice, written as language:speaker. Behaviour is data, so a
    # deployment adds a language by editing this record, not the pipeline.
    voices: dict[str, str] = field(default_factory=dict)
```

- [ ] **Step 6: Use the tracker in the session**

In `vaani/pipeline/session.py`, in `__init__`, after `self._profile` is assigned:

```python
        self._language = LanguageTracker(
            default=(self._profile.languages or ["hi-IN"])[0],
            voices=self._profile.voices,
        )
```

Add the import at the top: `from vaani.pipeline.language import LanguageTracker`.

In `_process_turn`, immediately after the transcript is obtained and before the agent is called, feed the detection and record it:

```python
        self._language.observe(transcript.language)
        self.record.language = self._language.current
```

Where TTS is invoked, pass the tracked voice, preferring it over the profile's single `voice`:

```python
        voice = self._language.voice() or self._profile.voice
```

and use `voice` in the `self._services.tts.stream(text, voice=voice)` call.

- [ ] **Step 7: Add `language` to `CallRecord`**

In `vaani/pipeline/session.py`, add to the `CallRecord` dataclass:

```python
    language: str | None = None
```

- [ ] **Step 8: Run the whole suite and commit**

Run: `.venv/Scripts/python.exe -m pytest -q && .venv/Scripts/python.exe -m ruff check .`
Expected: all pass, lint clean.

```bash
git add vaani/pipeline/language.py vaani/pipeline/session.py vaani/agent/prompt.py tests/test_language_tracker.py
git commit -m "feat: track caller language with hysteresis, select voice per language"
```

---

## Task 5: Call persistence models and repository

**Files:**
- Create: `vaani/db/models.py`, `vaani/db/repository.py`
- Test: `tests/test_call_repository.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `CallRepository(database_url: str)` with `async start()`, `create_call(record) -> None`, `append_turn(call_id: str, seq: int, role: str, text: str, language: str | None, metrics: dict[str, Any]) -> None` (synchronous, non-blocking), `async finish_call(record) -> None`, `async flush() -> None`, `async close() -> None`, `async get_call(call_id) -> dict | None`, `async turns_for(call_id) -> list[dict]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_call_repository.py`:

```python
"""Persistence must survive a restart and must never block the call path."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from vaani.db.repository import CallRepository


@dataclass
class FakeRecord:
    call_id: str = "c1"
    agent_key: str = "default"
    direction: str = "inbound"
    caller_number: str | None = "+919876543210"
    started_at: float = 1000.0
    ended_at: float | None = None
    outcome: str = "in_progress"
    summary: str | None = None
    language: str | None = "hi-IN"
    duration_s: float = 0.0


@pytest.fixture
async def repo(tmp_path):
    r = CallRepository(f"sqlite+aiosqlite:///{tmp_path / 'calls.db'}")
    await r.start()
    yield r
    await r.close()


async def test_creates_and_reads_back_a_call(repo):
    await repo.create_call(FakeRecord())
    await repo.flush()
    row = await repo.get_call("c1")
    assert row["caller_number"] == "+919876543210"
    assert row["outcome"] == "in_progress"


async def test_append_turn_does_not_block(repo):
    """It is called from the turn loop, so it must be synchronous and return
    immediately — the write happens on a background task."""
    await repo.create_call(FakeRecord())
    result = repo.append_turn(
        "c1", 0, "caller", "बिजली का बिल", "hi-IN",
        {"stt_ms": 300, "agent_ms": 400, "tts_first_chunk_ms": 200, "total_ms": 900},
    )
    assert result is None
    await repo.flush()
    turns = await repo.turns_for("c1")
    assert len(turns) == 1
    assert turns[0]["text"] == "बिजली का बिल"
    assert turns[0]["stt_ms"] == 300


async def test_turns_come_back_in_order(repo):
    await repo.create_call(FakeRecord())
    for i in range(5):
        repo.append_turn("c1", i, "caller", f"turn {i}", "hi-IN", {})
    await repo.flush()
    turns = await repo.turns_for("c1")
    assert [t["seq"] for t in turns] == [0, 1, 2, 3, 4]


async def test_finish_call_records_the_outcome(repo):
    await repo.create_call(FakeRecord())
    await repo.finish_call(
        FakeRecord(ended_at=1090.0, outcome="completed", summary="Bill query", duration_s=90.0)
    )
    row = await repo.get_call("c1")
    assert row["outcome"] == "completed"
    assert row["duration_s"] == 90.0
    assert row["summary"] == "Bill query"


async def test_data_survives_a_restart(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'calls.db'}"
    first = CallRepository(url)
    await first.start()
    await first.create_call(FakeRecord())
    first.append_turn("c1", 0, "caller", "नमस्ते", "hi-IN", {})
    await first.flush()
    await first.close()

    second = CallRepository(url)
    await second.start()
    assert (await second.get_call("c1"))["call_id"] == "c1"
    assert len(await second.turns_for("c1")) == 1
    await second.close()


async def test_a_full_queue_drops_rather_than_blocking(repo):
    """Losing a turn record is bad. Stalling every concurrent call because the
    disk is slow is worse."""
    await repo.create_call(FakeRecord())
    for i in range(5000):
        repo.append_turn("c1", i, "caller", "x", "hi-IN", {})
    await repo.flush()
    assert len(await repo.turns_for("c1")) > 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_call_repository.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vaani.db.repository'`

- [ ] **Step 3: Write the models**

Create `vaani/db/models.py`:

```python
"""Durable call records.

For a government deployment the transcript and its timings are the audit trail,
so they outlive the process. Deliberately two flat tables and no migrations:
phase 1 runs on SQLite at pilot scale, and Alembic arrives with Postgres.
"""

from __future__ import annotations

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class CallRow(Base):
    __tablename__ = "calls"

    call_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    agent_key: Mapped[str] = mapped_column(String(64), index=True)
    direction: Mapped[str] = mapped_column(String(16), default="inbound")
    caller_number: Mapped[str | None] = mapped_column(String(32), nullable=True)
    started_at: Mapped[float] = mapped_column(Float, index=True)
    ended_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    outcome: Mapped[str] = mapped_column(String(32), default="in_progress", index=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    recording_path: Mapped[str | None] = mapped_column(Text, nullable=True)


class TurnRow(Base):
    __tablename__ = "turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("calls.call_id"), index=True
    )
    seq: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(16))
    text: Mapped[str] = mapped_column(Text, default="")
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    stt_ms: Mapped[int] = mapped_column(Integer, default=0)
    agent_ms: Mapped[int] = mapped_column(Integer, default=0)
    tts_first_chunk_ms: Mapped[int] = mapped_column(Integer, default=0)
    total_ms: Mapped[int] = mapped_column(Integer, default=0)
    barged_in: Mapped[bool] = mapped_column(Boolean, default=False)
```

- [ ] **Step 4: Write the repository**

Create `vaani/db/repository.py`:

```python
"""Call persistence, kept off the call path.

One uvicorn worker carries every concurrent call on a single event loop, and
SQLite serialises writers. A synchronous insert inside the turn loop would
therefore stall every other live call for the duration of the write, so turn
records are queued and drained by one background task.

The queue is bounded and drops on overflow. Losing a turn record is bad; adding
latency to a live conversation because the disk is busy is worse.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from vaani.core.logging import get_logger
from vaani.db.models import Base, CallRow, TurnRow

log = get_logger(__name__)

_QUEUE_MAX = 2000


class CallRepository:
    def __init__(self, database_url: str) -> None:
        self._url = database_url
        self._engine: Any = None
        self._sessions: Any = None
        self._queue: asyncio.Queue[TurnRow] = asyncio.Queue(maxsize=_QUEUE_MAX)
        self._writer: asyncio.Task[None] | None = None
        self._dropped = 0

    async def start(self) -> None:
        self._engine = create_async_engine(self._url, future=True)
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self._sessions = async_sessionmaker(self._engine, expire_on_commit=False)
        self._writer = asyncio.create_task(self._drain(), name="call-writer")
        log.info("call repository ready")

    # -- write ---------------------------------------------------------------

    async def create_call(self, record: Any) -> None:
        async with self._sessions() as session, session.begin():
            session.add(
                CallRow(
                    call_id=record.call_id,
                    agent_key=record.agent_key,
                    direction=record.direction,
                    caller_number=record.caller_number,
                    started_at=record.started_at,
                    outcome=record.outcome,
                    language=getattr(record, "language", None),
                )
            )

    def append_turn(
        self,
        call_id: str,
        seq: int,
        role: str,
        text: str,
        language: str | None,
        metrics: dict[str, Any],
    ) -> None:
        """Synchronous by design: called from the turn loop, must not await."""
        row = TurnRow(
            call_id=call_id,
            seq=seq,
            role=role,
            text=text,
            language=language,
            stt_ms=int(metrics.get("stt_ms", 0)),
            agent_ms=int(metrics.get("agent_ms", 0)),
            tts_first_chunk_ms=int(metrics.get("tts_first_chunk_ms", 0)),
            total_ms=int(metrics.get("total_ms", 0)),
            barged_in=bool(metrics.get("barged_in", False)),
        )
        try:
            self._queue.put_nowait(row)
        except asyncio.QueueFull:
            self._dropped += 1
            log.warning("turn write queue full", extra={"dropped": self._dropped})

    async def finish_call(self, record: Any) -> None:
        await self.flush()
        async with self._sessions() as session, session.begin():
            row = await session.get(CallRow, record.call_id)
            if row is None:
                return
            row.ended_at = record.ended_at
            row.outcome = record.outcome
            row.summary = record.summary
            row.duration_s = getattr(record, "duration_s", 0.0)
            row.language = getattr(record, "language", None)
            row.recording_path = getattr(record, "recording_path", None)

    async def flush(self) -> None:
        """Wait for queued turns to land. Used at call end and in tests."""
        await self._queue.join()

    # -- read ----------------------------------------------------------------

    async def get_call(self, call_id: str) -> dict[str, Any] | None:
        async with self._sessions() as session:
            row = await session.get(CallRow, call_id)
            return _as_dict(row) if row else None

    async def turns_for(self, call_id: str) -> list[dict[str, Any]]:
        async with self._sessions() as session:
            result = await session.execute(
                select(TurnRow).where(TurnRow.call_id == call_id).order_by(TurnRow.seq)
            )
            return [_as_dict(row) for row in result.scalars()]

    # -- lifecycle -----------------------------------------------------------

    async def close(self) -> None:
        await self.flush()
        if self._writer is not None:
            self._writer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._writer
            self._writer = None
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None

    async def _drain(self) -> None:
        while True:
            row = await self._queue.get()
            try:
                async with self._sessions() as session, session.begin():
                    session.add(row)
            except Exception:
                log.exception("failed to persist turn", extra={"call_id": row.call_id})
            finally:
                self._queue.task_done()


def _as_dict(row: Any) -> dict[str, Any]:
    return {c.name: getattr(row, c.name) for c in row.__table__.columns}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_call_repository.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 6: Declare the retention setting**

A government deployment will be asked how long transcripts are kept, so the knob
is declared now even though enforcement lands in phase 3. Add to `vaani/config.py`
in the "Storage" group, next to `record_calls`:

```python
    retention_days: int = cfg(
        365, group="Storage", label="Retain records for (days)", ge=1, le=3650,
        restart=False,
        help="How long call records, transcripts and recordings are kept. "
             "Declared now for the audit conversation; automatic deletion "
             "arrives with the Postgres migration.",
    )
```

No `changed(...)` entry is needed: `restart=False` and nothing reads it yet.

- [ ] **Step 7: Lint and commit**

```bash
.venv/Scripts/python.exe -m pytest -q && .venv/Scripts/python.exe -m ruff check .
git add vaani/db/ vaani/config.py tests/test_call_repository.py
git commit -m "feat: durable call and turn records on SQLite"
```

---

## Task 6: Wire persistence into the call session

**Files:**
- Modify: `vaani/core/registry.py` (`Services` gains `calls`)
- Modify: `vaani/main.py` (start and close the repository)
- Modify: `vaani/pipeline/session.py` (record at start, per turn, at end)
- Modify: `vaani/api/routes.py` (serve history from the database)
- Test: `tests/test_session_persistence.py`

**Interfaces:**
- Consumes: `CallRepository` from Task 5; `LanguageTracker` from Task 4.
- Produces: `Services.calls: CallRepository | None`; `CallSession` persists automatically when it is set.

- [ ] **Step 1: Write the failing test**

Create `tests/test_session_persistence.py`:

```python
"""A completed call must be readable from the database afterwards."""

from __future__ import annotations

import pytest

from vaani.config import Settings
from vaani.core.registry import build_services
from vaani.db.repository import CallRepository
from vaani.pipeline.session import CallSession

from .test_pipeline import FakeTransport, silence, tone


@pytest.fixture
async def services_with_db(tmp_path):
    s = Settings(
        stt_provider="mock", llm_provider="mock", tts_provider="mock",
        vector_store="memory", embedding_provider="hash", record_calls=False,
        end_of_turn_silence_ms=200, idle_prompt_after_s=120, idle_hangup_after_s=600,
    )
    svc = build_services(s)
    svc.calls = CallRepository(f"sqlite+aiosqlite:///{tmp_path / 'calls.db'}")
    await svc.calls.start()
    await svc.start()
    yield svc
    await svc.close()
    await svc.calls.close()


async def test_a_call_is_persisted_with_its_turns(services_with_db):
    svc = services_with_db
    transport = FakeTransport()
    session = CallSession(
        transport=transport, services=svc, agent_key="default",
        caller_number="+919876543210",
    )
    import asyncio

    task = asyncio.create_task(session.run())
    await asyncio.sleep(0.1)
    await session.push_audio(tone(700))
    await session.push_audio(silence(500))
    await asyncio.sleep(0.6)
    await session.hangup("completed")
    await asyncio.wait_for(task, timeout=10)

    row = await svc.calls.get_call(session.call_id)
    assert row is not None
    assert row["caller_number"] == "+919876543210"
    assert row["outcome"] == "completed"
    assert row["ended_at"] is not None

    turns = await svc.calls.turns_for(session.call_id)
    assert turns, "expected at least one persisted turn"
    assert turns[0]["seq"] == 0


async def test_sessions_run_fine_without_a_repository(tmp_path):
    """Persistence is optional: a bare install must still place calls."""
    s = Settings(
        stt_provider="mock", llm_provider="mock", tts_provider="mock",
        record_calls=False, end_of_turn_silence_ms=200,
        idle_prompt_after_s=120, idle_hangup_after_s=600,
    )
    svc = build_services(s)
    await svc.start()
    assert svc.calls is None

    import asyncio

    session = CallSession(transport=FakeTransport(), services=svc)
    task = asyncio.create_task(session.run())
    await asyncio.sleep(0.1)
    await session.hangup("completed")
    await asyncio.wait_for(task, timeout=10)
    await svc.close()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_session_persistence.py -v`
Expected: FAIL — `AttributeError: 'Services' object has no attribute 'calls'`

- [ ] **Step 3: Add `calls` to `Services`**

In `vaani/core/registry.py`, add to the `Services` dataclass, after `profiles`:

```python
    # Optional so a bare install still places calls with nothing persisted.
    calls: Any = None
```

- [ ] **Step 4: Start and stop the repository in the app lifespan**

In `vaani/main.py`, inside `lifespan`, after `services = build_services(settings)` and before `await services.start()`:

```python
    repository = CallRepository(settings.database_url)
    await repository.start()
    services.calls = repository
```

and in the `finally` block, after `await services.close()`:

```python
        await repository.close()
```

Add the import: `from vaani.db.repository import CallRepository`.

- [ ] **Step 5: Record from the session**

In `vaani/pipeline/session.py`:

At the start of `run()`, after the call id context is set:

```python
        if self._services.calls is not None:
            await self._services.calls.create_call(self.record)
```

In `_process_turn`, after the turn is appended to `self.record.turns`, persist both sides:

```python
        if self._services.calls is not None:
            seq = len(self.record.turns) - 1
            self._services.calls.append_turn(
                self.call_id, seq * 2, "caller", transcript.text,
                self._language.current, asdict(metrics),
            )
            self._services.calls.append_turn(
                self.call_id, seq * 2 + 1, "agent", turn.text,
                self._language.current, asdict(metrics),
            )
```

In `_finish`, after `self.record.outcome` is settled and the summary is generated:

```python
        if self._services.calls is not None:
            with contextlib.suppress(Exception):
                await self._services.calls.finish_call(self.record)
```

- [ ] **Step 6: Serve history from the database**

In `vaani/api/routes.py`, change `call_history` to read persisted calls when a repository is present, falling back to the in-memory manager otherwise:

```python
@router.get("/calls", tags=["calls"])
async def call_history(
    request: Request, limit: int = Query(50, ge=1, le=500)
) -> list[dict[str, Any]]:
    repository = request.app.state.services.calls
    if repository is None:
        return request.app.state.calls.history(limit)
    return await repository.recent(limit)
```

Add `recent` to `CallRepository` in `vaani/db/repository.py`:

```python
    async def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        async with self._sessions() as session:
            result = await session.execute(
                select(CallRow).order_by(CallRow.started_at.desc()).limit(limit)
            )
            return [_as_dict(row) for row in result.scalars()]
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_session_persistence.py -v`
Expected: PASS, 2 tests.

- [ ] **Step 8: Run the whole suite and commit**

Run: `.venv/Scripts/python.exe -m pytest -q && .venv/Scripts/python.exe -m ruff check .`

```bash
git add vaani/core/registry.py vaani/main.py vaani/pipeline/session.py vaani/api/routes.py vaani/db/repository.py tests/test_session_persistence.py
git commit -m "feat: persist calls and turns from the session"
```

---

## Task 7: Exotel frame codec

**Files:**
- Create: `vaani/telephony/exotel_frames.py`
- Test: `tests/test_exotel_frames.py`

**Interfaces:**
- Consumes: `SAMPLE_RATE` from `vaani.config`; `ulaw_to_pcm16`, `pcm16_to_ulaw` from `vaani.audio.resample`.
- Produces: `parse_frame(raw: str | bytes) -> ExotelEvent` and `media_frame(stream_sid: str, pcm: bytes) -> str`, `clear_frame(stream_sid: str) -> str`. `ExotelEvent` is a dataclass with `kind: str`, `pcm: bytes | None`, `stream_sid: str | None`, `call_sid: str | None`, `caller: str | None`.

**Why this is its own module:** the wire format is the one thing in phase 1 not confirmed (Task 0). Isolating it means a format correction changes one small, fixture-tested file and nothing else.

- [ ] **Step 1: Write the failing test**

Create `tests/test_exotel_frames.py`:

```python
"""Exotel frame codec.

Fixtures encode the assumed format from Task 0. If the real format differs,
change these fixtures and the codec together — nothing else in the codebase
should need to move."""

from __future__ import annotations

import base64
import json

from vaani.config import SAMPLE_RATE
from vaani.telephony.exotel_frames import clear_frame, media_frame, parse_frame


def _pcm(ms: int) -> bytes:
    return b"\x02\x00" * int(SAMPLE_RATE * ms / 1000)


def test_parses_the_start_event_and_its_identifiers():
    raw = json.dumps({
        "event": "start",
        "stream_sid": "s-1",
        "start": {"call_sid": "c-1", "from": "+919876543210", "to": "+911800123456"},
    })
    event = parse_frame(raw)
    assert event.kind == "start"
    assert event.stream_sid == "s-1"
    assert event.call_sid == "c-1"
    assert event.caller == "+919876543210"


def test_parses_media_into_pipeline_pcm():
    payload = base64.b64encode(_pcm(100)).decode()
    raw = json.dumps({"event": "media", "stream_sid": "s-1",
                      "media": {"payload": payload}})
    event = parse_frame(raw)
    assert event.kind == "media"
    assert event.pcm == _pcm(100)


def test_parses_stop_and_connected():
    assert parse_frame(json.dumps({"event": "stop"})).kind == "stop"
    assert parse_frame(json.dumps({"event": "connected"})).kind == "connected"


def test_unknown_event_is_reported_not_raised():
    """An unrecognised event must not tear down a live call."""
    assert parse_frame(json.dumps({"event": "mark"})).kind == "mark"


def test_malformed_frame_is_reported_not_raised():
    assert parse_frame("not json").kind == "invalid"
    assert parse_frame(json.dumps({"no_event": 1})).kind == "invalid"


def test_media_frame_round_trips():
    frame = json.loads(media_frame("s-1", _pcm(20)))
    assert frame["event"] == "media"
    assert frame["stream_sid"] == "s-1"
    assert base64.b64decode(frame["media"]["payload"]) == _pcm(20)


def test_clear_frame_targets_the_stream():
    frame = json.loads(clear_frame("s-1"))
    assert frame["event"] == "clear"
    assert frame["stream_sid"] == "s-1"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_exotel_frames.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vaani.telephony.exotel_frames'`

- [ ] **Step 3: Write the implementation**

Create `vaani/telephony/exotel_frames.py`:

```python
"""Exotel streaming wire format.

Exotel's recommended media specification is 100 ms chunks of 3200 bytes, which
is 16 kHz PCM16 mono — the same format the pipeline uses, so this path needs no
resampling, unlike the 8 kHz AudioSocket bridge.

The exact envelope is confirmed in docs/exotel-protocol.md. It lives in its own
module so a correction touches one fixture-tested file and nothing downstream.
Every parse failure is reported as an event kind rather than raised: an
unrecognised frame must never tear down a live call.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass

from vaani.core.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class ExotelEvent:
    kind: str
    pcm: bytes | None = None
    stream_sid: str | None = None
    call_sid: str | None = None
    caller: str | None = None


def parse_frame(raw: str | bytes) -> ExotelEvent:
    try:
        frame = json.loads(raw)
    except (ValueError, TypeError):
        log.warning("unparseable exotel frame")
        return ExotelEvent(kind="invalid")

    if not isinstance(frame, dict) or "event" not in frame:
        return ExotelEvent(kind="invalid")

    kind = str(frame["event"])
    stream_sid = frame.get("stream_sid") or frame.get("streamSid")

    if kind == "media":
        payload = (frame.get("media") or {}).get("payload") or ""
        try:
            pcm = base64.b64decode(payload)
        except (ValueError, TypeError):
            log.warning("undecodable exotel media payload")
            return ExotelEvent(kind="invalid", stream_sid=stream_sid)
        return ExotelEvent(kind="media", pcm=pcm, stream_sid=stream_sid)

    if kind == "start":
        start = frame.get("start") or {}
        return ExotelEvent(
            kind="start",
            stream_sid=stream_sid,
            call_sid=start.get("call_sid") or start.get("callSid"),
            caller=start.get("from"),
        )

    return ExotelEvent(kind=kind, stream_sid=stream_sid)


def media_frame(stream_sid: str, pcm: bytes) -> str:
    return json.dumps({
        "event": "media",
        "stream_sid": stream_sid,
        "media": {"payload": base64.b64encode(pcm).decode()},
    })


def clear_frame(stream_sid: str) -> str:
    """Discard audio Exotel has buffered but not yet played.

    Without this, barge-in cancels synthesis locally while the caller keeps
    hearing several seconds of already-sent agent audio — which reads as the
    agent ignoring them.
    """
    return json.dumps({"event": "clear", "stream_sid": stream_sid})
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_exotel_frames.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Reconcile with the real format**

Compare against `docs/exotel-protocol.md` from Task 0. If the envelope differs — different key names, binary frames instead of base64, or 8 kHz μ-law — update this module and its fixtures together. If the audio is 8 kHz μ-law, convert with `ulaw_to_pcm16` / `pcm16_to_ulaw` from `vaani.audio.resample` inside `parse_frame` and `media_frame`, keeping every other module unchanged.

- [ ] **Step 6: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check .
git add vaani/telephony/exotel_frames.py tests/test_exotel_frames.py
git commit -m "feat: Exotel streaming frame codec"
```

---

## Task 8: Exotel transport and WebSocket route

**Files:**
- Create: `vaani/api/exotel_ws.py`
- Modify: `vaani/main.py` (include the router)
- Test: `tests/test_exotel_transport.py`

**Interfaces:**
- Consumes: `parse_frame`, `media_frame`, `clear_frame`, `ExotelEvent` from Task 7; `CallSession` from `vaani.pipeline.session`.
- Produces: `ExotelTransport(ws, stream_sid: str = "")` satisfying the `Transport` protocol; a `router` exposing `WS /ws/exotel`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_exotel_transport.py`:

```python
"""The transport is the only telephony-aware code in the call path. It must
convert both directions and cancel buffered carrier audio on barge-in."""

from __future__ import annotations

import base64
import json

from vaani.api.exotel_ws import ExotelTransport
from vaani.config import SAMPLE_RATE


class FakeWS:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed = False

    async def send_text(self, text: str) -> None:
        self.sent.append(text)

    async def close(self) -> None:
        self.closed = True


def _pcm(ms: int) -> bytes:
    return b"\x03\x00" * int(SAMPLE_RATE * ms / 1000)


async def test_audio_is_sent_as_media_frames():
    ws = FakeWS()
    transport = ExotelTransport(ws, stream_sid="s-1")
    await transport.send_audio(_pcm(20))

    frame = json.loads(ws.sent[0])
    assert frame["event"] == "media"
    assert base64.b64decode(frame["media"]["payload"]) == _pcm(20)


async def test_barge_in_event_clears_buffered_carrier_audio():
    """The agent stops synthesising locally, but Exotel is still holding queued
    audio. Without a clear it keeps playing and the caller is talked over."""
    ws = FakeWS()
    transport = ExotelTransport(ws, stream_sid="s-1")
    await transport.send_event({"type": "barge_in"})

    assert json.loads(ws.sent[0])["event"] == "clear"


async def test_non_barge_in_events_are_not_sent_to_the_carrier():
    """Exotel has no use for transcripts or metrics, and unknown frames risk
    the session being torn down."""
    ws = FakeWS()
    transport = ExotelTransport(ws, stream_sid="s-1")
    await transport.send_event({"type": "transcript", "text": "नमस्ते"})
    await transport.send_event({"type": "state", "state": "listening"})

    assert ws.sent == []


async def test_send_after_failure_is_silent():
    class BrokenWS(FakeWS):
        async def send_text(self, text: str) -> None:
            raise RuntimeError("socket gone")

    transport = ExotelTransport(BrokenWS(), stream_sid="s-1")
    await transport.send_audio(_pcm(20))
    await transport.send_audio(_pcm(20))  # must not raise


async def test_close_closes_the_socket():
    ws = FakeWS()
    await ExotelTransport(ws, stream_sid="s-1").close()
    assert ws.closed is True
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_exotel_transport.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vaani.api.exotel_ws'`

- [ ] **Step 3: Write the implementation**

Create `vaani/api/exotel_ws.py`:

```python
"""Exotel streaming transport.

This is the only telephony-aware code the call path touches: everything below
Transport is carrier-agnostic, which is why the same CallSession serves a
browser, an Asterisk trunk and Exotel without modification.

The stream identifier only arrives on the start event, so the transport is
constructed before it is known and told afterwards.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from fastapi import APIRouter, Query, WebSocket

from vaani.core.logging import get_logger
from vaani.pipeline.manager import CallCapacityError
from vaani.pipeline.session import CallSession
from vaani.telephony.exotel_frames import clear_frame, media_frame, parse_frame

log = get_logger(__name__)
router = APIRouter()


class ExotelTransport:
    def __init__(self, ws: Any, stream_sid: str = "") -> None:
        self._ws = ws
        self.stream_sid = stream_sid
        self._open = True
        # FastAPI's WebSocket is not safe for concurrent writes and audio
        # streaming races event emission constantly.
        self._lock = asyncio.Lock()

    async def send_audio(self, pcm: bytes) -> None:
        if not self._open or not self.stream_sid:
            return
        async with self._lock:
            try:
                await self._ws.send_text(media_frame(self.stream_sid, pcm))
            except Exception:
                self._open = False

    async def send_event(self, event: dict[str, Any]) -> None:
        # Exotel has no use for transcripts, state or metrics, and unrecognised
        # frames risk the stream being dropped. Barge-in is the one event that
        # must reach the carrier.
        if event.get("type") != "barge_in":
            return
        if not self._open or not self.stream_sid:
            return
        async with self._lock:
            try:
                await self._ws.send_text(clear_frame(self.stream_sid))
            except Exception:
                self._open = False

    async def close(self) -> None:
        self._open = False
        with contextlib.suppress(Exception):
            await self._ws.close()


@router.websocket("/ws/exotel")
async def exotel_stream(ws: WebSocket, agent: str = Query("default")) -> None:
    services = ws.app.state.services
    manager = ws.app.state.calls

    await ws.accept()
    transport = ExotelTransport(ws)
    session: CallSession | None = None
    session_task: asyncio.Task[Any] | None = None

    try:
        while True:
            raw = await ws.receive_text()
            event = parse_frame(raw)

            if event.kind == "start":
                transport.stream_sid = event.stream_sid or ""
                session = CallSession(
                    transport=transport,
                    services=services,
                    agent_key=agent,
                    caller_number=event.caller,
                    direction="inbound",
                )
                try:
                    await manager.register(session)
                except CallCapacityError:
                    # Every line is busy. Hanging up immediately is kinder than
                    # a caller listening to silence.
                    log.warning("rejected exotel call at capacity")
                    await transport.close()
                    return
                session_task = asyncio.create_task(
                    session.run(), name=f"exotel:{session.call_id}"
                )
                log.info(
                    "exotel call started",
                    extra={"call_id": session.call_id, "call_sid": event.call_sid},
                )

            elif event.kind == "media" and session is not None and event.pcm:
                await session.push_audio(event.pcm)

            elif event.kind == "stop":
                break

    except Exception:
        log.info("exotel stream closed")
    finally:
        if session is not None:
            await session.hangup("caller_disconnected")
            if session_task is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await asyncio.wait_for(session_task, timeout=10)
            await manager.unregister(session)
        await transport.close()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_exotel_transport.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Register the router**

In `vaani/main.py`, next to the existing router registrations:

```python
    app.include_router(exotel_ws.router)
```

and extend the import: `from vaani.api import exotel_ws, routes, settings as settings_api, ws_voice`.

- [ ] **Step 6: Declare the Exotel settings**

Add to `vaani/config.py` in the "Service" group:

```python
    exotel_enabled: bool = cfg(
        False, group="Service", label="Accept Exotel calls",
        help="Exposes the streaming endpoint at /ws/exotel. Leave off until the "
             "Voicebot applet is provisioned on your account.",
    )
    exotel_account_sid: str | None = cfg(
        None, group="Service", label="Exotel account SID",
        depends_on={"exotel_enabled": ["true"]},
    )
    exotel_api_key: str | None = cfg(
        None, group="Service", label="Exotel API key", secret=True,
        depends_on={"exotel_enabled": ["true"]},
    )
```

These configure the carrier side and are read at request time, so no
`changed(...)` entry is required.

Then gate the route registration in `vaani/main.py`:

```python
    if settings.exotel_enabled:
        app.include_router(exotel_ws.router)
```

Add a wiring test to `tests/test_exotel_transport.py`:

```python
def test_exotel_settings_exist_and_the_key_is_secret():
    from vaani.config import Settings

    assert Settings().exotel_enabled is False
    assert "exotel_api_key" in Settings.secret_fields()
```

- [ ] **Step 7: Confirm the session emits a barge-in event**

Barge-in only reaches the carrier if the session emits `{"type": "barge_in"}`. Check:

```bash
grep -n "barge_in" vaani/pipeline/session.py
```

If no such event is emitted, add one where playback is cancelled:

```python
            await self._transport.send_event({"type": "barge_in"})
```

- [ ] **Step 8: Run the whole suite and commit**

Run: `.venv/Scripts/python.exe -m pytest -q && .venv/Scripts/python.exe -m ruff check .`

```bash
git add vaani/api/exotel_ws.py vaani/main.py vaani/config.py vaani/pipeline/session.py tests/test_exotel_transport.py
git commit -m "feat: Exotel streaming transport and websocket route"
```

---

## Task 9: Golden audio and latency harness

**Files:**
- Create: `tests/golden/README.md`, `tests/test_golden_audio.py`
- Modify: `pyproject.toml` (register the `golden` marker)
- Test: itself

**Interfaces:**
- Consumes: `SarvamSTT` from Task 1; `Settings`, `build_services` from the registry.
- Produces: `pytest -m golden` measuring word error rate and time-to-first-audio.

**This is the acceptance gate.** Without it, "Hinglish works" is an opinion.

- [ ] **Step 1: Register the marker and document the corpus**

In `pyproject.toml`, under `[tool.pytest.ini_options]`:

```toml
markers = [
    "golden: real audio against live speech APIs; needs network and VAANI_SARVAM_API_KEY",
]
addopts = "-m 'not golden'"
```

Create `tests/golden/README.md`:

```markdown
# Golden audio set

Thirty to fifty real recorded utterances that decide whether this platform is
good enough for Indian callers. Not synthetic: record real speakers, over a
phone line if possible, including background noise.

## Layout

One WAV per utterance, PCM16 mono 16 kHz, alongside a `manifest.json`:

```json
[
  {"file": "hi-001.wav", "language": "hi-IN",
   "transcript": "मेरा बिजली का बिल कितना है"},
  {"file": "hinglish-001.wav", "language": "hi-IN",
   "transcript": "मेरा electricity bill pending hai kya"}
]
```

## Coverage required before phase 1 is accepted

- 10 Hindi, clean
- 10 code-mixed Hinglish — the case most likely to fail
- 5 English with an Indian accent
- 5 with background noise or a poor line
- 5 numbers and dates spoken naturally ("पंद्रह तारीख", "बारह सौ रुपये")

These files are recordings of real people. Do not commit them if they contain
personal data — keep them outside the repository and point `VAANI_GOLDEN_DIR`
at the directory.
```

- [ ] **Step 2: Write the harness**

Create `tests/test_golden_audio.py`:

```python
"""Real-audio accuracy and latency. Opt-in: `pytest -m golden`.

Excluded from the default suite because it needs network and a paid API key.
It is nonetheless the gate for phase 1 — everything else proves the plumbing
works, and only this proves the product does."""

from __future__ import annotations

import json
import os
import time
import wave
from pathlib import Path

import pytest

from vaani.config import SAMPLE_RATE
from vaani.providers.stt.sarvam import SarvamSTT

pytestmark = pytest.mark.golden

GOLDEN_DIR = Path(os.environ.get("VAANI_GOLDEN_DIR", "tests/golden"))
MAX_WER = float(os.environ.get("VAANI_MAX_WER", "0.25"))


def _manifest() -> list[dict]:
    path = GOLDEN_DIR / "manifest.json"
    if not path.exists():
        pytest.skip(f"no golden manifest at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        assert wav.getframerate() == SAMPLE_RATE, f"{path} must be {SAMPLE_RATE} Hz"
        assert wav.getnchannels() == 1, f"{path} must be mono"
        return wav.readframes(wav.getnframes())


def _wer(reference: str, hypothesis: str) -> float:
    """Levenshtein distance over words, normalised by reference length."""
    ref, hyp = reference.split(), hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i]
        for j, h in enumerate(hyp, 1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (r != h),
            ))
        previous = current
    return previous[-1] / len(ref)


@pytest.fixture(scope="module")
async def stt():
    key = os.environ.get("VAANI_SARVAM_API_KEY")
    if not key:
        pytest.skip("VAANI_SARVAM_API_KEY not set")
    provider = SarvamSTT(api_key=key)
    await provider.start()
    yield provider
    await provider.close()


async def test_word_error_rate_is_acceptable(stt):
    entries = _manifest()
    results = []
    for entry in entries:
        pcm = _read_wav(GOLDEN_DIR / entry["file"])
        started = time.perf_counter()
        result = await stt.transcribe(pcm)
        elapsed_ms = (time.perf_counter() - started) * 1000
        results.append({
            "file": entry["file"],
            "wer": _wer(entry["transcript"], result.text),
            "ms": elapsed_ms,
            "expected": entry["transcript"],
            "got": result.text,
        })

    worst = sorted(results, key=lambda r: -r["wer"])[:5]
    print("\nWorst five:")
    for r in worst:
        print(f"  {r['file']}  WER {r['wer']:.2f}  {r['ms']:.0f}ms")
        print(f"    expected: {r['expected']}")
        print(f"    got     : {r['got']}")

    mean_wer = sum(r["wer"] for r in results) / len(results)
    print(f"\nmean WER {mean_wer:.3f} over {len(results)} utterances")
    assert mean_wer <= MAX_WER, f"mean WER {mean_wer:.3f} exceeds {MAX_WER}"


async def test_transcription_latency_fits_the_turn_budget(stt):
    """The budget is ~300 ms of the 1.5 s p95 time-to-first-audio target."""
    entries = _manifest()
    timings = []
    for entry in entries[:10]:
        pcm = _read_wav(GOLDEN_DIR / entry["file"])
        started = time.perf_counter()
        await stt.transcribe(pcm)
        timings.append((time.perf_counter() - started) * 1000)

    timings.sort()
    p95 = timings[max(0, int(len(timings) * 0.95) - 1)]
    print(f"\nSTT p95 {p95:.0f}ms over {len(timings)} utterances")
    assert p95 <= 800, f"STT p95 {p95:.0f}ms leaves no room in the turn budget"


async def test_code_mixed_utterances_are_not_systematically_worse(stt):
    """Hinglish is the case most likely to fail, and the one that matters most.
    If it is much worse than clean Hindi, the language strategy is wrong."""
    entries = _manifest()
    mixed = [e for e in entries if "hinglish" in e["file"]]
    clean = [e for e in entries if e["file"].startswith("hi-")]
    if not mixed or not clean:
        pytest.skip("manifest lacks both hinglish and clean hindi samples")

    async def mean_wer(items):
        total = 0.0
        for entry in items:
            result = await stt.transcribe(_read_wav(GOLDEN_DIR / entry["file"]))
            total += _wer(entry["transcript"], result.text)
        return total / len(items)

    mixed_wer, clean_wer = await mean_wer(mixed), await mean_wer(clean)
    print(f"\nhinglish WER {mixed_wer:.3f} vs clean hindi {clean_wer:.3f}")
    assert mixed_wer <= clean_wer + 0.15, (
        f"code-mixed WER {mixed_wer:.3f} far worse than clean {clean_wer:.3f}"
    )
```

- [ ] **Step 3: Verify the default suite still excludes it**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: the previous test count, no network access, no skips from this file.

- [ ] **Step 4: Verify the marker selects it**

Run: `.venv/Scripts/python.exe -m pytest -m golden -v`
Expected: three tests, all skipped with "no golden manifest" or "VAANI_SARVAM_API_KEY not set".

- [ ] **Step 5: Commit**

```bash
.venv/Scripts/python.exe -m ruff check .
git add tests/test_golden_audio.py tests/golden/README.md pyproject.toml
git commit -m "test: golden audio accuracy and latency harness"
```

---

## Acceptance

Phase 1 is complete when all of the following hold on a real inbound PSTN call:

- [ ] The number answers within two rings.
- [ ] p95 time-to-first-audio ≤ 1.5 s, measured from the persisted `turns` table:
      `SELECT AVG(total_ms), MAX(total_ms) FROM turns WHERE role='agent';`
- [ ] Barge-in works over the carrier — interrupting mid-sentence stops agent audio within roughly 500 ms. This is the criterion most likely to fail; carrier jitter is nothing like a local WebSocket.
- [ ] Mean WER on the golden set is at or below the agreed threshold, and code-mixed utterances are not materially worse than clean Hindi.
- [ ] The LLM replies in the caller's script — verified by reading persisted agent turns, not by listening.
- [ ] The agent ends the call via `end_call` rather than looping.
- [ ] Call and turn records survive a process restart.
- [ ] `pytest -q` and `ruff check .` are both clean.

## Deferred to later phases

Outcome taxonomy and call objectives (phase 2), Postgres and retention enforcement (phase 3), the admin module (phase 4), the remaining five languages (phase 5), outbound and DLT compliance (phase 6).
