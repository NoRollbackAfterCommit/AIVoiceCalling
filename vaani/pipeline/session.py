"""The call session — the state machine that is the actual product.

One instance per live call, transport-agnostic: the same object drives a browser
WebRTC call, a SIP call from Asterisk, and a WhatsApp voice call, because all any
of them has to provide is the `Transport` protocol below.

    GREETING ─▶ LISTENING ─▶ THINKING ─▶ SPEAKING ─┐
                    ▲                       │      │
                    └───────────────────────┘      │
                         (barge-in cancels) ───────┘
                                                   ▼
                                                 ENDED

Two things here are worth reading closely, because they are what separates a
voice agent that people tolerate from one they hang up on:

1. **Barge-in.** Playback runs as a cancellable task. The moment the caller
   produces sustained speech, that task is cancelled and the buffered audio is
   dropped. Anything less and interrupting the agent does nothing, which callers
   find infuriating.

2. **Latency accounting.** Every turn records time-to-first-audio broken down by
   stage. When a pilot is "too slow", this is the data that says whether it is
   STT, the model, or TTS — and they need very different fixes.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from vaani.agent.runtime import ConversationAgent
from vaani.agent.tools.base import ToolContext
from vaani.audio.vad import BargeInDetector, TurnDetector, iter_frames
from vaani.config import FRAME_BYTES, SAMPLE_RATE, Settings
from vaani.core.logging import call_id_var, get_logger
from vaani.core.registry import Services
from vaani.pipeline.language import LanguageTracker

log = get_logger(__name__)


class CallState:
    GREETING = "greeting"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    TRANSFERRING = "transferring"
    ENDED = "ended"


class Transport(Protocol):
    """What a session needs from whatever is carrying the call."""

    async def send_audio(self, pcm: bytes) -> None: ...
    async def send_event(self, event: dict[str, Any]) -> None: ...
    async def close(self) -> None: ...


@dataclass(slots=True)
class TurnMetrics:
    stt_ms: int = 0
    agent_ms: int = 0
    tts_first_chunk_ms: int = 0
    total_ms: int = 0
    audio_s: float = 0.0
    barged_in: bool = False


@dataclass
class CallRecord:
    call_id: str
    agent_key: str
    direction: str = "inbound"
    caller_number: str | None = None
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    turns: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    languages: set[str] = field(default_factory=set)
    # The language the agent settled on, as opposed to every one detected.
    language: str | None = None
    outcome: str = "in_progress"
    summary: str = ""

    @property
    def duration_s(self) -> float:
        return (self.ended_at or time.time()) - self.started_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "agent_key": self.agent_key,
            "direction": self.direction,
            "caller_number": self.caller_number,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": round(self.duration_s, 2),
            "turn_count": len(self.turns),
            "turns": self.turns,
            "tool_calls": self.tool_calls,
            "languages": sorted(self.languages),
            "outcome": self.outcome,
            "summary": self.summary,
        }


class CallSession:
    def __init__(
        self,
        transport: Transport,
        services: Services,
        *,
        agent_key: str = "default",
        caller_number: str | None = None,
        direction: str = "inbound",
        call_id: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.call_id = call_id or uuid.uuid4().hex[:16]
        self._transport = transport
        self._services = services
        self._settings = settings or services.settings
        self._profile = services.profile(agent_key)

        self.state = CallState.GREETING
        self.record = CallRecord(
            call_id=self.call_id,
            agent_key=agent_key,
            direction=direction,
            caller_number=caller_number,
        )

        self._tool_ctx = ToolContext(
            call_id=self.call_id,
            caller_number=caller_number,
            agent_key=agent_key,
            services=services.as_tool_services(),
        )
        self._agent = ConversationAgent(
            profile=self._profile,
            llm=services.llm,
            tools=services.tools,
            ctx=self._tool_ctx,
        )

        # Keyed off voices, not profile.languages: the latter is prose for the
        # prompt ("Hindi, English"), while the tracker needs BCP-47 codes that
        # match what STT reports. The first configured voice is the default.
        self._language = LanguageTracker(
            default=next(iter(self._profile.voices), "hi-IN"),
            voices=self._profile.voices,
        )

        self._turns = TurnDetector(
            silence_ms=self._settings.end_of_turn_silence_ms,
            max_utterance_s=self._settings.max_turn_audio_s,
        )
        self._barge_in = BargeInDetector(trigger_ms=self._settings.barge_in_ms)

        # Frames arrive from the transport faster than they are consumed during
        # a THINKING pause; the queue absorbs that without blocking the reader.
        self._inbox: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=500)
        self._partial = bytearray()
        self._speaking_task: asyncio.Task[None] | None = None
        self._turn_task: asyncio.Task[None] | None = None
        # The event loop only holds a weak reference to a running task, so a
        # fire-and-forget state emit can be garbage collected before it is ever
        # sent — the console then silently misses a state change. Holding the
        # reference until it completes is what makes the emit reliable.
        self._emits: set[asyncio.Task[None]] = set()
        self._done = asyncio.Event()
        self._last_caller_audio = time.monotonic()
        self._idle_prompted = False
        self._recording = bytearray() if self._settings.record_calls else None
        self._playout_lead_s = self._settings.playout_lead_ms / 1000

    # -- inbound audio ------------------------------------------------------

    async def push_audio(self, pcm: bytes) -> None:
        """Called by the transport for every chunk of caller audio."""
        if self.state == CallState.ENDED:
            return
        try:
            self._inbox.put_nowait(pcm)
        except asyncio.QueueFull:
            # Dropping the oldest is the right failure mode: stale audio is
            # worthless, and blocking here would stall the whole transport.
            with contextlib.suppress(asyncio.QueueEmpty):
                self._inbox.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                self._inbox.put_nowait(pcm)

    async def push_dtmf(self, digit: str) -> None:
        """Keypad input. IVR menus and anything numeric a caller would rather
        type than say — account numbers over a bad line, in particular."""
        log.info("dtmf", extra={"digit": digit})
        self._tool_ctx.state.setdefault("dtmf", "")
        self._tool_ctx.state["dtmf"] += digit
        await self._transport.send_event({"type": "dtmf", "digit": digit})

    # -- lifecycle ----------------------------------------------------------

    async def run(self) -> CallRecord:
        token = call_id_var.set(self.call_id)
        log.info(
            "call started",
            extra={"agent": self.record.agent_key, "direction": self.record.direction},
        )
        await self._emit_state()
        watchdog = asyncio.create_task(self._watchdog())
        # The consumer runs concurrently with the greeting rather than after it.
        # Awaiting the greeting first leaves nothing draining the inbox while the
        # agent talks, so the opening seconds of every call are deaf and the
        # caller cannot interrupt "Namaste, how may I help you" — which is
        # precisely the moment an impatient caller talks over.
        consumer = asyncio.create_task(self._consume())
        ended = asyncio.create_task(self._done.wait())
        try:
            await self._speak(self._profile.greeting, kind="greeting")
            if self.state != CallState.ENDED:
                self._set_state(CallState.LISTENING)
            # The call is over when the caller's audio stream ends or something
            # hangs up.
            await asyncio.wait({consumer, ended}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            self.record.outcome = "cancelled"
            raise
        except Exception:
            log.exception("call failed")
            self.record.outcome = "error"
        finally:
            for task in (watchdog, consumer, ended):
                task.cancel()
            for task in (watchdog, consumer, ended):
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            await self._finish()
            call_id_var.reset(token)
        return self.record

    async def hangup(self, outcome: str = "completed") -> None:
        if self.state == CallState.ENDED:
            return
        self.record.outcome = outcome
        self._set_state(CallState.ENDED)
        await self._cancel_playback()
        await self._inbox.put(None)
        self._done.set()

    # -- main loop ----------------------------------------------------------

    async def _consume(self) -> None:
        while self.state != CallState.ENDED:
            chunk = await self._inbox.get()
            if chunk is None:
                break
            if self._recording is not None:
                self._recording.extend(chunk)

            # Re-frame to exact 20 ms boundaries; transports deliver arbitrary
            # sizes and the VAD requires exact frames.
            self._partial.extend(chunk)
            usable = len(self._partial) - (len(self._partial) % FRAME_BYTES)
            if usable <= 0:
                continue
            block = bytes(self._partial[:usable])
            del self._partial[:usable]

            for frame in iter_frames(block):
                await self._handle_frame(frame)
                if self.state == CallState.ENDED:
                    return

    async def _handle_frame(self, frame: bytes) -> None:
        if self.state == CallState.SPEAKING:
            if self._barge_in.push(frame):
                log.info("barge-in")
                await self._cancel_playback()
                await self._transport.send_event({"type": "barge_in"})
                self._set_state(CallState.LISTENING)
                self._turns.reset()
                self._last_caller_audio = time.monotonic()
                # Fall through: this frame is the start of their utterance.
            else:
                return

        if self.state != CallState.LISTENING:
            return  # mid-THINKING audio is buffered by the queue, not processed

        event = self._turns.push(frame)
        if event is None:
            return
        if event.state == "speaking":
            self._last_caller_audio = time.monotonic()
            self._idle_prompted = False
            await self._transport.send_event({"type": "speech_start"})
            return
        if event.state == "ended" and event.audio:
            self._last_caller_audio = time.monotonic()
            # A turn takes seconds — STT, then the model, then paced playback.
            # Running it inline would block this loop for that whole time, and a
            # blocked loop is a deaf agent: nothing drains the inbox, so the
            # caller cannot interrupt the reply they are currently listening to.
            # The state is set synchronously so the detector does not open a
            # second turn in the gap before the task is scheduled.
            self._set_state(CallState.THINKING)
            previous = self._turn_task
            if previous and not previous.done():
                previous.cancel()
                log.info("superseded an in-flight turn")
            self._turn_task = asyncio.create_task(
                self._process_turn(event.audio, event.duration_s)
            )

    async def _process_turn(self, audio: bytes, audio_s: float) -> None:
        metrics = TurnMetrics(audio_s=round(audio_s, 2))
        turn_started = time.monotonic()

        # 1. Speech to text
        t0 = time.monotonic()
        transcript = await self._services.stt.transcribe(audio)
        metrics.stt_ms = int((time.monotonic() - t0) * 1000)

        if not transcript.text.strip():
            self._set_state(CallState.LISTENING)
            return

        if transcript.language:
            self.record.languages.add(transcript.language)
            self._tool_ctx.language = transcript.language

        # Hysteresis lives in the tracker: one mis-detected utterance must not
        # flip the agent's voice mid-call.
        self._language.observe(transcript.language)
        self.record.language = self._language.current

        log.info(
            "caller",
            extra={"text": transcript.text, "lang": transcript.language,
                   "stt_ms": metrics.stt_ms},
        )
        await self._transport.send_event(
            {
                "type": "transcript",
                "role": "caller",
                "text": transcript.text,
                "language": transcript.language,
            }
        )

        # 2. Think
        t0 = time.monotonic()
        turn = await self._agent.respond(transcript.text)
        metrics.agent_ms = int((time.monotonic() - t0) * 1000)

        if turn.tool_calls:
            self.record.tool_calls.extend(turn.tool_calls)

        log.info(
            "agent",
            extra={"text": turn.text, "agent_ms": metrics.agent_ms,
                   "tools": [t["name"] for t in turn.tool_calls]},
        )
        await self._transport.send_event(
            {"type": "transcript", "role": "agent", "text": turn.text}
        )

        # 3. Speak
        if turn.text:
            metrics.tts_first_chunk_ms = await self._speak(turn.text, kind="reply")
        metrics.total_ms = int((time.monotonic() - turn_started) * 1000)
        metrics.barged_in = self.state == CallState.LISTENING and turn.text != ""

        self.record.turns.append(
            {
                "caller": transcript.text,
                "agent": turn.text,
                "language": transcript.language,
                "tools": [t["name"] for t in turn.tool_calls],
                "metrics": asdict(metrics),
            }
        )
        await self._transport.send_event({"type": "metrics", **asdict(metrics)})

        # 4. Act on anything a tool asked the call to do
        action = turn.control.get("action")
        if action == "transfer":
            await self._transfer(turn.control)
            return
        if action == "hangup":
            await self.hangup("completed")
            return

        if self.state != CallState.ENDED:
            self._set_state(CallState.LISTENING)

    # -- speaking -----------------------------------------------------------

    async def _speak(self, text: str, *, kind: str = "reply") -> int:
        """Stream TTS to the caller at real time. Returns time-to-first-audio.

        Playback runs as a cancellable task so barge-in can kill it, and the
        `finished` event is awaited rather than the task itself so cancellation
        unblocks us instead of propagating.

        The pacing is the important part. TTS renders far faster than real time,
        so sending chunks as fast as they are produced pushes an entire
        utterance down the wire in milliseconds. The session would then leave
        SPEAKING while the caller is still listening to five seconds of buffered
        audio — and anything they say during that window is treated as a fresh
        turn instead of an interruption, so barge-in silently never fires.
        Sending is therefore throttled to the audio's own duration, keeping at
        most `_playout_lead_s` of slack ahead so the far end never starves.
        Telephony needs this anyway: RTP has to be paced at real time.
        """
        if not text.strip() or self.state == CallState.ENDED:
            return 0

        self._set_state(CallState.SPEAKING)
        self._barge_in.reset()
        first_chunk_ms = 0
        started = time.monotonic()
        finished = asyncio.Event()

        async def play() -> None:
            nonlocal first_chunk_ms
            sent_s = 0.0
            try:
                first = True
                voice = self._language.voice() or self._profile.voice
                async for chunk in self._services.tts.stream(
                    text, voice=voice
                ):
                    if not chunk:
                        continue
                    if first:
                        first_chunk_ms = int((time.monotonic() - started) * 1000)
                        first = False
                    await self._transport.send_audio(chunk)
                    sent_s += len(chunk) / (SAMPLE_RATE * 2)

                    ahead = sent_s - (time.monotonic() - started)
                    # sleep(0) on the fast path still yields, so a cancellation
                    # lands between chunks rather than after the whole utterance.
                    await asyncio.sleep(max(0.0, ahead - self._playout_lead_s))
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("tts failed")
            finally:
                finished.set()

        self._speaking_task = asyncio.create_task(play())
        await self._transport.send_event(
            {"type": "speech", "role": "agent", "kind": kind, "text": text}
        )

        # Race playback against a caller interruption; _handle_frame cancels the
        # task and the wait returns.
        with contextlib.suppress(asyncio.CancelledError):
            await finished.wait()

        self._speaking_task = None
        if self.state == CallState.SPEAKING:
            self._set_state(CallState.LISTENING)
            self._turns.reset()
        await self._transport.send_event({"type": "speech_end"})
        return first_chunk_ms

    async def _cancel_playback(self) -> None:
        task = self._speaking_task
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._speaking_task = None

    # -- escalation ---------------------------------------------------------

    async def _transfer(self, control: dict[str, Any]) -> None:
        self._set_state(CallState.TRANSFERRING)
        self.record.outcome = "transferred"
        log.info(
            "transferring",
            extra={"department": control.get("department"), "reason": control.get("reason")},
        )
        await self._transport.send_event(
            {
                "type": "transfer",
                "department": control.get("department", "general"),
                "reason": control.get("reason", ""),
                # The human picking up needs the context, not a cold hand-off.
                "transcript": self._agent.transcript,
            }
        )
        await self.hangup("transferred")

    # -- housekeeping -------------------------------------------------------

    async def _watchdog(self) -> None:
        """Guards against the two ways a call hangs forever: a caller who has
        walked away, and a session that never terminates."""
        try:
            while self.state != CallState.ENDED:
                await asyncio.sleep(1.0)
                if self.record.duration_s > self._settings.max_call_duration_s:
                    log.warning("max call duration reached")
                    await self.hangup("timeout")
                    return
                if self.state != CallState.LISTENING:
                    continue
                idle = time.monotonic() - self._last_caller_audio
                if not self._idle_prompted and idle > self._settings.idle_prompt_after_s:
                    self._idle_prompted = True
                    await self._speak("Are you still there?", kind="idle_prompt")
                elif idle > self._settings.idle_hangup_after_s:
                    log.info("hanging up on silence")
                    await self._speak(self._profile.closing, kind="closing")
                    await self.hangup("abandoned")
                    return
        except asyncio.CancelledError:
            raise

    async def _finish(self) -> None:
        await self._cancel_playback()
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._turn_task
        # Give the final state emit a moment to land, then drop it. Bounded
        # rather than awaited outright: a transport that has gone away without
        # erroring would otherwise hold the call open here forever.
        if self._emits:
            _, pending = await asyncio.wait(set(self._emits), timeout=1.0)
            for task in pending:
                task.cancel()
        self.record.ended_at = time.time()
        if self.record.outcome == "in_progress":
            self.record.outcome = "completed"
        if self.record.turns:
            self.record.summary = await self._agent.summarise()
        if self._recording:
            await self._save_recording()
        log.info(
            "call ended",
            extra={
                "outcome": self.record.outcome,
                "duration_s": round(self.record.duration_s, 1),
                "turns": len(self.record.turns),
            },
        )
        with contextlib.suppress(Exception):
            await self._transport.send_event(
                {"type": "call_end", "record": self.record.to_dict()}
            )
        with contextlib.suppress(Exception):
            await self._transport.close()

    async def _save_recording(self) -> None:
        import wave
        from pathlib import Path

        try:
            directory = Path(self._settings.recordings_dir)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{self.call_id}.wav"

            def write() -> None:
                with wave.open(str(path), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(SAMPLE_RATE)
                    wav.writeframes(bytes(self._recording or b""))

            await asyncio.to_thread(write)
            log.info("recording saved", extra={"path": str(path)})
        except Exception:
            log.exception("failed to save recording")

    def _set_state(self, state: str) -> None:
        if state == self.state:
            return
        self.state = state
        task = asyncio.create_task(self._emit_state())
        self._emits.add(task)
        task.add_done_callback(self._emits.discard)

    async def _emit_state(self) -> None:
        with contextlib.suppress(Exception):
            await self._transport.send_event({"type": "state", "state": self.state})
