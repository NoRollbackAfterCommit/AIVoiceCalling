"""Place a synthetic call against a running server.

Verifies the media plane without a microphone: connects to /ws/call, streams
generated speech-shaped audio in real time, and asserts that the agent greets,
transcribes, replies with audio, and can be interrupted mid-sentence.

    python scripts/smoke_call.py [--url ws://localhost:8099/ws/call]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import struct
import sys

SAMPLE_RATE = 16_000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000


def frames(seconds: float, freq: float = 200.0, amplitude: float = 0.45) -> list[bytes]:
    """Amplitude-modulated tone. Not speech, but loud and continuous enough that
    the VAD treats it as an utterance."""
    total = int(SAMPLE_RATE * seconds)
    out: list[bytes] = []
    for start in range(0, total - FRAME_SAMPLES + 1, FRAME_SAMPLES):
        block = []
        for i in range(start, start + FRAME_SAMPLES):
            t = i / SAMPLE_RATE
            env = 0.6 + 0.4 * math.sin(2 * math.pi * 4 * t)
            block.append(int(amplitude * env * 32767 * math.sin(2 * math.pi * freq * t)))
        out.append(struct.pack(f"<{len(block)}h", *block))
    return out


def quiet(seconds: float) -> list[bytes]:
    count = int(seconds * 1000 / FRAME_MS)
    return [b"\x00\x00" * FRAME_SAMPLES] * count


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8099/ws/call?agent=default")
    parser.add_argument("--turns", type=int, default=2)
    args = parser.parse_args()

    try:
        import websockets
    except ImportError:
        print("pip install websockets", file=sys.stderr)
        return 1

    received_audio = 0
    events: list[dict] = []

    async with websockets.connect(args.url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", "client": "smoke"}))

        async def listen() -> None:
            nonlocal received_audio
            async for message in ws:
                if isinstance(message, bytes):
                    received_audio += len(message)
                    continue
                event = json.loads(message)
                events.append(event)
                kind = event.get("type")
                if kind == "state":
                    print(f"  [state] {event['state']}")
                elif kind == "transcript":
                    print(f"  {event['role']}: {event['text']}")
                elif kind == "speech":
                    print(f"  agent ({event.get('kind')}): {event['text']}")
                elif kind == "metrics":
                    print(
                        f"  [latency] stt {event['stt_ms']}ms · "
                        f"think {event['agent_ms']}ms · "
                        f"first audio {event['tts_first_chunk_ms']}ms · "
                        f"total {event['total_ms']}ms"
                    )
                elif kind == "barge_in":
                    print("  [barge-in] agent cut off by caller")
                elif kind == "call_end":
                    print(f"  [end] {event['record']['outcome']}")

        listener = asyncio.create_task(listen())

        async def speak(seconds: float, freq: float = 200.0) -> None:
            for frame in frames(seconds, freq):
                await ws.send(frame)
                await asyncio.sleep(FRAME_MS / 1000)  # real time, as a phone would

        async def pause(seconds: float) -> None:
            for frame in quiet(seconds):
                await ws.send(frame)
                await asyncio.sleep(FRAME_MS / 1000)

        await pause(1.2)  # let the greeting play
        for turn in range(args.turns):
            print(f"\n-- caller turn {turn + 1} --")
            await speak(1.0, freq=200 + turn * 40)
            await pause(1.2)  # trailing silence ends the turn
            await asyncio.sleep(0.4)

        # Talk over the agent mid-reply. Agent audio is paced to real time, so
        # by speaking immediately after its reply begins we land inside playback.
        print("\n-- interrupting the agent --")
        await speak(0.7)
        await pause(0.5)
        await speak(0.7)
        await pause(0.8)

        await ws.send(json.dumps({"type": "hangup"}))
        await asyncio.sleep(0.6)
        listener.cancel()

    kinds = {e.get("type") for e in events}
    checks = {
        "greeting spoken": any(
            e.get("type") == "speech" and e.get("kind") == "greeting" for e in events
        ),
        "caller transcribed": any(
            e.get("type") == "transcript" and e.get("role") == "caller" for e in events
        ),
        "agent replied": any(
            e.get("type") == "speech" and e.get("kind") == "reply" for e in events
        ),
        "agent audio received": received_audio > 0,
        "latency reported": "metrics" in kinds,
        "barge-in fired": "barge_in" in kinds,
    }

    print("\n" + "-" * 46)
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n  agent audio: {received_audio / (SAMPLE_RATE * 2):.1f}s")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
