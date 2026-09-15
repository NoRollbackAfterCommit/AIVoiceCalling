# Call quality: turn-taking, barge-in, and what concurrency does not cause

A voice agent fails in ways a chat agent cannot. The caller and the agent share
one channel, and every decision about *who is speaking now* is made from audio
energy alone. This documents the failure mode that has cost us the most time, the
settings that govern it, and the measurements behind the values we chose.

## The failure mode: talk-over collapse

A caller who keeps talking can end up hearing almost nothing. It looks like the
line is dead, and callers reasonably respond by talking more, which makes it
worse. Diagnosed 2026-09-15 from two production calls and reproduced in
isolation.

The cycle:

1. The agent starts a reply and audio reaches the caller.
2. The caller is still talking, so `barge_in_ms` of continuous speech trips the
   barge-in detector and the reply is cancelled.
3. The session returns to `LISTENING`, and the caller's turn only ends after
   `end_of_turn_silence_ms` of *continuous* silence. A caller mid-flow rarely
   pauses that long, so the turn runs for ten, twenty, thirty seconds.
4. Throughout that time the agent says nothing at all.
5. The turn finally settles, the agent replies — and step 2 happens again.

The caller never hears a complete sentence, and the line is silent most of the
time. Measured on a 45-second reproduction against production: **7.3 seconds of
agent audio, with two silent holes of 26 and 31 seconds.**

### Log signature

You are looking at talk-over collapse when agent replies are followed by
`barge-in` within a second, repeatedly:

```
07:41:08  caller ...
07:41:09  agent ...
07:41:09  barge-in          <- same second
07:41:31  caller ...
07:41:32  agent ...
07:41:33  barge-in          <- one second later
```

Compare against a healthy call, which completes many turns with at most an
occasional barge-in. A call that ends with `turns: 0` after sixty seconds is
almost always this.

## This is not caused by concurrent calls

It was first reported as "the second simultaneous call has no audio", and that
is not what is happening. Three measurements say so:

- Two concurrent calls driven by a simulated carrier both received agent audio
  promptly (644 and 524 media frames, first audio at 0.86 s and 0.66 s). Repeating
  it with an identical `streamSid` on both calls changed nothing.
- **One** call, with a caller who talks continuously, reproduces the failure
  exactly. No second call is required.
- A solo production call ended with `turns: 0` the same way.

The correlation with the second caller is behavioural: whoever dials into an
already-busy line tends to start talking immediately — "hello? can you hear
me?" — which is precisely the input that triggers the collapse.

**To verify for yourself:** make a single call and talk continuously from the
moment it connects. It will fail the same way.

Concurrency has its own limits, and they are elsewhere: `max_concurrent_calls`
(50) is enforced by `CallManager`, and a call past it is refused rather than
degraded. Per-call state — transports, detectors, sequence numbers — is
per-instance; there is no shared mutable state between calls in the call path.

## The settings that govern it

All three live under Conversation in the admin portal.

| Setting | What it does | Failure if too low | Failure if too high |
|---|---|---|---|
| `barge_in_ms` | Continuous caller speech needed to cut the agent off | Replies die on a cough, an echo, or a caller who has not finished their previous sentence | The caller cannot interrupt a long wrong answer |
| `end_of_turn_silence_ms` | Trailing silence that ends the caller's turn | Cuts callers off mid-sentence, and a pause for breath becomes a new turn | A caller who does not pause never gets a reply |
| `max_turn_audio_s` | Hard ceiling on one utterance | Long questions are truncated | A runaway turn holds the agent silent for its whole duration |

### Measured

Replaying a real caller's audio — one who talks over the agent — through a live
session, 50-second window:

| Configuration | Agent audio | Spurious barge-ins | Turns completed |
|---|---|---|---|
| `barge 240`, `silence 1200`, `turn 30` (as shipped) | 16.2 s | 3 | 2 |
| `silence 600` | 19.0 s | 3 | 3 |
| `barge 800` | 18.8 s | **0** | 2 |
| `turn 12` alone | 16.2 s | 3 | 2 |
| `barge 800` + `silence 600` | 18.8 s | 0 | 2 |
| `barge 800` + `silence 600` + `turn 12` | **25.2 s** | **0** | 3 |

`barge_in_ms` is the single biggest lever: raising it from 240 ms to 800 ms
removes every spurious interruption on its own. `max_turn_audio_s` does nothing
by itself and only pays off once the other two are right — it is the backstop
that caps a runaway turn, not a fix.

## Guidelines

1. **`barge_in_ms` belongs near 800, not 240.** A real interruption is someone
   speaking for most of a second. 240 ms is a syllable, an echo off a
   speakerphone, or the tail of what they were already saying. This is the
   change to make first, and it has no downside for turn-taking: a caller who
   genuinely wants to interrupt still can.

2. **Do not lower `end_of_turn_silence_ms` without re-testing barge-in.** It was
   raised from 700 ms to 1200 ms deliberately, because replies took around four
   seconds and a short window turned every pause into a new turn. Lowering it
   helps talk-over collapse and hurts callers who pause to think. Change it only
   with `barge_in_ms` already raised, and listen to a real call afterwards.

3. **Keep `max_turn_audio_s` well under the idle prompt.** A turn that can run
   for 30 seconds is 30 seconds of silence from the caller's side. Twelve
   seconds is long enough for any real question.

4. **Judge a change by agent audio delivered, not by average latency.** The
   metric that matters is how much of each reply the caller actually heard.
   `metrics.total_ms` is turn wall-clock and includes the reply playing out at
   real time — it is not latency, and tuning against it misleads. See the
   `waited … spoke …` split on the `/calls` console.

5. **Never tune from a single call.** Both directions of every setting here fail,
   and the failures look similar from the outside. Replay a recording through a
   session and measure, rather than redialling and forming an impression.

## Reproducing it

Recordings are inbound-only PCM16 at 16 kHz, written to
`data/recordings/<call_id>.wav` for every call when `record_calls` is on. That
file is the caller's side of a real conversation and is the right input for a
regression test: replay it into a `CallSession` and count how much agent audio
reaches the transport. A session driven this way runs against mock providers in
seconds, with no carrier and no API spend.

The two things worth asserting in such a test are that a continuously-talking
caller still receives complete replies, and that a genuine interruption still
cuts the agent off.
