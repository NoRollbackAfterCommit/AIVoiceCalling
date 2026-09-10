# Tata Smartflo voice streaming — design

**Date:** 2026-09-10
**Status:** approved, awaiting implementation plan
**Replaces:** Tasks 7–8 of `2026-08-08-indic-inbound-voice-slice`, which specify an
`ExotelTransport` carrying raw PCM16 at 16 kHz. Smartflo is a different wire format, so those
tasks are rewritten rather than adapted.

## Why

The pilot needs inbound PSTN calls from an Indian licensed operator. Tata Tele Business Services'
**Smartflo** product, with its Voice Streaming (voice bot) feature, was chosen over their raw SIP
Trunk product: Tata holds the licence and runs the media servers in India, and Vaani is reached
as a WebSocket bot endpoint. Asterisk is not needed on this path — it stays for the softphone
test rig and for a future on-premise SIP deployment.

Smartflo has cloned Twilio Media Streams, so Twilio examples transfer directly.

## Decisions taken

| Decision | Choice | Why |
|---|---|---|
| Endpoint mode | **Dynamic** | Per-call agent routing, the caller's number before audio arrives, and a per-call token. Costs one HTTP endpoint. |
| Playback completion | **Transport-local, `mark` instrumented only** | Keeps `session.py` and the `Transport` Protocol untouched. Adopting `mark` as load-bearing before a single live call would optimise against a document, not evidence. |
| Handshake auth | **Dedicated webhook secret + per-call token** | A leak in Tata's logs must not reach the control plane. |
| Done means | **Code + local test harness** | Channels Hub provisioning is a commercial step outside our control; the build must be provable without it. |

## Wire protocol (confirmed, Task 0)

- Audio is **G.711 mu-law (PCMU), 8 kHz, 8-bit, mono**, base64 inside JSON. Never binary frames.
  This differs from Exotel, which streams signed-linear PCM.
- Bot → Smartflo media must be **at least 160 bytes and a multiple of 160**. 160 bytes of mu-law
  at 8 kHz is exactly 20 ms, which matches Vaani's frame cadence.
- Smartflo sends `media` every 100 ms (800 bytes mu-law → 3200 bytes PCM16 at 16 kHz).
- Events from Smartflo: `connected`, `start` (accountSid, streamSid, callSid, from, to,
  direction, mediaFormat, customParameters), `media`, `dtmf`, `stop`, `mark`.
- Events from the bot: `media`, `mark`, `clear`. **`clear` is barge-in** — it empties Smartflo's
  buffered audio. `mark` echoes back when playback of that point finishes.
- Every message carries `streamSid` and an incrementing `sequenceNumber`.
- The dynamic endpoint must return exactly `{"success": true, "wss_url": "..."}` within
  **2000 ms**. Any extra key, bad JSON, or non-200 is refused and the call is dropped.

## Components

### `vaani/telephony/smartflo_frames.py` — codec, no I/O

Pure functions and one dataclass, so the wire format can be tested without a socket.

```python
@dataclass(slots=True)
class SmartfloEvent:
    kind: str  # connected | start | media | dtmf | stop | mark | invalid
    stream_sid: str | None = None
    call_sid: str | None = None
    pcm: bytes | None = None  # decoded to PCM16 mono 16 kHz
    caller: str | None = None  # `from`
    called: str | None = None  # `to`
    direction: str | None = None
    custom: dict[str, str] = field(default_factory=dict)
    digit: str | None = None
    mark_name: str | None = None


def parse_frame(raw: str | bytes) -> SmartfloEvent: ...
def media_frame(stream_sid: str, ulaw: bytes, seq: int) -> str: ...
def mark_frame(stream_sid: str, name: str, seq: int) -> str: ...
def clear_frame(stream_sid: str, seq: int) -> str: ...
```

`parse_frame` never raises: malformed JSON, a missing `event` key, or undecodable base64 all
return `kind="invalid"`. A telephony vendor sending one bad frame must not end a call. It decodes
inbound mu-law to PCM16/16 kHz with the existing `ulaw_to_pcm16`, so the pipeline only ever sees
one format.

`media_frame` takes **already-encoded mu-law** and does nothing but base64 and envelope it. The
codec module therefore owns the wire format and the transport owns rate conversion and the
160-byte rule — one responsibility each, and the 160-byte arithmetic stays next to the buffer
that enforces it.

### `vaani/telephony/smartflo.py` — `SmartfloTransport`

Implements the existing `Transport` Protocol (`send_audio`, `send_event`, `close`). Nothing in
`session.py` changes.

- `send_audio(pcm)` converts to mu-law with `pcm16_to_ulaw` and appends to a `bytearray`,
  emitting `media` frames while at least 160 bytes remain, holding the remainder — the same
  pattern as
  `AudioSocketTransport._pending`, which exists because agent audio arrives in 200 ms lumps and
  the carrier wants a steady cadence.
- `send_event(event)` reacts to what the session already emits:
  - `barge_in` → send `clear` **and discard the pending buffer**, or the flushed audio is
    immediately followed by our own stale remainder.
  - `speech_end` → send `mark` named `utt-<n>`; record the send time.
  - everything else → logged, matching `AudioSocketTransport.send_event`.
- Owns `stream_sid` and an incrementing `sequence_number`.
- A `mark` echo from Smartflo logs the round-trip against the recorded send time. This is the
  measurement that decides later whether to adopt `mark` as the playback-complete signal.

### `vaani/api/smartflo.py` — the two endpoints

**`GET|POST /api/telephony/smartflo/handshake`**

Reads `$callId`, `$fromNumber`, `$toNumber`, `$status`. Returns exactly:

```json
{"success": true, "wss_url": "wss://<host>/ws/smartflo?token=<per-call token>"}
```

Nothing else in the body — Smartflo refuses an extra key. It must do no database work, no
provider calls, and nothing that can block: the 2000 ms budget is a hangup, not a retry.

**`WS /ws/smartflo`**

Validates the per-call token, accepts, waits for `start` to learn `streamSid` and the caller's
number, then constructs a `CallSession` with a `SmartfloTransport` and pumps frames. `dtmf`
forwards to the existing `CallSession.push_dtmf`; `stop` hangs the session up.

## Authentication

`vaani/api/auth.py` deliberately refuses `?token=` on HTTP — "a token in a URL lands in access
logs and history" — and allows it only for WebSocket handshakes, which cannot carry a header.
The existing `OPEN_PATHS` exemption for `/api/telephony/announce` is justified by that path being
LAN-only. A Tata callback is neither: it is internet-facing HTTP, and its *response body* hands
out a WebSocket URL containing a token. Left open, it would be a token-disclosure hole.

Therefore, two paths are added to `OPEN_PATHS` — meaning exempt from the *shared* bearer guard —
and each authenticates itself more strictly than the guard would:

1. **`/api/telephony/smartflo/handshake`** requires `smartflo_webhook_secret`, a new setting
   distinct from `api_token`, compared with `hmac.compare_digest`. Tata puts it in the configured
   callback URL. If it leaks into Tata's logs it grants nothing but the ability to ask for a
   wss_url.
2. **`/ws/smartflo`** requires a **per-call token** minted by the handshake:
   `f"{call_id}.{exp}.{sig}"` where `sig = HMAC-SHA256(smartflo_webhook_secret, f"{call_id}:{exp}")`,
   `exp` 120 seconds out. Verified with `compare_digest`. It grants exactly one thing: the ability
   to start one call. It is not the admin token and cannot reach the control plane.

The docstring in `auth.py` is extended to explain both, so the next reader does not see two
unexplained holes in a deliberately small exemption list.

**Known limitation:** the per-call token is replayable until it expires. Binding it to
single use requires call-id state that would have to survive a restart; for a pilot, a 120-second
window on a token that can only open a call is an accepted risk. Recorded here so it is a
decision rather than an oversight.

## Settings

Added to `vaani/config.py` with `cfg()` metadata in a `Telephony` group, so `/api/settings/schema`
and the admin page render themselves. All `restart=False`.

| Setting | Default | Notes |
|---|---|---|
| `smartflo_enabled` | `False` | When false the handshake returns 404 and the WS refuses. |
| `smartflo_webhook_secret` | `""` | `secret=True`. Never echoed back by the settings API. |
| `smartflo_public_host` | `""` | Host for the returned `wss_url`. Not derived from request headers: behind Caddy those are attacker-influenced, and the URL must be exactly right. |
| `smartflo_agent` | `"default"` | Agent profile for Smartflo calls. |

Routes are registered unconditionally and gated on `smartflo_enabled` at request time, so the
setting genuinely takes effect without a restart rather than only appearing to.

Per-DID agent routing is deliberately omitted — the pilot has one DID. `$toNumber` is already
parsed and logged, so adding the mapping later is additive.

## Error handling

| Case | Behaviour |
|---|---|
| Malformed frame | `kind="invalid"`, logged at debug, call continues. |
| `smartflo_enabled` false | Handshake 404; WS closes without accepting. |
| Bad or expired per-call token | WS closes before `accept()`. |
| Capacity reached | `CallCapacityError` → close, matching `ws_voice`. |
| `stop` event | `session.hangup()`. |
| Socket dies mid-call | Transport marks itself closed; sends become no-ops, as in the other two transports. |

## Testing

`tests/test_smartflo.py`, against mock providers only — no GPU, no network, about a second, in
keeping with the rest of the suite.

- **Codec units:** mu-law round-trip within tolerance; `media_frame` output is base64 of a
  multiple of 160 bytes; sequence numbers increment; `parse_frame` returns `invalid` for bad
  JSON, a missing `event`, and undecodable base64.
- **Chunking:** feeding 200 ms lumps emits whole 160-byte multiples and retains the remainder.
- **Barge-in:** a `barge_in` event produces a `clear` frame and drops the pending buffer.
- **Handshake:** returns exactly the two required keys and nothing else; rejects a wrong secret;
  responds far inside the 2000 ms budget.
- **Per-call token:** accepted when fresh, rejected when tampered with or expired.
- **End-to-end:** a fake Smartflo client speaking the real wire format completes a call —
  `connected`, `start`, `media`, `stop` — and receives `media` frames back.

## Out of scope

- Outbound calling. Smartflo streaming does not support broadcast or outreach campaigns, and
  TRAI TCCCPR obligations (140-series, DND scrubbing) bite on outbound only.
- Adopting `mark` as the playback-complete signal. Instrumented now, decided later on live data.
- Per-DID agent routing.
- Removing the Asterisk AudioSocket path, which stays for the softphone rig and on-premise SIP.

## Commercial prerequisites (user-owned, block live testing only)

Tata must enable **Channels Hub** on the account. The bot is then registered under
Settings → Channels → Voice Bot with its name, description and the dynamic endpoint URL, and for
inbound the Voice Bot is mapped to a DID via Configure Destination in My Numbers.
