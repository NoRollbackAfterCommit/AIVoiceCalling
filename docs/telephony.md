# Accepting real phone calls

Vaani never speaks SIP or RTP itself. Asterisk does the telephony — signalling,
codecs, DTMF, the carrier relationship — and hands each answered call to Vaani
as raw audio over one TCP socket per call (the AudioSocket protocol). That
split is deliberate: the telephony stack changes per deployment (SIP trunk
here, PRI there, LiveKit at scale), while everything from VAD to TTS stays
identical to a web call.

```
caller's mobile ──▶ carrier (PSTN) ──▶ SIP trunk ──▶ Asterisk ──▶ AudioSocket ──▶ Vaani
                                       (or GSM gateway / PRI)      TCP :9092
```

No GPU is needed anywhere in this path. With hosted providers (Sarvam STT/TTS
plus a hosted LLM), a 4-core VM runs Asterisk and Vaani side by side.

There are now two routes onto the PSTN. Everything below through "When this
outgrows Asterisk" is the self-hosted route just described: you own the SIP
trunk and run Asterisk. **Tata Smartflo**, covered at the end of this
document, is the hosted alternative — Tata runs the media servers and reaches
Vaani over a plain WebSocket, no Asterisk box at all. A hosted pilot with no
telephony infrastructure of its own wants Smartflo; an operator who already
holds a SIP trunk, or needs carrier-grade scale later, wants Asterisk.

## What you need from a carrier

Three things, for inbound:

1. **A DID number** — the number callers dial.
2. **A SIP trunk** delivering that number's calls to your Asterisk box. In
   India that means a licensed operator (Airtel, Tata Communications, Jio
   Business, BSNL) or a cloud-telephony provider that hands off via SIP
   (Exotel, Ozonetel, Servetel). You will get either username/password
   credentials or IP-whitelist authentication.
3. **Channels** — trunks are sold by concurrent calls. Ten channels means ten
   simultaneous callers; match it to `VAANI_MAX_CONCURRENT_CALLS`.

Ask the provider for: G.711 A-law codec (the Indian default), the IP:port
their SIP traffic originates from, and their RTP port range. You need a static
public IP, or their VPN/MPLS handoff.

Alternatives to a SIP trunk:

- **GSM gateway** — a box holding SIM cards, presented to Asterisk as a SIP
  peer. Fine for a pilot; do not scale a business on it (TRAI treats
  commercial SIM banks harshly).
- **PRI/E1** — 30 channels on legacy copper, needs a Digium/Sangoma card.
  Only worth it where the customer already has the line.

For **outbound** campaigns later: TRAI's TCCCPR rules apply (140-series
telemarketing numbering, DND scrubbing). Inbound helplines carry no such
burden — launch inbound first.

## Asterisk setup

Any Asterisk 18+ with `chan_pjsip` and `app_audiosocket` (both in the standard
Debian/Ubuntu `asterisk` package) will do.

`/etc/asterisk/pjsip.conf` — the trunk. Credentials variant shown; an
IP-whitelist trunk drops the auth section and adds `match=` on the identify:

```ini
[trunk]
type = registration
outbound_auth = trunk-auth
server_uri = sip:sip.your-carrier.in
client_uri = sip:YOUR_USERNAME@sip.your-carrier.in

[trunk-auth]
type = auth
auth_type = userpass
username = YOUR_USERNAME
password = YOUR_PASSWORD

[trunk-endpoint]
type = endpoint
context = vaani-inbound          ; where incoming calls land in the dialplan
disallow = all
allow = alaw                     ; G.711 A-law — the Indian carrier default
aors = trunk-aor

[trunk-aor]
type = aor
contact = sip:sip.your-carrier.in

[trunk-identify]
type = identify
endpoint = trunk-endpoint
match = sip.your-carrier.in      ; accept INVITEs from the carrier only
```

`/etc/asterisk/extensions.conf` — answer and hand the audio to Vaani:

```ini
[vaani-inbound]
exten => _X.,1,Answer()
 same  => n,Set(CALL_UUID=${UUID()})
 same  => n,Set(R=${CURL(http://VAANI_HOST:8080/api/telephony/announce,uuid=${CALL_UUID}&caller=${URIENCODE(${CALLERID(num)})}&did=${EXTEN})})
 same  => n,AudioSocket(${CALL_UUID},VAANI_HOST:9092)
 same  => n,Hangup()
```

Replace `VAANI_HOST` with the address Vaani listens on. When both run on the
same machine, `127.0.0.1`. Asterisk sends 8 kHz signed-linear audio down
that socket; Vaani converts to its 16 kHz pipeline format at this edge and
nowhere else.

The `CURL()` line tells Vaani who is calling. AudioSocket itself carries only
the UUID, so without it every call is answered by the default agent and the
call record has no caller number — which means no number to ring back when
the agent promises a callback. The announce accepts:

| Field | Value | Effect |
|---|---|---|
| `uuid` | the same `${CALL_UUID}` passed to `AudioSocket()` | required; pairs the announce with the socket |
| `caller` | `${CALLERID(num)}` | stored on the call record, visible to tools and the live console |
| `did` | `${EXTEN}`, the number dialled | logged; useful when one trunk carries several numbers |
| `agent` | an agent profile key | which profile answers; unknown keys fall back to `default` |

One Asterisk box fronting several helplines gives each DID its own `agent=`
in the dialplan, so routing stays where the rest of the telephony
configuration already is. `func_curl` is in the standard Asterisk package. An
announce that is never followed by a socket expires after a minute.

## Vaani setup

```bash
VAANI_AUDIOSOCKET_ENABLED=true
VAANI_AUDIOSOCKET_HOST=0.0.0.0    # or the LAN interface Asterisk reaches
VAANI_AUDIOSOCKET_PORT=9092
```

Restart the server; the boot log prints `audiosocket listening`. The same
settings are on the admin page under **Telephony**.

**Security**: AudioSocket is unauthenticated TCP — whoever can reach the port
can place a call. The network is the access control. Bind to a LAN interface,
keep the port firewalled from the internet, and put Asterisk and Vaani on the
same machine or VLAN.

If `VAANI_API_TOKEN` is set, `/api/telephony/announce` stays reachable without
it — it's LAN-only like AudioSocket itself, and a token there would only end
up sitting in cleartext in the dialplan file. Keep port 8080 firewalled to the
LAN as the table below says; that's the actual access control for this route.

## Firewall

| Port | Protocol | Direction | Who |
|---|---|---|---|
| 5060 | UDP/TCP | carrier ⇄ Asterisk | SIP signalling |
| 10000–20000 | UDP | carrier ⇄ Asterisk | RTP media (range per carrier) |
| 9092 | TCP | Asterisk → Vaani | AudioSocket, LAN only |
| 8080 | TCP | operators and Asterisk → Vaani | web console/API and the call announce (announce is exempt from `VAANI_API_TOKEN`), not the carrier |

## Testing before the trunk exists

Register a softphone (Zoiper, Linphone) directly against Asterisk as a local
extension and point its dialplan at the same `vaani-inbound` context. That
exercises the entire path — SIP, codecs, AudioSocket, barge-in — with no
carrier involved. `tests/test_audiosocket.py` covers the wire format itself,
and `scripts/smoke_call.py` checks the pipeline end to end without telephony.

## Capacity

Each call costs one trunk channel, one Asterisk channel, and one Vaani session
slot. Size in this order: trunk channels ≤ `VAANI_MAX_CONCURRENT_CALLS`, and
with hosted STT/TTS/LLM check the provider's concurrent-request ceiling —
that, not CPU, is the usual bottleneck. When callers past capacity should hear
a busy message rather than ringing out, add a `GotoIf` on
`${CURL(http://VAANI_HOST:8080/api/ready)}` before the `AudioSocket()` line.

## When this outgrows Asterisk

For carrier-grade NAT traversal, geographic redundancy, or thousands of
channels, put LiveKit (or another media server) in front and hand its PCM to
the same `CallSession` — the transport is the only layer that changes.

## Tata Smartflo (voice bot streaming)

Smartflo is the hosted path: Tata holds the licence and runs the media servers
in India, and Vaani is reached as a WebSocket bot endpoint. Asterisk is not
involved. Audio is G.711 mu-law at 8 kHz, base64 inside JSON.

No live Smartflo call has been made against this implementation — the wire
format here comes from Tata's published documentation, and the test suite
(`tests/test_smartflo.py`) exercises it against a fake client, not a real
account. The steps below have not been run.

### What Tata must do (they cannot be done from here)

1. Enable **Channels Hub** on the account.
2. Register the bot under **Settings → Channels → Voice Bot**: name, description,
   and the dynamic endpoint URL below.
3. For inbound, map the Voice Bot to a DID with **Configure Destination** in
   **My Numbers**.

Broadcast and outreach campaigns are not supported by Smartflo streaming, which
suits the inbound-first plan.

### What to configure here

On `/settings`, under Telephony:

| Setting | Value |
|---|---|
| Tata Smartflo voice bot | on |
| Smartflo webhook secret | a long random string; also put it in the endpoint URL |
| Smartflo public host | the public hostname, e.g. `voice.example.in` |
| Smartflo agent profile | which agent answers |

The dynamic endpoint URL given to Tata is:

    https://<your host>/api/telephony/smartflo/handshake?key=<webhook secret>

It answers with `{"success": true, "wss_url": "..."}` and nothing else — Smartflo
refuses any extra key and drops the call — inside its 2000 ms budget.

### Why the secret is not the API token

That URL sits in Tata's configuration and its logs. The webhook secret grants
only the ability to ask for a socket URL, and the per-call token in the reply
names one call and stops verifying two minutes after it is minted. Nothing marks
a token as used, so within that window whoever holds it can open the stream
again — an accepted risk, since the window only has to cover the network hop
between the handshake and Smartflo's connect. Neither credential can reach the
settings API, the knowledge base, or a live call.
