# SamparkAI — frequently asked questions

Every question below came from something that actually happened during the build
or the pilot, and the answers carry the measurements that settled them. Invented
questions have been left out; a made-up FAQ is worse than none, because it reads
as though the ground has been covered.

---

## Calls

### A caller says the bot talked over them, or that they heard nothing

This is the failure mode that has cost the most time. It is documented in full
in `docs/call-quality.md`; the short version:

The agent starts a reply, the caller is still talking, and 240 ms of their
continuing speech cancelled it. They then keep talking, and their own turn does
not end until they pause for a continuous 1.2 seconds — which somebody mid-flow
never does. So the agent is silent for ten, twenty, thirty seconds, replies, and
is cancelled again.

Measured on a reproduction against production: **7.3 seconds of agent audio in a
45-second call**, with silent holes of 26 and 31 seconds.

The fix was raising `barge_in_ms` from 240 to 800. A real interruption is
somebody speaking for most of a second; 240 ms is a syllable, an echo off a
speakerphone, or the tail of what they were already saying.

### Is that caused by two calls happening at once?

No, and it was reported that way. One call reproduces it exactly. Two simulated
concurrent calls both received audio promptly (644 and 524 media frames, first
audio at 0.86 s and 0.66 s), and forcing an identical stream id on both changed
nothing. A solo production call had already failed the same way.

The correlation is behavioural: whoever dials into an already-busy line tends to
start talking immediately — *"hello? can you hear me?"* — which is precisely the
input that triggers the collapse.

**To check for yourself:** make a single call and talk continuously from the
moment it connects.

### How many calls can it take at once?

Our server is not the constraint. During a load test it sat at **0.13% CPU and
169 MB** while thirty simultaneous calls ran with zero failures.

The limit is the speech vendor. At fifty simultaneous calls, twenty-one got
`429 Too Many Requests` from Sarvam's synthesis endpoint. So the practical
ceiling today is **comfortably thirty, breaking somewhere between thirty and
fifty**, and raising it is a conversation with Sarvam rather than a change here.

A rate-limited request is now retried twice with backoff. If it still fails the
caller gets silence for that reply, and the live console marks the call.

### The call rings and then disconnects

Almost always the carrier. When this happened on the pilot number, our handshake
was answering in 0.14 seconds and the carrier was not calling it at all — the
call was being dropped inside their platform before it ever reached us.

Check, in the carrier's dashboard: the endpoint is registered **Dynamic** and not
Static; the number's call flow actually reaches the voice bot; and the parameters
are set in the key/value table rather than typed into the URL.

### Why does the caller hear a soft tone while the bot thinks?

Because a line that goes completely silent reads as a dropped call, and callers
say "hello? hello?" into the gap — which then arrives as a fresh question and
derails the conversation. The tone is deliberately below speech level.

---

## Reporting

### "Caller waited 2.6 s" but calls feel slower than that

Both are true, and the difference is the most misread thing in the system.

A turn's total time includes **the agent speaking**. Speech is played at real
time, so a six-second answer takes six seconds. What the caller *waits* for is
recognition plus the model plus the first audio.

Measured over 86 real turns: **2,647 ms waited** (recognition 511, model 1,815,
first audio 321) and **6,005 ms spoken**. Reporting the total as latency
overstates it roughly threefold, and doing exactly that once sent an afternoon
hunting for seconds that were never lost.

If you want the wait shorter, the model is 70% of it.

### Some calls show no organisation

They arrived on a number nobody has mapped. They were answered — by the fallback
agent — and recorded as unattributed rather than assigned to whichever
organisation happened to be first.

Map the number under **Organisations → Numbers** and future calls will be
attributed. Past ones keep the attribution they had, which is deliberate: numbers
get reassigned between customers, and last quarter's figures must not rewrite
themselves.

### Why do my reports show fewer calls than a colleague's?

If you are an organisation administrator or a supervisor you see only your own
organisation's calls. A platform administrator sees every organisation. That is
the isolation working.

### Most of our calls say `caller_disconnected`

It means the caller hung up rather than the conversation reaching an end. On the
pilot, 248 of 259 calls ended that way — which is the talk-over problem above
written down in numbers, and it predates the fix for it. Watch the share fall as
a measure of whether things are improving.

---

## Accounts and access

### What is the default password?

There is not one. A migration that creates an account with a known password
would be a backdoor in every deployment that runs it, so a new deployment has no
accounts at all until somebody creates the first from the command line. Every
account after that is created in the portal.

### Somebody left. Should I delete their account?

Suspend it. Suspending removes access on their next click and keeps everything
they did attributed to them; deleting would orphan it.

### Can an organisation administrator see another organisation's data?

No, and it is tested as an attacker rather than as a user: by direct id, by query
parameter, by crafted filter, by asking the export for it, and by trying to hang
up a call they can name. Requests for another organisation's data answer **404**
rather than 403, because a 403 confirms the record exists.

### The role list was blank and data could be added without signing in

That was a real fault, now fixed. The console used to attach the shared API token
to its requests. That token is a machine credential and the server treats it as a
platform administrator, so a browser that had ever stored one bypassed the
sign-in screen: writes succeeded while the console could not say who was signed
in, so pages drew empty lists.

The server was never open — unauthenticated requests were refused throughout. The
console no longer uses the token at all, and discards any it finds stored.

### What is the API token for, then?

Machines: the deploy script, health probes, the simulated carrier used for
testing. Anyone holding it has full API access, so treat it like a password and
rotate it if it has been shared around. It is not a login and the portal does not
use it.

---

## Running it

### Can I look at the logs after a deployment?

Only from the deployment onwards. Redeploying recreates the container, which
**wipes `docker logs`**. Never deploy while somebody is mid-test — one call's
only evidence was lost that way.

### The timestamps do not line up

The host clock is IST; the application logs in UTC. `journalctl` and
`docker inspect` read IST, the application's own JSON logs read `+0000`. Reading
them side by side misleads by five and a half hours.

### Will upgrading lose our calls?

No. Migrations run at boot and the data volume survives a redeploy. The tenancy
migration was rehearsed on a copy of the production database — 257 calls and 174
turns through the upgrade and back down again, unchanged — before it was applied.

Backups are **not** automatic. `data/` holds the database and the recordings and
is deliberately excluded from the repository.

### Does it work without internet access?

The architecture allows it: every provider is imported only when selected, so an
install that never fetched the cloud extras is structurally incapable of calling
out. In that configuration you would run local speech and a local model instead
of Sarvam and OpenAI. The current deployment does use hosted services, so it
needs the network.
