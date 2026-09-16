# SamparkAI — operator's manual

For the person running a call centre on this system. No programming required;
everything below is done from the portal in a browser.

The portal is at **https://samparkai.demosites.co.in/**. Sign in at `/login`.

---

## Who you are, and what you can do

Your role decides what you see. If a page or a menu item is missing, that is
your role, not a fault.

| Role | What it is for |
|---|---|
| **Platform administrator** | Runs the whole deployment. Creates organisations, maps phone numbers, changes provider settings. Usually one or two people. |
| **Organisation administrator** | Runs one organisation's call centre: its agents, its knowledge, its numbers, its people. |
| **Supervisor** | Watches live calls and can end one. Reads reports. Changes nothing. |
| **Viewer** | Reads reports. Nothing else. |

Your name appears at the top right of every page. Click it to sign out.

---

## Setting up a new call centre

Five steps, in this order. Steps 1 and 4 need a platform administrator,
because creating a customer and assigning a phone number are the platform's
business. An organisation administrator can do the other three.

### 1. Create the organisation

**Organisations → Name and short code → Add organisation.**

The short code is used in exported filenames and never changes, so keep it
plain: `health`, `wb-portal`. The name is what everyone sees and can be
corrected later.

### 2. Create the agent that answers

**Agents → New**, or click an existing agent to change it.

An agent is the bot's personality and rules for one line. It carries:

- **Key** — a short internal name, `health-admissions`. It is chosen once and
  fixed afterwards, because every document you upload is filed under it.
- **Name and organisation** — what the bot calls itself out loud. "I am Vaani,
  the voice assistant for the Health University." Write them as speech.
- **Role** — one or two sentences on what this line is for.
- **Greeting and closing** — the first and last thing a caller hears.
- **Languages and voices** — a voice per language, so a Bengali caller gets a
  Bengali voice rather than an English one reading Bengali. The first voice is
  the language a call starts in, and a caller can only be moved into a language
  listed here.
- **Policies** — the rules of this service: hours, eligibility, fees, what to
  refuse. Write them as you would tell a new member of staff, one per line.
- **Will not discuss** and **hand to a person when** — where the bot stops.
- **Tools** — what it is allowed to do besides talk. Only the ones this
  deployment has installed are offered.

**What the bot is actually told**, at the foot of the page, shows the finished
instruction assembled from everything above. Open it after a change you are
unsure of; it is the fastest way to see whether a policy landed the way you
meant it.

One organisation can have several agents. A university with an admissions line
and an examinations line should have two, because each answers from its own
documents.

> Changes take effect on the next call. A call already in progress finishes on
> the agent it started with.

### 3. Teach it

**Knowledge → choose the line → choose the scope → paste text or upload a file.**

There are two training sets, and every call reads both.

| | Who answers from it |
|---|---|
| **Whole organisation** | Every line that organisation runs. Upload the fee schedule once and all its numbers have it. |
| **This line only** | That line alone, on top of the shared set. The examination timetable belongs here, not on the admissions line. |

The bar at the top of the page names the scope you are adding to and lists the
exact phone numbers it will be answered on. Read it before you upload; it is
the difference between correcting one line and correcting five.

Put anything the whole call centre shares — hours, fees, contacts, eligibility,
the general FAQ — in the organisation's set. Keep a line's own set for what only
its callers should hear.

> An organisation's set needs the line to belong to an organisation. A number
> nobody has mapped is answered by the fallback agent, which belongs to nobody
> and reads only its own documents — deliberately, so an unconfigured number can
> never be read a customer's material.

Give every upload a source name you will recognise (`fees-2026.pdf`), because
that is what appears in reports when the bot cites it, and it is how you replace
or delete it later.

**To correct a document, upload it again under the same source name.** The old
version is removed as the new one goes in, so the bot cannot quote the
superseded text. To withdraw one entirely, use **Delete** on its row — the row
says which scope it lives in, and deleting a shared one takes it off every line
of that organisation.

**Test retrieval** at the foot of the page shows what that line would actually
find for a question — both sets together, exactly as a real call reads them.

### 4. Map the phone number

**Organisations → Numbers → number, organisation, answered by, label → Map
number.**

Type the number however your carrier shows it — `+91 80 6560 5873`,
`08065605873`, `918065605873` all work. The page echoes back the stored form;
seeing `+918065605873` come back is how you know it matched.

> A number that is not mapped here is **still answered**, by the fallback agent
> in Settings, and its calls appear in reports as unattributed. That is
> deliberate: a missing mapping should never drop a real caller. But it means an
> unattributed call in your reports is a number that needs mapping.

The number must also exist on the carrier's side and be routed to this system.
Mapping a number here that the carrier does not send us is just a row in a table.

### 5. Add the people

**People → Add someone.** Give them the least role that lets them do their job.
Set a temporary password and ask them to change it at **People → Reset password**
on their own row when they first sign in.

Suspending somebody (**Suspend**) removes their access on their next click and
keeps everything they did attributed to them. Prefer it to deleting.

---

## Day to day

### Watching calls as they happen

**Live calls.** Every call in progress, with a pulsing dot for what the agent is
doing — listening, thinking, speaking. Click one to read the conversation as it
happens, with what the caller said, what the agent replied, what it looked up,
and how long each turn took.

**Hang up** ends a call that has gone wrong. There is a confirmation, because
there is a real person on the other end.

Below the live list are recent calls; click one to read its transcript
afterwards.

A **⚠ beside a caller's number** means a reply reached them as silence. Usually
the speech service refusing a burst of calls. One is bad luck; several in an
hour is worth reporting.

### Reading the reports

**Reports.** Filter by organisation, by number, and by period.

- **Calls** — how many, in the period.
- **Handled by the bot** — the share finished without passing to a person. This
  is the number to watch week on week.
- **Average call** — how long a caller was on the line.
- **Caller waited** — how long, on average, between the caller finishing and the
  agent starting to reply. **This is not how long the agent spoke**; the two are
  shown separately, because adding them together is the most common way to
  misread this system's speed.
- **Questions with no answer** — what callers asked that the bot could not
  answer. This is the most valuable thing on the page: every line is a gap in
  your documents, and filling it is the cheapest improvement available.

**Export CSV** gives you the call list for the same filters, for a spreadsheet or
a report.

### Trying the bot yourself

**Console → Start call** talks to the agent through your browser's microphone.
Use it to hear a greeting change or test a new document before a real caller
does.

---

## When something is wrong

### The agent talks over people, or callers hear nothing

Read `docs/call-quality.md`. The short version: if agent replies are cut off
within a second, the barge-in threshold is too low; and a caller who talks
continuously can starve themselves of replies. Do not change these settings
one at a time by feel — that document has measurements.

### A call rings and then disconnects

Almost always the carrier, not this system. Check that the number is registered
on the carrier as a **Dynamic** endpoint, that its call flow reaches the voice
bot, and that the parameters are set in the carrier's key/value table rather than
typed into the URL.

### Somebody cannot sign in

Check **People** — the account may be suspended, or the email may differ from the
one they are typing. The sign-in page says "email or password is incorrect" for
every failure on purpose, so it never confirms which addresses exist.

If **nobody** can sign in, an administrator with server access can create a fresh
platform administrator only while no accounts exist at all. Otherwise recovery is
through the database.

### The reports look empty

Check the period — the default is 30 days — and the organisation filter. If you
are an organisation administrator you only ever see your own calls, which is
working as intended.

---

## Things worth knowing

- **Times in the portal are your browser's local time. Server logs are UTC.**
- **A deployment restarts the application**, which ends calls in progress and
  signs nobody out. Reports are unaffected — they come from the database.
- **Recordings and transcripts are kept** for the retention window in Settings,
  then deleted. That window is a policy decision; set it deliberately.
- **Suspending an organisation** takes all of its numbers out of service at once.
  Calls to them fall back to the default agent rather than failing.
