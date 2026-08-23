# Architecture — AI Stock Research Assistant

This document explains how the system works, in plain language. If you are new to the
codebase, read sections 1–5 and you will understand the whole thing.

---

## 1. What this is

You ask a question about a company. Three specialist agents go and look things up —
company financials, price trends, and recent news — and a final step combines what they
found into one written answer with citations, so every claim points back to the data
behind it.

You watch the agents work live instead of staring at a spinner, and you can keep asking
follow-up questions afterwards.

**What it can do**

- Research one company: *"Analyse NVIDIA"*
- Compare several: *"Compare NVIDIA and AMD"*
- Review a set of holdings: *"How is my portfolio of NVDA, AAPL and MSFT doing?"*
- Answer follow-ups, including ones that need fresh data: *"any news on AMD today?"*

**What it deliberately does not do**

- Find stocks for you. It analyses companies you name; it does not screen a market or
  sector for candidates.
- Give personal financial advice.
- Trade anything.
- Cover non-US listings (a limitation of the free data plans — see §8).

---

## 2. The one idea that shapes everything

**The language model reports what it sees. The code decides what to do.**

This sounds small but it is the main design decision, and most of the structure follows
from it.

The model gets one job per message: read it and describe what is in it. Which companies
are named? Is the message pointing back at something we discussed earlier? Which kind of
analysis is being asked for? These are observations — a careful human could check each one
against the message text alone.

Everything else is worked out in ordinary code: which companies this answer covers, what
form the answer takes, what data needs re-fetching, and whether to research, answer from
memory, ask a question, or just reply.

Why it matters: decisions made in code can be tested exactly, with no API key and no
guessing. Decisions made by a model can only be sampled and hoped for. So we give the
model the smallest possible job and compute the rest.

An earlier version asked the model to pick one of six "paths". That fused several separate
judgments into one answer, so a mistake anywhere ruined everything downstream. It also hid
real bugs: two of those six paths (*refresh this company* vs *add a new company*) differ
only by whether we already have the company — which is a lookup, not a judgment.

---

## 3. How a message flows through

Every message — first or fiftieth, research or small talk — takes the same route.

```
                          Your message
                               │
                               ▼
                        ┌─────────────┐
                        │    plan     │   1 model call + lookups + plain code
                        └─────────────┘
                               │
        ┌──────────────┬───────┴───────┬──────────────────┐
        ▼              ▼               ▼                  ▼
      chat          clarify          recall            research
   "not something  "which company   answer from      go and fetch
    I can help      did you mean?"  what we already   what's missing
    with"                           have                   │
        │              │               │                   ▼
        │              │               │          fundamentals · technical · news
        │              │               │            (one per company, in parallel)
        │              │               │                   │
        │              │               │                   ▼
        │              │               │              wait for all
        │              │               │                   │
        │              │               │                   ▼
        │              │               │            write the report
        │              │               │                   │
        └──────────────┴───────┬───────┴───────────────────┘
                               ▼
                        ┌─────────────┐
                        │    emit     │   the only place an answer is sent
                        └─────────────┘
                               │
                               ▼
                          Your answer
```

Four outcomes, one exit. That is the whole graph.

### The `plan` step, in order

**1. Read the message** (one model call). Produces a short list of observations:

| Observation | Meaning |
|---|---|
| `companies` | Each company named, and **how** it appears (see below) |
| `refers_to_prior` | Does this point back at something? *"which one is better?"* |
| `screening_scope` | Is it asking us to *find* stocks? *"best Indian stocks"* |
| `shape_hint` | Does the wording ask for a comparison, a portfolio view, or one company? |
| `aspects` | Does it ask about only financials, or only price, or only news? |
| `off_domain_topic` | Is part of this something we can't help with? |

The one real judgment here is **how** each company appears:

- **research_subject** — something is being asked *about* the company.
  *"How is Amazon stock doing?"*
- **incidental** — the company is just background; the real subject is something else.
  *"Amazon is not doing well, I might switch jobs"* — that question is about a career.
- **unclear** — genuinely can't tell. *"How is Amazon doing?"* with nothing else to go on.

Note that financial-sounding words do not make something a research request. *"Doing
badly"*, *"falling"*, *"laying people off"* all describe a company without asking anything
about it.

**2. Turn names into ticker symbols.** The model says "Amazon"; this step turns that into
`AMZN` and checks it is real. "Real" means the data provider actually has price history
for it — not just that the string looks like a ticker. This matters: bare `TCS` matches a
delisted company, while Tata Consultancy Services is really `TCS.NS`.

Anything that doesn't resolve is dropped with a reason we can show the user.

**3. Work out the plan** (plain code, no model). See §4.

---

## 4. Deciding scope, shape, and what to fetch

Three questions, answered in order. All in plain code.

### Which companies does this answer cover? (`scope`)

Work down the list and stop at the first that applies:

1. **Companies named in this message.** A named company always wins. Nothing is inherited.
2. **The previous answer's companies**, if the message points backwards. Note: the
   *previous answer's*, not everything in the session. If the session holds NVDA, AMD and
   INTC but the last answer was about the first two, *"which one is better?"* means those
   two.
3. **The session's only company**, if there is exactly one. No ambiguity possible.
4. **Nothing matched → ask.** We never guess, and we never quietly fall back to "all of
   them".

### What form should the answer take? (`shape`)

- **One company in scope → a single-company report.** Always. A company cannot be compared
  with itself.
- Otherwise, if the wording asked for a comparison or a portfolio view, use that.
- Otherwise, if the last answer was a portfolio view of the same companies, keep it.
- Otherwise, compare them.

The first rule is important. It is what lets an answer get *narrower*. Ask *"how is NVDA
doing now?"* in the middle of an NVDA-vs-AMD session and you get an answer about NVDA — not
the whole comparison again.

### What data do we need to go and get? (`fetch`)

We keep results in small units called **cells**. One cell = one company + one kind of
analysis. So NVDA has a fundamentals cell, a technical cell, and a news cell.

A cell needs re-fetching if any of these is true:

| Reason | Meaning |
|---|---|
| **missing** | We have never fetched it |
| **failed** | We tried and it failed — worth another go |
| **empty** | It worked but found nothing usable |
| **stale** | It is older than that kind of data stays good for |

The **failed** case matters more than it looks. Every attempt records when it happened,
including failures — so a company whose lookups all just failed has three recent
timestamps. If you only checked timestamps, it would look perfectly fresh and would never
be retried.

**How long each kind of data stays good:**

| Data | Good for | Why |
|---|---|---|
| News | 5 minutes | New stories arrive constantly |
| Price / technical | 15 minutes **while the market is open**, otherwise until it next opens | These come from *daily* bars. Once the market closes, the last bar is final and cannot change — so re-fetching every 15 minutes all evening and all weekend gets you identical data |
| Company financials | 1 hour | Margins and growth change when results are filed, roughly quarterly |

Nothing may be shorter than 5 minutes, because the data layer keeps its own 5-minute cache
— asking again inside that window just hands back the same thing.

Finally:

- **Nothing to fetch → `recall`.** Answer from what we already have. No API calls.
- **Something to fetch → `research`.** Go and get exactly those cells and write a report.

That is a calculation, not an opinion. An earlier version let the model decide "is this
already covered?", which had no way to notice the data had quietly gone out of date.

---

## 5. What is remembered between messages

```python
class SessionState:
    session_id: str
    user_question: str

    # Kept and added to
    researched:   dict[ticker][kind] -> cell   # everything we've looked up
    conversation: list[Message]                # the transcript

    # Kept, but overwritten each time we answer about something
    last_scope: list[str]     # which companies the last answer covered
    last_shape: str           # what form that answer took

    # Kept only while a question is open
    pending: {question, original_question, attempts} | None

    # Thrown away and rebuilt on every single message
    turn: TurnPlan
```

The split is the point. **`turn` never survives a message.** Scope, shape, what to fetch,
warnings, the answer itself — all of it is rebuilt from scratch every time. Nothing from
the last message can quietly leak into this one.

`last_scope` and `last_shape` are the one deliberate exception, and only because *"which
one is better?"* needs something to point at. They are a *fallback* — consulted only when
the message names no companies of its own, and always overridden when it does.

`pending` holds an open clarifying question. It carries a counter: after two tries we stop
asking and say plainly *"name the ticker directly and I'll take it from there"*, instead of
asking a third differently-worded question forever.

There is no separate handling for replies to clarifying questions. A reply is just another
message, classified with the question it answers visible in the context. If the user
ignores the question and asks something else, that is handled too — because it is handled
the same way as any other message.

---

## 6. The three specialist agents

They run in parallel, one instance per company. They do not know about each other, and they
never know how many others are running.

| Agent | Where the data comes from | What it produces |
|---|---|---|
| **fundamentals** | Finnhub | Sector, margins, growth, valuation multiples, debt, cash |
| **technical** | Twelve Data | SMA 20/50/200, RSI, MACD, 52-week range, momentum, volatility |
| **news** | DuckDuckGo web search | Recent headlines and what they suggest about sentiment |

Every agent follows the same three steps: fetch its one data source, ask the model to
summarise it, and return a result. It never raises an error upward — it returns a result
marked either `ok` or `failed`, always with a timestamp.

Each produces **findings**. A finding is the smallest citable unit:

```python
{ "id": "NVDA-fundamentals-1",
  "claim": "Profit margins are strong",
  "evidence": "Profit margin: 0.31",
  "source": { "label": "NVDA fundamentals (Finnhub)", "url": ..., "as_of": ... } }
```

Every non-obvious statement in a report carries one of these ids in brackets, and every id
used appears in the sources list at the bottom.

**A note on technical indicators:** the maths (SMA, RSI, MACD) is plain pandas, and it is
checked in tests against a second, independent implementation over a fixed price series.
That is a different kind of correctness from "is this report good", and it gets a different
kind of test.

---

## 7. Writing the answer

`render` takes the turn's scope and shape **as arguments** and writes the matching report:
one company, a comparison, or a portfolio view.

It does not read anything about the session. That is deliberate — an earlier version read
the session's stored form directly, which is exactly why a narrowing follow-up used to
re-print the whole comparison.

**When there isn't enough to report.** A cell counts as usable only if it succeeded *and*
produced findings. Succeeding while finding nothing is a fact, not evidence. If too few
companies are usable for the requested shape, you get a plain sentence explaining what
failed — never an empty report with a heading and a verdict line and nothing behind it.

**Then `emit` finishes the job.** It is the only place in the whole system that sends an
answer, and it assembles the final text in this order:

1. A hedge, if we had to assume what you meant (*"Taking that as a question about the
   stock —"*)
2. Which companies this covers, on short answers (reports already say so in their heading)
3. The answer itself
4. A coverage line, if anything was unavailable:
   `Coverage: NVDA — technical unavailable (request timed out).`
5. Any warnings, e.g. a company we couldn't find
6. An acknowledgement of anything off-topic in your message

Two things about `emit` are structural rather than stylistic.

The **coverage line is generated from the data**, not written by the model. Asking a model
to remember to mention gaps mostly works, which is the problem — "mostly" is not a
guarantee, and this way the disclosure is always there.

And because `emit` is the *only* node that writes to the transcript, sending an answer and
recording it are one action. No future node can send something and forget to record it.
That used to be possible, and it happened.

---

## 8. Data sources

**Finnhub** — company financials.

**Twelve Data** — daily price history, technical indicators, and ticker checking.

**DuckDuckGo** — news. Titles, snippets, links and dates only; we do not download full
articles. Fetching and cleaning arbitrary news pages means fighting paywalls, bot blocks
and page furniture, which is a lot of fragility for this project's scope.

Both market-data providers are on free plans that effectively cover **US-listed stocks
only**. A real non-US company is turned away with an honest *"isn't currently supported"* —
not a message implying you typed it wrong.

Some shared behaviour worth knowing:

- Both providers report a bad ticker with a **success** response and an empty body, not an
  error. So validity is judged on the *shape of what came back*.
- Retries only happen for temporary problems (network blips, rate limits). "This ticker
  doesn't exist" is permanent — retrying it just burns quota.
- Everything is cached for 5 minutes and has a hard timeout.

> The project originally used `yfinance`. Yahoo's unofficial API blocks cloud/datacenter
> IPs, which retries cannot fix, so it broke in production and was replaced.

---

## 9. When things go wrong

Failures are handled at each level, because no single level can catch everything.

| Level | What happens |
|---|---|
| **Data fetch** | Timeout, then retry — but only for temporary failures |
| **One agent** | Catches its own errors and returns a `failed` result. It never crashes the run |
| **One company** | A ticker that can't be resolved is dropped with a reason; the others carry on |
| **The whole message** | Off-topic, screening and unclear messages never reach the agents at all |
| **The report** | Runs once every agent has finished, succeeded or failed |
| **The model itself** | If the message can't even be classified, we say so — we do not quietly answer from old data |

The rule throughout: **a failure is always visible and explained.** Never a silent gap,
never a bare error page.

---

## 10. Live progress and logs

Two separate things, for two different audiences.

**Live progress (for you).** As the graph runs, the API turns each step into an event and
streams it to the browser using Server-Sent Events. Events are saved to SQLite *before*
being sent, which is what makes reconnecting work: if your connection drops, the browser
reports the last event it saw and gets everything after it, with nothing missed and nothing
repeated.

Events are derived at the API layer from what the graph reports, not published by the nodes
themselves. So the nodes stay free of streaming concerns, and the browser never depends on
LangGraph's internals.

**The log file (for developers).** `logs/dump.log` records every step, every external call
with timing, every retry, every fallback taken — one structured line each:

```
timestamp | level | session_id | component | message | extra fields as JSON
```

You can answer "what did session X do?" with `grep`, without reading any code. It rotates
by size so eval runs don't fill the disk.

---

## 11. Testing

**Unit tests** cover the pieces individually. The important thing here is that all the
decision-making — scope, shape, freshness, what to fetch — is plain code with no I/O, so
those tests need no API key, no mocking, and no network. They run in milliseconds.

**Graph tests** run the whole thing end to end with the providers and model stubbed, and
cover the things that only appear once it's actually running: parallel results merging
correctly, partial failures, and dispatching only what was asked for.

**The eval harness** (`eval/`) runs real questions against the real system and scores them
two ways:

*Automatic checks* — things a computer can verify exactly:

| Check | Blocks the case? | What it catches |
|---|---|---|
| Valid structure | Yes | Malformed output |
| Citation integrity | Yes | Every `[id]` in the report is a real finding — catches invented citations |
| Well-formed findings | Yes | Empty claims, missing evidence |
| Total failure is stated | Yes | A failure silently dressed up as a report |
| Correct type / companies / route | No | Recorded, but these are judgment calls, not guarantees |
| Speed | No | Tracked against a budget |

*A model judge* — for the things a computer can't check: is it well-grounded, does it
actually answer the question, is a comparison a real comparison or three reports glued
together. This uses **Gemini** rather than Groq, deliberately: the harness should not have
a model marking its own homework.

---

## 12. Tech stack

**Backend** — Python 3.11+, FastAPI, LangGraph, `langchain-groq`, `requests`, `ddgs`,
`pandas`, `tenacity` (retries), `cachetools`, `pydantic`, SQLite (session state and event
log). `langchain-google-genai` is used only by the eval judge.

**Frontend** — Vite, React, TypeScript, Tailwind, and the browser's built-in `EventSource`
for live updates.

**Config** — `.env`, never committed. `.env.example` lists what you need:
`GROQ_API_KEY`, `FINNHUB_API_KEY`, `TWELVEDATA_API_KEY`, plus optional model names,
timeouts and `MAX_TICKERS`.

---

## 13. Known limits

Real, and worth knowing before you rely on any of it.

- **The classifier can still get it wrong.** Everything downstream is exact, but if the
  model decides a message is about the wrong company, the rest of the system will execute
  that mistake perfectly. This is why short answers state which companies they cover — so a
  wrong guess is obvious immediately rather than buried.
- **Restarting the server loses a run in progress.** Saved state and past answers survive;
  the specific request that was mid-flight does not.
- **One process only.** Live updates are held in memory, so running a second copy behind a
  load balancer needs a shared message layer (Redis or similar) first. The saved event log
  is fine — only the live push is affected.
- **US-listed stocks only** (§8).
- **News is headlines and snippets**, not full articles.
- **Market holidays are not modelled.** On a holiday the system treats the market as open,
  so price data refreshes a few extra times. Harmless, and no worse than a plain timer.
- **No portfolio maths** — no correlations, no optimisation, no Sharpe ratios. That is a
  different project with a different core skill.
