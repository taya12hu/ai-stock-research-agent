# Architecture — AI Stock Research Assistant

## 1. Problem & Goal

Researching a company means pulling together fundamentals, price/technical data, and
recent news/sentiment, then synthesizing all of it into one coherent view — normally a
slow, manual, inconsistent process. This project builds a small multi-agent system that
does that automatically:

- Specialized agents independently analyze fundamentals, technicals, and news/sentiment.
- Agents use real external data (Finnhub, Twelve Data, DuckDuckGo web search) rather than
  relying on model knowledge alone.
- Findings are combined into one cited research summary — every non-trivial claim traces
  back to a source.
- The user watches the agents work in real time (live progress, not a single opaque
  response) and can ask grounded follow-up questions afterward.
- A single failed data source or agent degrades gracefully instead of crashing the run.
- A small evaluation harness measures whether changes actually improve quality,
  reliability, and response time — this is a learning project, not a production trading
  platform, so the point is to demonstrate the system *works* and is *measurable*, not to
  maximize feature surface.

## 2. Confirmed Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Backend language | Python 3.11+ | `ddgs` (DuckDuckGo search) is a native Python library; avoids HTTP-wrapper overhead. Market data was originally `yfinance` for the same reason, but that's since been replaced — see below and §6. |
| Market data | Finnhub (fundamentals) + Twelve Data (price history, technicals, ticker resolution) | Originally `yfinance`/Yahoo Finance for everything. Yahoo's unofficial API actively blocks cloud/datacenter IPs (confirmed via `YFRateLimitError` in production on Render, badly enough that retries didn't help), so ticker resolution and technicals moved to Twelve Data first, fundamentals followed onto Finnhub once a key was available. Both are US-market-only on their free tiers — see §6. |
| Frontend | TypeScript + React (Vite) | Separate app, consumes the backend over HTTP/SSE. |
| LLM provider | Groq (via `langchain-groq`) | Fast inference, tool-calling support on Llama models. |
| Orchestration | LangGraph | Explicit state graph, native support for parallel fan-out/fan-in and checkpointed session state — a good fit for "N agents run independently, then merge." |
| User-facing transport | Server-Sent Events (SSE) | One-directional live progress stream is simpler than WebSockets for this use case and has native browser reconnection support (`Last-Event-ID`). |
| Query scope | Single-stock, portfolio, and comparison queries, plus off-topic/clarification/no-tickers handling | See §4 and §7. |

## 3. High-Level Architecture

```
┌─────────────────────────┐        SSE stream         ┌──────────────────────────────┐
│  React/TS Frontend       │◄───────────────────────────│  FastAPI Backend              │
│  - AgentCards, grouped    │   POST /research            │  - /research (start job)      │
│    per ticker (live)      │   GET  /research/{id}/stream│  - /research/{id}/stream (SSE)│
│  - Inline citations +     │   POST /research/{id}/ask   │  - /research/{id}/ask (follow)│
│    sources in the report  │───────────────────────────►│                              │
│  - Follow-up chat feed    │                             └───────────┬──────────────────┘
└─────────────────────────┘                             LangGraph StateGraph (per session)
                                                                      │
                                                             entry_router
                                                  (state check, not classification: fresh
                                                   session -> router, awaiting-clarification
                                                   session -> the matching clarification
                                                   resolver, else -> followup_router)
                                                                      │
                                                          router_node / clarification_response
                                              (classify query_type, extract tickers, validate
                                               each via Twelve Data — bad or non-US tickers are
                                               dropped with a note, not a hard failure; a
                                               stock-related but nameless request asks a
                                               clarifying question instead of guessing; an
                                               off-topic or stock-discovery request gets a
                                               plain reply, no agents run)
                                                                      │
                                          dynamic fan-out (LangGraph Send / map-reduce),
                                          one branch per resolved ticker
                        ┌────────────────────────────┴────────────────────────────┐
                        ▼                                                         ▼
              per-ticker subgraph (NVDA)                              per-ticker subgraph (AMD)
        ┌───────────┬───────────┬───────────┐                  ┌───────────┬───────────┬───────────┐
        │Fundamentals│ Technical │   News    │        ...       │Fundamentals│ Technical │   News    │
        │ (Finnhub) │(Twelve Data)│(DuckDuckGo)│                 │ (Finnhub) │(Twelve Data)│(DuckDuckGo)│
        └───────────┴───────────┴───────────┘                  └───────────┴───────────┴───────────┘
        each node: try/except + timeout, never raises — returns status ok/failed
                        └────────────────────────────┬────────────────────────────┘
                                                       ▼
                                   type-aware synthesis (single / portfolio / comparison)
                                          merges available per-ticker results,
                                          notes any gaps explicitly, compiles citations
                                                       ▼
                                            Final Report (streamed)
```

Follow-up turns re-enter the same graph through `followup_router_node` (or, if a follow-up
was itself too vague, `followup_clarification_response_node`) instead of `router_node` —
see §7-§8 for the full entry-routing logic, including the off-topic/no-tickers/clarification
short-circuits that end a turn without ever reaching the specialist nodes.

Progress events (`run_started`, `router_completed`/`followup_classified`, `agent_started`,
`agent_completed`, `report_ready`/`followup_answer_ready`, `run_completed`, `run_failed` —
see §9) are derived at the API layer from LangGraph's `astream(stream_mode="updates")`
output, not published by node code directly, onto a per-session, persisted
event log. The SSE endpoint streams that log to the browser live and can replay it on
reconnect. This event layer is independent of LangGraph's own internals, so the frontend
never depends on framework-specific event shapes, and every function/flow that emits one
of these events also writes to the structured log described in §9 — the event stream is
the user-facing observability layer, the log file is the developer-facing one.

## 4. Query Scope

1. **Single-stock analysis** — "Analyze NVIDIA" / "What is the financial health of Apple?"
2. **Portfolio analysis** — "Analyze my portfolio of NVIDIA, Apple and Microsoft" → a
   per-stock health rollup plus qualitative portfolio-level notes (sector concentration /
   overlap). **Not in scope**: quantitative portfolio math (correlation matrices,
   MPT-style optimization, Sharpe ratio) — a different, much larger project.
3. **Comparison** — "Compare NVIDIA and AMD" → structured side-by-side comparison
   (relative valuation, momentum, sentiment), explicitly framed as informational, not
   investment advice.

Requests are capped at a configurable max of 5 tickers (`MAX_TICKERS`) to keep cost and
latency bounded.

## 5. Repository Layout

```
stock-research/
├── ARCHITECTURE.md              # this file
├── README.md
├── Makefile
├── logs/
│   └── dump.log                 # structured application log (gitignored, see §9)
├── backend/
│   ├── app/
│   │   ├── main.py                      # FastAPI app, CORS, startup, /health
│   │   ├── logging_config.py            # central logging setup -> logs/dump.log
│   │   ├── config.py                    # env vars: GROQ_API_KEY, FINNHUB_API_KEY,
│   │   │                                # TWELVEDATA_API_KEY, model names, timeouts, MAX_TICKERS
│   │   ├── api/
│   │   │   └── research_routes.py       # POST /research, GET .../stream, POST .../ask
│   │   ├── graph/
│   │   │   ├── state.py                 # ResearchState (multi-ticker + clarification/off-topic fields)
│   │   │   ├── build_graph.py           # entry_router -> router/clarification/followup -> fan-out -> synthesis
│   │   │   └── nodes/
│   │   │       ├── _shared.py                          # shared specialist-node LLM/Finding helpers
│   │   │       ├── _synthesis_shared.py                 # shared rendering/citation helpers for synthesis_*
│   │   │       ├── router_node.py                       # query_type + ticker extraction/validation
│   │   │       ├── clarification_response_node.py       # resolves a fresh-session clarification reply
│   │   │       ├── followup_router_node.py               # answer-from-context / refresh / add-ticker
│   │   │       ├── followup_clarification_response_node.py # resolves a follow-up clarification reply
│   │   │       ├── answer_from_context.py               # follow-up path 1: no tool calls
│   │   │       ├── fundamentals_node.py
│   │   │       ├── technical_node.py
│   │   │       ├── news_node.py
│   │   │       ├── synthesis_single.py
│   │   │       ├── synthesis_portfolio.py
│   │   │       └── synthesis_comparison.py
│   │   ├── tools/
│   │   │   ├── yahoo_finance.py         # Finnhub (fundamentals) + Twelve Data (price/technicals/
│   │   │   │                            # ticker resolution) + TTL cache + retries; filename is
│   │   │   │                            # legacy (`yfinance` itself is no longer imported — see §6)
│   │   │   ├── errors.py
│   │   │   └── web_search.py            # DuckDuckGo (ddgs) wrapper + retries
│   │   ├── llm/
│   │   │   ├── groq_client.py
│   │   │   └── errors.py                # rate-limit detection + shared fallback message
│   │   ├── streaming/
│   │   │   ├── events.py
│   │   │   └── session_bus.py           # per-session event log + live subscribers
│   │   └── memory/
│   │       └── checkpointer.py          # LangGraph checkpointer (SQLite) per session
│   ├── eval/
│   │   ├── dataset.yaml                 # single/portfolio/comparison/edge/follow-up cases
│   │   ├── run_eval.py
│   │   ├── objective_checks.py          # structural/numeric/citation-integrity checks
│   │   ├── judge.py                     # LLM-as-judge scoring rubric
│   │   └── results/                     # timestamped run outputs (gitignored)
│   ├── tests/
│   │   ├── test_health.py
│   │   ├── test_tools.py
│   │   ├── test_shared.py
│   │   ├── test_synthesis_shared.py
│   │   ├── test_graph.py                # incl. partial-failure, multi-ticker, clarification/off-topic cases
│   │   ├── test_followup.py             # answer/refresh/add-ticker + follow-up clarification
│   │   ├── test_research_routes.py
│   │   └── test_streaming.py
│   ├── pyproject.toml
│   └── .env.example
└── frontend/
    ├── src/
    │   ├── api/client.ts
    │   ├── hooks/useResearchStream.ts   # SSE + Last-Event-ID reconnection
    │   ├── lib/history.ts               # per-session conversation persistence (localStorage)
    │   ├── components/
    │   │   ├── Hero.tsx                 # landing state: pitch + example prompts
    │   │   ├── QuestionInput.tsx
    │   │   ├── Sidebar.tsx
    │   │   ├── StatusBanner.tsx
    │   │   ├── AgentCard.tsx
    │   │   ├── TickerGroup.tsx
    │   │   ├── FinalReport.tsx
    │   │   ├── ConversationFeed.tsx      # follow-up Q&A transcript
    │   │   └── Markdown.tsx              # shared report/answer renderer (citations, sources)
    │   ├── types.ts
    │   └── App.tsx
    ├── package.json
    └── vite.config.ts
```

## 6. Data Sources

Both providers below are accessed from `backend/app/tools/yahoo_finance.py` — the filename
is legacy (kept to limit diff churn; see git history). `yfinance`/Yahoo Finance is no longer
used anywhere: Yahoo's unofficial API actively blocks requests from cloud/datacenter IPs
(confirmed via `YFRateLimitError` in production on Render, badly enough that retries didn't
fix it), so market data moved to two paid-tier-free HTTP APIs instead.

**Finnhub — fundamentals** (`get_fundamentals`, `get_company_name`, `ticker_exists`)
- `/stock/profile2`, `/quote`, `/stock/metric` (`metric=all`) → `FundamentalsData` (sector,
  industry, margins, growth, yield, ROE, valuation multiples, etc.).
- Finnhub's basic-financials numbers are percentages (e.g. `netProfitMarginTTM: 27.62`);
  they're divided by 100 in this layer to match the decimal-fraction convention
  (`profitMargins: 0.276`) so nothing downstream needs to know the provider changed.

**Twelve Data — price history, technicals, ticker resolution** (`get_technical_data`,
`resolve_ticker`)
- Daily OHLC history → SMA20/50/200, RSI14, MACD, 52-week high/low, 1-month momentum,
  annualized volatility (plain pandas rolling-window math, no heavy TA dependency).
- `resolve_ticker`/`aresolve_ticker` requires real price history for a bare symbol before
  accepting it, falling back to well-known non-US exchange suffixes on the same symbol when
  it doesn't hold up — a bare symbol can otherwise silently collide with the wrong company
  (observed live: bare `"TCS"` resolved to a delisted company; Tata Consultancy Services is
  actually `"TCS.NS"`). No per-company lookup table.

**Coverage limitation: US-listed stocks only.** Both providers' free tiers are effectively
US-market-scoped. A ticker that resolves to a real, non-US company is still dropped —
`router_node`/`followup_router_node` surface this as "isn't currently supported" rather than
implying a typo (see `_no_tickers_node` in `build_graph.py` and the Hero UI's coverage note),
distinct from a ticker that just doesn't exist at all.

**Shared provider-integration notes**
- Both providers signal "invalid ticker" the same way `yfinance` used to: HTTP 200 with an
  empty/near-empty body, not an exception (Finnhub's `/stock/profile2` returns `{}` for a bad
  symbol; Twelve Data returns `status: "error"`). Validity is checked on the *shape* of the
  response, not on exceptions, and that contract is normalized across both providers so
  callers don't need to special-case which one a function talks to.
- Retries apply only to transient network errors (and Twelve Data rate limits), never to
  "ticker doesn't exist" — that's permanent, not transient, and retrying it would just burn
  through the free-tier rate limit for nothing.
- Every public fetch function is wrapped with a short TTL in-memory cache (avoids re-hitting
  the provider repeatedly, e.g. during eval runs) and, on the async side, a hard timeout.
- Sync core + `asyncio.to_thread` async wrappers, since neither underlying HTTP call is
  async and agent nodes run in an async LangGraph.

**Web search (DuckDuckGo via `ddgs`)**
- Query pattern: `"<company/ticker> stock news"`, recency-filtered where supported.
- Evidence unit = title + snippet + url + date. No full-page scraping in v1 — fetching
  and cleaning arbitrary article HTML (paywalls, bot blocks, boilerplate stripping) is
  fragility disproportionate to this project's scope; documented as a future extension.
  Every sentiment claim is grounded in a specific snippet + URL, not a vague summary.

## 7. Graph Design (LangGraph)

**State** (`ResearchState`) — ticker-list-based from the start, so single-stock is simply
the N=1 case rather than a special path that would need a later rewrite:

```python
class ResearchState(TypedDict):
    tickers: list[str]
    query_type: Literal["single", "portfolio", "comparison"]
    user_question: str
    per_ticker_results: Annotated[dict[str, TickerResults], merge_per_ticker_results]
    final_report: str | None
    conversation_history: Annotated[list[Message], operator.add]
    session_id: str
    # Router/agent-level warnings surfaced to the user (invalid ticker dropped, a
    # non-US ticker outside coverage, the MAX_TICKERS cap trimmed the request).
    notes: Annotated[list[str], operator.add]
    # Follow-up classification + its targets/answer — set by followup_router_node.
    followup_path: Literal["answer", "refresh", "add_ticker", "unrelated",
                            "discovery", "needs_clarification"] | None
    followup_targets: list[FollowUpTarget]      # [{ticker, agents: [AgentName, ...]}, ...]
    followup_answer: str | None
    # Set when a request is clearly stock-related but names no company; while True,
    # entry_router routes the *next* turn deterministically to the matching resolver
    # instead of letting an LLM re-classify it as a new, unrelated request.
    awaiting_clarification: bool
    clarification_question: str | None
    pending_question: str | None                # the original ambiguous question, for LLM context only
    pending_intent: QueryType | None             # the query_type already decided; not re-guessed on resolve
    # Set when a message isn't stock-related at all, or asks the app to screen/discover
    # stocks (a capability it doesn't have) — a plain reply, not a research trigger.
    # Always explicitly cleared back to None on every other outcome.
    off_topic_reply: str | None
    # Which entry path last set awaiting_clarification=True ("router" vs. "followup") —
    # tells entry_router which of the two clarification resolvers to send the next turn
    # to, since a fresh-session reply replaces state["tickers"] outright while a
    # follow-up-session reply must merge into the session's existing tickers instead.
    clarification_origin: Literal["router", "followup"] | None

class TickerResults(TypedDict, total=False):
    fundamentals: AgentResult
    technical: AgentResult
    news: AgentResult

class Finding(TypedDict):
    id: str            # stable id, referenced by citation markers in the final report
    claim: str
    evidence: str
    source: dict        # {type, label, url | None, as_of}

class AgentResult(TypedDict):
    status: Literal["ok", "failed"]
    summary: str
    findings: list[Finding]
    error: str | None
```

**Flow**

0. **`entry_router`** — a state check, not an LLM classification: `awaiting_clarification`
   routes deterministically to whichever clarification resolver matches
   `clarification_origin`; an existing `per_ticker_results` means the session has already
   run once, so the turn goes to `followup_router` instead of `router`. This is what keeps
   a short clarification reply like "TCS and Infosys" from ever being at risk of being
   misread as an unrelated new topic.
1. **`router_node`** (fresh sessions) / **`clarification_response_node`** (resolving a
   pending clarification) — classify `query_type` and extract tickers from free text,
   validating each via Twelve Data (`resolve_ticker`). Invalid or non-US tickers are
   dropped with a recorded reason rather than failing the whole request; `MAX_TICKERS` is
   enforced. The same LLM call also decides three short-circuits that skip the specialist
   nodes entirely: **ask a clarifying question** (stock-related, no company named — sets
   `awaiting_clarification`), **off-topic / stock-discovery reply** (sets
   `off_topic_reply`), or **no resolvable tickers** (a plain reply built from the drop
   reasons, not a report-shaped one — see `_no_tickers_node`).
2. **Dynamic fan-out** (LangGraph `Send`) spawns one per-ticker subgraph per resolved
   ticker; each subgraph fans out internally to `fundamentals_node` / `technical_node` /
   `news_node`. The three node implementations are reused regardless of query type —
   comparison and portfolio differ only at the synthesis layer, not in data gathering.
   Every node: timeout + try/except, **never raises**, always returns a valid
   `AgentResult` (`status: "ok" | "failed"`). Nodes don't publish events themselves — the
   API layer (§9) emits `agent_started` right before dispatching each (ticker, agent) pair
   and `agent_completed` from the resulting `per_ticker_results` update, independent of
   LangGraph's own internals.
3. **`collect_results`** — a no-op join barrier: LangGraph runs it once every
   dynamically-spawned specialist instance has settled, regardless of how many
   (ticker, agent) pairs were fanned out.
4. **Type-aware synthesis**, dispatched on `query_type`:
   - `synthesis_single` — one company's report.
   - `synthesis_portfolio` — per-stock rollup + qualitative concentration/overlap notes.
   - `synthesis_comparison` — structured side-by-side (valuation / momentum / sentiment),
     explicit non-advice framing.
   - All three note missing per-ticker sections explicitly rather than omitting them,
     compile the final citation list from every `Finding.source` used, and — if literally
     nothing came back usable for any ticker — fall back to a plain reply instead of a
     report-shaped one with no real content in it.

## 8. Follow-Up Conversation Design

Follow-ups **can** trigger fresh agent/tool calls — necessary once portfolio/comparison
queries exist (e.g. "now add Intel to the comparison" or "any news on AMD today?" cannot
be answered from stored text alone). `followup_router_node` classifies each follow-up into
one of six `followup_path` outcomes, doing only as much work as needed:

1. **`answer`** — fully answerable from the session's stored `per_ticker_results` /
   `final_report` / conversation history. Routed to `answer_from_context_node`: one
   grounded LLM call, no tool calls.
2. **`refresh`** — needs updated data for an *existing* ticker (e.g. "any news today for
   NVDA?"). Re-enters the graph at just the relevant specialist node(s) via `Send`, updates
   `per_ticker_results`, re-runs the appropriate synthesis node.
3. **`add_ticker`** — introduces a new ticker (e.g. "compare that with Intel too"). Spawns
   a new per-ticker subgraph, merges results, re-runs synthesis with the updated ticker
   list (may flip `query_type`, e.g. single → comparison).
4. **`needs_clarification`** — the follow-up is clearly about a stock in/around this
   session but too vague to act on (e.g. "how's the other one doing?" with 3+ tickers in
   the session). Sets `awaiting_clarification` + `clarification_origin="followup"`; the
   *next* turn resolves through `followup_clarification_response_node`, a sibling of
   `clarification_response_node` that merges into or selects from the session's *existing*
   tickers instead of replacing them outright (reusing the fresh-session resolver here
   would silently wipe out prior research the first time this path was exercised).
5. **`unrelated`** / **`discovery`** — not stock-related, or asks the app to screen/select
   candidate stocks (a capability this app doesn't have — it analyzes stocks it's given, it
   doesn't discover them). Both route to the same `off_topic` reply as the router-level
   equivalent.

All research-producing paths reuse the same specialist/synthesis nodes as the initial run —
there is no parallel implementation of the research logic for follow-ups. Session state is
loaded via the LangGraph checkpointer keyed on `session_id`.

## 9. Observability: Streaming Events + Application Log

Two distinct layers, serving two different audiences:

**a) SSE event stream (user-facing, per session)**
- Backed by a persisted, ordered event log (SQLite — the same store used by the
  checkpointer). Every event gets a monotonically increasing id.
- SSE responses set `id:` per event. On reconnect, the browser sends `Last-Event-ID`
  automatically; the endpoint replays events after that id, then either continues live
  (run still in progress) or closes after replay (run already finished).
- Idle sessions (default 1h TTL) are evicted to bound storage growth.
- If the backend process restarts mid-run, the in-flight LangGraph run is lost (accepted
  limitation for this project's scope) but the event log and last-good state survive via
  SQLite, so Q&A over the last completed report still works.

**b) Application log file (developer-facing, `logs/dump.log`)**
- Every node, tool call, and API request logs a structured line via the shared logger
  configured in `backend/app/logging_config.py` — not just errors, but the flow itself:
  entry/exit of every agent node, every external call (Finnhub/Twelve Data/DuckDuckGo/Groq)
  with timing, every retry, every fallback taken, and every SSE event published.
- Format: `timestamp | level | session_id | component | message | extra_fields(json)` —
  one line per event, so `grep`/`findstr` against `logs/dump.log` can answer "what did
  session X do" or "how long did the AMD fundamentals call take" without reading code.
- Rotates by size (e.g. 10MB, 5 backups) so it doesn't grow unbounded during eval runs.
- `logs/` is gitignored; `logs/.gitkeep` keeps the directory present in the repo.

## 10. Fault Tolerance Strategy

- **Tool-level**: timeout + try/except + limited retries on transient errors only (no
  retry on hard failures like an invalid ticker).
- **Node-level**: every node catches its own exceptions, always returns a valid
  `AgentResult` — never propagates an exception into the graph.
- **Ticker-level**: an invalid, unresolvable, or non-US ticker is dropped with a recorded
  reason at `router_node`/`followup_router_node` (§6), not a fatal error for the whole
  (possibly multi-ticker) request.
- **Query-level**: an off-topic message, a stock-discovery request, or a stock-related but
  nameless request never reaches the specialist nodes at all — it's answered directly with
  a plain reply or a clarifying question (§7-§8), rather than running (and likely failing)
  agents against tickers the router had to guess.
- **Graph-level**: synthesis runs once all branches *settle* (success or failure),
  bounded by per-node timeouts so nothing blocks indefinitely.
- **User-facing**: partial failures are visible and explained in both the live UI (agent
  card shows `failed: <reason>`) and the final report — never a silent drop or a hard 500.

## 11. Evaluation Harness

**Objective / structural checks** (deterministic — catch real bugs an LLM judge might
rate fine anyway):
- **Citation integrity** — every citation marker in the final report resolves to a real
  `Finding.id` that was actually produced (catches fabricated citations mechanically).
- **Citation coverage** — ratio of findings-with-a-source to total findings (~1.0 expected).
- **Schema validation** — `AgentResult` / `Finding` / final report conform to shape.
- **Numeric sanity** — computed indicators within plausible bounds (RSI ∈ [0,100], price
  > 0, etc.).
- **Partial-failure handling** — for deliberately-broken-ticker cases, the run still
  completes with a non-empty `final_report` that explicitly notes the failure.
- **Latency thresholds** — pass/fail against target SLAs (e.g. time-to-first-event < 2s;
  total time scaled by ticker count).

**LLM-as-judge** (graded, for what can't be checked mechanically): a separate Groq call
scores each run 1-5 on grounding, relevance, completeness across the three research
dimensions, and (for comparisons) whether the output is a genuine structured comparison
rather than three reports stapled together.

- `eval/dataset.yaml`: ~12-15 cases spanning all three query types, an invalid ticker, a
  thin-news-coverage company, and a follow-up scenario (including the "add a ticker" path).
- `eval/run_eval.py` runs the graph in-process (bypassing HTTP for speed), writes
  timestamped JSON to `eval/results/`; a diff script compares two runs (baseline vs. after
  a change) across both objective checks and judge scores.

## 12. Tech Stack Summary

- **Backend**: Python 3.11+, FastAPI, `uvicorn`, LangGraph, `langchain-groq`, `requests`
  (Finnhub + Twelve Data), `ddgs` (DuckDuckGo), `pandas`, `tenacity` (retries),
  `cachetools` (TTL caching), `pydantic`, `pydantic-settings`, `python-dotenv`, SQLite
  (checkpointer + event log). Config via `.env` (`GROQ_API_KEY`, `FINNHUB_API_KEY`,
  `TWELVEDATA_API_KEY`, model names, timeouts, `MAX_TICKERS`) — never committed
  (`.env.example` documents required keys).
- **Frontend**: Vite + React + TypeScript, native `EventSource` (SSE, reconnect via
  `Last-Event-ID`), Tailwind for styling.
- **Model routing**: one capable Groq-hosted model (e.g. `openai/gpt-oss-120b`) for
  agent reasoning/tool-use, configurable via env.

**Explicitly out of scope**: trading execution, user accounts/auth, multi-user scaling,
quantitative portfolio optimization (MPT, correlation matrices, Sharpe ratio), full-page
news scraping, billing, non-US-listed equities (§6).

## 13. Build Order

1. Backend skeleton: FastAPI app, config, logging, `.env` handling, health check.
2. Tools layer: `yahoo_finance.py` + `web_search.py`, unit-tested against mocked responses.
3. Graph v0, single-ticker path (`query_type="single"`, N=1): three agent nodes +
   `synthesis_single`, verified via a direct script including a forced-failure case.
4. Extend to N tickers: dynamic fan-out, `synthesis_portfolio`, `synthesis_comparison`.
5. Streaming: SQLite-backed event log + SSE route with `Last-Event-ID` replay.
6. Follow-up endpoint: `followup_router_node` with all three paths + checkpointer.
7. Frontend: agent cards (grouped per ticker), sources panel, report views, follow-up chat.
8. Eval harness: dataset + objective checks + judge; run once as the baseline.

## 14. Verification

- Backend: `pytest backend/tests` (`test_health`, `test_tools`, `test_shared`,
  `test_synthesis_shared`, `test_graph`, `test_followup`, `test_research_routes`,
  `test_streaming` — provider calls mocked; graph tests include a forced
  single-agent-failure case, a multi-ticker case with one invalid ticker, and
  clarification/off-topic cases).
- Manual smoke test: run all three query types end-to-end in the browser, confirm live
  per-ticker agent cards update, citations are traceable, a reconnect mid-run resumes
  correctly, and a follow-up that adds a new ticker updates the report.
- Eval: `python -m eval.run_eval` (from `backend/`) produces a results JSON; re-run after
  any future change and diff against the prior baseline.
