# Architecture — AI Stock Research Assistant

## 1. Problem & Goal

Researching a company means pulling together fundamentals, price/technical data, and
recent news/sentiment, then synthesizing all of it into one coherent view — normally a
slow, manual, inconsistent process. This project builds a small multi-agent system that
does that automatically:

- Specialized agents independently analyze fundamentals, technicals, and news/sentiment.
- Agents use real external data (Yahoo Finance, DuckDuckGo web search) rather than
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
| Backend language | Python 3.11+ | `yfinance` and `ddgs` (DuckDuckGo search) are native Python libraries; avoids HTTP-wrapper overhead. |
| Frontend | TypeScript + React (Vite) | Separate app, consumes the backend over HTTP/SSE. |
| LLM provider | Groq (via `langchain-groq`) | Fast inference, tool-calling support on Llama models. |
| Orchestration | LangGraph | Explicit state graph, native support for parallel fan-out/fan-in and checkpointed session state — a good fit for "N agents run independently, then merge." |
| User-facing transport | Server-Sent Events (SSE) | One-directional live progress stream is simpler than WebSockets for this use case and has native browser reconnection support (`Last-Event-ID`). |
| Query scope | Single-stock, portfolio, and comparison queries | See §4. |

## 3. High-Level Architecture

```
┌─────────────────────────┐        SSE stream         ┌──────────────────────────────┐
│  React/TS Frontend       │◄───────────────────────────│  FastAPI Backend              │
│  - AgentCards, grouped    │   POST /research            │  - /research (start job)      │
│    per ticker (live)      │   GET  /research/{id}/stream│  - /research/{id}/stream (SSE)│
│  - Sources / citations    │   POST /research/{id}/ask   │  - /research/{id}/ask (follow)│
│  - Final report            │───────────────────────────►│                              │
│    (single/portfolio/     │                             └───────────┬──────────────────┘
│     comparison view)      │                                         │
└─────────────────────────┘                             LangGraph StateGraph (per session)
                                                                      │
                                                            router_node
                                                    (classify query_type, extract tickers,
                                                     validate each via yfinance lookup —
                                                     bad tickers are dropped with a note,
                                                     not a hard failure)
                                                                      │
                                          dynamic fan-out (LangGraph Send / map-reduce),
                                          one branch per resolved ticker
                        ┌────────────────────────────┴────────────────────────────┐
                        ▼                                                         ▼
              per-ticker subgraph (NVDA)                              per-ticker subgraph (AMD)
        ┌───────────┬───────────┬───────────┐                  ┌───────────┬───────────┬───────────┐
        │Fundamentals│ Technical │   News    │        ...       │Fundamentals│ Technical │   News    │
        │ (yfinance) │(yfinance) │(DuckDuckGo)│                  │ (yfinance) │(yfinance) │(DuckDuckGo)│
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

Every node — specialist and synthesis alike — publishes structured progress events
(`started`, `tool_call`, `finding`, `completed`, `failed`) onto a per-session, persisted
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
ai-stock-research-agent/
├── ARCHITECTURE.md              # this file
├── README.md
├── TODO.md
├── Makefile
├── logs/
│   └── dump.log                 # structured application log (gitignored, see §9)
├── backend/
│   ├── app/
│   │   ├── main.py                      # FastAPI app, CORS, startup, health check
│   │   ├── logging_config.py            # central logging setup -> logs/dump.log
│   │   ├── config.py                    # env vars: GROQ_API_KEY, model names, timeouts, MAX_TICKERS
│   │   ├── api/
│   │   │   └── research_routes.py       # POST /research, GET .../stream, POST .../ask
│   │   ├── graph/
│   │   │   ├── state.py                 # ResearchState (multi-ticker from the start)
│   │   │   ├── build_graph.py           # router -> dynamic fan-out -> synthesis
│   │   │   └── nodes/
│   │   │       ├── router_node.py       # query_type + ticker extraction/validation
│   │   │       ├── fundamentals_node.py
│   │   │       ├── technical_node.py
│   │   │       ├── news_node.py
│   │   │       ├── synthesis_single.py
│   │   │       ├── synthesis_portfolio.py
│   │   │       ├── synthesis_comparison.py
│   │   │       └── followup_router.py   # answer-from-context / re-run / add-ticker
│   │   ├── tools/
│   │   │   ├── yahoo_finance.py         # yfinance wrappers + TTL cache + retries
│   │   │   └── web_search.py            # DuckDuckGo (ddgs) wrapper + retries
│   │   ├── llm/
│   │   │   └── groq_client.py
│   │   ├── streaming/
│   │   │   ├── events.py
│   │   │   └── session_bus.py           # per-session event log + live subscribers
│   │   └── memory/
│   │       └── checkpointer.py          # LangGraph checkpointer (SQLite) per session
│   ├── eval/
│   │   ├── dataset.yaml                 # single/portfolio/comparison/edge cases
│   │   ├── run_eval.py
│   │   ├── objective_checks.py          # structural/numeric/citation-integrity checks
│   │   ├── judge.py                     # LLM-as-judge scoring rubric
│   │   └── results/                     # timestamped run outputs (gitignored)
│   ├── tests/
│   │   ├── test_health.py
│   │   ├── test_tools.py
│   │   ├── test_nodes.py
│   │   └── test_graph.py                # incl. partial-failure, multi-ticker cases
│   ├── pyproject.toml
│   └── .env.example
└── frontend/
    ├── src/
    │   ├── api/client.ts
    │   ├── hooks/useResearchStream.ts   # SSE + Last-Event-ID reconnection
    │   ├── components/
    │   │   ├── AgentCard.tsx
    │   │   ├── TickerGroup.tsx
    │   │   ├── SourcesPanel.tsx
    │   │   ├── ComparisonTable.tsx
    │   │   ├── PortfolioOverview.tsx
    │   │   ├── FinalReport.tsx
    │   │   └── FollowUpChat.tsx
    │   └── App.tsx
    ├── package.json
    └── vite.config.ts
```

## 6. Data Sources

**Yahoo Finance (`yfinance`)**
- Fundamentals: `Ticker.info`, `.financials` / `.balance_sheet` / `.cashflow`, `.earnings`.
- Technical: `Ticker.history(period="1y", interval="1d")` → SMA20/50/200, RSI14, MACD,
  52-week high/low, momentum/volatility (plain pandas rolling-window math, no heavy TA
  dependency).
- Wrapped with a short TTL in-memory cache and 1-2 retries with backoff — `yfinance`
  scrapes Yahoo Finance and is occasionally flaky.

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
    per_ticker_results: dict[str, dict[str, "AgentResult"]]   # ticker -> {fundamentals, technical, news}
    final_report: str | None
    conversation_history: list[Message]
    session_id: str

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

1. **`router_node`** — classifies `query_type` and extracts tickers from free text;
   validates each via a cheap `yfinance` lookup. Invalid tickers are dropped with a
   recorded reason (e.g. `"XYZQ not found"`) rather than failing the whole request.
   Enforces `MAX_TICKERS`.
2. **Dynamic fan-out** (LangGraph `Send`) spawns one per-ticker subgraph per resolved
   ticker; each subgraph fans out internally to `fundamentals_node` / `technical_node` /
   `news_node`. The three node implementations are reused regardless of query type —
   comparison and portfolio differ only at the synthesis layer, not in data gathering.
   Every node: timeout + try/except, **never raises**, always returns a valid
   `AgentResult`, and emits `started` / `tool_call` / `finding` / `completed` / `failed`
   events as it works.
3. **Map-reduce join** collects all per-ticker results into `per_ticker_results`.
4. **Type-aware synthesis**, dispatched on `query_type`:
   - `synthesis_single` — one company's report.
   - `synthesis_portfolio` — per-stock rollup + qualitative concentration/overlap notes.
   - `synthesis_comparison` — structured side-by-side (valuation / momentum / sentiment),
     explicit non-advice framing.
   - All three note missing per-ticker sections explicitly rather than omitting them, and
     compile the final citation list from every `Finding.source` used.

## 8. Follow-Up Conversation Design

Follow-ups **can** trigger fresh agent/tool calls — necessary once portfolio/comparison
queries exist (e.g. "now add Intel to the comparison" or "any news on AMD today?" cannot
be answered from stored text alone). `followup_router_node` classifies each follow-up
into exactly one of three paths, doing only as much work as needed:

1. **Answer from existing context** — fully answerable from the session's stored
   `per_ticker_results` / `final_report` / conversation history. One grounded LLM call,
   no tool calls.
2. **Refresh specific data** — needs updated data for an *existing* ticker (e.g. "any news
   today for NVDA?"). Re-enters the graph at just the relevant specialist node(s), updates
   `per_ticker_results`, re-runs the appropriate synthesis node.
3. **Add a ticker** — introduces a new ticker (e.g. "compare that with Intel too").
   Spawns a new per-ticker subgraph, merges results, re-runs synthesis with the updated
   ticker list (may flip `query_type`, e.g. single → comparison).

All three paths reuse the same specialist/synthesis nodes as the initial run. Session
state is loaded via the LangGraph checkpointer keyed on `session_id`.

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
  entry/exit of every agent node, every external call (yfinance/DuckDuckGo/Groq) with
  timing, every retry, every fallback taken, and every SSE event published.
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
- **Ticker-level**: an invalid/unresolvable ticker is dropped with a recorded reason at
  `router_node`, not a fatal error for the whole (possibly multi-ticker) request.
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

- **Backend**: Python 3.11+, FastAPI, `uvicorn`, LangGraph, `langchain-groq`, `yfinance`,
  `ddgs`, `pandas`, `pydantic`, `pydantic-settings`, `python-dotenv`, SQLite (checkpointer
  + event log). Config via `.env` (`GROQ_API_KEY`, model names, timeouts, `MAX_TICKERS`) —
  never committed (`.env.example` documents required keys).
- **Frontend**: Vite + React + TypeScript, native `EventSource` (SSE, reconnect via
  `Last-Event-ID`), Tailwind for styling.
- **Model routing**: one capable Groq-hosted model (e.g. `llama-3.3-70b-versatile`) for
  agent reasoning/tool-use, configurable via env.

**Explicitly out of scope**: trading execution, user accounts/auth, multi-user scaling,
quantitative portfolio optimization (MPT, correlation matrices, Sharpe ratio), full-page
news scraping, billing.

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

- Backend: `pytest backend/tests` (tool wrappers mocked; graph tests include a forced
  single-agent-failure case and a multi-ticker case with one invalid ticker).
- Manual smoke test: run all three query types end-to-end in the browser, confirm live
  per-ticker agent cards update, citations are traceable, a reconnect mid-run resumes
  correctly, and a follow-up that adds a new ticker updates the report.
- Eval: `python -m eval.run_eval` (from `backend/`) produces a results JSON; re-run after
  any future change and diff against the prior baseline.
