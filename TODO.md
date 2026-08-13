# TODO

Tracks progress against the build order in [ARCHITECTURE.md §13](./ARCHITECTURE.md#13-build-order).
Check items off as they're completed; add sub-items as work is broken down further.

## 0. Scaffolding
- [x] `ARCHITECTURE.md`
- [x] `README.md`
- [x] `TODO.md`
- [x] Logging setup (`backend/app/logging_config.py` → `logs/dump.log`)
- [x] `Makefile` with common commands
- [x] Backend skeleton (FastAPI app, config, health check)
- [x] Initial test (`test_health.py`) passing via `make test`

## 1. Tools Layer
- [x] `tools/yahoo_finance.py` — fundamentals + price history, TTL cache, retries
- [x] `tools/web_search.py` — DuckDuckGo (`ddgs`) wrapper, retries
- [x] Unit tests for both against mocked responses (17 tests passing)
- [x] Live smoke test against real Yahoo Finance + DuckDuckGo data

## 2. Graph v0 — Single-Ticker Path
- [x] `ResearchState` / `AgentResult` / `Finding` schemas (`graph/state.py`), incl. custom
      reducer for `per_ticker_results` so parallel branches merge instead of overwriting
- [x] `fundamentals_node`, `technical_node`, `news_node`
- [x] `synthesis_single` (with deterministic all-failed and LLM-failure fallback paths)
- [x] Direct script (`scripts/verify_graph.py`) verifying a full live run produces a cited report
- [x] Forced-failure case: bad ticker still yields a report with a noted gap (verified live)
- [x] Automated mocked tests (`tests/test_graph.py`, 5 tests incl. citation-integrity check)

## 3. Multi-Ticker (Portfolio / Comparison)
- [x] `router_node`: query_type classification + ticker extraction/validation + `MAX_TICKERS` cap
- [x] Dynamic fan-out (LangGraph `Send`) per ticker — same 3 specialist nodes reused unchanged
- [x] `synthesis_portfolio`
- [x] `synthesis_comparison`
- [x] `collect_results` join barrier verified to wait for all 3×N branches (live: 2- and
      3-ticker requests; mocked regression test `test_multi_ticker_fan_out_merges_every_ticker`)
- [x] Live-verified: single/portfolio/comparison queries, invalid-ticker drop +
      query_type downgrade, and `MAX_TICKERS` cap enforcement (6-company request → capped
      at 5 with a note)
- [x] Mocked tests for router classification, cap enforcement, no-valid-tickers
      short-circuit (confirms zero agent calls made), and multi-ticker merge (26 tests total)

## 4. Streaming
- [x] `streaming/events.py` — event types (run_started, router_completed, agent_started,
      agent_completed, report_ready, run_completed, run_failed) derived from
      `astream(stream_mode="updates")` at the API layer — no node code touched
- [x] `streaming/session_bus.py` — SQLite-backed event log + in-memory live fan-out
- [x] `POST /research` (starts run as background task) + `GET /research/{id}/stream` (SSE)
- [x] SSE route with `Last-Event-ID` reconnection/replay (dedup via last-sent-id tracking)
- [x] Session idle-TTL eviction (opportunistic, ~2% of publishes, `SESSION_TTL_SECONDS`)
- [x] Live-verified with `curl -N`: full run streaming end-to-end, reconnect-after-completion
      (pure replay), and reconnect-mid-run (disconnected after 3 events, reconnected with
      `Last-Event-ID`, received the remaining 13 events with no gaps or duplicates)
- [x] Automated tests (`tests/test_streaming.py`, 7 tests: replay, per-session scoping,
      terminal detection, live queue + sentinel, multi-subscriber fan-out, unregister)

## 5. Follow-Up Conversation
- [x] `memory/checkpointer.py` (`AsyncSqliteSaver`, keyed by `session_id` = `thread_id`,
      same `sessions.db` file as the event log)
- [x] Unified graph entry point (`_entry_router`): fresh session -> `router`,
      checkpointed session with existing results -> `followup_router` — no separate graph
- [x] `followup_router_node`: answer-from-context / refresh-data / add-ticker paths, all
      three reusing the same fundamentals/technical/news + synthesis_* nodes via `Send`
- [x] `answer_from_context_node` for the "answer" path (no tool calls)
- [x] `conversation_history` now actually populated (every user turn + assistant reply)
- [x] `/research/{id}/ask` endpoint (404s if no prior research exists for the session)
- [x] Live-verified (`scripts/verify_followup.py`) across a real 4-turn conversation:
      initial comparison -> answer-from-context -> refresh (news only, verified only 1
      of 3 agents re-ran) -> add_ticker (flipped query_type, added a 3rd stock).
      Caught and fixed a real bug this way: the LLM initially extracted "INTEL" instead
      of "INTC" — fixed by tightening the `new_tickers` field description to explicitly
      require ticker symbols with an example, matching `router_node`'s working pattern.
      Also organically validated the LLM-failure fallback path when a real Groq 429
      (daily token quota) hit mid-run — the run still completed via the deterministic
      fallback report instead of crashing.
- [x] Mocked tests (`tests/test_followup.py`, 4 tests): answer path makes zero new tool
      calls, refresh path calls only the targeted agent (verified via call counts) and
      preserves untouched agents' prior results, add_ticker flips query_type and runs
      all 3 agents for the new ticker, and MAX_TICKERS cap is enforced on follow-ups too

## 6. Frontend
- [x] Vite + React + TS scaffold (Tailwind + typography plugin, `react-markdown`)
- [x] `useResearchStream` hook — opens a fresh `EventSource` per turn, closes it on
      `run_completed`/`run_failed` rather than holding one persistent connection open for
      the whole session (simpler than an idle-keepalive design; mid-turn reconnection via
      `Last-Event-ID` still works exactly as verified with `curl -N` in Phase 4)
- [x] `AgentCard` (queued/running/done/failed, inline error, findings + source links),
      `TickerGroup` (groups 3 agent cards per ticker)
- [x] `FinalReport` — single markdown renderer for all 3 query types, since the backend
      already emits one adaptively-structured report string rather than separate
      structured comparison/portfolio data; skipped the separately-proposed
      `ComparisonTable`/`PortfolioOverview` components as there's no structured data
      source for them to render (documented simplification, not a silent scope cut)
- [x] `FollowUpChat` (Q&A transcript + input, distinguishes a short answer from "the
      report above was updated")
- [x] `npx tsc -b` (type-check) and `npm run build` (production build) both clean
- [x] Browser-verified two ways (Playwright, no project browser-run skill existed yet):
      (1) **live**, against the real backend — confirmed the question form, live query-type
      header, and agent cards transitioning queued→running→done/failed with real data,
      including a real Groq 429 rendering inline exactly as designed; a full success-path
      run wasn't captured live because the Groq daily quota was exhausted during testing.
      (2) **network-mocked** (Playwright route interception, no LLM/backend dependency) —
      confirmed the full pipeline: agent cards (incl. one deliberately failed agent),
      markdown report with inline citations and a rendered Sources list, and a follow-up
      Q&A exchange in the chat panel. Zero browser console errors in either run.

## 7. Evaluation Harness
- [x] `eval/dataset.yaml` — 12 cases: 3 single-stock (incl. a real thin-news-coverage
      ticker, UNFI), 2 comparisons, 1 portfolio (+1 natural-language portfolio phrasing),
      an invalid ticker, a comparison with one invalid ticker, a `MAX_TICKERS`-cap case,
      and 2 follow-up scenarios (answer + refresh paths) plus an add-ticker follow-up
      nested in the first comparison case
- [x] `eval/objective_checks.py` — hard gates (schema, citation integrity, findings
      well-formed, the deterministic total-failure message) + soft/informational checks
      (partial-failure acknowledgment, follow-up path match, latency budget)
- [x] `eval/judge.py` (LLM-as-judge: grounding/relevance/completeness/structure).
      Caught and fixed a real bug here: Groq's tool-call schema validation rejected its
      own model's output ~every time for short reports (score fields returned as quoted
      strings, failing Groq's *server-side* schema check before even reaching client
      code) — deterministic at temperature=0, so retries alone never helped. Fixed by
      changing score fields from constrained `int` (`ge=1,le=5`) to `Literal[1,2,3,4,5]`;
      confirmed via a smoke test that the same case which failed 3/3 times before the fix
      succeeded on the first attempt after
- [x] `eval/run_eval.py` (`python -m eval.run_eval` from `backend/` — needs `-m` so the
      `eval` package's own internal imports resolve) — runs in-process via `InMemorySaver`
      (follow-ups need a checkpointer even for a single eval process), writes timestamped
      JSON to `eval/results/`
- [x] Ran the full dataset once. Real-world result: the first 3 cases ran clean (5/5/4/5
      judge scores), then the Groq daily quota was exhausted mid-run — every remaining
      case hit 429s, including at the router-classification stage. **Every single one
      still degraded gracefully**: schema-valid report, non-empty, clearly explaining the
      failure, 100% hard-check pass rate throughout, zero crashes — an unplanned but real
      end-to-end proof of the project's core fault-tolerance goal. Not a usable *quality*
      baseline though (9/12 cases got no real research done) — re-run once quota resets
      for a clean baseline to diff future changes against.

## Known limitations (tracked, not blocking)
- Backend restart mid-run loses the in-flight LangGraph run (event log/state survive via SQLite).
- No full-page news scraping — news evidence is title/snippet/url only.
- No quantitative portfolio optimization (correlation matrices, MPT, Sharpe ratio).
