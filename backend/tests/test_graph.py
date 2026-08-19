from __future__ import annotations

import re
import uuid
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

import app.graph.nodes.clarification_response_node as clarification_mod
import app.graph.nodes.fundamentals_node as fundamentals_mod
import app.graph.nodes.news_node as news_mod
import app.graph.nodes.router_node as router_mod
import app.graph.nodes.synthesis_comparison as synthesis_comparison_mod
import app.graph.nodes.synthesis_portfolio as synthesis_portfolio_mod
import app.graph.nodes.synthesis_single as synthesis_mod
import app.graph.nodes.technical_node as technical_mod
from app.config import settings
from app.graph.build_graph import build_research_graph
from app.graph.nodes._shared import LLMFinding, NodeAnalysis
from app.graph.nodes.news_node import NewsAnalysis, NewsLLMFinding
from app.graph.nodes.router_node import RouterDecision, TickerCandidate
from app.graph.state import failed_result, new_state, ok_result
from app.llm.errors import LLMAnalysisError
from app.tools.errors import WebSearchError, YahooFinanceError
from app.tools.web_search import SearchResult
from app.tools.yahoo_finance import FundamentalsData, ResolvedTicker, TechnicalData

FAKE_FUNDAMENTALS = FundamentalsData(
    ticker="AAPL", name="Apple Inc.", sector="Technology", industry="Consumer Electronics",
    market_cap=4_000_000_000_000, trailing_pe=30.0, forward_pe=28.0, price_to_book=50.0,
    dividend_yield=0.4, profit_margin=0.25, revenue_growth=0.1, earnings_growth=0.15,
    return_on_equity=1.4, total_debt=1e11, total_cash=6e10, current_price=300.0,
    recommendation="buy", as_of="2026-08-12T00:00:00+00:00",
)

FAKE_TECHNICAL = TechnicalData(
    ticker="AAPL", last_close=300.0, sma_20=298.0, sma_50=295.0, sma_200=280.0,
    rsi_14=55.0, macd={"macd": 1.0, "signal": 0.8, "histogram": 0.2},
    momentum_1m_pct=2.0, volatility_annualized_pct=20.0,
    fifty_two_week_high=320.0, fifty_two_week_low=200.0, as_of="2026-08-12T00:00:00+00:00",
)

FAKE_NEWS_RESULTS = [
    SearchResult(
        title="Apple news", url="https://example.com/a", snippet="Apple did something.",
        date="2026-08-11", source="Example",
    )
]


class FakeAIMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLM:
    def __init__(self, content: str = "Synthesized report body mentioning findings.") -> None:
        self._content = content

    async def ainvoke(self, prompt: str) -> FakeAIMessage:  # noqa: ARG002
        return FakeAIMessage(self._content)


def _async_return(value: Any):
    async def _fn(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        return value

    return _fn


def _async_raise(exc: Exception):
    async def _fn(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        raise exc

    return _fn


def _mock_analysis(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    *,
    summary: str = "Summary text.",
    claims: list[tuple[str, str]] | None = None,
) -> None:
    claims = claims or [("Claim one", "evidence one")]

    async def _fake(prompt: str, schema: type = NodeAnalysis) -> NodeAnalysis:  # noqa: ARG001
        if schema is NewsAnalysis:
            return NewsAnalysis(
                summary=summary,
                findings=[NewsLLMFinding(claim=c, evidence=e, article_index=1) for c, e in claims],
                overall_sentiment="neutral",
            )
        return NodeAnalysis(summary=summary, findings=[LLMFinding(claim=c, evidence=e) for c, e in claims])

    monkeypatch.setattr(module, "run_structured_analysis", _fake)


@pytest.fixture(autouse=True)
def _mock_synthesis_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = lambda temperature=0.3: FakeLLM()  # noqa: ARG005, E731
    monkeypatch.setattr(synthesis_mod, "get_chat_model", fake)
    monkeypatch.setattr(synthesis_portfolio_mod, "get_chat_model", fake)
    monkeypatch.setattr(synthesis_comparison_mod, "get_chat_model", fake)


@pytest.fixture(autouse=True)
def _mock_company_name_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    # news_node looks up the company's display name (a search-query disambiguation aid,
    # see news_node.py) before every search — stub it so no test in this file makes a
    # live Yahoo Finance call regardless of whether it happens to exercise news_node.
    monkeypatch.setattr(news_mod, "aget_company_name", _async_return(None))


def _new_state() -> Any:
    return new_state(
        tickers=["AAPL"], query_type="single", user_question="Analyze AAPL",
        session_id=str(uuid.uuid4()),
    )


async def test_full_success_path_merges_all_three_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression guard for the per_ticker_results reducer: without it, only one of
    the three parallel branches' updates would survive (last write wins)."""
    monkeypatch.setattr(fundamentals_mod, "aget_fundamentals", _async_return(FAKE_FUNDAMENTALS))
    monkeypatch.setattr(technical_mod, "aget_technical_data", _async_return(FAKE_TECHNICAL))
    monkeypatch.setattr(news_mod, "asearch_news", _async_return(FAKE_NEWS_RESULTS))
    _mock_analysis(monkeypatch, fundamentals_mod)
    _mock_analysis(monkeypatch, technical_mod)
    _mock_analysis(monkeypatch, news_mod)

    result = await build_research_graph().ainvoke(_new_state())

    ticker_results = result["per_ticker_results"]["AAPL"]
    assert set(ticker_results) == {"fundamentals", "technical", "news"}
    assert all(r["status"] == "ok" for r in ticker_results.values())
    assert result["final_report"]
    assert "Sources" in result["final_report"]


async def test_single_agent_failure_still_produces_report(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        fundamentals_mod, "aget_fundamentals", _async_raise(YahooFinanceError("no data"))
    )
    monkeypatch.setattr(technical_mod, "aget_technical_data", _async_return(FAKE_TECHNICAL))
    monkeypatch.setattr(news_mod, "asearch_news", _async_return(FAKE_NEWS_RESULTS))
    _mock_analysis(monkeypatch, technical_mod)
    _mock_analysis(monkeypatch, news_mod)

    result = await build_research_graph().ainvoke(_new_state())

    ticker_results = result["per_ticker_results"]["AAPL"]
    assert ticker_results["fundamentals"]["status"] == "failed"
    assert ticker_results["fundamentals"]["error"] == "no data"
    assert ticker_results["technical"]["status"] == "ok"
    assert ticker_results["news"]["status"] == "ok"
    assert result["final_report"]


async def test_agent_llm_analysis_failure_marks_agent_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tool fetch succeeds but the LLM summarization step fails: the agent is still
    marked failed (not a partial/garbage success), and the run still completes."""
    monkeypatch.setattr(fundamentals_mod, "aget_fundamentals", _async_return(FAKE_FUNDAMENTALS))
    monkeypatch.setattr(
        fundamentals_mod, "run_structured_analysis", _async_raise(LLMAnalysisError("groq down"))
    )
    monkeypatch.setattr(technical_mod, "aget_technical_data", _async_return(FAKE_TECHNICAL))
    monkeypatch.setattr(news_mod, "asearch_news", _async_return(FAKE_NEWS_RESULTS))
    _mock_analysis(monkeypatch, technical_mod)
    _mock_analysis(monkeypatch, news_mod)

    result = await build_research_graph().ainvoke(_new_state())

    assert result["per_ticker_results"]["AAPL"]["fundamentals"]["status"] == "failed"
    assert result["per_ticker_results"]["AAPL"]["fundamentals"]["error"] == "groq down"
    assert result["final_report"]


async def test_all_agents_failed_produces_deterministic_report_without_llm_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fundamentals_mod, "aget_fundamentals", _async_raise(YahooFinanceError("no fundamentals"))
    )
    monkeypatch.setattr(
        technical_mod, "aget_technical_data", _async_raise(YahooFinanceError("no technical"))
    )
    monkeypatch.setattr(news_mod, "asearch_news", _async_raise(WebSearchError("search down")))

    calls = {"count": 0}

    class TrackingFakeLLM:
        async def ainvoke(self, prompt: str) -> FakeAIMessage:  # noqa: ARG002
            calls["count"] += 1
            return FakeAIMessage("should not be used")

    monkeypatch.setattr(synthesis_mod, "get_chat_model", lambda temperature=0.3: TrackingFakeLLM())  # noqa: ARG005

    result = await build_research_graph().ainvoke(_new_state())

    ticker_results = result["per_ticker_results"]["AAPL"]
    assert all(r["status"] == "failed" for r in ticker_results.values())
    # A plain reply, not a fake "Research Report" card — nothing was actually researched.
    assert result["final_report"] is None
    assert "wasn't able to complete research" in result["followup_answer"]
    assert calls["count"] == 0  # total failure is handled deterministically, no LLM call needed


async def test_fundamentals_node_skips_llm_when_data_is_too_thin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ticker resolving to a company with almost no populated fields (e.g. Yahoo
    matching the wrong, data-thin listing for an ambiguous symbol) must fail
    deterministically rather than send a near-empty prompt to a forced-tool-call LLM —
    which tends to refuse the call outright and surface a raw provider API error."""
    thin_data = FundamentalsData(
        ticker="TCS", name="Some Co", sector=None, industry=None, market_cap=None,
        trailing_pe=None, forward_pe=None, price_to_book=None, dividend_yield=None,
        profit_margin=None, revenue_growth=None, earnings_growth=None,
        return_on_equity=None, total_debt=None, total_cash=None, current_price=None,
        recommendation=None, as_of="2026-08-19T00:00:00+00:00",
    )
    monkeypatch.setattr(fundamentals_mod, "aget_fundamentals", _async_return(thin_data))

    calls = {"count": 0}

    async def _tracking_analysis(*args: Any, **kwargs: Any) -> NodeAnalysis:  # noqa: ARG001
        calls["count"] += 1
        raise AssertionError("should not be called with insufficient data")

    monkeypatch.setattr(fundamentals_mod, "run_structured_analysis", _tracking_analysis)

    state = new_state(tickers=["TCS"], query_type="single", user_question="Analyze TCS", session_id=str(uuid.uuid4()))
    result = await fundamentals_mod.fundamentals_node(state)

    fundamentals_result = result["per_ticker_results"]["TCS"]["fundamentals"]
    assert fundamentals_result["status"] == "failed"
    assert "Not enough fundamentals data" in fundamentals_result["error"]
    assert calls["count"] == 0


async def test_technical_node_skips_llm_when_data_is_too_thin(monkeypatch: pytest.MonkeyPatch) -> None:
    thin_data = TechnicalData(
        ticker="TCS", last_close=100.0, sma_20=None, sma_50=None, sma_200=None,
        rsi_14=None, macd=None, momentum_1m_pct=None, volatility_annualized_pct=None,
        fifty_two_week_high=None, fifty_two_week_low=None, as_of="2026-08-19T00:00:00+00:00",
    )
    monkeypatch.setattr(technical_mod, "aget_technical_data", _async_return(thin_data))

    calls = {"count": 0}

    async def _tracking_analysis(*args: Any, **kwargs: Any) -> NodeAnalysis:  # noqa: ARG001
        calls["count"] += 1
        raise AssertionError("should not be called with insufficient data")

    monkeypatch.setattr(technical_mod, "run_structured_analysis", _tracking_analysis)

    state = new_state(tickers=["TCS"], query_type="single", user_question="Analyze TCS", session_id=str(uuid.uuid4()))
    result = await technical_mod.technical_node(state)

    technical_result = result["per_ticker_results"]["TCS"]["technical"]
    assert technical_result["status"] == "failed"
    assert "Not enough price/indicator data" in technical_result["error"]
    assert calls["count"] == 0


async def test_citation_markers_in_report_resolve_to_real_findings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Foreshadows the eval harness's citation-integrity check (ARCHITECTURE.md §11):
    every [id] marker the synthesis LLM emits must resolve to a real Finding.id."""
    monkeypatch.setattr(fundamentals_mod, "aget_fundamentals", _async_return(FAKE_FUNDAMENTALS))
    monkeypatch.setattr(technical_mod, "aget_technical_data", _async_return(FAKE_TECHNICAL))
    monkeypatch.setattr(news_mod, "asearch_news", _async_return(FAKE_NEWS_RESULTS))
    _mock_analysis(monkeypatch, fundamentals_mod, claims=[("c1", "e1"), ("c2", "e2")])
    _mock_analysis(monkeypatch, technical_mod, claims=[("c3", "e3")])
    _mock_analysis(monkeypatch, news_mod, claims=[("c4", "e4")])
    monkeypatch.setattr(
        synthesis_mod,
        "get_chat_model",
        lambda temperature=0.3: FakeLLM("Report citing [AAPL-fundamentals-1] and [AAPL-news-1]."),  # noqa: ARG005
    )

    result = await build_research_graph().ainvoke(_new_state())

    all_ids = {
        f["id"]
        for agent_result in result["per_ticker_results"]["AAPL"].values()
        for f in agent_result["findings"]
    }
    body = result["final_report"].split("**Sources**")[0]
    cited_ids = set(re.findall(r"\[([\w-]+)\]", body))

    assert cited_ids, "expected at least one citation marker in the report body"
    assert cited_ids <= all_ids


# --- router + multi-ticker fan-out --------------------------------------------------


def _mock_router_decision(
    monkeypatch: pytest.MonkeyPatch,
    query_type: str,
    tickers: list[str],
    exists_map: dict[str, bool] | None = None,
    needs_clarification: bool = False,
    clarifying_question: str | None = None,
    is_stock_related: bool = True,
    off_topic_reply: str | None = None,
    is_discovery_request: bool = False,
    discovery_reply: str | None = None,
    unaddressed_note: str | None = None,
) -> None:
    exists_map = exists_map if exists_map is not None else dict.fromkeys(tickers, True)

    async def _fake_decision(prompt: str, schema: type = RouterDecision) -> RouterDecision:  # noqa: ARG001
        return RouterDecision(
            is_stock_related=is_stock_related,
            query_type=query_type,  # type: ignore[arg-type]
            tickers=[TickerCandidate(ticker=t) for t in tickers],
            needs_clarification=needs_clarification,
            clarifying_question=clarifying_question,
            off_topic_reply=off_topic_reply,
            is_discovery_request=is_discovery_request,
            discovery_reply=discovery_reply,
            unaddressed_note=unaddressed_note,
        )

    async def _fake_resolve(ticker: str) -> ResolvedTicker:
        return ResolvedTicker(ticker if exists_map.get(ticker, True) else None)

    monkeypatch.setattr(router_mod, "run_structured_analysis", _fake_decision)
    monkeypatch.setattr(router_mod, "aresolve_ticker", _fake_resolve)


def _mock_clarification_decision(
    monkeypatch: pytest.MonkeyPatch,
    query_type: str,
    tickers: list[str],
    exists_map: dict[str, bool] | None = None,
    resolves_pending_clarification: bool = True,
    is_stock_related: bool = True,
    off_topic_reply: str | None = None,
    needs_clarification: bool = False,
    clarifying_question: str | None = None,
    is_discovery_request: bool = False,
    discovery_reply: str | None = None,
    unaddressed_note: str | None = None,
) -> None:
    exists_map = exists_map if exists_map is not None else dict.fromkeys(tickers, True)

    async def _fake_decision(prompt: str, schema: type = RouterDecision) -> RouterDecision:  # noqa: ARG001
        return RouterDecision(
            is_stock_related=is_stock_related,
            query_type=query_type,  # type: ignore[arg-type]
            tickers=[TickerCandidate(ticker=t) for t in tickers],
            needs_clarification=needs_clarification,
            clarifying_question=clarifying_question,
            off_topic_reply=off_topic_reply,
            resolves_pending_clarification=resolves_pending_clarification,
            is_discovery_request=is_discovery_request,
            discovery_reply=discovery_reply,
            unaddressed_note=unaddressed_note,
        )

    async def _fake_resolve(ticker: str) -> ResolvedTicker:
        return ResolvedTicker(ticker if exists_map.get(ticker, True) else None)

    monkeypatch.setattr(clarification_mod, "run_structured_analysis", _fake_decision)
    monkeypatch.setattr(router_mod, "aresolve_ticker", _fake_resolve)


def _mock_all_specialist_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fundamentals_mod, "aget_fundamentals", _async_return(FAKE_FUNDAMENTALS))
    monkeypatch.setattr(technical_mod, "aget_technical_data", _async_return(FAKE_TECHNICAL))
    monkeypatch.setattr(news_mod, "asearch_news", _async_return(FAKE_NEWS_RESULTS))
    _mock_analysis(monkeypatch, fundamentals_mod)
    _mock_analysis(monkeypatch, technical_mod)
    _mock_analysis(monkeypatch, news_mod)


async def test_router_drops_invalid_ticker_and_downgrades_query_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_router_decision(
        monkeypatch, "comparison", ["AAPL", "ZZZINVALID"],
        exists_map={"AAPL": True, "ZZZINVALID": False},
    )
    _mock_all_specialist_tools(monkeypatch)

    state = new_state(user_question="Compare Apple and ZZZINVALID", session_id=str(uuid.uuid4()))
    result = await build_research_graph().ainvoke(state)

    assert result["tickers"] == ["AAPL"]
    assert result["query_type"] == "single"  # downgraded: only one valid ticker survived
    assert any("ZZZINVALID" in note for note in result["notes"])
    assert "AAPL" in result["per_ticker_results"]
    assert result["final_report"]


async def test_router_enforces_max_tickers_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    many_tickers = [f"T{i}" for i in range(settings.max_tickers + 2)]
    _mock_router_decision(monkeypatch, "comparison", many_tickers)
    _mock_all_specialist_tools(monkeypatch)

    state = new_state(user_question="Compare a lot of companies", session_id=str(uuid.uuid4()))
    result = await build_research_graph().ainvoke(state)

    assert len(result["tickers"]) == settings.max_tickers
    assert any("limit reached" in note for note in result["notes"])


async def test_router_no_valid_tickers_short_circuits_without_running_agents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_router_decision(monkeypatch, "single", ["ZZZINVALID"], exists_map={"ZZZINVALID": False})
    calls = {"count": 0}

    async def _tracking_fetch(*args: Any, **kwargs: Any) -> FundamentalsData:  # noqa: ARG001
        calls["count"] += 1
        return FAKE_FUNDAMENTALS

    monkeypatch.setattr(fundamentals_mod, "aget_fundamentals", _tracking_fetch)

    state = new_state(user_question="Analyze ZZZINVALID", session_id=str(uuid.uuid4()))
    result = await build_research_graph().ainvoke(state)

    assert result["tickers"] == []
    assert calls["count"] == 0
    # A plain reply, not a fake "Research Report" card.
    assert result["final_report"] is None
    assert "ZZZINVALID" in result["followup_answer"]


async def test_router_llm_failure_routes_through_off_topic_reply_not_notes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When classification itself fails (any LLMAnalysisError — its message is always
    clean by construction, see _shared.py), the reply should go out through the same
    plain-reply channel as off-topic/discovery — not get buried in `notes` under a
    misleading 'no tickers found' dead end, which is a different failure entirely."""
    async def _raise(prompt: str, schema: type = RouterDecision) -> RouterDecision:  # noqa: ARG001
        raise LLMAnalysisError("The analysis service is temporarily unavailable.")

    monkeypatch.setattr(router_mod, "run_structured_analysis", _raise)

    state = new_state(user_question="Should I buy TCS?", session_id=str(uuid.uuid4()))
    result = await build_research_graph().ainvoke(state)

    assert result["tickers"] == []
    assert result["notes"] == []
    assert result["followup_answer"] == "The analysis service is temporarily unavailable."
    assert result["final_report"] is None


async def test_synthesis_portfolio_all_failed_produces_plain_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    state = new_state(
        tickers=["AAPL", "MSFT"], query_type="portfolio", user_question="my portfolio",
        session_id=str(uuid.uuid4()),
    )
    state["per_ticker_results"] = {
        "AAPL": {"fundamentals": failed_result("no data"), "technical": failed_result("no data"), "news": failed_result("no data")},
        "MSFT": {"fundamentals": failed_result("no data"), "technical": failed_result("no data"), "news": failed_result("no data")},
    }
    result = await synthesis_portfolio_mod.synthesis_portfolio_node(state)

    assert result.get("final_report") is None
    assert "wasn't able to complete this portfolio research" in result["followup_answer"]


async def test_synthesis_comparison_insufficient_usable_produces_plain_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = new_state(
        tickers=["AAPL", "MSFT"], query_type="comparison", user_question="compare",
        session_id=str(uuid.uuid4()),
    )
    state["per_ticker_results"] = {
        "AAPL": {"fundamentals": failed_result("no data"), "technical": failed_result("no data"), "news": failed_result("no data")},
        "MSFT": {"fundamentals": ok_result("ok", []), "technical": ok_result("ok", []), "news": ok_result("ok", [])},
    }
    result = await synthesis_comparison_mod.synthesis_comparison_node(state)

    assert result.get("final_report") is None
    assert "wasn't able to complete this comparison" in result["followup_answer"]


def test_router_decision_schema_allows_omitting_irrelevant_fields_when_off_topic() -> None:
    """Regression guard for a real observed failure: Groq's structured-output
    enforcement rejects a tool call that's missing any field without a default. The
    model naturally omits query_type/tickers when is_stock_related is false (they're
    irrelevant to an off-topic reply) — those two fields must have defaults, or a
    genuinely correct off-topic classification gets rejected as a malformed tool call
    and falls back to the generic 'analysis service unavailable' message instead of the
    off_topic_reply the model actually produced."""
    decision = RouterDecision(is_stock_related=False, off_topic_reply="not a stock question")
    assert decision.query_type == "single"
    assert decision.tickers == []


# --- discovery detection ---------------------------------------------------------------


async def test_router_discovery_request_explains_limitation_without_research(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'Which Indian stocks should I buy?' names a scope (a market) with no company —
    this app doesn't screen a market for candidates, so it should explain that plainly
    instead of either fabricating a shortlist or interrogating the user for preferences."""
    _mock_router_decision(
        monkeypatch, "single", [],
        is_discovery_request=True,
        discovery_reply="I can research specific Indian stocks, but I don't screen the market for candidates.",
    )
    calls = {"count": 0}

    async def _tracking_fetch(*args: Any, **kwargs: Any) -> FundamentalsData:  # noqa: ARG001
        calls["count"] += 1
        return FAKE_FUNDAMENTALS

    monkeypatch.setattr(fundamentals_mod, "aget_fundamentals", _tracking_fetch)

    state = new_state(user_question="Which Indian stocks should I buy?", session_id=str(uuid.uuid4()))
    result = await build_research_graph().ainvoke(state)

    assert result["tickers"] == []
    assert result["followup_answer"] == "I can research specific Indian stocks, but I don't screen the market for candidates."
    assert result["awaiting_clarification"] is False  # terminal reply, not another question
    assert result["final_report"] is None
    assert calls["count"] == 0  # no fabricated candidate was ever researched


async def test_router_scope_free_vague_question_still_asks_clarification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard for the discovery/clarification boundary: 'which stock has
    stronger sentiment?' names no scope at all, so it must stay on the existing
    needs_clarification path (ask which companies) rather than being swept into
    discovery just because no ticker was found."""
    _mock_router_decision(
        monkeypatch, "comparison", [],
        needs_clarification=True, clarifying_question="Which stocks would you like me to compare?",
        is_discovery_request=False,
    )

    state = new_state(
        user_question="Which stock has stronger recent market sentiment?",
        session_id=str(uuid.uuid4()),
    )
    result = await build_research_graph().ainvoke(state)

    assert result["awaiting_clarification"] is True
    assert result["followup_answer"] == "Which stocks would you like me to compare?"


async def test_router_named_ticker_overrides_discovery_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backend safety net: even if the model inconsistently sets is_discovery_request
    while also naming a real ticker, concrete information wins — research proceeds
    rather than discarding a real company because of a fuzzy classification flag."""
    _mock_router_decision(
        monkeypatch, "single", ["AAPL"],
        is_discovery_request=True,  # deliberately inconsistent with a real ticker present
        discovery_reply="should never be shown",
    )
    _mock_all_specialist_tools(monkeypatch)

    state = new_state(user_question="Should I buy Apple?", session_id=str(uuid.uuid4()))
    result = await build_research_graph().ainvoke(state)

    assert result["tickers"] == ["AAPL"]
    assert result["final_report"]
    assert result["followup_answer"] is None


async def test_router_unaddressed_note_appended_when_tickers_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'TCS, Infosys, and a few other good IT stocks' names real companies AND an
    open-ended, undiscoverable tail — research proceeds on the named companies, with a
    transparent note about what's being skipped, rather than silently dropping it."""
    _mock_router_decision(
        monkeypatch, "comparison", ["TCS", "INFY"],
        unaddressed_note="Only TCS and Infosys were analyzed — the app doesn't screen for 'other good IT stocks'.",
    )
    _mock_all_specialist_tools(monkeypatch)

    state = new_state(
        user_question="Compare TCS, Infosys, and a few other good IT stocks",
        session_id=str(uuid.uuid4()),
    )
    result = await build_research_graph().ainvoke(state)

    assert set(result["tickers"]) == {"TCS", "INFY"}
    assert any("other good IT stocks" in note for note in result["notes"])
    assert result["final_report"]


async def test_router_unaddressed_note_covers_unrelated_side_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard for a real observed gap: 'how do I make pasta, and how's TCS
    doing?' has a company AND a completely unrelated tangent — the tangent must be
    acknowledged via unaddressed_note, not silently dropped, when research succeeds."""
    _mock_router_decision(
        monkeypatch, "single", ["TCS"],
        unaddressed_note="I can't help with recipes — here's what I found on TCS.",
    )
    _mock_all_specialist_tools(monkeypatch)

    state = new_state(
        user_question="how do I make pasta, and how's TCS doing?", session_id=str(uuid.uuid4()),
    )
    result = await build_research_graph().ainvoke(state)

    assert result["tickers"] == ["TCS"]
    assert any("recipes" in note for note in result["notes"])
    assert result["final_report"]


async def test_unaddressed_note_surfaces_in_no_tickers_reply_when_ticker_also_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact reported scenario: an unrelated tangent (pasta) mixed with a company
    that ALSO fails to resolve (not found) — the tangent still must not be silently
    dropped just because the ticker lookup also failed."""
    _mock_router_decision(
        monkeypatch, "single", ["FRESHAGRO"], exists_map={"FRESHAGRO": False},
        unaddressed_note="I can't help with recipes.",
    )

    state = new_state(
        user_question="how do I make pasta, and how's Freshara Agro Export doing?",
        session_id=str(uuid.uuid4()),
    )
    result = await build_research_graph().ainvoke(state)

    assert result["tickers"] == []
    assert result["final_report"] is None
    assert "I can't help with recipes." in result["followup_answer"]
    assert "FRESHAGRO" in result["followup_answer"]


async def test_discovery_reply_does_not_block_a_later_specific_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a discovery reply, no clarification is pending (`awaiting_clarification`
    stays False) — so the user naming specific stocks in their next message goes through
    a normal fresh classification and researches them, exactly as if it were a new chat."""
    checkpointer = InMemorySaver()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    _mock_router_decision(
        monkeypatch, "single", [],
        is_discovery_request=True, discovery_reply="I don't screen the market for candidates.",
    )
    graph = build_research_graph(checkpointer=checkpointer)
    first = await graph.ainvoke(
        new_state(user_question="What are the best Indian stocks right now?", session_id=thread_id),
        config=config,
    )
    assert first["tickers"] == []
    assert first["awaiting_clarification"] is False
    assert first["per_ticker_results"] == {}

    _mock_router_decision(monkeypatch, "comparison", ["TCS", "INFY", "HCLTECH"])
    _mock_all_specialist_tools(monkeypatch)

    second = await graph.ainvoke({"user_question": "TCS, Infosys and HCLTech"}, config=config)

    assert set(second["tickers"]) == {"TCS", "INFY", "HCLTECH"}
    assert second["final_report"]


# --- clarification flow --------------------------------------------------------------


async def test_router_asks_for_clarification_when_no_ticker_but_stock_related(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_router_decision(
        monkeypatch, "comparison", [],
        needs_clarification=True, clarifying_question="Which stocks do you want me to compare?",
    )

    state = new_state(
        user_question="Which stock has stronger recent market sentiment?",
        session_id=str(uuid.uuid4()),
    )
    result = await build_research_graph().ainvoke(state)

    assert result["tickers"] == []
    assert result["awaiting_clarification"] is True
    assert result["pending_question"] == "Which stock has stronger recent market sentiment?"
    assert result["pending_intent"] == "comparison"
    assert result["followup_answer"] == "Which stocks do you want me to compare?"
    assert result["final_report"] is None


async def test_router_off_topic_message_gets_polite_reply_without_research(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuinely unrelated message shouldn't produce a 'Research Report' card, and
    shouldn't run any research even if the model mistakenly extracted a ticker."""
    _mock_router_decision(
        monkeypatch, "single", ["GOOGL"],  # e.g. "I work at Google" mis-extracted
        is_stock_related=False, off_topic_reply="I'm focused on stock research.",
    )
    calls = {"count": 0}

    async def _tracking_fetch(*args: Any, **kwargs: Any) -> FundamentalsData:  # noqa: ARG001
        calls["count"] += 1
        return FAKE_FUNDAMENTALS

    monkeypatch.setattr(fundamentals_mod, "aget_fundamentals", _tracking_fetch)

    state = new_state(user_question="I work at Google, is it going to rain today?", session_id=str(uuid.uuid4()))
    result = await build_research_graph().ainvoke(state)

    assert result["tickers"] == []
    assert result["followup_answer"] == "I'm focused on stock research."
    assert result["final_report"] is None
    assert calls["count"] == 0  # no research was triggered by the incidental mention


async def test_clarification_reply_reuses_pending_intent_without_reclassifying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point of storing `pending_intent`: a resolving reply only supplies the
    missing tickers, it never asks the LLM to re-decide the request type. Mocking the
    reply's own `query_type` as something else proves it's ignored."""
    _mock_clarification_decision(monkeypatch, query_type="single", tickers=["AAPL", "MSFT"])
    _mock_all_specialist_tools(monkeypatch)

    state = new_state(
        user_question="TCS and Infosys",  # placeholder text; tickers come from the mock
        session_id=str(uuid.uuid4()),
    )
    state["awaiting_clarification"] = True
    state["pending_question"] = "Which stock has stronger recent market sentiment?"
    state["clarification_question"] = "Which stocks do you want me to compare?"
    state["pending_intent"] = "comparison"

    result = await build_research_graph().ainvoke(state)

    assert result["tickers"] == ["AAPL", "MSFT"]
    assert result["query_type"] == "comparison"  # from pending_intent, NOT the mock's "single"
    assert result["awaiting_clarification"] is False
    assert result["pending_question"] is None
    assert result["pending_intent"] is None
    assert result["final_report"]
    assert "AAPL" in result["per_ticker_results"]
    assert "MSFT" in result["per_ticker_results"]


async def test_clarification_reply_with_no_tickers_falls_through_without_looping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_clarification_decision(monkeypatch, query_type="single", tickers=[])

    state = new_state(user_question="I don't know, whichever is best", session_id=str(uuid.uuid4()))
    state["awaiting_clarification"] = True
    state["pending_question"] = "Which stock has stronger recent market sentiment?"
    state["clarification_question"] = "Which stocks do you want me to compare?"
    state["pending_intent"] = "comparison"

    result = await build_research_graph().ainvoke(state)

    assert result["tickers"] == []
    assert result["awaiting_clarification"] is False
    assert result["pending_intent"] is None
    # A plain reply, not a fake "Research Report" card — nothing was ever going to be
    # researched here, so the note IS the whole answer, not a caveat under a report.
    assert result["final_report"] is None
    assert "try naming one directly" in result["followup_answer"]


async def test_clarification_abandoned_reply_is_reclassified_as_a_new_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The escape hatch: a reply that isn't answering the pending question at all (here,
    a meta question about the app) should never be forced through the ticker extractor."""
    _mock_clarification_decision(
        monkeypatch, query_type="single", tickers=[],
        resolves_pending_clarification=False, is_stock_related=False,
        off_topic_reply="I use sources such as company filings, exchange data, and news.",
    )

    state = new_state(user_question="never mind, what's your data source?", session_id=str(uuid.uuid4()))
    state["awaiting_clarification"] = True
    state["pending_question"] = "Which stock has stronger recent market sentiment?"
    state["clarification_question"] = "Which stocks do you want me to compare?"
    state["pending_intent"] = "comparison"

    result = await build_research_graph().ainvoke(state)

    assert result["tickers"] == []
    assert result["awaiting_clarification"] is False
    assert result["pending_intent"] is None
    assert result["followup_answer"] == "I use sources such as company filings, exchange data, and news."
    assert result["final_report"] is None


async def test_clarification_reply_revealing_discovery_gets_limitation_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact reported bug: 'which stock is more affected by recent negative news?'
    -> asked which stocks -> user replies 'any Indian stock I can purchase' / 'healthcare
    sector'. That reply names a scope, not a company — it must NOT trigger another round
    of preference questions (market cap, dividend, risk tolerance); it should explain the
    discovery limitation once and stop."""
    _mock_clarification_decision(
        monkeypatch, query_type="comparison", tickers=[],
        resolves_pending_clarification=False, is_stock_related=True,
        is_discovery_request=True,
        discovery_reply="I can compare specific Indian stocks, but I don't screen the market for candidates.",
    )

    state = new_state(user_question="any Indian stock I can purchase", session_id=str(uuid.uuid4()))
    state["awaiting_clarification"] = True
    state["pending_question"] = "Which stock is more affected by recent negative news?"
    state["clarification_question"] = "Which stocks would you like me to compare?"
    state["pending_intent"] = "comparison"

    result = await build_research_graph().ainvoke(state)

    assert result["tickers"] == []
    assert result["awaiting_clarification"] is False  # does NOT loop into another question
    assert result["pending_intent"] is None
    assert result["followup_answer"] == (
        "I can compare specific Indian stocks, but I don't screen the market for candidates."
    )
    assert result["final_report"] is None


async def test_clarification_abandoned_reply_with_named_stock_proceeds_to_research(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'Never mind, analyze TCS' — abandons the pending clarification for a fully
    specific request, which should research TCS directly, not dead-end or re-ask."""
    _mock_clarification_decision(
        monkeypatch, query_type="single", tickers=["TCS"],
        resolves_pending_clarification=False, is_stock_related=True,
    )
    _mock_all_specialist_tools(monkeypatch)

    state = new_state(user_question="never mind, analyze TCS", session_id=str(uuid.uuid4()))
    state["awaiting_clarification"] = True
    state["pending_question"] = "Which stock is more affected by recent negative news?"
    state["clarification_question"] = "Which stocks would you like me to compare?"
    state["pending_intent"] = "comparison"

    result = await build_research_graph().ainvoke(state)

    assert result["tickers"] == ["TCS"]
    assert result["awaiting_clarification"] is False
    assert result["final_report"]


async def test_multi_ticker_fan_out_merges_every_ticker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression guard for the map-reduce join barrier at N>1: `collect_results` must
    wait for all 3*N dynamically-spawned specialist invocations, not just the first."""
    _mock_all_specialist_tools(monkeypatch)

    state = new_state(
        tickers=["AAPL", "MSFT"], query_type="comparison",
        user_question="Compare AAPL and MSFT", session_id=str(uuid.uuid4()),
    )
    result = await build_research_graph().ainvoke(state)

    assert set(result["per_ticker_results"]) == {"AAPL", "MSFT"}
    for ticker in ("AAPL", "MSFT"):
        assert set(result["per_ticker_results"][ticker]) == {"fundamentals", "technical", "news"}
    assert "Comparison" in result["final_report"]


# --- news_node: ambiguous ticker/company-name disambiguation ------------------------


async def test_news_node_query_includes_company_name_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression guard for a real observed bug: a bare ticker/word (e.g. 'TITAN') is
    ambiguous across unrelated companies (Titan Company Limited vs. Titan Mining Corp
    vs. Titan International) — the company's actual name narrows the search query
    itself, on top of the prompt-level check below."""
    captured: dict[str, str] = {}

    async def _fake_search(query: str, max_results: int = 6) -> list[SearchResult]:  # noqa: ARG001
        captured["query"] = query
        return FAKE_NEWS_RESULTS

    monkeypatch.setattr(news_mod, "asearch_news", _fake_search)
    monkeypatch.setattr(news_mod, "aget_company_name", _async_return("Titan Company Limited"))
    _mock_analysis(monkeypatch, news_mod)

    await news_mod.news_node({"tickers": ["TITAN.NS"], "session_id": str(uuid.uuid4())})

    assert "Titan Company Limited" in captured["query"]
    assert "TITAN" in captured["query"]


async def test_news_node_query_falls_back_to_bare_ticker_when_name_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    async def _fake_search(query: str, max_results: int = 6) -> list[SearchResult]:  # noqa: ARG001
        captured["query"] = query
        return FAKE_NEWS_RESULTS

    monkeypatch.setattr(news_mod, "asearch_news", _fake_search)
    monkeypatch.setattr(news_mod, "aget_company_name", _async_return(None))
    _mock_analysis(monkeypatch, news_mod)

    await news_mod.news_node({"tickers": ["ZZZINVALID"], "session_id": str(uuid.uuid4())})

    assert captured["query"] == "ZZZINVALID stock news"


async def test_news_node_prompt_warns_the_llm_about_ambiguous_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    """The narrower search query helps, but search engines still aren't perfect (see the
    TITAN case above) — the LLM synthesizing findings must itself be told to verify each
    article's identity rather than trust whatever the search returned."""
    captured: dict[str, str] = {}

    async def _fake_analysis(prompt: str, schema: type = NodeAnalysis) -> NewsAnalysis:  # noqa: ARG001
        captured["prompt"] = prompt
        return NewsAnalysis(summary="s", findings=[], overall_sentiment="neutral")

    monkeypatch.setattr(news_mod, "asearch_news", _async_return(FAKE_NEWS_RESULTS))
    monkeypatch.setattr(news_mod, "aget_company_name", _async_return("Titan Company Limited"))
    monkeypatch.setattr(news_mod, "run_structured_analysis", _fake_analysis)

    await news_mod.news_node({"tickers": ["TITAN.NS"], "session_id": str(uuid.uuid4())})

    assert "Titan Company Limited" in captured["prompt"]
    assert "different company" in captured["prompt"].lower()


# --- fundamentals/technical: verifiable link on an otherwise link-less source ---------


async def test_fundamentals_and_technical_findings_link_to_the_yahoo_quote_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """fundamentals/technical data comes from Finnhub/Twelve Data's token-authed APIs,
    not a web page — there's nothing to literally link back to. The public Yahoo Finance
    quote page shows the same figures and is a real, clickable, verifiable link, which
    beats leaving these two source types as the only ones with no link at all."""
    monkeypatch.setattr(fundamentals_mod, "aget_fundamentals", _async_return(FAKE_FUNDAMENTALS))
    monkeypatch.setattr(technical_mod, "aget_technical_data", _async_return(FAKE_TECHNICAL))
    monkeypatch.setattr(news_mod, "asearch_news", _async_return(FAKE_NEWS_RESULTS))
    _mock_analysis(monkeypatch, fundamentals_mod)
    _mock_analysis(monkeypatch, technical_mod)
    _mock_analysis(monkeypatch, news_mod)

    result = await build_research_graph().ainvoke(_new_state())

    ticker_results = result["per_ticker_results"]["AAPL"]
    assert ticker_results["fundamentals"]["findings"][0]["source"]["url"] == "https://finance.yahoo.com/quote/AAPL"
    assert ticker_results["technical"]["findings"][0]["source"]["url"] == "https://finance.yahoo.com/quote/AAPL"
