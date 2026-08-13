from __future__ import annotations

import re
import uuid
from typing import Any

import pytest

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
from app.graph.state import new_state
from app.llm.errors import LLMAnalysisError
from app.tools.errors import WebSearchError, YahooFinanceError
from app.tools.web_search import SearchResult
from app.tools.yahoo_finance import FundamentalsData, TechnicalData

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
    assert result["final_report"]
    assert "Unable to complete research" in result["final_report"]
    assert calls["count"] == 0  # total failure is handled deterministically, no LLM call needed


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
) -> None:
    exists_map = exists_map if exists_map is not None else dict.fromkeys(tickers, True)

    async def _fake_decision(prompt: str, schema: type = RouterDecision) -> RouterDecision:  # noqa: ARG001
        return RouterDecision(
            query_type=query_type,  # type: ignore[arg-type]
            tickers=[TickerCandidate(ticker=t) for t in tickers],
        )

    async def _fake_exists(ticker: str) -> bool:
        return exists_map.get(ticker, True)

    monkeypatch.setattr(router_mod, "run_structured_analysis", _fake_decision)
    monkeypatch.setattr(router_mod, "aticker_exists", _fake_exists)


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
    assert "Unable to identify any valid stock tickers" in result["final_report"]


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
