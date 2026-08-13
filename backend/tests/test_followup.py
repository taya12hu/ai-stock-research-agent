from __future__ import annotations

import uuid
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

import app.graph.nodes.answer_from_context as answer_mod
import app.graph.nodes.followup_router_node as followup_mod
import app.graph.nodes.fundamentals_node as fundamentals_mod
import app.graph.nodes.news_node as news_mod
import app.graph.nodes.synthesis_comparison as synthesis_comparison_mod
import app.graph.nodes.synthesis_portfolio as synthesis_portfolio_mod
import app.graph.nodes.synthesis_single as synthesis_mod
import app.graph.nodes.technical_node as technical_mod
from app.config import settings
from app.graph.build_graph import build_research_graph
from app.graph.nodes._shared import LLMFinding, NodeAnalysis
from app.graph.nodes.followup_router_node import FollowUpDecision
from app.graph.nodes.news_node import NewsAnalysis, NewsLLMFinding
from app.graph.state import new_state
from app.tools.yahoo_finance import FundamentalsData, TechnicalData
from app.tools.web_search import SearchResult

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
    def __init__(self, content: str = "answer text") -> None:
        self._content = content
        self.call_count = 0

    async def ainvoke(self, prompt: str) -> FakeAIMessage:  # noqa: ARG002
        self.call_count += 1
        return FakeAIMessage(self._content)


def _async_return(value: Any):
    async def _fn(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        return value

    return _fn


def _counting(value: Any, counter: dict[str, int], key: str):
    async def _fn(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        counter[key] = counter.get(key, 0) + 1
        return value

    return _fn


def _mock_analysis(monkeypatch: pytest.MonkeyPatch, module: Any, *, claims: list[tuple[str, str]] | None = None) -> None:
    claims = claims or [("Claim one", "evidence one")]

    async def _fake(prompt: str, schema: type = NodeAnalysis) -> NodeAnalysis:  # noqa: ARG001
        if schema is NewsAnalysis:
            return NewsAnalysis(
                summary="news summary",
                findings=[NewsLLMFinding(claim=c, evidence=e, article_index=1) for c, e in claims],
                overall_sentiment="neutral",
            )
        return NodeAnalysis(summary="summary", findings=[LLMFinding(claim=c, evidence=e) for c, e in claims])

    monkeypatch.setattr(module, "run_structured_analysis", _fake)


@pytest.fixture(autouse=True)
def _mock_all_llm_calls(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    shared_llm = FakeLLM()
    fake = lambda temperature=0.3: shared_llm  # noqa: ARG005, E731
    monkeypatch.setattr(synthesis_mod, "get_chat_model", fake)
    monkeypatch.setattr(synthesis_portfolio_mod, "get_chat_model", fake)
    monkeypatch.setattr(synthesis_comparison_mod, "get_chat_model", fake)
    monkeypatch.setattr(answer_mod, "get_chat_model", fake)
    return shared_llm


def _mock_tools(monkeypatch: pytest.MonkeyPatch, call_counts: dict[str, int] | None = None) -> None:
    if call_counts is None:
        monkeypatch.setattr(fundamentals_mod, "aget_fundamentals", _async_return(FAKE_FUNDAMENTALS))
        monkeypatch.setattr(technical_mod, "aget_technical_data", _async_return(FAKE_TECHNICAL))
        monkeypatch.setattr(news_mod, "asearch_news", _async_return(FAKE_NEWS_RESULTS))
    else:
        monkeypatch.setattr(fundamentals_mod, "aget_fundamentals", _counting(FAKE_FUNDAMENTALS, call_counts, "fundamentals"))
        monkeypatch.setattr(technical_mod, "aget_technical_data", _counting(FAKE_TECHNICAL, call_counts, "technical"))
        monkeypatch.setattr(news_mod, "asearch_news", _counting(FAKE_NEWS_RESULTS, call_counts, "news"))
    _mock_analysis(monkeypatch, fundamentals_mod)
    _mock_analysis(monkeypatch, technical_mod)
    _mock_analysis(monkeypatch, news_mod)


def _mock_followup_decision(monkeypatch: pytest.MonkeyPatch, **fields: Any) -> None:
    decision = FollowUpDecision(**fields)

    async def _fake(prompt: str, schema: type = FollowUpDecision) -> FollowUpDecision:  # noqa: ARG001
        return decision

    monkeypatch.setattr(followup_mod, "run_structured_analysis", _fake)


async def _initial_run(
    monkeypatch: pytest.MonkeyPatch, checkpointer: InMemorySaver, thread_id: str
) -> tuple[dict, Any, dict]:
    _mock_tools(monkeypatch)
    graph = build_research_graph(checkpointer=checkpointer)
    state = new_state(
        tickers=["AAPL"], query_type="single", user_question="Analyze AAPL", session_id=thread_id,
    )
    config = {"configurable": {"thread_id": thread_id}}
    result = await graph.ainvoke(state, config=config)
    return result, graph, config


async def test_followup_answer_path_makes_no_new_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    checkpointer = InMemorySaver()
    thread_id = str(uuid.uuid4())
    initial, graph, config = await _initial_run(monkeypatch, checkpointer, thread_id)
    original_report = initial["final_report"]

    call_counts: dict[str, int] = {}
    _mock_tools(monkeypatch, call_counts)
    _mock_followup_decision(monkeypatch, path="answer")

    result = await graph.ainvoke({"user_question": "What's the P/E ratio?"}, config=config)

    assert result["followup_path"] == "answer"
    assert result["followup_answer"] == "answer text"
    assert result["final_report"] == original_report  # untouched
    assert call_counts == {}  # no specialist tool was re-invoked
    assert [m["role"] for m in result["conversation_history"][-2:]] == ["user", "assistant"]


async def test_followup_refresh_path_only_calls_targeted_agent(
    monkeypatch: pytest.MonkeyPatch, _mock_all_llm_calls: FakeLLM,
) -> None:
    checkpointer = InMemorySaver()
    thread_id = str(uuid.uuid4())
    initial, graph, config = await _initial_run(monkeypatch, checkpointer, thread_id)
    synthesis_calls_before = _mock_all_llm_calls.call_count

    call_counts: dict[str, int] = {}
    _mock_tools(monkeypatch, call_counts)
    _mock_followup_decision(
        monkeypatch, path="refresh", refresh_tickers=["AAPL"], refresh_agents=["news"]
    )

    result = await graph.ainvoke({"user_question": "Any fresh news today?"}, config=config)

    assert result["followup_path"] == "refresh"
    assert call_counts == {"news": 1}  # fundamentals/technical NOT re-run
    assert result["per_ticker_results"]["AAPL"]["fundamentals"]["status"] == "ok"  # preserved from turn 1
    # synthesis_single re-ran exactly once more (news's own LLM analysis is mocked
    # separately via _mock_analysis/run_structured_analysis, not through get_chat_model)
    assert _mock_all_llm_calls.call_count == synthesis_calls_before + 1


async def test_followup_add_ticker_flips_query_type_and_runs_all_agents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpointer = InMemorySaver()
    thread_id = str(uuid.uuid4())
    initial, graph, config = await _initial_run(monkeypatch, checkpointer, thread_id)
    assert initial["query_type"] == "single"

    call_counts: dict[str, int] = {}
    _mock_tools(monkeypatch, call_counts)

    async def _fake_exists(ticker: str) -> bool:  # noqa: ARG001
        return True

    monkeypatch.setattr(followup_mod, "aticker_exists", _fake_exists)
    _mock_followup_decision(monkeypatch, path="add_ticker", new_tickers=["MSFT"])

    result = await graph.ainvoke({"user_question": "Also add Microsoft"}, config=config)

    assert result["followup_path"] == "add_ticker"
    assert set(result["tickers"]) == {"AAPL", "MSFT"}
    assert result["query_type"] == "comparison"  # flipped from single
    assert call_counts == {"fundamentals": 1, "technical": 1, "news": 1}  # all 3 for the new ticker
    assert set(result["per_ticker_results"]) == {"AAPL", "MSFT"}


async def test_followup_add_ticker_respects_max_tickers_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    checkpointer = InMemorySaver()
    thread_id = str(uuid.uuid4())
    _mock_tools(monkeypatch)
    graph = build_research_graph(checkpointer=checkpointer)
    existing = [f"T{i}" for i in range(settings.max_tickers)]
    state = new_state(
        tickers=existing, query_type="comparison", user_question="Compare a lot", session_id=thread_id,
    )
    config = {"configurable": {"thread_id": thread_id}}
    initial = await graph.ainvoke(state, config=config)
    assert len(initial["tickers"]) == settings.max_tickers

    call_counts: dict[str, int] = {}
    _mock_tools(monkeypatch, call_counts)
    _mock_followup_decision(monkeypatch, path="add_ticker", new_tickers=["EXTRA"])

    result = await graph.ainvoke({"user_question": "Add one more"}, config=config)

    assert result["followup_path"] == "answer"  # fell back: no room to add
    assert "EXTRA" not in result["tickers"]
    assert call_counts == {}
    assert any("limit" in note.lower() for note in result["notes"])
