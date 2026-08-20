from __future__ import annotations

import uuid
from typing import Any

import groq
import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver

import app.graph.nodes.answer_from_context as answer_mod
import app.graph.nodes.followup_clarification_response_node as followup_clarification_mod
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
from app.llm.errors import LLMAnalysisError
from app.tools.yahoo_finance import FundamentalsData, ResolvedTicker, TechnicalData
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
    # news_node also looks up the company's display name (as a search-query
    # disambiguation aid) before searching — stub it so tests never hit the network.
    monkeypatch.setattr(news_mod, "aget_company_name", _async_return(None))
    _mock_analysis(monkeypatch, fundamentals_mod)
    _mock_analysis(monkeypatch, technical_mod)
    _mock_analysis(monkeypatch, news_mod)


def _mock_followup_decision(monkeypatch: pytest.MonkeyPatch, **fields: Any) -> None:
    decision = FollowUpDecision(**fields)

    async def _fake(prompt: str, schema: type = FollowUpDecision) -> FollowUpDecision:  # noqa: ARG001
        return decision

    monkeypatch.setattr(followup_mod, "run_structured_analysis", _fake)


def _mock_followup_clarification_decision(monkeypatch: pytest.MonkeyPatch, **fields: Any) -> None:
    decision = FollowUpDecision(**fields)

    async def _fake(prompt: str, schema: type = FollowUpDecision) -> FollowUpDecision:  # noqa: ARG001
        return decision

    monkeypatch.setattr(followup_clarification_mod, "run_structured_analysis", _fake)


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


async def _initial_run_two_tickers(
    monkeypatch: pytest.MonkeyPatch, checkpointer: InMemorySaver, thread_id: str
) -> tuple[dict, Any, dict]:
    _mock_tools(monkeypatch)
    graph = build_research_graph(checkpointer=checkpointer)
    state = new_state(
        tickers=["AAPL", "MSFT"], query_type="comparison",
        user_question="Compare AAPL and MSFT", session_id=thread_id,
    )
    config = {"configurable": {"thread_id": thread_id}}
    result = await graph.ainvoke(state, config=config)
    return result, graph, config


async def test_answer_from_context_failure_message_is_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression guard for a real observed leak: this node calls the LLM directly
    (not through run_structured_analysis), so it needs its own guard against putting a
    raw provider error — e.g. a rate-limit body — straight into the user-facing answer."""
    class _RaisingLLM:
        async def ainvoke(self, prompt: str) -> None:  # noqa: ARG002
            raise RuntimeError("Rate limit reached ... {'error': {'code': 'rate_limit_exceeded'}}")

    monkeypatch.setattr(answer_mod, "get_chat_model", lambda temperature=0.2: _RaisingLLM())  # noqa: ARG005

    state = new_state(
        tickers=["AAPL"], query_type="single", user_question="Any updates?", session_id=str(uuid.uuid4()),
    )
    result = await answer_mod.answer_from_context_node(state)

    assert "couldn't generate an answer" in result["followup_answer"]
    assert "rate_limit_exceeded" not in result["followup_answer"]
    assert "{'error'" not in result["followup_answer"]


async def test_answer_from_context_rate_limit_gets_specific_friendly_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuinely exhausted quota gets a distinct, honest message — not the generic
    'couldn't generate an answer' fallback used for other failures."""
    def _rate_limit_error() -> groq.RateLimitError:
        request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        response = httpx.Response(429, request=request)
        body = {"error": {"code": "rate_limit_exceeded", "message": "Rate limit reached ... tokens per day"}}
        return groq.RateLimitError("rate limited", response=response, body=body)

    class _RateLimitedLLM:
        async def ainvoke(self, prompt: str) -> None:  # noqa: ARG002
            raise _rate_limit_error()

    monkeypatch.setattr(answer_mod, "get_chat_model", lambda temperature=0.2: _RateLimitedLLM())  # noqa: ARG005

    state = new_state(
        tickers=["AAPL"], query_type="single", user_question="Any updates?", session_id=str(uuid.uuid4()),
    )
    result = await answer_mod.answer_from_context_node(state)

    assert "temporarily rate-limited" in result["followup_answer"]
    assert "rate_limit_exceeded" not in result["followup_answer"]


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


async def test_followup_answer_path_escalates_to_refresh_when_data_is_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard for the context-sufficiency gap: whether a follow-up needs fresh
    data is otherwise decided entirely by one soft LLM judgment call
    (`FollowUpDecision.path`), which can misjudge "already covered" for something that's
    quietly gone stale (e.g. "what's the RSI *right now*"). A ticker whose stored data has
    aged past the freshness window must be refreshed before being answered from, not
    silently served as current, regardless of what the classifier decided."""
    checkpointer = InMemorySaver()
    thread_id = str(uuid.uuid4())
    _, graph, config = await _initial_run(monkeypatch, checkpointer, thread_id)

    old_timestamp = "2000-01-01T00:00:00+00:00"
    await graph.aupdate_state(
        config,
        {"per_ticker_fetched_at": {"AAPL": {"fundamentals": old_timestamp, "technical": old_timestamp, "news": old_timestamp}}},
    )

    call_counts: dict[str, int] = {}
    _mock_tools(monkeypatch, call_counts)
    _mock_followup_decision(monkeypatch, path="answer")

    result = await graph.ainvoke({"user_question": "What's the current RSI?"}, config=config)

    assert result["followup_path"] == "refresh"  # escalated, not "answer"
    assert call_counts == {"fundamentals": 1, "technical": 1, "news": 1}
    assert any("freshness" in note.lower() for note in result["notes"])


async def test_followup_answer_path_stays_answer_when_only_partially_stale_but_fresh_enough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The freshness guard checks actual elapsed time, not just presence of a timestamp —
    data fetched moments ago (as in a normal same-session follow-up) must not be treated
    as stale just because some time technically passed."""
    checkpointer = InMemorySaver()
    thread_id = str(uuid.uuid4())
    _, graph, config = await _initial_run(monkeypatch, checkpointer, thread_id)

    call_counts: dict[str, int] = {}
    _mock_tools(monkeypatch, call_counts)
    _mock_followup_decision(monkeypatch, path="answer")

    result = await graph.ainvoke({"user_question": "What's the P/E ratio?"}, config=config)

    assert result["followup_path"] == "answer"
    assert call_counts == {}  # no escalation — data was just fetched


async def test_followup_unrelated_path_gets_polite_reply_without_running_agents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpointer = InMemorySaver()
    thread_id = str(uuid.uuid4())
    initial, graph, config = await _initial_run(monkeypatch, checkpointer, thread_id)
    original_report = initial["final_report"]

    call_counts: dict[str, int] = {}
    _mock_tools(monkeypatch, call_counts)
    _mock_followup_decision(
        monkeypatch, path="unrelated", off_topic_reply="I can help with the stock research in this chat."
    )

    result = await graph.ainvoke({"user_question": "What's the weather today?"}, config=config)

    assert result["followup_path"] == "unrelated"
    assert result["followup_answer"] == "I can help with the stock research in this chat."
    assert result["final_report"] == original_report  # untouched
    assert call_counts == {}  # no specialist tool was invoked


async def test_followup_discovery_path_gets_polite_reply_without_fabricating_tickers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'What other good IT stocks should I look at?' in an ongoing session must not be
    forced into add_ticker (which would require inventing a company) or answer (which
    risks improvising a suggestion) — it should explain the limitation, same as a fresh
    discovery request would."""
    checkpointer = InMemorySaver()
    thread_id = str(uuid.uuid4())
    initial, graph, config = await _initial_run(monkeypatch, checkpointer, thread_id)
    original_report = initial["final_report"]

    call_counts: dict[str, int] = {}
    _mock_tools(monkeypatch, call_counts)
    _mock_followup_decision(
        monkeypatch, path="discovery",
        discovery_reply="I analyze stocks you provide — I don't screen for similar candidates.",
    )

    result = await graph.ainvoke({"user_question": "What other good IT stocks should I look at?"}, config=config)

    assert result["followup_path"] == "discovery"
    assert result["followup_answer"] == "I analyze stocks you provide — I don't screen for similar candidates."
    assert result["final_report"] == original_report  # untouched
    assert result["tickers"] == ["AAPL"]  # no fabricated ticker was added to the session
    assert call_counts == {}  # no specialist tool was invoked


async def test_followup_refresh_matches_bare_ticker_against_resolved_suffixed_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session's stored ticker may carry a resolved exchange suffix (e.g. 'TCS.NS' —
    see `aresolve_ticker`), but a user talking about it in a follow-up says 'TCS', not
    'TCS.NS'. Refresh targeting must still match."""
    checkpointer = InMemorySaver()
    thread_id = str(uuid.uuid4())
    _mock_tools(monkeypatch)
    graph = build_research_graph(checkpointer=checkpointer)
    state = new_state(
        tickers=["TCS.NS"], query_type="single", user_question="Analyze TCS", session_id=thread_id,
    )
    config = {"configurable": {"thread_id": thread_id}}
    await graph.ainvoke(state, config=config)

    call_counts: dict[str, int] = {}
    _mock_tools(monkeypatch, call_counts)
    _mock_followup_decision(monkeypatch, path="refresh", refresh_tickers=["TCS"], refresh_agents=["news"])

    result = await graph.ainvoke({"user_question": "Any fresh news on TCS?"}, config=config)

    assert result["followup_path"] == "refresh"
    assert call_counts == {"news": 1}
    assert result["per_ticker_results"]["TCS.NS"]["fundamentals"]["status"] == "ok"  # preserved


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

    async def _fake_resolve(ticker: str) -> ResolvedTicker:
        return ResolvedTicker(ticker)

    monkeypatch.setattr(followup_mod, "aresolve_ticker", _fake_resolve)
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


# --- follow-up "cannot tell which ticker" -> notes, not a silent no-op -------------------


async def test_followup_refresh_naming_a_ticker_outside_the_session_explains_why(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard for a real gap: `_plan_refresh` used to fall back to 'answer'
    with zero notes when the named ticker didn't match anything in the session — the
    user's refresh request would silently do nothing with no explanation."""
    checkpointer = InMemorySaver()
    thread_id = str(uuid.uuid4())
    _, graph, config = await _initial_run(monkeypatch, checkpointer, thread_id)

    call_counts: dict[str, int] = {}
    _mock_tools(monkeypatch, call_counts)
    _mock_followup_decision(monkeypatch, path="refresh", refresh_tickers=["MSFT"], refresh_agents=["news"])

    result = await graph.ainvoke({"user_question": "Any fresh news on MSFT?"}, config=config)

    assert result["followup_path"] == "answer"
    assert call_counts == {}
    assert any("MSFT" in note and "AAPL" in note for note in result["notes"])


async def test_followup_add_ticker_already_in_session_explains_why(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same gap as refresh, on the add_ticker side: asking to add a ticker that's already
    in the session used to silently do nothing (no room-exceeded note, since the ticker
    was filtered out before the cap check even ran)."""
    checkpointer = InMemorySaver()
    thread_id = str(uuid.uuid4())
    _, graph, config = await _initial_run(monkeypatch, checkpointer, thread_id)

    call_counts: dict[str, int] = {}
    _mock_tools(monkeypatch, call_counts)
    _mock_followup_decision(monkeypatch, path="add_ticker", new_tickers=["AAPL"])

    result = await graph.ainvoke({"user_question": "Add AAPL too"}, config=config)

    assert result["followup_path"] == "answer"
    assert result["tickers"] == ["AAPL"]  # unchanged
    assert call_counts == {}
    assert any("AAPL" in note and "already" in note for note in result["notes"])


# --- follow-up clarification: a follow-up too vague to act on -----------------------


async def test_followup_needs_clarification_asks_without_touching_the_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The headline gap this closes: before this, an ambiguous follow-up in a
    multi-ticker session had no way to ask back and was forced into 'answer', which
    could confidently answer about the wrong ticker."""
    checkpointer = InMemorySaver()
    thread_id = str(uuid.uuid4())
    initial, graph, config = await _initial_run_two_tickers(monkeypatch, checkpointer, thread_id)

    call_counts: dict[str, int] = {}
    _mock_tools(monkeypatch, call_counts)
    _mock_followup_decision(
        monkeypatch, path="needs_clarification", clarifying_question="Did you mean AAPL or MSFT?"
    )

    result = await graph.ainvoke({"user_question": "How's the other one doing?"}, config=config)

    assert result["followup_path"] == "needs_clarification"
    assert result["awaiting_clarification"] is True
    assert result["clarification_origin"] == "followup"
    assert result["clarification_question"] == "Did you mean AAPL or MSFT?"
    assert result["pending_question"] == "How's the other one doing?"
    assert result["followup_answer"] == "Did you mean AAPL or MSFT?"  # ask_clarification's reply
    assert set(result["tickers"]) == {"AAPL", "MSFT"}  # session untouched
    assert call_counts == {}  # no specialist invoked
    assert initial["final_report"] == result["final_report"]  # untouched


async def test_followup_clarification_resolving_to_refresh_only_targets_the_named_ticker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpointer = InMemorySaver()
    thread_id = str(uuid.uuid4())
    await _initial_run_two_tickers(monkeypatch, checkpointer, thread_id)
    graph = build_research_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    _mock_followup_decision(
        monkeypatch, path="needs_clarification", clarifying_question="Did you mean AAPL or MSFT?"
    )
    await graph.ainvoke({"user_question": "How's the other one doing?"}, config=config)

    call_counts: dict[str, int] = {}
    _mock_tools(monkeypatch, call_counts)
    _mock_followup_clarification_decision(
        monkeypatch, path="refresh", resolves_pending_clarification=True,
        refresh_tickers=["AAPL"], refresh_agents=["news"],
    )

    result = await graph.ainvoke({"user_question": "The Apple one"}, config=config)

    assert result["followup_path"] == "refresh"
    assert result["awaiting_clarification"] is False
    assert result["clarification_origin"] is None
    assert call_counts == {"news": 1}  # only the resolved ticker's requested agent ran
    assert set(result["tickers"]) == {"AAPL", "MSFT"}  # merged, not replaced


async def test_followup_clarification_resolving_to_add_ticker_merges_into_the_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The specific risk a naive fix would have: resolving a follow-up clarification by
    reusing `apply_router_decision` (fresh-session shape) would REPLACE `state["tickers"]`
    outright, wiping out the session's existing research. This must merge instead."""
    checkpointer = InMemorySaver()
    thread_id = str(uuid.uuid4())
    await _initial_run_two_tickers(monkeypatch, checkpointer, thread_id)
    graph = build_research_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    _mock_followup_decision(
        monkeypatch, path="needs_clarification", clarifying_question="Which company did you mean?"
    )
    await graph.ainvoke({"user_question": "What about the other tech company?"}, config=config)

    call_counts: dict[str, int] = {}
    _mock_tools(monkeypatch, call_counts)

    async def _fake_resolve(ticker: str) -> ResolvedTicker:
        return ResolvedTicker(ticker)

    monkeypatch.setattr(followup_mod, "aresolve_ticker", _fake_resolve)
    _mock_followup_clarification_decision(
        monkeypatch, path="add_ticker", resolves_pending_clarification=True, new_tickers=["GOOGL"],
    )

    result = await graph.ainvoke({"user_question": "Google"}, config=config)

    assert result["followup_path"] == "add_ticker"
    assert set(result["tickers"]) == {"AAPL", "MSFT", "GOOGL"}  # merged, AAPL/MSFT preserved
    assert call_counts == {"fundamentals": 1, "technical": 1, "news": 1}  # only for the new ticker


async def test_followup_clarification_abandoned_for_something_unrelated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpointer = InMemorySaver()
    thread_id = str(uuid.uuid4())
    await _initial_run_two_tickers(monkeypatch, checkpointer, thread_id)
    graph = build_research_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    _mock_followup_decision(
        monkeypatch, path="needs_clarification", clarifying_question="Did you mean AAPL or MSFT?"
    )
    await graph.ainvoke({"user_question": "How's the other one doing?"}, config=config)

    call_counts: dict[str, int] = {}
    _mock_tools(monkeypatch, call_counts)
    _mock_followup_clarification_decision(
        monkeypatch, path="unrelated", resolves_pending_clarification=False,
        off_topic_reply="I can only help with the stocks in this session.",
    )

    result = await graph.ainvoke({"user_question": "What's the weather today?"}, config=config)

    assert result["followup_path"] == "unrelated"
    assert result["followup_answer"] == "I can only help with the stocks in this session."
    assert result["awaiting_clarification"] is False
    assert call_counts == {}
    assert set(result["tickers"]) == {"AAPL", "MSFT"}  # untouched


async def test_followup_clarification_still_ambiguous_asks_again_with_a_fresh_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Abandoning the first clarification for a second, equally vague question must not
    loop back to the SAME question — it asks a fresh one, exactly like the router-side
    clarification_response_node does for a fresh session."""
    checkpointer = InMemorySaver()
    thread_id = str(uuid.uuid4())
    await _initial_run_two_tickers(monkeypatch, checkpointer, thread_id)
    graph = build_research_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    _mock_followup_decision(
        monkeypatch, path="needs_clarification", clarifying_question="Did you mean AAPL or MSFT?"
    )
    await graph.ainvoke({"user_question": "How's the other one doing?"}, config=config)

    call_counts: dict[str, int] = {}
    _mock_tools(monkeypatch, call_counts)
    _mock_followup_clarification_decision(
        monkeypatch, path="needs_clarification", resolves_pending_clarification=False,
        clarifying_question="Sorry, still not sure which — AAPL or MSFT?",
    )

    result = await graph.ainvoke({"user_question": "the one I mentioned earlier"}, config=config)

    assert result["followup_path"] == "needs_clarification"
    assert result["awaiting_clarification"] is True
    assert result["clarification_origin"] == "followup"
    assert result["clarification_question"] == "Sorry, still not sure which — AAPL or MSFT?"
    assert result["pending_question"] == "the one I mentioned earlier"
    assert call_counts == {}


async def test_followup_clarification_llm_failure_gives_clean_message_and_clears_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpointer = InMemorySaver()
    thread_id = str(uuid.uuid4())
    await _initial_run_two_tickers(monkeypatch, checkpointer, thread_id)
    graph = build_research_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    _mock_followup_decision(
        monkeypatch, path="needs_clarification", clarifying_question="Did you mean AAPL or MSFT?"
    )
    await graph.ainvoke({"user_question": "How's the other one doing?"}, config=config)

    async def _raise(prompt: str, schema: type = FollowUpDecision) -> FollowUpDecision:  # noqa: ARG001
        raise LLMAnalysisError("The analysis service is temporarily unavailable.")

    monkeypatch.setattr(followup_clarification_mod, "run_structured_analysis", _raise)

    result = await graph.ainvoke({"user_question": "AAPL"}, config=config)

    assert result["awaiting_clarification"] is False
    assert result["clarification_origin"] is None
    assert result["clarification_question"] is None
    assert result["pending_question"] is None
    assert any("temporarily unavailable" in note for note in result["notes"])
