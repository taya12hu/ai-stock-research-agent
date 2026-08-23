"""End-to-end graph behaviour on the turn-based architecture.

Routing and classification decisions are tested at their own level (`test_plan_turn.py`
for the ladders, `test_planning.py` for the pipeline around them), so what's exercised here
is what only shows up once the graph actually runs: parallel-branch merging, partial
failure, fan-out scoping, and the single-exit guarantee.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

import app.graph.nodes.answer_from_context as answer_mod
import app.graph.nodes.fundamentals_node as fundamentals_mod
import app.graph.nodes.news_node as news_mod
import app.graph.nodes.plan_node as plan_mod
import app.graph.nodes.render as render_mod
import app.graph.nodes.technical_node as technical_mod
import app.graph.resolve_scope as resolve_mod
from app.graph.build_graph import build_research_graph
from app.graph.intent import CompanyRef, TurnIntent
from app.graph.nodes._shared import LLMFinding, NodeAnalysis
from app.graph.nodes.news_node import NewsAnalysis, NewsLLMFinding
from app.graph.session import AGENT_NAMES, new_session_state
from app.llm.errors import LLMAnalysisError
from app.tools.errors import WebSearchError, MarketDataError
from app.tools.web_search import SearchResult
from app.tools.market_data import FundamentalsData, ResolvedTicker, TechnicalData

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
def _mock_llms(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = lambda temperature=0.3: FakeLLM()  # noqa: ARG005, E731
    monkeypatch.setattr(render_mod, "get_chat_model", fake)
    monkeypatch.setattr(answer_mod, "get_chat_model", fake)


@pytest.fixture(autouse=True)
def _mock_company_name_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    # news_node looks up the company's display name before every search — stub it so no
    # test here makes a live provider call regardless of which nodes it exercises.
    monkeypatch.setattr(news_mod, "aget_company_name", _async_return(None))


def _mock_all_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fundamentals_mod, "aget_fundamentals", _async_return(FAKE_FUNDAMENTALS))
    monkeypatch.setattr(technical_mod, "aget_technical_data", _async_return(FAKE_TECHNICAL))
    monkeypatch.setattr(news_mod, "asearch_news", _async_return(FAKE_NEWS_RESULTS))
    for module in (fundamentals_mod, technical_mod, news_mod):
        _mock_analysis(monkeypatch, module)


def _mock_intent(monkeypatch: pytest.MonkeyPatch, intent: TurnIntent, mapping: dict[str, str]) -> None:
    async def _classify(state):  # noqa: ARG001
        return intent

    async def _resolve(ticker: str) -> ResolvedTicker:
        return ResolvedTicker(mapping.get(ticker))

    monkeypatch.setattr(plan_mod, "classify_turn", _classify)
    monkeypatch.setattr(resolve_mod, "aresolve_ticker", _resolve)


def _subjects(*names: str) -> TurnIntent:
    return TurnIntent(
        companies=[
            CompanyRef(name=n, role="research_subject", ticker=n) for n in names
        ]
    )


def _state(question: str = "Analyze AAPL") -> Any:
    return new_session_state(user_question=question, session_id=str(uuid.uuid4()))


async def _run(monkeypatch: pytest.MonkeyPatch, intent: TurnIntent, mapping: dict[str, str], question: str = "q"):
    _mock_intent(monkeypatch, intent, mapping)
    graph = build_research_graph()
    return await graph.ainvoke(_state(question))


# ─────────────────────── parallel merge and partial failure ───────────────────────


async def test_all_three_branches_merge_into_one_ticker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reducer guard: the three specialists run in parallel and each returns a partial
    update for its own cell only. Without `merge_cells`, last-write-wins would leave one.
    """
    _mock_all_tools(monkeypatch)
    result = await _run(monkeypatch, _subjects("AAPL"), {"AAPL": "AAPL"})

    assert set(result["researched"]["AAPL"]) == set(AGENT_NAMES)
    assert all(cell["status"] == "ok" for cell in result["researched"]["AAPL"].values())
    assert result["turn"]["output"]["kind"] == "report"


async def test_one_failed_agent_still_produces_a_report(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_all_tools(monkeypatch)
    monkeypatch.setattr(news_mod, "asearch_news", _async_raise(WebSearchError("search down")))

    result = await _run(monkeypatch, _subjects("AAPL"), {"AAPL": "AAPL"})
    cells = result["researched"]["AAPL"]

    assert cells["news"]["status"] == "failed"
    assert cells["fundamentals"]["status"] == "ok"
    assert result["turn"]["output"]["kind"] == "report"


async def test_partial_failure_is_disclosed_deterministically(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gap statement no longer depends on the synthesis model remembering to narrate
    it. `emit` assembles a coverage line from the cells themselves, so the disclosure
    exists whether the model mentions it or not — and the harness can assert on it.
    """
    _mock_all_tools(monkeypatch)
    monkeypatch.setattr(technical_mod, "aget_technical_data", _async_raise(MarketDataError("timed out")))

    result = await _run(monkeypatch, _subjects("AAPL"), {"AAPL": "AAPL"})
    text = result["turn"]["output"]["text"]

    assert "Coverage:" in text
    assert "technical unavailable" in text
    assert "timed out" in text


async def test_llm_failure_marks_the_agent_failed_not_partially_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_all_tools(monkeypatch)

    async def _boom(prompt: str, schema: type = NodeAnalysis):  # noqa: ARG001
        raise LLMAnalysisError("The analysis service is temporarily unavailable.")

    monkeypatch.setattr(fundamentals_mod, "run_structured_analysis", _boom)

    result = await _run(monkeypatch, _subjects("AAPL"), {"AAPL": "AAPL"})
    cell = result["researched"]["AAPL"]["fundamentals"]

    assert cell["status"] == "failed"
    assert cell["findings"] == []
    assert cell["fetched_at"]  # stamped on failure too, so it can be retried later


async def test_every_agent_failing_produces_a_plain_reply_not_a_report_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A-10 end to end: nothing usable came back, so a "Research Report" card would present
    a failure as a completed deliverable.
    """
    monkeypatch.setattr(fundamentals_mod, "aget_fundamentals", _async_raise(MarketDataError("down")))
    monkeypatch.setattr(technical_mod, "aget_technical_data", _async_raise(MarketDataError("down")))
    monkeypatch.setattr(news_mod, "asearch_news", _async_raise(WebSearchError("down")))

    result = await _run(monkeypatch, _subjects("AAPL"), {"AAPL": "AAPL"})
    output = result["turn"]["output"]

    assert output["kind"] == "answer"
    assert "# Research Report" not in output["text"]
    assert "wasn't able to complete" in output["text"]


async def test_news_returning_no_articles_does_not_count_as_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A-10's precise shape: news runs cleanly and finds nothing, while the other two fail.
    `status == "ok"` was the old usability test, so this produced a full report — verdict
    line included — from two error strings and a "no news found".
    """
    monkeypatch.setattr(fundamentals_mod, "aget_fundamentals", _async_raise(MarketDataError("down")))
    monkeypatch.setattr(technical_mod, "aget_technical_data", _async_raise(MarketDataError("down")))
    monkeypatch.setattr(news_mod, "asearch_news", _async_return([]))

    result = await _run(monkeypatch, _subjects("AAPL"), {"AAPL": "AAPL"})

    assert result["researched"]["AAPL"]["news"]["status"] == "ok"
    assert result["researched"]["AAPL"]["news"]["findings"] == []
    assert result["turn"]["output"]["kind"] == "answer"
    assert "# Research Report" not in result["turn"]["output"]["text"]


# ─────────────────────── fan-out scoping ───────────────────────


async def test_fan_out_dispatches_only_the_cells_in_the_fetch_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A-04/A-05 end to end. The turn asks about one aspect of one company, so exactly one
    specialist runs — regardless of what else the session holds.
    """
    _mock_all_tools(monkeypatch)
    calls: list[str] = []

    async def _tracked_fundamentals(ticker: str):
        calls.append(ticker)
        return FAKE_FUNDAMENTALS

    async def _tracked_technical(ticker: str):  # noqa: ARG001
        calls.append("technical")
        return FAKE_TECHNICAL

    monkeypatch.setattr(fundamentals_mod, "aget_fundamentals", _tracked_fundamentals)
    monkeypatch.setattr(technical_mod, "aget_technical_data", _tracked_technical)

    intent = TurnIntent(
        companies=[CompanyRef(name="AAPL", role="research_subject", ticker="AAPL")],
        aspects=["fundamentals"],
    )
    result = await _run(monkeypatch, intent, {"AAPL": "AAPL"})

    assert calls == ["AAPL"]
    assert set(result["researched"]["AAPL"]) == {"fundamentals"}


async def test_multi_ticker_run_fans_out_per_ticker(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_all_tools(monkeypatch)
    result = await _run(
        monkeypatch, _subjects("AAPL", "MSFT"), {"AAPL": "AAPL", "MSFT": "MSFT"}
    )

    assert set(result["researched"]) == {"AAPL", "MSFT"}
    assert result["turn"]["shape"] == "comparison"
    assert "# Comparison" in result["turn"]["output"]["text"]


# ─────────────────────── the single exit ───────────────────────


@pytest.mark.parametrize(
    ("intent", "mapping"),
    [
        (_subjects("AAPL"), {"AAPL": "AAPL"}),                                    # research
        (TurnIntent(off_domain_topic="a pasta recipe"), {}),                       # chat
        (TurnIntent(screening_scope="Indian stocks"), {}),                         # screening
        (_subjects("Wakanda"), {}),                                                # unresolvable
    ],
)
async def test_every_lane_records_exactly_one_assistant_turn(
    monkeypatch: pytest.MonkeyPatch, intent: TurnIntent, mapping: dict
) -> None:
    """A-02 regression, as a structural guarantee rather than a per-node habit.

    `synthesis_comparison` and `synthesis_portfolio` returned `{"final_report": report}` on
    their success paths while `synthesis_single` returned the report *and* the history
    entry — so every comparison session's follow-ups classified against a transcript
    containing only the user's own messages. Emission and recording are now one operation
    in one node, so no lane can drop it.
    """
    _mock_all_tools(monkeypatch)
    result = await _run(monkeypatch, intent, mapping, question="something")

    assistant_turns = [m for m in result["conversation"] if m["role"] == "assistant"]
    assert len(assistant_turns) == 1
    assert assistant_turns[0]["content"]
    assert assistant_turns[0]["gist"]
    assert result["conversation"][0]["role"] == "user"


async def test_chat_lane_runs_no_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fundamentals_mod, "aget_fundamentals", _async_raise(AssertionError("must not run")))
    monkeypatch.setattr(technical_mod, "aget_technical_data", _async_raise(AssertionError("must not run")))
    monkeypatch.setattr(news_mod, "asearch_news", _async_raise(AssertionError("must not run")))

    result = await _run(monkeypatch, TurnIntent(off_domain_topic="a pasta recipe"), {})

    assert result["researched"] == {}
    assert result["turn"]["kind"] == "chat"
    assert "a pasta recipe" in result["turn"]["output"]["text"]


# ─────────────────────── citations ───────────────────────


async def test_every_citation_marker_resolves_to_a_real_finding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirrors the eval harness's hard-gated citation-integrity check: a bracket marker
    that doesn't resolve to a produced `Finding.id` is a fabricated citation.
    """
    _mock_all_tools(monkeypatch)
    monkeypatch.setattr(
        render_mod, "get_chat_model",
        lambda temperature=0.3: FakeLLM("Body citing [AAPL-fundamentals-1] and [AAPL-news-1]."),  # noqa: ARG005
    )

    result = await _run(monkeypatch, _subjects("AAPL"), {"AAPL": "AAPL"})
    text = result["turn"]["output"]["text"]

    produced = {
        f["id"]
        for cell in result["researched"]["AAPL"].values()
        for f in cell["findings"]
    }
    body = text.split("**Sources**")[0]
    for marker in re.findall(r"\[([A-Za-z0-9\-]+)\]", body):
        assert marker in produced, f"unresolved citation: {marker}"


async def test_sources_are_listed_once_per_unique_source(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_all_tools(monkeypatch)
    _mock_analysis(
        monkeypatch, fundamentals_mod,
        claims=[("c1", "e1"), ("c2", "e2"), ("c3", "e3")],
    )

    result = await _run(monkeypatch, _subjects("AAPL"), {"AAPL": "AAPL"})
    sources_block = result["turn"]["output"]["text"].split("**Sources**")[1]

    # Three fundamentals findings share one fetch, so one line carries all three ids.
    assert sources_block.count("fundamentals (Finnhub)") == 1


# ─────────────────────── multi-turn behaviour ───────────────────────


async def test_a_second_turn_reuses_fresh_research_without_refetching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recall lane: everything in scope passed the freshness check, so no tool call is
    made. Decided by a computation over timestamps, not by a classifier's opinion.
    """
    _mock_all_tools(monkeypatch)
    _mock_intent(monkeypatch, _subjects("AAPL"), {"AAPL": "AAPL"})

    graph = build_research_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t1"}}
    await graph.ainvoke(_state("Analyze AAPL"), config=config)

    calls: list[str] = []

    async def _tracked(ticker: str):
        calls.append(ticker)
        return FAKE_FUNDAMENTALS

    monkeypatch.setattr(fundamentals_mod, "aget_fundamentals", _tracked)

    second = await graph.ainvoke({"user_question": "what's the P/E?"}, config=config)

    assert calls == []
    assert second["turn"]["kind"] == "recall"
    assert second["turn"]["output"]["kind"] == "answer"


async def test_narrowing_follow_up_does_not_re_render_the_whole_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A-01 end to end — the bug that started the refactor.

    Turn 1 compares two companies. Turn 2 asks about one of them. Previously
    `_dispatch_synthesis` read the session's persisted `query_type`, which only ever
    ratcheted upward, so turn 2 produced the full comparison again with stale data for the
    company nobody asked about.
    """
    _mock_all_tools(monkeypatch)
    _mock_intent(monkeypatch, _subjects("AAPL", "MSFT"), {"AAPL": "AAPL", "MSFT": "MSFT"})

    graph = build_research_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t2"}}
    first = await graph.ainvoke(_state("compare AAPL and MSFT"), config=config)
    assert first["turn"]["shape"] == "comparison"

    _mock_intent(monkeypatch, _subjects("AAPL"), {"AAPL": "AAPL"})
    second = await graph.ainvoke({"user_question": "how is AAPL doing now?"}, config=config)

    assert second["turn"]["scope"] == ["AAPL"]
    assert second["turn"]["shape"] == "single"
    assert "MSFT" not in second["turn"]["output"]["text"]
    # The other company's research is retained, just not rendered.
    assert "MSFT" in second["researched"]


async def test_last_scope_survives_a_chat_interjection(monkeypatch: pytest.MonkeyPatch) -> None:
    """A chat turn mid-session must not wipe the antecedent a later "which one?" resolves
    against — which is why `emit` updates `last_scope` only on turns that had a scope.
    """
    _mock_all_tools(monkeypatch)
    _mock_intent(monkeypatch, _subjects("AAPL", "MSFT"), {"AAPL": "AAPL", "MSFT": "MSFT"})

    graph = build_research_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t3"}}
    await graph.ainvoke(_state("compare AAPL and MSFT"), config=config)

    _mock_intent(monkeypatch, TurnIntent(off_domain_topic="a pasta recipe"), {})
    after_chat = await graph.ainvoke({"user_question": "how do I make pasta?"}, config=config)

    assert after_chat["last_scope"] == ["AAPL", "MSFT"]
