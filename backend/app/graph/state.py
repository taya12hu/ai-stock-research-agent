"""Shared LangGraph state schema (ARCHITECTURE.md §7).

`tickers` is a list from the start so the single-stock case is simply N=1 rather than a
special path that would need a rewrite once portfolio/comparison queries are added.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

AgentName = Literal["fundamentals", "technical", "news"]
AGENT_NAMES: tuple[AgentName, ...] = ("fundamentals", "technical", "news")
QueryType = Literal["single", "portfolio", "comparison"]
FollowUpPath = Literal["answer", "refresh", "add_ticker"]


class Source(TypedDict):
    type: Literal["yahoo_finance", "web"]
    label: str
    url: str | None
    as_of: str


class Finding(TypedDict):
    id: str
    claim: str
    evidence: str
    source: Source


class AgentResult(TypedDict):
    status: Literal["ok", "failed"]
    summary: str
    findings: list[Finding]
    error: str | None


class TickerResults(TypedDict, total=False):
    fundamentals: AgentResult
    technical: AgentResult
    news: AgentResult


def merge_per_ticker_results(
    existing: dict[str, TickerResults] | None,
    update: dict[str, TickerResults] | None,
) -> dict[str, TickerResults]:
    """Reducer for `ResearchState.per_ticker_results`.

    Fundamentals/technical/news nodes run in parallel and each returns a partial update
    for just its own key (e.g. `{"AAPL": {"fundamentals": ...}}`). Without a custom
    reducer, LangGraph's default "last write wins" behavior would let one parallel
    branch's update overwrite the others'. This reducer deep-merges at the
    ticker -> agent level instead, regardless of arrival order.
    """
    merged: dict[str, TickerResults] = {k: dict(v) for k, v in (existing or {}).items()}  # type: ignore[misc]
    for ticker, agent_updates in (update or {}).items():
        merged[ticker] = {**merged.get(ticker, {}), **agent_updates}  # type: ignore[typeddict-item]
    return merged


class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class FollowUpTarget(TypedDict):
    ticker: str
    agents: list[AgentName]


class ResearchState(TypedDict):
    tickers: list[str]
    query_type: QueryType
    user_question: str
    per_ticker_results: Annotated[dict[str, TickerResults], merge_per_ticker_results]
    final_report: str | None
    # Every user turn and assistant reply (initial report or follow-up answer) is
    # appended here, grounding subsequent follow-ups — hence `operator.add`.
    conversation_history: Annotated[list[Message], operator.add]
    session_id: str
    # Router/agent-level warnings surfaced to the user (e.g. an invalid ticker dropped,
    # the MAX_TICKERS cap trimmed the request). Nodes only ever append, hence `operator.add`.
    notes: Annotated[list[str], operator.add]
    # Set by `followup_router_node` on follow-up turns; unset (None/[]) on the initial run.
    followup_path: FollowUpPath | None
    followup_targets: list[FollowUpTarget]
    followup_answer: str | None


def new_state(
    *,
    user_question: str,
    session_id: str,
    tickers: list[str] | None = None,
    query_type: QueryType | None = None,
) -> ResearchState:
    """`tickers`/`query_type` are optional: leave them unset for a free-text request
    (the router node will classify it), or pass them to skip classification — used by
    tests and by follow-up flows that already know which tickers to (re)run.
    """
    return ResearchState(
        tickers=tickers or [],
        query_type=query_type or "single",
        user_question=user_question,
        per_ticker_results={},
        final_report=None,
        conversation_history=[],
        session_id=session_id,
        notes=[],
        followup_path=None,
        followup_targets=[],
        followup_answer=None,
    )


def ok_result(summary: str, findings: list[Finding]) -> AgentResult:
    return AgentResult(status="ok", summary=summary, findings=findings, error=None)


def failed_result(error: str) -> AgentResult:
    return AgentResult(status="failed", summary="", findings=[], error=error)
