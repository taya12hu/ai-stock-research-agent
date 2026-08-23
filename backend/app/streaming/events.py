"""SSE event shapes published to the browser during a research run (ARCHITECTURE.md §10).

Events are derived from LangGraph's `astream(state, stream_mode="updates")` output at
the API layer (see `app/api/research_routes.py`) — no node code publishes events
directly, which keeps the event schema independent of LangGraph's internal shapes and
keeps the (already-tested) node functions free of streaming concerns.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict, Union

from app.graph.state import Finding

EventType = Literal[
    "run_started",
    "router_completed",
    "agent_started",
    "agent_completed",
    "report_ready",
    "followup_answer_ready",
    "run_completed",
    "run_failed",
]

TERMINAL_EVENT_TYPES = {"run_completed", "run_failed"}


class RunStartedEvent(TypedDict):
    type: Literal["run_started"]


class RouterCompletedEvent(TypedDict):
    type: Literal["router_completed"]
    query_type: str
    tickers: list[str]
    notes: list[str]


class AgentStartedEvent(TypedDict):
    type: Literal["agent_started"]
    ticker: str
    agent: str


class AgentCompletedEvent(TypedDict):
    type: Literal["agent_completed"]
    ticker: str
    agent: str
    status: Literal["ok", "failed"]
    summary: str | None
    findings: list[Finding]
    error: str | None


class ReportReadyEvent(TypedDict):
    type: Literal["report_ready"]
    final_report: str


class FollowUpAnswerReadyEvent(TypedDict):
    type: Literal["followup_answer_ready"]
    answer: str


class RunCompletedEvent(TypedDict):
    type: Literal["run_completed"]


class RunFailedEvent(TypedDict):
    type: Literal["run_failed"]
    error: str


Event = Union[
    RunStartedEvent,
    RouterCompletedEvent,
    AgentStartedEvent,
    AgentCompletedEvent,
    ReportReadyEvent,
    FollowUpAnswerReadyEvent,
    RunCompletedEvent,
    RunFailedEvent,
]


def run_started() -> RunStartedEvent:
    return {"type": "run_started"}


def router_completed(query_type: str, tickers: list[str], notes: list[str]) -> RouterCompletedEvent:
    return {"type": "router_completed", "query_type": query_type, "tickers": tickers, "notes": notes}


def agent_started(ticker: str, agent: str) -> AgentStartedEvent:
    return {"type": "agent_started", "ticker": ticker, "agent": agent}


def agent_completed(ticker: str, agent: str, result: dict[str, Any]) -> AgentCompletedEvent:
    ok = result["status"] == "ok"
    return {
        "type": "agent_completed",
        "ticker": ticker,
        "agent": agent,
        "status": result["status"],
        "summary": result.get("summary") if ok else None,
        "findings": result["findings"] if ok else [],
        "error": None if ok else result.get("error"),
    }


def report_ready(final_report: str) -> ReportReadyEvent:
    return {"type": "report_ready", "final_report": final_report}


def followup_answer_ready(answer: str) -> FollowUpAnswerReadyEvent:
    return {"type": "followup_answer_ready", "answer": answer}


def run_completed() -> RunCompletedEvent:
    return {"type": "run_completed"}


def run_failed(error: str) -> RunFailedEvent:
    return {"type": "run_failed", "error": error}
