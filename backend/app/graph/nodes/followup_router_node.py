"""Classifies a follow-up question into one of three paths (ARCHITECTURE.md §8):

1. answer     — fully answerable from what's already in this session; no new tool calls.
2. refresh    — needs updated data for a ticker already in the session.
3. add_ticker — introduces a ticker not yet in the session.

Runs only on follow-up turns (routed here by `_entry_router` in build_graph.py when the
checkpointed session already has `per_ticker_results`). Both `refresh` and `add_ticker`
resolve to the same shape — a list of `{ticker, agents}` targets — which
`_fan_out_after_followup` in build_graph.py turns into `Send` calls at the *same*
fundamentals/technical/news nodes used by the initial run. No parallel implementation of
the research logic for follow-ups.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.config import settings
from app.graph.nodes._shared import run_structured_analysis
from app.graph.state import AGENT_NAMES, AgentName, FollowUpPath, FollowUpTarget, QueryType, ResearchState
from app.llm.errors import LLMAnalysisError
from app.logging_config import get_logger, log_event
from app.tools.yahoo_finance import aticker_exists

logger = get_logger("app.graph.nodes.followup_router")


class FollowUpDecision(BaseModel):
    path: FollowUpPath = Field(
        description=(
            "'answer' if the question is fully answerable from research already gathered "
            "in this session (no new data needed); 'refresh' if it needs updated/fresh "
            "data for a ticker ALREADY researched in this session; 'add_ticker' if it "
            "introduces a company/ticker not yet researched in this session."
        )
    )
    refresh_tickers: list[str] = Field(
        default_factory=list,
        description="For path='refresh': which of the session's EXISTING tickers need fresh data",
    )
    refresh_agents: list[AgentName] = Field(
        default_factory=list,
        description=(
            "For path='refresh': which analyses to refresh. If the question implies all "
            "of them, include all three."
        ),
    )
    new_tickers: list[str] = Field(
        default_factory=list,
        description=(
            "For path='add_ticker': the stock TICKER SYMBOL (not the company name) for "
            "each new company to add, e.g. 'Intel' -> 'INTC', 'Google' -> 'GOOGL'."
        ),
    )


def _build_prompt(state: ResearchState) -> str:
    history = "\n".join(
        f"{m['role']}: {m['content'][:300]}" for m in state.get("conversation_history", [])[-6:]
    )
    return (
        "This is a follow-up question in an ongoing stock research session. The session "
        f"has already researched: {', '.join(state['tickers'])} (query type: "
        f"{state['query_type']}).\n\n"
        f"Recent conversation:\n{history}\n\n"
        f'New question: "{state["user_question"]}"\n\n'
        "Classify how to handle it."
    )


async def followup_router_node(state: ResearchState) -> dict:
    user_turn = [{"role": "user", "content": state["user_question"]}]

    try:
        decision = await run_structured_analysis(_build_prompt(state), schema=FollowUpDecision)
    except LLMAnalysisError as exc:
        log_event(
            logger, "followup classification failed", level=logging.ERROR,
            session_id=state["session_id"], error=str(exc),
        )
        return {
            "followup_path": "answer",
            "followup_targets": [],
            "notes": [f"Could not classify the follow-up: {exc}"],
            "conversation_history": user_turn,
        }

    if decision.path == "refresh":
        result = _plan_refresh(state, decision)
    elif decision.path == "add_ticker":
        result = await _plan_add_ticker(state, decision)
    else:
        result = {"followup_path": "answer", "followup_targets": []}

    result["conversation_history"] = user_turn
    log_event(
        logger, "followup routed", session_id=state["session_id"],
        path=result["followup_path"], target_count=len(result.get("followup_targets", [])),
    )
    return result


def _plan_refresh(state: ResearchState, decision: FollowUpDecision) -> dict:
    existing = set(state["tickers"])
    refresh_tickers = [t.strip().upper() for t in decision.refresh_tickers if t.strip().upper() in existing]
    agents: list[AgentName] = decision.refresh_agents or list(AGENT_NAMES)

    if not refresh_tickers:
        return {"followup_path": "answer", "followup_targets": []}

    targets: list[FollowUpTarget] = [{"ticker": t, "agents": agents} for t in refresh_tickers]
    return {"followup_path": "refresh", "followup_targets": targets}


async def _plan_add_ticker(state: ResearchState, decision: FollowUpDecision) -> dict:
    existing = state["tickers"]
    notes: list[str] = []

    candidates = list(dict.fromkeys(t.strip().upper() for t in decision.new_tickers if t.strip()))
    candidates = [t for t in candidates if t not in existing]

    room = settings.max_tickers - len(existing)
    if room <= 0:
        notes.append(f"Cannot add more tickers — the {settings.max_tickers}-ticker limit is already reached.")
        return {"followup_path": "answer", "followup_targets": [], "notes": notes}

    if len(candidates) > room:
        dropped, candidates = candidates[room:], candidates[:room]
        notes.append(
            f"Only {', '.join(candidates)} could be added — the {settings.max_tickers}-ticker "
            f"limit was reached; dropped: {', '.join(dropped)}."
        )

    valid_new: list[str] = []
    for ticker in candidates:
        if await aticker_exists(ticker):
            valid_new.append(ticker)
        else:
            notes.append(f"'{ticker}' could not be found and was skipped.")

    if not valid_new:
        return {"followup_path": "answer", "followup_targets": [], "notes": notes}

    new_tickers = existing + valid_new
    query_type: QueryType = state["query_type"]
    if len(new_tickers) > 1 and query_type == "single":
        query_type = "comparison"

    targets: list[FollowUpTarget] = [{"ticker": t, "agents": list(AGENT_NAMES)} for t in valid_new]
    return {
        "followup_path": "add_ticker",
        "followup_targets": targets,
        "tickers": new_tickers,
        "query_type": query_type,
        "notes": notes,
    }
