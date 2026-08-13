"""Classifies a free-text request into a query type + validated ticker list.

Runs first in the graph. If the caller already supplied `tickers` (tests, or a later
follow-up flow that already knows which tickers to (re)run), classification is skipped
entirely — this node is a no-op passthrough in that case.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.config import settings
from app.graph.nodes._shared import run_structured_analysis
from app.graph.state import QueryType, ResearchState
from app.llm.errors import LLMAnalysisError
from app.logging_config import get_logger, log_event
from app.tools.yahoo_finance import aticker_exists

logger = get_logger("app.graph.nodes.router")


class TickerCandidate(BaseModel):
    ticker: str = Field(description="Stock ticker symbol in caps, e.g. AAPL, NVDA")


class RouterDecision(BaseModel):
    query_type: QueryType = Field(
        description=(
            "'single' if exactly one company is mentioned; 'portfolio' if the user "
            "describes multiple stocks they hold together as a portfolio/holdings; "
            "'comparison' if the user wants two or more companies compared, or asks "
            "which is better / one 'vs' another. When multiple tickers are mentioned "
            "with no portfolio framing, default to 'comparison'."
        )
    )
    tickers: list[TickerCandidate] = Field(
        description="Every distinct publicly-traded company/stock ticker mentioned or implied"
    )


def _build_prompt(user_question: str) -> str:
    return (
        "Identify every publicly-traded company mentioned in the request below, convert "
        'each to its stock ticker symbol, and classify the request type.\n\n'
        f'Request: "{user_question}"'
    )


async def router_node(state: ResearchState) -> dict:
    if state.get("tickers"):
        return {}

    try:
        decision = await run_structured_analysis(
            _build_prompt(state["user_question"]), schema=RouterDecision
        )
    except LLMAnalysisError as exc:
        log_event(
            logger, "router classification failed", level=logging.ERROR,
            session_id=state["session_id"], error=str(exc),
        )
        return {
            "tickers": [],
            "query_type": "single",
            "notes": [f"Could not understand the request: {exc}"],
            "conversation_history": [{"role": "user", "content": state["user_question"]}],
        }

    raw_tickers = [c.ticker.strip().upper() for c in decision.tickers if c.ticker.strip()]
    deduped = list(dict.fromkeys(raw_tickers))

    notes: list[str] = []
    if len(deduped) > settings.max_tickers:
        dropped, deduped = deduped[settings.max_tickers :], deduped[: settings.max_tickers]
        notes.append(
            f"Only the first {settings.max_tickers} tickers were analyzed (limit reached); "
            f"dropped: {', '.join(dropped)}."
        )

    valid_tickers: list[str] = []
    for ticker in deduped:
        if await aticker_exists(ticker):
            valid_tickers.append(ticker)
        else:
            notes.append(f"'{ticker}' could not be found and was skipped.")

    query_type: QueryType = decision.query_type
    if len(valid_tickers) <= 1 and query_type != "single":
        # e.g. a requested comparison where one side turned out to be invalid
        query_type = "single"

    log_event(
        logger, "router completed", session_id=state["session_id"],
        query_type=query_type, tickers=valid_tickers, dropped_count=len(notes),
    )
    return {
        "tickers": valid_tickers,
        "query_type": query_type,
        "notes": notes,
        "conversation_history": [{"role": "user", "content": state["user_question"]}],
    }
