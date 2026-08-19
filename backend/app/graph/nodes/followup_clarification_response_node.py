"""Handles the turn immediately after `followup_router_node` set `awaiting_clarification=True`
with `clarification_origin="followup"` — i.e. a follow-up question was clearly about a stock
in/around this session but too vague to act on (e.g. "how's the other one doing?" with 3+
tickers in the session), so it asked which one instead of guessing.

Sibling of `clarification_response_node`, not a reuse of it, because resolving the two kinds
of clarification means different things: a *fresh*-session clarification reply supplies the
tickers for a request that had none yet, so resolving it means replacing `state["tickers"]`
outright (see `router_node.apply_router_decision`). A *follow-up* clarification reply is
about a session that already has tickers/results — resolving it must never touch that list
wholesale, only merge into it (a new company) or select a subset of it (an existing ticker
the user wants refreshed or answered about). Reusing `apply_router_decision` here would
silently wipe out the session's prior research the first time this path was exercised.

Structurally this mirrors `clarification_response_node` exactly, one level down: a single LLM
call decides `resolves_pending_clarification`, and either way the decision comes back shaped
as a `FollowUpDecision` — the *same* schema `followup_router_node` uses for a fresh follow-up
— so resolving or abandoning both flow through `apply_followup_decision`, never a second,
parallel interpretation of what "refresh"/"add_ticker"/"needs_clarification" mean.
"""

from __future__ import annotations

import logging

from app.graph.nodes._shared import run_structured_analysis
from app.graph.nodes.followup_router_node import FollowUpDecision, apply_followup_decision
from app.graph.state import ResearchState
from app.llm.errors import LLMAnalysisError
from app.logging_config import get_logger, log_event

logger = get_logger("app.graph.nodes.followup_clarification_response")


def _build_prompt(state: ResearchState, original_question: str, clarification_question: str, reply: str) -> str:
    history = "\n".join(
        f"{m['role']}: {m['content'][:300]}" for m in state.get("conversation_history", [])[-6:]
    )
    return (
        "This is a follow-up question in an ongoing stock research session. The session "
        f"has already researched: {', '.join(state['tickers'])} (query type: "
        f"{state['query_type']}).\n\n"
        f"Recent conversation:\n{history}\n\n"
        f'The user asked: "{original_question}"\n'
        f'They were asked to clarify: "{clarification_question}"\n'
        f'They replied: "{reply}"\n\n'
        "First decide whether this reply actually answers that question — names or "
        "otherwise identifies which stock(s) it meant — and set "
        "resolves_pending_clarification to true. If instead the user has moved on to "
        "something else entirely (a different question, something unrelated, or a new "
        "company that has nothing to do with the original question), set it to false.\n\n"
        "If true: classify the now-disambiguated request exactly as you would classify a "
        "fresh follow-up question — 'refresh' if they picked one of this session's "
        "existing tickers (fill refresh_tickers/refresh_agents), 'add_ticker' if they "
        "named a company not yet in this session (fill new_tickers), or 'answer' if it's "
        "now answerable from what's already in this session. needs_clarification/"
        "clarifying_question/unrelated/discovery do not apply to a resolved reply.\n\n"
        "If false: classify this reply exactly as you would a brand new follow-up "
        "question, including needs_clarification with a fresh clarifying_question of "
        "your own (never a repeat of the previous one) if it's itself still too vague to "
        "act on, or unrelated/discovery if that's what it actually is."
    )


async def followup_clarification_response_node(state: ResearchState) -> dict:
    reply = state["user_question"]
    original_question = state.get("pending_question") or reply
    clarification_question = state.get("clarification_question") or ""

    try:
        decision = await run_structured_analysis(
            _build_prompt(state, original_question, clarification_question, reply),
            schema=FollowUpDecision,
        )
    except LLMAnalysisError as exc:
        log_event(
            logger, "followup clarification classification failed", level=logging.ERROR,
            session_id=state["session_id"], error=str(exc),
        )
        return {
            "followup_path": "answer",
            "followup_targets": [],
            "notes": [f"Could not classify the reply: {exc}"],
            "conversation_history": [{"role": "user", "content": reply}],
            "awaiting_clarification": False,
            "clarification_question": None,
            "pending_question": None,
            "clarification_origin": None,
        }

    result = await apply_followup_decision(decision, state)
    result["conversation_history"] = [{"role": "user", "content": reply}]
    log_event(
        logger, "followup clarification resolved", session_id=state["session_id"],
        resolved=decision.resolves_pending_clarification, path=result["followup_path"],
    )
    return result
