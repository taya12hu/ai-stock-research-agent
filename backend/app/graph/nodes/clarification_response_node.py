"""Handles the turn immediately after `router_node` set `awaiting_clarification=True` (it
asked the user which company/companies they meant).

This is deliberately NOT another LLM classification of "is this a new request or a reply
to my question" at the *routing* level — `_entry_router` sends every next turn here purely
off the persisted state flag, so a short reply like "TCS and Infosys" is never at risk of
being misread as an unrelated new topic (see `build_graph.py`). What IS asked of the LLM,
in a single call, is the narrower judgment call state alone can't make: does *this specific
reply* answer the pending question, or has the user abandoned it for something else
(a different stock question, a meta question about the app, or something unrelated)?
`RouterDecision.resolves_pending_clarification` carries that answer.

When it resolves: the request's query_type was already decided when the clarification was
first asked (`pending_intent`) and is reused as-is — the LLM here only extracts the missing
tickers, it is not asked to re-derive intent from the combined text a second time.

When it doesn't (abandoned): the same LLM call already classified the reply as if it were
a brand-new message (same fields `RouterDecision` always carries), so `apply_router_decision`
handles it exactly like `router_node` would — including opening a fresh clarification, a
polite off-topic reply, or (see that module) a discovery-limitation reply, if that's what
this new message calls for. This is how "any Indian stock I can purchase" or "healthcare
sector" — replies that reveal the user wants the app to find candidates rather than naming
any — get treated: they don't *resolve* the pending question (no company named), so they
fall through to the same discovery check a fresh message would get, instead of triggering
another round of "which stocks?"/"what criteria?" Either way this resolves in one LLM call,
never a loop back into re-asking the *same* question.

Only reachable when `clarification_origin == "router"` — a fresh session, or a clarification
reply that itself got reclassified as a new, equally-ambiguous message (still no
`per_ticker_results` yet). A follow-up-turn clarification (`clarification_origin ==
"followup"`) is resolved by `followup_clarification_response_node` instead, since resolving
it means merging into the session's existing tickers rather than replacing them — see that
module's docstring.
"""

from __future__ import annotations

import logging

from app.graph.nodes._shared import run_structured_analysis
from app.graph.nodes.router_node import RouterDecision, apply_router_decision, resolve_tickers
from app.graph.state import QueryType, ResearchState
from app.llm.errors import LLMAnalysisError
from app.logging_config import get_logger, log_event

logger = get_logger("app.graph.nodes.clarification_response")


def _build_prompt(original_question: str, clarification_question: str, reply: str) -> str:
    return (
        "A user asked a stock research question that didn't name a specific company:\n"
        f'"{original_question}"\n\n'
        f'They were asked: "{clarification_question}"\n'
        f'They replied: "{reply}"\n\n'
        "First decide whether this reply actually answers that question — names the "
        "missing company/companies, or otherwise resolves it — and set "
        "resolves_pending_clarification to true. If instead the user has moved on to "
        "something else entirely (a different stock question, a question about this app "
        "itself, or something unrelated), set it to false. Naming a scope/criterion "
        "instead of a company (e.g. 'any Indian stock', 'healthcare sector', 'the best "
        "IT stocks') does NOT resolve the question — that reveals the user wants the "
        "app to find candidates, which is a discovery request, not an answer naming "
        "companies; set resolves_pending_clarification to false for that case.\n\n"
        "If true: extract every company/ticker they named into tickers, and set "
        "unaddressed_note if part of the reply names companies but also asks for more "
        "via an open-ended criterion (see that field's description). The request's type "
        "was already decided earlier, so query_type/needs_clarification/"
        "clarifying_question/off_topic_reply/is_discovery_request/discovery_reply are "
        "ignored — just fill tickers and (if applicable) unaddressed_note.\n\n"
        "If false: classify this reply exactly as you would classify a brand new "
        "message — set is_stock_related, query_type, tickers, is_discovery_request "
        "(and discovery_reply if so — see that field's description for the discovery "
        "vs. needs_clarification test), and if it's itself a vague stock question with "
        "no scope, needs_clarification with a fresh clarifying_question of your own (not "
        "a repeat of the previous question); or, if it isn't about stocks at all, "
        "off_topic_reply."
    )


async def clarification_response_node(state: ResearchState) -> dict:
    reply = state["user_question"]
    original_question = state.get("pending_question") or reply
    clarification_question = state.get("clarification_question") or ""
    pending_intent: QueryType = state.get("pending_intent") or "single"

    try:
        decision = await run_structured_analysis(
            _build_prompt(original_question, clarification_question, reply), schema=RouterDecision
        )
    except LLMAnalysisError as exc:
        log_event(
            logger, "clarification classification failed", level=logging.ERROR,
            session_id=state["session_id"], error=str(exc),
        )
        # `str(exc)` is guaranteed clean (see the equivalent comment in router_node.py).
        return {
            "tickers": [],
            "query_type": "single",
            "notes": [],
            "conversation_history": [{"role": "user", "content": reply}],
            "awaiting_clarification": False,
            "clarification_question": None,
            "pending_question": None,
            "pending_intent": None,
            "off_topic_reply": str(exc),
            "clarification_origin": None,
        }

    if not decision.resolves_pending_clarification:
        log_event(
            logger, "clarification abandoned, reclassifying reply as a new message",
            session_id=state["session_id"],
        )
        return await apply_router_decision(decision, reply, state["session_id"])

    raw_tickers = [c.ticker for c in decision.tickers]
    valid_tickers, query_type, notes = await resolve_tickers(raw_tickers, pending_intent)
    if decision.unaddressed_note:
        notes = [*notes, decision.unaddressed_note]

    if not valid_tickers:
        notes = [
            *notes,
            "Still couldn't tell which company you meant from that reply — try naming "
            "one directly, e.g. 'AAPL' or 'Apple'.",
        ]

    log_event(
        logger, "clarification resolved", session_id=state["session_id"],
        tickers=valid_tickers, resolved=bool(valid_tickers),
    )
    return {
        "tickers": valid_tickers,
        "query_type": query_type,
        "notes": notes,
        "conversation_history": [{"role": "user", "content": reply}],
        "awaiting_clarification": False,
        "clarification_question": None,
        "pending_question": None,
        "pending_intent": None,
        "off_topic_reply": None,
        "clarification_origin": None,
    }
