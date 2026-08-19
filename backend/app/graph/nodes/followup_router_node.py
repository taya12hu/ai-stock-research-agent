"""Classifies a follow-up question into one of six paths (ARCHITECTURE.md §8):

1. answer               — fully answerable from what's already in this session; no new tool calls.
2. refresh               — needs updated data for a ticker already in the session.
3. add_ticker            — introduces a ticker not yet in the session.
4. unrelated             — not about stocks/this research session at all.
5. discovery             — asks the app to find/select/rank candidate stocks; unsupported.
6. needs_clarification   — clearly about a stock in/around this session, but too vague to
                            act on (e.g. "how's the other one doing?" with 3+ tickers in
                            the session) — mirrors `router_node.RouterDecision.
                            needs_clarification`, one level down.

Runs only on follow-up turns (routed here by `_entry_router` in build_graph.py when the
checkpointed session already has `per_ticker_results`). Both `refresh` and `add_ticker`
resolve to the same shape — a list of `{ticker, agents}` targets — which
`_fan_out_after_followup` in build_graph.py turns into `Send` calls at the *same*
fundamentals/technical/news nodes used by the initial run. No parallel implementation of
the research logic for follow-ups.

`unrelated` and `discovery` both exist so those messages get the same kind of plain,
honest reply here as they would on a brand-new chat (`router_node`), instead of being
forced through `answer_from_context_node`'s research-grounded prompt (which risks
hallucinating a suggestion since its own instructions only guard against contradicting
the *existing* report, not against inventing new candidates) or `add_ticker` (which would
otherwise have to invent a `new_tickers` value for a company nobody actually named) — see
build_graph.py's `_off_topic_node`, reused by both paths so the reply reads identically
regardless of whether this is the first message or the tenth.

`needs_clarification` exists for the same reason `router_node` has it: without it, a
follow-up that's too vague to act on would be forced into one of the other five —
overwhelmingly `answer`, which then has no signal that the question was ambiguous and
may confidently answer about the wrong ticker. Choosing it sets `awaiting_clarification`
exactly like `router_node` does, with `clarification_origin="followup"` so `_entry_router`
sends the *next* turn to `followup_clarification_response_node` rather than
`clarification_response_node` — the two need different resolution logic (merge into the
session's existing tickers vs. replace them wholesale) — see that module's docstring.
`apply_followup_decision` below is shared by both this node and that one, exactly the way
`router_node.apply_router_decision` is shared with `clarification_response_node`.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.config import settings
from app.graph.nodes._shared import run_structured_analysis
from app.graph.nodes.router_node import DEFAULT_DISCOVERY_REPLY, DEFAULT_OFF_TOPIC_REPLY
from app.graph.state import AGENT_NAMES, AgentName, FollowUpPath, FollowUpTarget, QueryType, ResearchState
from app.llm.errors import LLMAnalysisError
from app.logging_config import get_logger, log_event
from app.tools.yahoo_finance import aresolve_ticker

logger = get_logger("app.graph.nodes.followup_router")


class FollowUpDecision(BaseModel):
    path: FollowUpPath = Field(
        description=(
            "'answer' if the question is fully answerable from research already gathered "
            "in this session (no new data needed); 'refresh' if it needs updated/fresh "
            "data for a ticker ALREADY researched in this session; 'add_ticker' if it "
            "introduces a SPECIFIC, NAMED company/ticker not yet researched in this "
            "session; 'unrelated' if the question isn't about stocks or this research "
            "session at all; 'discovery' if it asks the app to find, select, rank, or "
            "recommend candidate stocks from some scope (a market, sector, theme, or "
            "criterion like 'best'/'undervalued') with no specific new company named — "
            "e.g. 'what other good IT stocks should I look at?', 'any similar companies "
            "worth considering?'. This app analyzes stocks it's given, it doesn't screen "
            "for candidates — never use 'add_ticker' to fill in a company you'd merely "
            "guess fits an implied 'find more like this' request; that's 'discovery'. "
            "'needs_clarification' if the question is clearly about a stock in or around "
            "this session but you genuinely can't tell which one it means and answering "
            "anyway risks guessing wrong — e.g. 'how's the other one doing?' or 'what "
            "about the second one?' when the session has more than one ticker and "
            "nothing in the recent conversation disambiguates it. Do not use this when "
            "the reference is actually clear from context (a session with exactly one "
            "ticker, or the conversation just discussed a specific one) — that's "
            "'answer'/'refresh', not 'needs_clarification'."
        )
    )
    off_topic_reply: str | None = Field(
        default=None,
        description=(
            "Only when path='unrelated': a brief, friendly reply. If the user asked "
            "something about this app itself (e.g. what data it uses), answer directly "
            "using: this app researches stocks using Yahoo Finance data (fundamentals "
            "and price/technical indicators) and web search for recent news. Otherwise, "
            "note that this chat is for the stock(s) already being researched and give "
            "one example, e.g. 'ask me about a specific stock'. Otherwise null."
        ),
    )
    discovery_reply: str | None = Field(
        default=None,
        description=(
            "Only when path='discovery': a brief, natural reply explaining that this app "
            "analyzes stocks the user provides rather than screening a market/sector for "
            "candidates, tailored to what they asked — then invite them to name a "
            "specific company to add to this session. Do NOT name any specific company "
            "in it, not even as an example. Otherwise null."
        ),
    )
    refresh_tickers: list[str] = Field(
        default_factory=list,
        description=(
            "For path='refresh': which of the session's EXISTING tickers (listed above "
            "in the prompt) need fresh data — only tickers already in that list, never "
            "a new one."
        ),
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
            "For path='add_ticker' ONLY: the stock TICKER SYMBOL (not the company name) "
            "for each new company the user explicitly named, e.g. 'Intel' -> 'INTC', "
            "'Google' -> 'GOOGL'. Never populate this for a company you're inferring "
            "fits some criterion the user described — an unnamed 'similar' or 'other "
            "good' company is a 'discovery' path, not 'add_ticker'."
        ),
    )
    clarifying_question: str | None = Field(
        default=None,
        description=(
            "If path='needs_clarification': a short question tailored to what was "
            "actually asked, naming the session's own tickers as the choices where that "
            "makes sense (e.g. 'Did you mean AAPL or MSFT?'). Write it fresh each time; "
            "never reuse a stock template phrase. Otherwise null."
        ),
    )
    resolves_pending_clarification: bool = Field(
        default=True,
        description=(
            "Only meaningful when classifying a REPLY to an earlier follow-up "
            "clarifying question (see that prompt): true if this reply answers it, "
            "false if the user has abandoned it for something else. Ignored otherwise."
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
        "Classify how to handle it. If it's not about stocks or this research session at "
        "all — even if it happens to mention a company name incidentally — use "
        "'unrelated' and write off_topic_reply rather than forcing it into 'answer'. If "
        "it asks for candidate stocks from a scope/criterion with no specific new "
        "company named (e.g. 'what other good IT stocks should I look at?', 'any "
        "similar companies?') use 'discovery' and write discovery_reply — never guess a "
        "company to satisfy it via 'add_ticker', and never let 'answer' improvise a "
        "suggestion beyond what's already in this session's research. If the session has "
        "more than one ticker and the question refers to one of them ambiguously (e.g. "
        "'how's the other one doing?', 'what about the second one?') with nothing in the "
        "recent conversation to disambiguate it, use 'needs_clarification' and write "
        "clarifying_question rather than guessing which ticker via 'answer' or 'refresh'."
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
            "awaiting_clarification": False,
            "clarification_question": None,
            "pending_question": None,
            "clarification_origin": None,
        }

    result = await apply_followup_decision(decision, state)
    result["conversation_history"] = user_turn
    log_event(
        logger, "followup routed", session_id=state["session_id"],
        path=result["followup_path"], target_count=len(result.get("followup_targets", [])),
    )
    return result


async def apply_followup_decision(decision: FollowUpDecision, state: ResearchState) -> dict:
    """Turns a `FollowUpDecision` into a state update for a fresh follow-up turn. Used by
    `followup_router_node` directly, and reused by `followup_clarification_response_node`
    when a reply abandons a pending follow-up clarification — in that case the same LLM
    call that noticed the abandonment already classified the reply as if it were a
    brand-new follow-up, so there's no second LLM round-trip (mirrors
    `router_node.apply_router_decision`, one level down).
    """
    if decision.path == "needs_clarification":
        log_event(logger, "followup needs clarification", session_id=state["session_id"])
        return {
            "followup_path": "needs_clarification",
            "followup_targets": [],
            "awaiting_clarification": True,
            "clarification_question": (
                decision.clarifying_question or "Which of the stocks in this session did you mean?"
            ),
            "pending_question": state["user_question"],
            "clarification_origin": "followup",
        }

    if decision.path == "refresh":
        result = _plan_refresh(state, decision)
    elif decision.path == "add_ticker":
        result = await _plan_add_ticker(state, decision)
    elif decision.path == "unrelated":
        result = {
            "followup_path": "unrelated",
            "followup_targets": [],
            "off_topic_reply": decision.off_topic_reply or DEFAULT_OFF_TOPIC_REPLY,
        }
    elif decision.path == "discovery":
        result = {
            "followup_path": "discovery",
            "followup_targets": [],
            "off_topic_reply": decision.discovery_reply or DEFAULT_DISCOVERY_REPLY,
        }
    else:
        result = {"followup_path": "answer", "followup_targets": []}

    # Explicit resets, same discipline as `router_node.apply_router_decision` — every
    # branch above can be reached right after a resolved/abandoned clarification, so a
    # stale `awaiting_clarification` (or its question/origin) must never survive past it.
    # `setdefault` rather than repeating these four keys in every branch/helper above
    # (`_plan_refresh`/`_plan_add_ticker` each have several return points of their own):
    # one place to keep the reset invariant correct, not six.
    result.setdefault("awaiting_clarification", False)
    result.setdefault("clarification_question", None)
    result.setdefault("pending_question", None)
    result.setdefault("clarification_origin", None)
    return result


def _bare(ticker: str) -> str:
    """Strips an exchange suffix ('TCS.NS' -> 'TCS'). The LLM extracts tickers from
    natural conversation, where a user says 'TCS', not 'TCS.NS' — but `state["tickers"]`
    may hold the resolved, suffixed symbol (see `aresolve_ticker`). Matching on the bare
    form is what lets a follow-up mentioning 'TCS' still find the session's 'TCS.NS'.
    """
    return ticker.split(".")[0]


def _plan_refresh(state: ResearchState, decision: FollowUpDecision) -> dict:
    existing_by_bare = {_bare(t): t for t in state["tickers"]}
    requested = [t.strip().upper() for t in decision.refresh_tickers if t.strip()]
    refresh_tickers = list(
        dict.fromkeys(existing_by_bare[_bare(t)] for t in requested if _bare(t) in existing_by_bare)
    )
    agents: list[AgentName] = decision.refresh_agents or list(AGENT_NAMES)

    if not refresh_tickers:
        # Not a silent no-op: either the model named something outside this session
        # (told apart here, not left for the user to guess) or extracted nothing at
        # all — either way the user asked for a refresh and nothing happened, so that
        # needs to be said, not swallowed into a plain context-grounded answer.
        unmatched = list(dict.fromkeys(t for t in requested if _bare(t) not in existing_by_bare))
        note = (
            f"{', '.join(unmatched)} isn't part of this session yet — I can only refresh "
            f"{', '.join(state['tickers'])}. Ask me to add it if you'd like it researched."
            if unmatched else
            "Couldn't tell which of this session's stocks you wanted refreshed — try "
            f"naming one directly, e.g. {state['tickers'][0]}."
        )
        return {"followup_path": "answer", "followup_targets": [], "notes": [note]}

    targets: list[FollowUpTarget] = [{"ticker": t, "agents": agents} for t in refresh_tickers]
    return {"followup_path": "refresh", "followup_targets": targets}


async def _plan_add_ticker(state: ResearchState, decision: FollowUpDecision) -> dict:
    existing = state["tickers"]
    notes: list[str] = []

    requested = list(dict.fromkeys(t.strip().upper() for t in decision.new_tickers if t.strip()))
    existing_bare = {_bare(t) for t in existing}
    already_present = [t for t in requested if _bare(t) in existing_bare]
    candidates = [t for t in requested if _bare(t) not in existing_bare]

    if already_present:
        verb = "is" if len(already_present) == 1 else "are"
        notes.append(f"{', '.join(already_present)} {verb} already in this session.")

    if not candidates:
        # Same silent-no-op concern as `_plan_refresh`: if nothing was named at all
        # (as opposed to everything named already being in-session, which the note
        # above already explains), say so rather than quietly falling through to a
        # generic context-grounded answer with no acknowledgment of the request.
        if not already_present:
            notes.append("Couldn't tell which company to add — try naming one directly, e.g. 'AAPL' or 'Apple'.")
        return {"followup_path": "answer", "followup_targets": [], "notes": notes}

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
        resolved = await aresolve_ticker(ticker)
        if resolved.symbol:
            valid_new.append(resolved.symbol)
        elif resolved.unsupported_market:
            notes.append(f"'{ticker}' isn't a US-listed stock, so it isn't currently supported.")
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
