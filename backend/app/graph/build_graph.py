"""LangGraph wiring — the full multi-ticker research graph, initial run + follow-ups.

    START -> entry_router
        (fresh session, no per_ticker_results yet)
            -> router -> {fan-out per ticker x 3 agents} -> collect_results -> dispatch -> synthesis_* -> END
                      |-> ask_clarification ------------------------------------------------------> END
                      |-> off_topic -------------------------------------------------------------> END
                      `-> no_tickers ---------------------------------------------------------------> END
        (awaiting_clarification=True: previous turn asked "which company?")
            -> clarification_response -> {fan-out, same as router} -> collect_results -> ... -> END
                                       |-> off_topic -----------------------------------------> END
                                       `-> no_tickers ------------------------------------------> END
        (follow-up: checkpointed session already has per_ticker_results)
            -> followup_router -> {fan-out to just the (ticker, agent) pairs that need (re)running}
                                      -> collect_results -> dispatch -> synthesis_* -> END
                                `-> answer_from_context -------------------------------------------> END
                                `-> ask_clarification -----------------------------------------------> END
                                `-> off_topic ---------------------------------------------------> END
        (awaiting_clarification=True AND clarification_origin="followup": previous turn was
         a follow-up too vague to act on, e.g. "how's the other one doing?")
            -> followup_clarification_response -> {fan-out, same as followup_router} -> ... -> END
                                                 `-> answer_from_context / ask_clarification / off_topic

    `router_node` and `followup_router_node` each set `awaiting_clarification=True` instead
    of guessing when a request is clearly stock-related but too ambiguous to act on —
    `entry_router` then routes the *next* turn deterministically to whichever resolver
    matches where the question came from (a state check, not an LLM classification of "is
    this a reply or a new topic"). A fresh-session clarification resolves through
    `clarification_response_node`, which replaces `state["tickers"]` outright since there
    was nothing there yet; a follow-up-session clarification resolves through
    `followup_clarification_response_node`, which must merge into or select from the
    session's *existing* tickers instead — see both nodes' docstrings for why they aren't
    the same node.

    `off_topic` is reached whenever a message (fresh, a resolved-as-abandoned
    clarification reply, or a follow-up) isn't about stocks at all, OR asks the app to
    find/select/rank candidate stocks — a capability this app doesn't have (it analyzes
    stocks it's given, it doesn't screen a market/sector). `router_node` and
    `followup_router_node` each decide both cases as part of their one classification
    call rather than a separate step, setting `off_topic_reply` /
    `followup_path in ("unrelated", "discovery")` respectively; `off_topic` itself is a
    dumb passthrough shared by all these entry paths, so the reply reads identically
    regardless of which turn produced it or which of the two reasons triggered it.

    `no_tickers` is the same idea applied to a request that *did* name a company, just
    not one that resolved to anything researchable (e.g. delisted, mistyped) — a plain
    reply built from why each attempted ticker failed, never a "Research Report" card:
    there's nothing in it to report. The three synthesis_* nodes carry the identical
    principle one step later in the graph — if literally nothing came back usable for
    any ticker, they also return a plain reply rather than a report-shaped one with no
    real content in it.

`router`/`followup_router` fan out dynamically via `Send`. Both routes target the exact
same fundamentals/technical/news nodes and the exact same synthesis_* nodes — there is no
parallel implementation of the research logic for follow-ups (ARCHITECTURE.md §8).

`collect_results` is a no-op join barrier: it has a static incoming edge from each
specialist node, so LangGraph only runs it once every dynamically-spawned instance of
all three has completed (the standard LangGraph map-reduce pattern), regardless of how
many (ticker, agent) pairs were fanned out to.

A session's conversation persists across calls via the LangGraph checkpointer, keyed by
`thread_id` = `session_id` (see `app/memory/checkpointer.py`) — pass one in for any
caller that needs follow-ups (the API layer); omit it for one-shot use (tests, the
`scripts/verify_graph.py` dev script, the eval harness).
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from app.graph.nodes.answer_from_context import answer_from_context_node
from app.graph.nodes.clarification_response_node import clarification_response_node
from app.graph.nodes.followup_clarification_response_node import followup_clarification_response_node
from app.graph.nodes.followup_router_node import followup_router_node
from app.graph.nodes.fundamentals_node import fundamentals_node
from app.graph.nodes.news_node import news_node
from app.graph.nodes.router_node import DEFAULT_OFF_TOPIC_REPLY, router_node
from app.graph.nodes.synthesis_comparison import synthesis_comparison_node
from app.graph.nodes.synthesis_portfolio import synthesis_portfolio_node
from app.graph.nodes.synthesis_single import synthesis_single_node
from app.graph.nodes.technical_node import technical_node
from app.graph.state import ResearchState

SPECIALIST_NODES = {
    "fundamentals": fundamentals_node,
    "technical": technical_node,
    "news": news_node,
}


def _entry_router(state: ResearchState) -> str:
    if state.get("awaiting_clarification"):
        # Which resolver depends on where the question came from, not just that one is
        # pending: a fresh-session clarification reply replaces `state["tickers"]`
        # outright (`clarification_response_node`), while a follow-up-session one must
        # merge into the session's existing tickers instead of wiping them out
        # (`followup_clarification_response_node`) — see that module's docstring.
        if state.get("clarification_origin") == "followup":
            return "followup_clarification_response"
        return "clarification_response"
    return "followup_router" if state.get("per_ticker_results") else "router"


def router_outcome(update: dict) -> str:
    """Which of the four outcomes a `router`/`clarification_response` update routes to —
    the exact precedence `_fan_out_after_router` acts on, factored out so
    `research_routes.py` can ask the identical question when deciding whether these
    same `notes` are about to be folded into `_no_tickers_node`'s reply (and so
    shouldn't also go out as a separate event) instead of re-deriving its own
    approximation that could quietly drift out of sync with the real routing rule.
    Takes a plain dict (not `ResearchState`) since the SSE-publishing caller only ever
    has a node's partial return, not the full graph state.
    """
    if update.get("awaiting_clarification"):
        return "ask_clarification"
    if update.get("off_topic_reply"):
        return "off_topic"
    if not update.get("tickers"):
        return "no_tickers"
    return "research"


def _fan_out_after_router(state: ResearchState) -> str | list[Send]:
    # Shared by both `router` and `clarification_response`'s outgoing edge — they return
    # the same shape (tickers/query_type/notes/awaiting_clarification/off_topic_reply),
    # so the same four-way fork (ask again / off-topic / give up / fan out to
    # specialists) applies to both.
    outcome = router_outcome(state)
    if outcome == "ask_clarification":
        return "ask_clarification"
    if outcome == "off_topic":
        return "off_topic"
    if outcome == "no_tickers":
        return "no_tickers"
    return [
        Send(agent, {**state, "tickers": [ticker]})
        for ticker in state["tickers"]
        for agent in SPECIALIST_NODES
    ]


def _fan_out_after_followup(state: ResearchState) -> str | list[Send]:
    # Shared by `followup_router` and `followup_clarification_response` — both return the
    # same shape (followup_path/followup_targets/off_topic_reply/awaiting_clarification),
    # so the same fork applies to either's output.
    if state.get("followup_path") == "needs_clarification":
        return "ask_clarification"
    if state.get("followup_path") in ("unrelated", "discovery"):
        return "off_topic"
    targets = state.get("followup_targets") or []
    if state.get("followup_path") != "answer" and targets:
        return [
            Send(agent, {**state, "tickers": [target["ticker"]]})
            for target in targets
            for agent in target["agents"]
        ]
    return "answer_from_context"


def _no_tickers_node(state: ResearchState) -> dict:
    # A plain reply, not a fake "Research Report" — nothing was actually researched here,
    # so a report-styled card would misrepresent a decline as a completed deliverable
    # (see `notes`' own dual purpose: normally an aside alongside a real report, but the
    # sole content of the reply when it's this one). `conversation_history` still gets
    # the plain assistant text, matching every other plain-reply node.
    detail = " ".join(state.get("notes", []))
    reply = (
        f"{detail} Could you double-check the name or ticker, or try a different company?"
        if detail
        else "I couldn't tell which company you meant — could you name one directly, e.g. 'AAPL' or 'Apple'?"
    )
    return {
        "followup_answer": reply,
        "conversation_history": [{"role": "assistant", "content": reply}],
    }


def _ask_clarification_node(state: ResearchState) -> dict:
    # Reuses the `followup_answer` field/event rather than inventing a parallel one — to
    # the frontend this is the same thing an answer_from_context reply is: a plain text
    # assistant message, not a research report.
    question = state.get("clarification_question") or "Which company or stock ticker would you like me to research?"
    return {
        "followup_answer": question,
        "conversation_history": [{"role": "assistant", "content": question}],
    }


def _off_topic_node(state: ResearchState) -> dict:
    # Same reuse of `followup_answer` as `_ask_clarification_node`, and shared verbatim
    # by the router/clarification-reply path (via `off_topic_reply`) and the follow-up
    # path (via `followup_path == "unrelated"`) — see module docstring.
    reply = state.get("off_topic_reply") or DEFAULT_OFF_TOPIC_REPLY
    return {
        "followup_answer": reply,
        "conversation_history": [{"role": "assistant", "content": reply}],
    }


def _collect_results_node(state: ResearchState) -> dict:  # noqa: ARG001 - join barrier, no-op
    return {}


def _dispatch_synthesis(state: ResearchState) -> str:
    return state["query_type"]


def build_research_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    graph = StateGraph(ResearchState)

    graph.add_node("router", router_node)
    graph.add_node("clarification_response", clarification_response_node)
    graph.add_node("ask_clarification", _ask_clarification_node)
    graph.add_node("off_topic", _off_topic_node)
    graph.add_node("followup_router", followup_router_node)
    graph.add_node("followup_clarification_response", followup_clarification_response_node)
    graph.add_node("answer_from_context", answer_from_context_node)
    for name, fn in SPECIALIST_NODES.items():
        graph.add_node(name, fn)
    graph.add_node("no_tickers", _no_tickers_node)
    graph.add_node("collect_results", _collect_results_node)
    graph.add_node("synthesis_single", synthesis_single_node)
    graph.add_node("synthesis_portfolio", synthesis_portfolio_node)
    graph.add_node("synthesis_comparison", synthesis_comparison_node)

    graph.add_conditional_edges(START, _entry_router)
    graph.add_conditional_edges("router", _fan_out_after_router)
    graph.add_conditional_edges("clarification_response", _fan_out_after_router)
    graph.add_conditional_edges("followup_router", _fan_out_after_followup)
    graph.add_conditional_edges("followup_clarification_response", _fan_out_after_followup)

    for name in SPECIALIST_NODES:
        graph.add_edge(name, "collect_results")

    graph.add_conditional_edges(
        "collect_results",
        _dispatch_synthesis,
        {
            "single": "synthesis_single",
            "portfolio": "synthesis_portfolio",
            "comparison": "synthesis_comparison",
        },
    )

    graph.add_edge("no_tickers", END)
    graph.add_edge("ask_clarification", END)
    graph.add_edge("off_topic", END)
    graph.add_edge("answer_from_context", END)
    graph.add_edge("synthesis_single", END)
    graph.add_edge("synthesis_portfolio", END)
    graph.add_edge("synthesis_comparison", END)

    return graph.compile(checkpointer=checkpointer)
