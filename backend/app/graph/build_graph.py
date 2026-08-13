"""LangGraph wiring — the full multi-ticker research graph, initial run + follow-ups.

    START -> entry_router
        (fresh session, no per_ticker_results yet)
            -> router -> {fan-out per ticker x 3 agents} -> collect_results -> dispatch -> synthesis_* -> END
                      `-> no_tickers ---------------------------------------------------------------> END
        (follow-up: checkpointed session already has per_ticker_results)
            -> followup_router -> {fan-out to just the (ticker, agent) pairs that need (re)running}
                                      -> collect_results -> dispatch -> synthesis_* -> END
                                `-> answer_from_context -------------------------------------------> END

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
from app.graph.nodes.followup_router_node import followup_router_node
from app.graph.nodes.fundamentals_node import fundamentals_node
from app.graph.nodes.news_node import news_node
from app.graph.nodes.router_node import router_node
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
    return "followup_router" if state.get("per_ticker_results") else "router"


def _fan_out_after_router(state: ResearchState) -> str | list[Send]:
    if not state["tickers"]:
        return "no_tickers"
    return [
        Send(agent, {**state, "tickers": [ticker]})
        for ticker in state["tickers"]
        for agent in SPECIALIST_NODES
    ]


def _fan_out_after_followup(state: ResearchState) -> str | list[Send]:
    targets = state.get("followup_targets") or []
    if state.get("followup_path") != "answer" and targets:
        return [
            Send(agent, {**state, "tickers": [target["ticker"]]})
            for target in targets
            for agent in target["agents"]
        ]
    return "answer_from_context"


def _no_tickers_node(state: ResearchState) -> dict:
    reason = "; ".join(state.get("notes", [])) or "No valid stock tickers were identified in the request."
    return {"final_report": f"# Research Report\n\nUnable to identify any valid stock tickers to research. {reason}"}


def _collect_results_node(state: ResearchState) -> dict:  # noqa: ARG001 - join barrier, no-op
    return {}


def _dispatch_synthesis(state: ResearchState) -> str:
    return state["query_type"]


def build_research_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    graph = StateGraph(ResearchState)

    graph.add_node("router", router_node)
    graph.add_node("followup_router", followup_router_node)
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
    graph.add_conditional_edges("followup_router", _fan_out_after_followup)

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
    graph.add_edge("answer_from_context", END)
    graph.add_edge("synthesis_single", END)
    graph.add_edge("synthesis_portfolio", END)
    graph.add_edge("synthesis_comparison", END)

    return graph.compile(checkpointer=checkpointer)
