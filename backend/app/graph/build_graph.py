"""LangGraph wiring.

    START -> plan
        chat     -> emit                                     (off-domain, screening, failure)
        clarify  -> emit                                     (asked which company)
        recall   -> answer_from_context -> emit              (everything in scope is fresh)
        research -> Send(fetch) -> collect -> render -> emit

Four lanes, one exit. The previous graph had eleven nodes and five entry paths; the
reduction is not tidying, it comes from moving decisions out of the model and into
`plan_turn`, at which point most of those nodes had nothing left to decide:

- `router` / `followup_router` collapse into `plan`. They differed only in which
  classification schema they used, and the fresh-session-versus-follow-up distinction they
  encoded is now just whether `researched` happens to be empty — which nothing needs to
  branch on, because `needs_fetch` gives the same answer either way.
- `clarification_response` / `followup_clarification_response` (and `clarification_origin`,
  which existed only to choose between them) are gone entirely: a reply to a clarifying
  question is an ordinary message classified with that question in view. See
  `plan_node`'s docstring.
- `no_tickers` / `off_topic` / `ask_clarification` collapse into the `chat` lane, since all
  three were passthrough nodes whose only job was to move a pre-built string into a state
  field.
- The three `synthesis_*` nodes collapse into `render`, which takes scope and shape as
  arguments instead of reading them off session state — the change that fixes A-01.

`collect` remains a no-op join barrier with a static incoming edge from each specialist, so
LangGraph runs it once every dynamically-spawned branch has settled, regardless of how many
were dispatched. That pattern was correct before and is unchanged.

A session's conversation persists across calls via the checkpointer keyed by
`thread_id` = `session_id`. Pass one in for any caller that needs follow-ups (the API
layer); omit it for one-shot use (tests, the eval harness).
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from app.graph.nodes.answer_from_context import answer_from_context_node
from app.graph.nodes.emit import emit_node
from app.graph.nodes.fundamentals_node import fundamentals_node
from app.graph.nodes.news_node import news_node
from app.graph.nodes.plan_node import plan_node
from app.graph.nodes.render import render_node
from app.graph.nodes.technical_node import technical_node
from app.graph.session import SessionState

SPECIALIST_NODES = {
    "fundamentals": fundamentals_node,
    "technical": technical_node,
    "news": news_node,
}


def route_turn(state: SessionState) -> str | list[Send]:
    """The graph's only branch, driven entirely by the plan.

    Note there is no separate `refresh` and `add_ticker` handling. Whether a ticker is
    already in the session is set membership, resolved when `fetch` was computed — by the
    time routing happens the distinction has no consequences left.
    """
    turn = state["turn"]
    if turn["kind"] in ("chat", "clarify"):
        return "emit"
    if turn["kind"] == "recall":
        return "answer_from_context"

    # One branch per (ticker, agent) cell that failed the freshness check — never per
    # ticker and never for the whole session. `scope` is narrowed to the single ticker each
    # specialist should work on, so nodes don't need to know how many branches exist.
    return [
        Send(cell["agent"], {**state, "turn": {**turn, "scope": [cell["ticker"]]}})
        for cell in turn["fetch"]
    ]


def _collect_node(state: SessionState) -> dict:  # noqa: ARG001 - join barrier, no-op
    return {}


def build_research_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    graph = StateGraph(SessionState)

    graph.add_node("plan", plan_node)
    graph.add_node("answer_from_context", answer_from_context_node)
    for name, fn in SPECIALIST_NODES.items():
        graph.add_node(name, fn)
    graph.add_node("collect", _collect_node)
    graph.add_node("render", render_node)
    graph.add_node("emit", emit_node)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges(
        "plan",
        route_turn,
        ["emit", "answer_from_context", *SPECIALIST_NODES],
    )

    for name in SPECIALIST_NODES:
        graph.add_edge(name, "collect")
    graph.add_edge("collect", "render")
    graph.add_edge("render", "emit")
    graph.add_edge("answer_from_context", "emit")
    graph.add_edge("emit", END)

    return graph.compile(checkpointer=checkpointer)
