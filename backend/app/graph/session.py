"""Session state for the turn-based graph.

The organising rule, and the one the previous schema broke: **durable session memory and
per-turn decisions are different things and live in different places.**

`query_type`, `tickers` and `notes` each did three jobs at once — session memory, this
turn's scope, and this turn's output shape — and eleven of the audit's fourteen findings
trace back to that conflation. Most visibly A-01: `query_type` was persisted state that
only ever ratcheted upward (`single → comparison`, never back), and synthesis dispatched
straight off it, so a narrowing follow-up in a comparison session re-rendered the whole
comparison with stale data for the ticker nobody asked about.

Here the split is structural:

- **Durable, additive** — `researched`, `conversation`. Only ever grown, never re-derived
  from a classification.
- **Durable, overwritten by `emit`** — `last_scope`, `last_shape`. The deliberate,
  minimal exception to "nothing is inherited": a backward reference ("which one is
  better?") needs the *previous turn's* scope, which is not the same as the session's full
  ticker list — a session holding NVDA, AMD and INTC where the last turn discussed only
  the first two must resolve "which one" to those two. These are *fallback inputs* to a
  fresh derivation, consulted only when the current message names no companies of its own,
  and a message that names one always overrides them. Read by `plan_turn`; never by
  `render`, which takes its scope as an argument.
- **Per-turn, replaced wholesale** — `turn`. Everything the audit found leaking between
  turns (`notes` accumulating via `operator.add`, `off_topic_reply` never reset,
  `final_report` shadowing `followup_answer`) lives in here and cannot survive the turn.

One caveat that has to be enforced rather than assumed: on a follow-up the API passes only
`{"user_question": ...}` and LangGraph merges it into the checkpointed state, so the
*previous* turn's `turn` is still present until something overwrites it. `entry_gate`
returns `fresh_turn()` as its first act for exactly that reason — otherwise a node reading
`state["turn"]` before `plan_turn` had run would see stale scope, which is the precise bug
class this schema exists to remove.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from app.graph.state import AGENT_NAMES, AgentName, Finding, Message

Shape = Literal["single", "comparison", "portfolio"]

# What this turn is doing. Replaces the six-value `followup_path` enum: `refresh` and
# `add_ticker` were never distinct intents — the difference between them is set membership
# against `researched`, which code answers exactly — and `answer` vs the rest is
# `fetch == []`, a computation over timestamps. Only these four are genuinely different
# control flow.
TurnKind = Literal["research", "recall", "clarify", "chat"]

# How a company appears in the message. The single real judgment call left to the model.
# `unclear` is a first-class value rather than a confidence score: self-reported confidence
# from an LLM is poorly calibrated and doubles as an escape hatch from the hard call, where
# an explicit third role forces the ambiguity into the open where code can route on it.
CompanyRole = Literal["research_subject", "incidental", "unclear"]

OutputKind = Literal["report", "answer", "clarify", "chat"]


class TickerCell(TypedDict):
    """One (ticker, agent) result. Merges the old `AgentResult` with the separate
    `per_ticker_fetched_at` map — they were always written together by `node_result` and
    always read together by the freshness check, so keeping them apart bought nothing and
    cost a second reducer that had to stay in sync with the first.
    """

    status: Literal["ok", "failed"]
    summary: str
    findings: list[Finding]
    error: str | None
    # Stamped on success *and* failure, so "we tried and it failed" is distinguishable
    # from "never attempted" — see `freshness.fetch_reason`.
    fetched_at: str


class CellRef(TypedDict):
    """A (ticker, agent) address. A TypedDict rather than a tuple because state is
    serialised through the SQLite checkpointer and tuple keys do not survive JSON.
    """

    ticker: str
    agent: AgentName


class PendingClarification(TypedDict):
    """An open clarifying question. `attempts` is what bounds the loop (A-07): the previous
    design could re-arm clarification from inside its own resolver with no counter, so a
    user replying vaguely could be asked a differently-worded question indefinitely.
    """

    question: str
    original_question: str
    attempts: int


class TurnOutput(TypedDict):
    """The one slot for what this turn said. Previously two independent fields
    (`final_report`, `followup_answer`), neither ever cleared, so which one was current
    depended on which node happened to run last — knowledge no reader had. That is how a
    two-turn-old report ended up being handed to `answer_from_context` under the heading
    "most recently written report" (A-09).
    """

    kind: OutputKind
    text: str


class TurnPlan(TypedDict):
    """Everything decided about the current turn. Built by `plan_turn` from the model's
    observations plus session state; replaced wholesale on every message.
    """

    kind: TurnKind
    # The tickers THIS answer covers — a subset or superset of the session's, never
    # implicitly the whole session.
    scope: list[str]
    shape: Shape
    aspects: list[AgentName]
    # Exactly the cells that failed the freshness check. Empty means `recall`.
    fetch: list[CellRef]
    notes: list[str]
    # Fully-built reply text for `chat` and `clarify` turns — assembled in `app/replies.py`
    # from code-owned frames, never free-form model prose.
    reply: str | None
    # True when scope was resolved from an ambiguous subject that we chose to answer
    # anyway because no fetch was required; `emit` prepends the hedge prefix.
    hedged: bool
    # The non-stock half of a mixed message ("...and I'm thinking of leaving Amazon"),
    # acknowledged by `emit` after the report rather than silently dropped.
    off_domain_topic: str | None
    output: TurnOutput | None


def merge_cells(
    existing: dict[str, dict[AgentName, TickerCell]] | None,
    update: dict[str, dict[AgentName, TickerCell]] | None,
) -> dict[str, dict[AgentName, TickerCell]]:
    """Reducer for `researched`.

    The three specialist nodes run in parallel and each returns a partial update for its
    own cell only (`{"NVDA": {"news": ...}}`). LangGraph's default last-write-wins would
    let one branch's update discard its siblings', so this deep-merges at the
    ticker → agent level regardless of arrival order.
    """
    merged: dict[str, dict[AgentName, TickerCell]] = {
        ticker: dict(cells) for ticker, cells in (existing or {}).items()
    }
    for ticker, cells in (update or {}).items():
        merged[ticker] = {**merged.get(ticker, {}), **cells}
    return merged


class SessionState(TypedDict):
    session_id: str
    user_question: str
    researched: Annotated[dict[str, dict[AgentName, TickerCell]], merge_cells]
    conversation: Annotated[list[Message], operator.add]
    last_scope: list[str]
    last_shape: Shape
    pending: PendingClarification | None
    turn: TurnPlan


def fresh_turn() -> TurnPlan:
    """An empty turn. Returned by `entry_gate` so the previous turn's plan cannot survive
    into this one — see the module docstring on why this is explicit rather than implied.
    """
    return TurnPlan(
        kind="chat",
        scope=[],
        shape="single",
        aspects=list(AGENT_NAMES),
        fetch=[],
        notes=[],
        reply=None,
        hedged=False,
        off_domain_topic=None,
        output=None,
    )


def new_session_state(*, user_question: str, session_id: str) -> SessionState:
    return SessionState(
        session_id=session_id,
        user_question=user_question,
        researched={},
        conversation=[],
        last_scope=[],
        last_shape="single",
        pending=None,
        turn=fresh_turn(),
    )


def cell_ok(summary: str, findings: list[Finding], fetched_at: str) -> TickerCell:
    return TickerCell(
        status="ok", summary=summary, findings=findings, error=None, fetched_at=fetched_at
    )


def cell_failed(error: str, fetched_at: str) -> TickerCell:
    return TickerCell(
        status="failed", summary="", findings=[], error=error, fetched_at=fetched_at
    )


def session_tickers(state: SessionState) -> list[str]:
    """Every ticker this session has researched, in insertion order."""
    return list((state.get("researched") or {}).keys())
