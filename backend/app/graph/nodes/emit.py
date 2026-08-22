"""The single exit. Every path in the graph ends here, and nothing else writes to
`conversation`.

That is the structural fix for A-02. Previously each terminal node was trusted to append
its own assistant turn, and two of them didn't: `synthesis_comparison` and
`synthesis_portfolio` returned `{"final_report": report}` on their success paths while
`synthesis_single` returned the report *and* the history entry. Both files appended history
correctly on their LLM-failure fallbacks, which is how it went unnoticed — it was a
copy-paste omission on the happy path, not a decision. The consequence was that every
follow-up in a comparison session classified against a transcript containing only the
user's own messages.

Making emission and recording one indivisible operation means no future node can reintroduce
that bug by forgetting a key.

Assembly this node owns, in order: the hedge prefix, the scope echo, the body, the coverage
line, per-turn notes, and the off-domain acknowledgment. Each is placed deliberately — see
the inline notes.
"""

from __future__ import annotations

from app import replies
from app.graph.freshness import is_usable
from app.graph.nodes._synthesis_shared import AGENT_LABELS
from app.graph.session import ConversationMessage, SessionState, TurnOutput, TurnPlan
from app.logging_config import get_logger, log_event

logger = get_logger("app.graph.nodes.emit")


def _coverage_line(turn: TurnPlan, researched: dict) -> str:
    """A deterministic statement of what was and wasn't available.

    Previously this was asked of the LLM ("if a section says 'Unavailable', mention that
    gap") and checked as a *soft* eval signal, which meant nothing guaranteed it happened.
    Assembling it from the cells themselves means the disclosure exists whether the model
    narrates it or not, and gives the harness something to assert on mechanically.
    """
    lines: list[str] = []
    for ticker in turn["scope"]:
        cells = researched.get(ticker) or {}
        parts: list[str] = []
        for agent in turn["aspects"]:
            cell = cells.get(agent)
            label = AGENT_LABELS[agent].lower()
            if is_usable(cell):
                continue
            if cell is None:
                parts.append(f"{label} not run")
            elif cell["status"] == "failed":
                parts.append(f"{label} unavailable ({cell['error']})")
            else:
                parts.append(f"{label} returned nothing")
        if parts:
            lines.append(f"{ticker} — {'; '.join(parts)}")

    if not lines:
        return ""
    return "\n\n*Coverage: " + ". ".join(lines) + ".*"


def _gist(turn: TurnPlan, output: TurnOutput) -> str:
    """The one-line summary of this turn that later classifications will see instead of the
    body — see `ConversationMessage.gist` for why the body is withheld.
    """
    scope = replies.join_human(turn["scope"])
    if output["kind"] == "report":
        return f"({turn['shape']} report on {scope})"
    if output["kind"] == "answer":
        return f"(answered about {scope})" if scope else "(answered)"
    if output["kind"] == "clarify":
        return "(asked which company was meant)"
    return "(declined — outside stock research)"


async def emit_node(state: SessionState) -> dict:
    turn = state["turn"]
    researched = state.get("researched") or {}

    output = turn.get("output")
    if output is None:
        # Chat and clarify turns carry their fully-built reply from `plan_node`; there is
        # no producer node in between.
        output = TurnOutput(
            kind="clarify" if turn["kind"] == "clarify" else "chat",
            text=turn.get("reply") or "",
        )

    parts: list[str] = []

    # Hedge first: it qualifies everything that follows.
    if turn["hedged"]:
        parts.append(replies.hedge_prefix())

    # Scope echo on recall answers only. Reports already state their scope in the title,
    # but a recall answer carries no header at all — and that is also where a wrong scope
    # is hardest to notice, since there is no report structure to look wrong. Cheapest
    # available mitigation for a misclassified scope: it turns a silently wrong answer into
    # an obviously wrong one the user corrects in a turn.
    if output["kind"] == "answer" and turn["scope"]:
        parts.append(replies.scope_echo(turn["scope"]))

    parts.append(output["text"])

    if output["kind"] == "report":
        parts.append(_coverage_line(turn, researched))

    # Per-turn notes (a dropped ticker, a trimmed request). Only on turns that produced
    # research — a chat turn's notes are already folded into its reply by
    # `replies.unresolved_tickers`, and appending them again would say it twice.
    if turn["kind"] in ("research", "recall") and turn["notes"]:
        parts.append("\n\n" + " ".join(turn["notes"]))

    # The off-domain half of a mixed message, last. Leading with a decline would bury the
    # report the user actually asked for behind a caveat about the part they didn't.
    parts.append(replies.mixed_acknowledgment(turn["off_domain_topic"]))

    text = "".join(p for p in parts if p).strip()
    final = TurnOutput(kind=output["kind"], text=text)

    message: ConversationMessage = {
        "role": "assistant",
        "content": text,
        "gist": _gist(turn, final),
    }
    update: dict = {
        "turn": {**turn, "output": final},
        "conversation": [message],
    }

    # `last_scope`/`last_shape` are the antecedent for the next backward reference, so they
    # are updated only by turns that actually had a scope. A chat interjection mid-session
    # must not wipe out what "which one is better?" should resolve against.
    if turn["kind"] in ("research", "recall") and turn["scope"]:
        update["last_scope"] = list(turn["scope"])
        update["last_shape"] = turn["shape"]

    log_event(
        logger, "turn emitted", session_id=state["session_id"],
        kind=turn["kind"], output_kind=final["kind"], scope=turn["scope"],
    )
    return update
