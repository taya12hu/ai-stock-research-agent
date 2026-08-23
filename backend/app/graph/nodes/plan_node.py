"""The graph's entry node: classify the message, resolve its companies, plan the turn.

Three separately-testable pieces composed in one node — one LLM call, one resolution
round-trip, one pure computation — rather than three graph nodes, which would put the
intermediate `TurnIntent` into checkpointed state for no benefit.

**There is no separate clarification resolver, and no `entry_gate`.** Both existed in the
previous design and both turned out to be unnecessary here:

- The old graph had `clarification_response_node` and
  `followup_clarification_response_node` (near-identical siblings, plus a
  `clarification_origin` field to pick between them) because it feared an LLM would misread
  a terse reply like "TCS and Infosys" as an unrelated new topic. Under this design that
  fear dissolves: misreading it as a new message produces *exactly the same outcome*, since
  a new message naming two companies is a two-company research turn either way. What the
  old resolvers really added — the already-decided intent — now comes from the transcript,
  which `_projection` surfaces explicitly. Two nodes, one state field, and a whole class of
  merge-versus-replace bugs all go away.
- `entry_gate` existed to reset the per-turn state before anything could read it. This node
  runs first and returns a complete `turn` built from `fresh_turn()`, so the previous
  turn's plan is replaced before any reader exists.

`pending` therefore carries only what the transcript cannot: how many times we have already
asked. That counter is what bounds the loop (A-07).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from app import replies
from app.graph.nodes.classify_turn import classify_turn
from app.graph.plan_turn import plan_turn
from app.graph.resolve_scope import resolve_scope
from app.graph.session import (
    ConversationMessage,
    PendingClarification,
    SessionState,
    fresh_turn,
    session_tickers,
)
from app.llm.errors import RATE_LIMIT_MESSAGE, LLMAnalysisError
from app.logging_config import get_logger, log_event

logger = get_logger("app.graph.nodes.plan")

# Two questions, then stop asking. The previous design bounded nothing: both clarification
# resolvers could re-arm `awaiting_clarification` from inside their own resolution path, so
# a user replying vaguely could be asked a differently-worded question indefinitely. The
# docstring's claim that it "never loops back into re-asking the same question" was true
# and beside the point — a different question forever is still a loop.
MAX_CLARIFY_ATTEMPTS = 2


async def plan_node(state: SessionState) -> dict:
    user_turn: list[ConversationMessage] = [
        {"role": "user", "content": state["user_question"]}
    ]

    try:
        intent = await classify_turn(state)
    except LLMAnalysisError as exc:
        # A-06: the old fallback returned `followup_path="answer"`, which skipped the
        # freshness guard and answered from arbitrarily old context without ever telling
        # the user classification had failed. Say so instead.
        log_event(
            logger,
            "turn classification failed",
            level=logging.ERROR,
            session_id=state["session_id"],
            error=str(exc),
        )
        turn = fresh_turn()
        turn["reply"] = replies.classification_failed(
            rate_limited=str(exc) == RATE_LIMIT_MESSAGE
        )
        return {"turn": turn, "pending": None, "conversation": user_turn}

    known = session_tickers(state)
    resolution = await resolve_scope(intent, known)
    plan = plan_turn(
        intent=intent, resolution=resolution, state=state, now=datetime.now(timezone.utc)
    )

    update: dict = {"turn": plan, "conversation": user_turn, "pending": None}

    if plan["kind"] == "clarify":
        prior = state.get("pending")
        attempts = (prior["attempts"] if prior else 0) + 1
        if attempts > MAX_CLARIFY_ATTEMPTS:
            # Out of attempts. Hand over a concrete way forward rather than producing a
            # third differently-worded question.
            log_event(
                logger,
                "clarification attempts exhausted",
                session_id=state["session_id"],
                attempts=attempts,
            )
            plan["kind"] = "chat"
            plan["reply"] = replies.clarify_exhausted(known)
        else:
            update["pending"] = PendingClarification(
                question=plan["reply"] or "",
                # Keep the *original* ambiguous question across repeated attempts, so the
                # transcript shown to the classifier still says what was actually being
                # asked rather than the last restatement of it.
                original_question=(
                    prior["original_question"] if prior else state["user_question"]
                ),
                attempts=attempts,
            )

    log_event(
        logger,
        "turn planned",
        session_id=state["session_id"],
        kind=plan["kind"],
        scope=plan["scope"],
        shape=plan["shape"],
        fetch=len(plan["fetch"]),
        hedged=plan["hedged"],
    )
    return update
