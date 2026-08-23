"""The recall lane: answer from research this session already holds, with no tool calls.

Reached when `turn.fetch` is empty — every cell in scope passed the freshness check. That
is a computation over timestamps and statuses, not a classifier's opinion that the question
was "already covered", which is the distinction that matters: the old design let a single
soft LLM judgment decide this, with nothing checking whether the underlying data had gone
stale in the meantime.

Two scoping details carry weight:

- Only `turn.scope` cells are shown, not the whole session. A question about NVDA in a
  three-ticker session should not be answered against context for two companies nobody
  asked about.
- The raw findings are shown alongside the last written answer. A report only ever carries
  a subset of what was found (`MAX_FINDINGS_PER_AGENT` caps it for readability), so
  answering from the prose alone produced a wrong "that wasn't covered" for anything
  fetched that didn't make the cut.
"""

from __future__ import annotations

import logging

from app.graph.nodes._synthesis_shared import PROSE_STYLE, ticker_section_block
from app.graph.session import SessionState, TurnOutput
from app.llm.errors import RATE_LIMIT_MESSAGE, is_rate_limited
from app.llm.groq_client import get_chat_model
from app.logging_config import get_logger, log_event
from app.replies import join_human

logger = get_logger("app.graph.nodes.answer_from_context")

_HISTORY_MESSAGES = 8


def _build_prompt(state: SessionState) -> str:
    turn = state["turn"]
    scope = turn["scope"]
    researched = state.get("researched") or {}

    blocks = "\n\n".join(
        ticker_section_block(ticker, researched.get(ticker) or {}, turn["aspects"])
        for ticker in scope
    ) or "(no underlying findings available)"

    history = "\n".join(
        f"{m['role']}: {m.get('gist') if m['role'] == 'assistant' and m.get('gist') else m['content']}"
        for m in (state.get("conversation") or [])[-_HISTORY_MESSAGES:]
    )

    return (
        "You are answering a follow-up question in an ongoing stock research session, "
        "grounded ONLY in the research below. If the answer genuinely isn't covered by "
        "it, say so plainly rather than guessing or inventing facts. If asked for a "
        "buy/sell/hold view, give one grounded only in these findings, with a brief "
        "rationale and a confidence qualifier, and note that it is not personalized "
        f"financial advice.{PROSE_STYLE}\n\n"
        f"This answer covers: {join_human(scope)}\n\n"
        f"Research gathered for those companies:\n{blocks}\n\n"
        f"Recent conversation:\n{history}\n\n"
        f'Question: "{state["user_question"]}"'
    )


async def answer_from_context_node(state: SessionState) -> dict:
    turn = state["turn"]
    try:
        llm = get_chat_model(temperature=0.2)
        response = await llm.ainvoke(_build_prompt(state))
        answer = response.content
    except Exception as exc:
        # The raw provider exception is logged in full for debugging and never shown —
        # same discipline as `run_structured_analysis`.
        log_event(
            logger, "answer_from_context LLM call failed", level=logging.ERROR,
            session_id=state["session_id"], error=str(exc),
        )
        answer = (
            RATE_LIMIT_MESSAGE
            if is_rate_limited(exc)
            else "Sorry, I couldn't answer that right now. The research above is still available."
        )

    log_event(
        logger, "answer_from_context completed",
        session_id=state["session_id"], scope=turn["scope"],
    )
    return {"turn": {**turn, "output": TurnOutput(kind="answer", text=answer)}}
