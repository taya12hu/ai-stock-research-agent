"""Path 1 of follow-up handling: answer a question grounded only in what this session
has already gathered — no tool calls, no re-running specialist agents.
"""

from __future__ import annotations

import logging

from app.graph.state import ResearchState
from app.llm.errors import RATE_LIMIT_MESSAGE, is_rate_limited
from app.llm.groq_client import get_chat_model
from app.logging_config import get_logger, log_event

logger = get_logger("app.graph.nodes.answer_from_context")


def _build_prompt(state: ResearchState) -> str:
    history = "\n".join(
        f"{m['role']}: {m['content']}" for m in state.get("conversation_history", [])[-8:]
    )
    report = state.get("final_report") or "(no prior report available)"
    return (
        "You are answering a follow-up question in an ongoing stock research session, "
        "grounded ONLY in the research already gathered below. If the answer genuinely "
        "isn't covered by this context, say so plainly rather than guessing or inventing "
        "facts. If asked for a buy/sell/hold view, give one grounded only in the findings "
        "already gathered, with a brief rationale and a confidence qualifier, and note "
        "that this is not personalized financial advice.\n\n"
        f"Tickers researched: {', '.join(state['tickers'])}\n\n"
        f"Most recent full report:\n{report}\n\n"
        f"Recent conversation:\n{history}\n\n"
        f'Follow-up question: "{state["user_question"]}"'
    )


async def answer_from_context_node(state: ResearchState) -> dict:
    try:
        llm = get_chat_model(temperature=0.2)
        response = await llm.ainvoke(_build_prompt(state))
        answer = response.content
    except Exception as exc:
        # The raw provider exception (e.g. a rate-limit error body) is logged here, in
        # full, for debugging — never shown to the user; see the equivalent guard in
        # `_shared.py`'s run_structured_analysis for why.
        log_event(
            logger, "answer_from_context LLM call failed", level=logging.ERROR,
            session_id=state["session_id"], error=str(exc),
        )
        answer = (
            RATE_LIMIT_MESSAGE if is_rate_limited(exc) else
            "Sorry, I couldn't generate an answer to that follow-up right now. "
            "The most recent full report is still available above."
        )

    log_event(logger, "answer_from_context completed", session_id=state["session_id"])
    return {
        "followup_answer": answer,
        "conversation_history": [{"role": "assistant", "content": answer}],
    }
