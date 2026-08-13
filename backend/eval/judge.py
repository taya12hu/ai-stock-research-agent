"""LLM-as-judge scoring (ARCHITECTURE.md §11) — graded quality checks that can't be
verified mechanically; see objective_checks.py for what can be.
"""

from __future__ import annotations

from typing import Literal

import groq
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.llm.groq_client import get_chat_model

# Groq's tool-call schema validation occasionally rejects its own model's output when the
# model stringifies a numeric field (e.g. "3" instead of 3) — observed to be deterministic
# per prompt at temperature=0 (so a same-input retry alone never helps), and specifically
# tied to `int` fields with min/max constraints in the JSON schema. Score = Literal[1..5]
# (an enum, not a bounded integer) avoids the pattern that was triggering it; the retry
# below remains as a safety net for genuine transient API errors.
_RETRYABLE = (groq.BadRequestError, groq.APIConnectionError, groq.APITimeoutError, groq.RateLimitError)

Score = Literal[1, 2, 3, 4, 5]


class JudgeScores(BaseModel):
    grounding: Score = Field(
        description="Are claims plausibly grounded in the report's own cited evidence, with no obvious fabrication? 1=poor, 5=excellent.",
    )
    relevance: Score = Field(description="Does the report actually answer the user's question? 1=poor, 5=excellent.")
    completeness: Score = Field(
        description="Does it appropriately cover fundamentals, technical, and news/sentiment, or explain why a section is missing? 1=poor, 5=excellent.",
    )
    structure: Score = Field(
        description=(
            "For a comparison: is it a genuine structured comparison rather than reports "
            "stapled together? For single/portfolio: is it well-organized? 1=poor, 5=excellent."
        ),
    )
    rationale: str = Field(description="1-2 sentence justification for the scores")


def _build_prompt(question: str, query_type: str, final_report: str) -> str:
    return (
        "You are grading an AI-generated stock research report. Score it 1 (poor) to 5 "
        "(excellent) on each dimension, based only on the report text given — do not "
        "reward or penalize investment advice framing either way, just answer whether "
        "the report is well-grounded, relevant, complete, and well-structured.\n\n"
        f'User\'s question: "{question}"\n'
        f"Query type: {query_type}\n\n"
        f"Report:\n{final_report or '(no report was produced)'}"
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=6),
    retry=retry_if_exception_type(_RETRYABLE),
    reraise=True,
)
async def judge_report(question: str, query_type: str, final_report: str) -> JudgeScores:
    llm = get_chat_model(temperature=0).with_structured_output(JudgeScores)
    return await llm.ainvoke(_build_prompt(question, query_type, final_report))
