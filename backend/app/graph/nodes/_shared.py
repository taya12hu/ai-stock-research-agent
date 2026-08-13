"""Shared helpers for the specialist agent nodes (fundamentals/technical/news).

Each specialist node follows the same shape: fetch its one data source deterministically
(no tool-selection ambiguity — a fundamentals node always needs fundamentals data), then
ask the LLM to turn that data into a short summary plus a handful of grounded findings.
This module holds the piece all three share: the structured-output LLM call, its retry
policy, and conversion of LLM output into `Finding` objects with stable ids.
"""

from __future__ import annotations

from typing import TypeVar

import groq
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.graph.state import AgentName, Finding, Source
from app.llm.errors import LLMAnalysisError
from app.llm.groq_client import get_chat_model
from app.logging_config import get_logger

logger = get_logger("app.graph.nodes")

_RETRYABLE_GROQ_ERRORS = (
    groq.APIConnectionError,
    groq.APITimeoutError,
    groq.RateLimitError,
    groq.InternalServerError,
)

MAX_FINDINGS_PER_AGENT = 5


class LLMFinding(BaseModel):
    claim: str = Field(description="A single specific, factual claim about the company")
    evidence: str = Field(description="The exact number or fact from the provided data that supports the claim")


class NodeAnalysis(BaseModel):
    summary: str = Field(description="A 2-4 sentence plain-language summary")
    findings: list[LLMFinding] = Field(
        description=f"Up to {MAX_FINDINGS_PER_AGENT} specific, data-grounded findings"
    )


T = TypeVar("T", bound=BaseModel)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(_RETRYABLE_GROQ_ERRORS),
    reraise=True,
)
def _invoke_structured(prompt: str, schema: type[T]) -> T:
    llm = get_chat_model().with_structured_output(schema)
    result = llm.invoke(prompt)
    if not isinstance(result, schema):
        raise LLMAnalysisError(f"unexpected structured output type: {type(result)}")
    return result


async def run_structured_analysis(prompt: str, schema: type[T] = NodeAnalysis) -> T:  # type: ignore[assignment]
    try:
        return _invoke_structured(prompt, schema)
    except LLMAnalysisError:
        raise
    except Exception as exc:
        raise LLMAnalysisError(f"LLM analysis failed: {exc}") from exc


def build_findings(
    ticker: str, agent: AgentName, analysis: NodeAnalysis, source: Source
) -> list[Finding]:
    return [
        Finding(
            id=f"{ticker}-{agent}-{i + 1}",
            claim=f.claim,
            evidence=f.evidence,
            source=source,
        )
        for i, f in enumerate(analysis.findings[:MAX_FINDINGS_PER_AGENT])
    ]
