"""Shared helpers for the specialist agent nodes (fundamentals/technical/news).

Each specialist node follows the same shape: fetch its one data source deterministically
(no tool-selection ambiguity — a fundamentals node always needs fundamentals data), then
ask the LLM to turn that data into a short summary plus a handful of grounded findings.
This module holds the piece all three share: the structured-output LLM call, its retry
policy, and conversion of LLM output into `Finding` objects with stable ids.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TypeVar

import groq
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.graph.state import AgentName, AgentResult, Finding, Source
from app.llm.errors import RATE_LIMIT_MESSAGE, LLMAnalysisError, is_rate_limited
from app.llm.groq_client import get_chat_model
from app.logging_config import get_logger, log_event

logger = get_logger("app.graph.nodes")

_RETRYABLE_GROQ_ERRORS = (
    groq.APIConnectionError,
    groq.APITimeoutError,
    groq.RateLimitError,
    groq.InternalServerError,
)


def _is_tool_use_failed(exc: BaseException) -> bool:
    """True for Groq's 400 'the model didn't call the required tool' — distinct from a
    genuinely malformed request (also a 400, but not retryable: it would fail identically
    every time). This one is model sampling variance, not a persistent block: the model
    sometimes answers a forced-tool-call prompt in plain text instead of invoking the
    tool, and asking again — same prompt, same schema — usually succeeds. Confirmed from
    a real failure: Groq's `failed_generation` field showed the model HAD produced a
    complete, well-grounded analysis, it just didn't format it as a tool call.
    """
    if not isinstance(exc, groq.BadRequestError):
        return False
    body = exc.body if isinstance(exc.body, dict) else {}
    return (body.get("error") or {}).get("code") == "tool_use_failed"

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
    retry=retry_if_exception_type(_RETRYABLE_GROQ_ERRORS) | retry_if_exception(_is_tool_use_failed),
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
        # The raw provider exception (e.g. a Groq API error body) is logged here, in
        # full, for debugging — but never becomes the message on `LLMAnalysisError`
        # itself. Every caller that surfaces `str(exc)` to a user (a failed `AgentResult`,
        # a `notes` entry) does so via this one exception type, so keeping its message
        # human-safe here is what keeps it human-safe everywhere downstream.
        log_event(logger, "structured analysis call failed", error=str(exc))
        if is_rate_limited(exc):
            raise LLMAnalysisError(RATE_LIMIT_MESSAGE) from exc
        raise LLMAnalysisError("The analysis service is temporarily unavailable.") from exc


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def node_result(ticker: str, agent: AgentName, result: AgentResult) -> dict:
    """Every specialist node's return value, in one place — wraps an `AgentResult` into
    the state update and stamps when this (ticker, agent) attempt completed, success or
    failure alike. That timestamp (`per_ticker_fetched_at`, see state.py) is what lets a
    follow-up's "answer from context" path tell fresh results from stale ones instead of
    trusting a single classification LLM call's guess about whether a refresh is needed.
    """
    return {
        "per_ticker_results": {ticker: {agent: result}},
        "per_ticker_fetched_at": {ticker: {agent: now_iso()}},
    }


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
