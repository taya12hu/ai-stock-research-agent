from __future__ import annotations

import groq
import httpx
import pytest

import app.graph.nodes._shared as shared_mod
from app.graph.nodes._shared import NodeAnalysis, run_structured_analysis
from app.llm.errors import RATE_LIMIT_MESSAGE, LLMAnalysisError


def _bad_request_error(code: str) -> groq.BadRequestError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(400, request=request)
    body = {"error": {"code": code, "message": "boom"}}
    return groq.BadRequestError("bad request", response=response, body=body)


def _rate_limit_error() -> groq.RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, request=request)
    body = {"error": {"code": "rate_limit_exceeded", "message": "Rate limit reached ... tokens per day (TPD)"}}
    return groq.RateLimitError("rate limited", response=response, body=body)


# The doubles expose `ainvoke`, not `invoke`, because that is what the specialists call.
# The synchronous form blocked the event loop and serialized the parallel `Send` branches;
# a double that only implements `invoke` would let that regress silently.
class _RateLimitedLLM:
    async def ainvoke(self, prompt: str) -> NodeAnalysis:  # noqa: ARG002
        raise _rate_limit_error()


class _FakeStructuredLLM:
    def __init__(self, calls_before_success: int, error_code: str) -> None:
        self._calls_before_success = calls_before_success
        self._error_code = error_code
        self.call_count = 0

    async def ainvoke(self, prompt: str) -> NodeAnalysis:  # noqa: ARG002
        self.call_count += 1
        if self.call_count <= self._calls_before_success:
            raise _bad_request_error(self._error_code)
        return NodeAnalysis(summary="ok", findings=[])


class _FakeChatModel:
    def __init__(self, structured: _FakeStructuredLLM) -> None:
        self._structured = structured

    def with_structured_output(self, schema: type) -> _FakeStructuredLLM:  # noqa: ARG002
        return self._structured


async def test_tool_use_failed_is_retried_and_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression guard for a real observed failure: Groq's `failed_generation` showed
    the model HAD produced a complete, well-grounded analysis, it just answered in plain
    text instead of calling the required tool. That's sampling variance, not a
    persistent block — a bare retry with the same prompt should succeed."""
    structured = _FakeStructuredLLM(calls_before_success=1, error_code="tool_use_failed")
    monkeypatch.setattr(shared_mod, "get_chat_model", lambda: _FakeChatModel(structured))

    result = await run_structured_analysis("prompt")

    assert result.summary == "ok"
    assert structured.call_count == 2  # failed once, succeeded on retry


async def test_other_bad_request_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A genuinely malformed request would fail identically every time — only the
    specific 'model didn't call the tool' case is worth retrying."""
    structured = _FakeStructuredLLM(calls_before_success=99, error_code="invalid_request")
    monkeypatch.setattr(shared_mod, "get_chat_model", lambda: _FakeChatModel(structured))

    with pytest.raises(LLMAnalysisError):
        await run_structured_analysis("prompt")

    assert structured.call_count == 1


async def test_exhausted_retries_produce_clean_message_not_raw_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even when every retry attempt fails, the exception shown to the user must never
    contain the raw provider error body."""
    structured = _FakeStructuredLLM(calls_before_success=99, error_code="tool_use_failed")
    monkeypatch.setattr(shared_mod, "get_chat_model", lambda: _FakeChatModel(structured))

    with pytest.raises(LLMAnalysisError) as exc_info:
        await run_structured_analysis("prompt")

    assert str(exc_info.value) == "The analysis service is temporarily unavailable."
    assert "tool_use_failed" not in str(exc_info.value)


async def test_rate_limit_produces_specific_friendly_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """A genuinely exhausted quota deserves a distinct, honest message — not the generic
    'temporarily unavailable' fallback, and never the raw provider error body."""
    monkeypatch.setattr(shared_mod, "get_chat_model", lambda: _FakeChatModel(_RateLimitedLLM()))

    with pytest.raises(LLMAnalysisError) as exc_info:
        await run_structured_analysis("prompt")

    assert str(exc_info.value) == RATE_LIMIT_MESSAGE
    assert "rate_limit_exceeded" not in str(exc_info.value)
    assert "tokens per day" not in str(exc_info.value)
