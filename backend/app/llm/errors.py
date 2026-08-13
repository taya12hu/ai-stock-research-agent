from __future__ import annotations


class LLMAnalysisError(Exception):
    """Raised when an LLM analysis call fails after retries."""
