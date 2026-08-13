"""Exceptions raised by the tools layer (external data source failures).

Agent nodes (built in a later phase) are expected to catch these, never let them
propagate, and turn them into a failed `AgentResult` instead — see ARCHITECTURE.md §10.
"""

from __future__ import annotations


class ToolError(Exception):
    """Base class for tool-layer errors."""


class YahooFinanceError(ToolError):
    """Raised when fundamentals/price data can't be fetched or parsed for a ticker."""


class WebSearchError(ToolError):
    """Raised when a web search fails after retries (not raised for zero results)."""
