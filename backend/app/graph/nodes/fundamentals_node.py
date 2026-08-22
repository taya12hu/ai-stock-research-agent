from __future__ import annotations

from app.graph.nodes._shared import (
    build_findings,
    node_failed,
    node_ok,
    run_structured_analysis,
    target_ticker,
)
from app.graph.session import SessionState
from app.graph.state import Finding, Source
from app.llm.errors import LLMAnalysisError
from app.logging_config import get_logger, log_event
from app.tools.errors import MarketDataError
from app.tools.market_data import FundamentalsData, aget_fundamentals

logger = get_logger("app.graph.nodes.fundamentals")

# Below this many populated fields, the data is too thin to write a real summary from —
# a forced-tool-call LLM asked to analyze near-nothing tends to refuse the tool call
# entirely rather than say so gracefully, which surfaces as a raw provider API error.
# Give up deterministically instead, the same way `synthesis_single` does when every
# agent failed outright.
MIN_FACTS_FOR_ANALYSIS = 3


def _facts(data: FundamentalsData) -> list[tuple[str, object]]:
    return [
        (label, value)
        for label, value in [
            ("Company", data.name),
            ("Sector", data.sector),
            ("Industry", data.industry),
            ("Current price (USD)", data.current_price),
            ("Market cap (USD)", data.market_cap),
            ("Trailing P/E", data.trailing_pe),
            ("Forward P/E", data.forward_pe),
            ("Price-to-book", data.price_to_book),
            ("Dividend yield", data.dividend_yield),
            ("Profit margin", data.profit_margin),
            ("Revenue growth (YoY)", data.revenue_growth),
            ("Earnings growth (YoY)", data.earnings_growth),
            ("Return on equity", data.return_on_equity),
            ("Total debt (USD)", data.total_debt),
            ("Total cash (USD)", data.total_cash),
            ("Analyst recommendation", data.recommendation),
        ]
        if value is not None
    ]


def _build_prompt(data: FundamentalsData, facts: list[tuple[str, object]]) -> str:
    facts_text = "\n".join(f"- {label}: {value}" for label, value in facts)
    return (
        "You are a fundamentals research analyst. Based ONLY on the data below for "
        f"{data.ticker}, write a short summary of the company's financial health and "
        "list specific findings. Every finding's evidence must be a number or fact "
        "literally present in the data — do not invent figures, and do not give "
        "investment advice (buy/sell/hold recommendations of your own).\n\n"
        f"Data (as of {data.as_of}):\n{facts_text}"
    )


async def fundamentals_node(state: SessionState) -> dict:
    ticker = target_ticker(state)

    try:
        data = await aget_fundamentals(ticker)
    except MarketDataError as exc:
        log_event(logger, "fundamentals data fetch failed", session_id=state["session_id"], ticker=ticker, error=str(exc))
        return node_failed(ticker, "fundamentals", str(exc))

    facts = _facts(data)
    if len(facts) < MIN_FACTS_FOR_ANALYSIS:
        log_event(
            logger, "fundamentals: too little data to analyze", session_id=state["session_id"],
            ticker=ticker, fact_count=len(facts),
        )
        return node_failed(
            ticker, "fundamentals",
            f"Not enough fundamentals data was available for {ticker} to analyze.",
        )

    try:
        analysis = await run_structured_analysis(_build_prompt(data, facts))
    except LLMAnalysisError as exc:
        log_event(logger, "fundamentals analysis failed", session_id=state["session_id"], ticker=ticker, error=str(exc))
        return node_failed(ticker, "fundamentals", str(exc))

    source = Source(
        type="market_data",
        label=f"{ticker} fundamentals (Finnhub)",
        # A verification reference, not the fetch URL: Finnhub's API is token-authed
        # and has no public per-ticker page, so there is nothing to link to directly.
        # The label names the real provider; this gives the reader somewhere to check
        # the same figures. Swap or drop it if that distinction ever needs to be
        # sharper than the label alone makes it.
        url=f"https://finance.yahoo.com/quote/{ticker}",
        as_of=data.as_of,
    )
    findings: list[Finding] = build_findings(ticker, "fundamentals", analysis, source)

    log_event(
        logger, "fundamentals node completed", session_id=state["session_id"],
        ticker=ticker, finding_count=len(findings),
    )
    return node_ok(ticker, "fundamentals", analysis.summary, findings)
