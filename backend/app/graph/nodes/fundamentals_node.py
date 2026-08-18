from __future__ import annotations

from app.graph.nodes._shared import build_findings, run_structured_analysis
from app.graph.state import Finding, ResearchState, Source, failed_result, ok_result
from app.llm.errors import LLMAnalysisError
from app.logging_config import get_logger, log_event
from app.tools.errors import YahooFinanceError
from app.tools.yahoo_finance import FundamentalsData, aget_fundamentals

logger = get_logger("app.graph.nodes.fundamentals")

_METRIC_FIELDS = (
    "market_cap", "trailing_pe", "forward_pe", "price_to_book", "profit_margin",
    "revenue_growth", "earnings_growth", "return_on_equity", "total_debt", "total_cash",
)


def _has_usable_metrics(data: FundamentalsData) -> bool:
    return any(getattr(data, field) is not None for field in _METRIC_FIELDS)


def _build_prompt(data: FundamentalsData) -> str:
    facts = "\n".join(
        f"- {label}: {value}"
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
    )
    return (
        "You are a fundamentals research analyst. Based ONLY on the data below for "
        f"{data.ticker}, write a short summary of the company's financial health and "
        "list specific findings. Every finding's evidence must be a number or fact "
        "literally present in the data — do not invent figures, and do not give "
        "investment advice (buy/sell/hold recommendations of your own).\n\n"
        f"Data (as of {data.as_of}):\n{facts}"
    )


async def fundamentals_node(state: ResearchState) -> dict:
    ticker = state["tickers"][0]

    try:
        data = await aget_fundamentals(ticker)
    except YahooFinanceError as exc:
        log_event(logger, "fundamentals data fetch failed", session_id=state["session_id"], ticker=ticker, error=str(exc))
        return {"per_ticker_results": {ticker: {"fundamentals": failed_result(str(exc))}}}

    if not _has_usable_metrics(data):
        message = f"no usable financial metrics were available for {ticker} (data may be unavailable for this ticker)"
        log_event(logger, "fundamentals data lacked usable metrics", session_id=state["session_id"], ticker=ticker)
        return {"per_ticker_results": {ticker: {"fundamentals": failed_result(message)}}}

    try:
        analysis = await run_structured_analysis(_build_prompt(data))
    except LLMAnalysisError as exc:
        log_event(logger, "fundamentals analysis failed", session_id=state["session_id"], ticker=ticker, error=str(exc))
        return {"per_ticker_results": {ticker: {"fundamentals": failed_result(str(exc))}}}

    source = Source(
        type="yahoo_finance",
        label=f"{ticker} fundamentals (Yahoo Finance)",
        url=None,
        as_of=data.as_of,
    )
    findings: list[Finding] = build_findings(ticker, "fundamentals", analysis, source)

    log_event(
        logger, "fundamentals node completed", session_id=state["session_id"],
        ticker=ticker, finding_count=len(findings),
    )
    return {"per_ticker_results": {ticker: {"fundamentals": ok_result(analysis.summary, findings)}}}
