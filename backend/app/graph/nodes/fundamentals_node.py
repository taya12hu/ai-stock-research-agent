from __future__ import annotations

from app.graph.nodes._shared import build_findings, run_structured_analysis
from app.graph.state import Finding, ResearchState, Source, failed_result, ok_result
from app.llm.errors import LLMAnalysisError
from app.logging_config import get_logger, log_event
from app.tools.errors import YahooFinanceError
from app.tools.yahoo_finance import FundamentalsData, aget_fundamentals

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


async def fundamentals_node(state: ResearchState) -> dict:
    ticker = state["tickers"][0]

    try:
        data = await aget_fundamentals(ticker)
    except YahooFinanceError as exc:
        log_event(logger, "fundamentals data fetch failed", session_id=state["session_id"], ticker=ticker, error=str(exc))
        return {"per_ticker_results": {ticker: {"fundamentals": failed_result(str(exc))}}}

    facts = _facts(data)
    if len(facts) < MIN_FACTS_FOR_ANALYSIS:
        log_event(
            logger, "fundamentals: too little data to analyze", session_id=state["session_id"],
            ticker=ticker, fact_count=len(facts),
        )
        return {
            "per_ticker_results": {
                ticker: {"fundamentals": failed_result(f"Not enough fundamentals data was available for {ticker} to analyze.")}
            }
        }

    try:
        analysis = await run_structured_analysis(_build_prompt(data, facts))
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
