from __future__ import annotations

from app.graph.nodes._shared import build_findings, run_structured_analysis
from app.graph.state import Finding, ResearchState, Source, failed_result, ok_result
from app.llm.errors import LLMAnalysisError
from app.logging_config import get_logger, log_event
from app.tools.errors import YahooFinanceError
from app.tools.yahoo_finance import TechnicalData, aget_technical_data

logger = get_logger("app.graph.nodes.technical")

# See MIN_FACTS_FOR_ANALYSIS in fundamentals_node.py for why this guard exists: a
# forced-tool-call LLM asked to analyze near-empty data tends to refuse the tool call
# outright rather than say so gracefully, which surfaces as a raw provider API error.
MIN_FACTS_FOR_ANALYSIS = 2


def _facts(data: TechnicalData) -> list[tuple[str, object]]:
    return [
        (label, value)
        for label, value in [
            ("Last close (USD)", data.last_close),
            ("20-day SMA", data.sma_20),
            ("50-day SMA", data.sma_50),
            ("200-day SMA", data.sma_200),
            ("RSI (14-day)", data.rsi_14),
            ("MACD", data.macd),
            ("1-month momentum (%)", data.momentum_1m_pct),
            ("Annualized volatility (%)", data.volatility_annualized_pct),
            ("52-week high (USD)", data.fifty_two_week_high),
            ("52-week low (USD)", data.fifty_two_week_low),
        ]
        if value is not None
    ]


def _build_prompt(data: TechnicalData, facts: list[tuple[str, object]]) -> str:
    facts_text = "\n".join(f"- {label}: {value}" for label, value in facts)
    return (
        "You are a technical analyst. Based ONLY on the price/indicator data below for "
        f"{data.ticker}, write a short summary of the stock's price trend and momentum, "
        "and list specific findings (e.g. what the RSI level or moving-average "
        "relationship suggests). Every finding's evidence must be a number literally "
        "present in the data — do not invent figures, and do not give investment "
        "advice (buy/sell/hold recommendations of your own).\n\n"
        f"Data (as of {data.as_of}):\n{facts_text}"
    )


async def technical_node(state: ResearchState) -> dict:
    ticker = state["tickers"][0]

    try:
        data = await aget_technical_data(ticker)
    except YahooFinanceError as exc:
        log_event(logger, "technical data fetch failed", session_id=state["session_id"], ticker=ticker, error=str(exc))
        return {"per_ticker_results": {ticker: {"technical": failed_result(str(exc))}}}

    facts = _facts(data)
    if len(facts) < MIN_FACTS_FOR_ANALYSIS:
        log_event(
            logger, "technical: too little data to analyze", session_id=state["session_id"],
            ticker=ticker, fact_count=len(facts),
        )
        return {
            "per_ticker_results": {
                ticker: {"technical": failed_result(f"Not enough price/indicator data was available for {ticker} to analyze.")}
            }
        }

    try:
        analysis = await run_structured_analysis(_build_prompt(data, facts))
    except LLMAnalysisError as exc:
        log_event(logger, "technical analysis failed", session_id=state["session_id"], ticker=ticker, error=str(exc))
        return {"per_ticker_results": {ticker: {"technical": failed_result(str(exc))}}}

    source = Source(
        type="yahoo_finance",
        label=f"{ticker} price history (Yahoo Finance)",
        url=None,
        as_of=data.as_of,
    )
    findings: list[Finding] = build_findings(ticker, "technical", analysis, source)

    log_event(
        logger, "technical node completed", session_id=state["session_id"],
        ticker=ticker, finding_count=len(findings),
    )
    return {"per_ticker_results": {ticker: {"technical": ok_result(analysis.summary, findings)}}}
