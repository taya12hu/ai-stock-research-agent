from __future__ import annotations

import logging

from app.graph.nodes._synthesis_shared import (
    citation_instruction,
    collect_findings,
    sources_section,
    ticker_all_failed,
    ticker_section_block,
)
from app.graph.state import ResearchState
from app.llm.groq_client import get_chat_model
from app.logging_config import get_logger, log_event

logger = get_logger("app.graph.nodes.synthesis_portfolio")


async def synthesis_portfolio_node(state: ResearchState) -> dict:
    tickers = state["tickers"]
    per_ticker = state["per_ticker_results"]

    usable = [t for t in tickers if not ticker_all_failed(per_ticker.get(t, {}))]
    failed = [t for t in tickers if t not in usable]

    if not usable:
        report = (
            "# Portfolio Research Report\n\n"
            "Unable to complete research — every holding failed to return data: "
            f"{', '.join(failed)}."
        )
        log_event(logger, "portfolio synthesis: all tickers failed", session_id=state["session_id"], tickers=tickers)
        return {"final_report": report, "conversation_history": [{"role": "assistant", "content": report}]}

    ticker_blocks = "\n\n".join(ticker_section_block(t, per_ticker.get(t, {})) for t in tickers)
    all_findings = [f for t in tickers for f in collect_findings(per_ticker.get(t, {}))]
    failed_note = (
        f" Note: no usable data was found for {', '.join(failed)}; excluded from this analysis."
        if failed else ""
    )

    prompt = (
        f"You are writing a portfolio research report for a holding of {len(usable)} "
        f"stocks: {', '.join(usable)}. Below, for each stock, are findings from three "
        "independent analysts (fundamentals, technical, news/sentiment). Write a report "
        "with: (1) a brief per-stock summary for each holding ending in a 'Verdict: "
        "Buy/Sell/Hold' line with a one-sentence rationale grounded only in the findings "
        "above, and (2) a portfolio-level section noting any sector concentration or "
        "overlap across the holdings (purely qualitative — do not compute correlations, "
        f"optimal weights, or other quantitative portfolio math). "
        f"{citation_instruction(all_findings)}{failed_note} If a section says "
        "'Unavailable', mention that gap rather than omitting it. Note that this is not "
        "personalized financial advice.\n\n"
        f"{ticker_blocks}"
    )

    try:
        llm = get_chat_model(temperature=0.3)
        response = await llm.ainvoke(prompt)
        body = response.content
    except Exception as exc:
        log_event(
            logger, "portfolio synthesis LLM call failed", level=logging.ERROR,
            session_id=state["session_id"], tickers=tickers, error=str(exc),
        )
        report = f"# Portfolio Research Report\n\n{ticker_blocks}{sources_section(all_findings)}"
        return {"final_report": report, "conversation_history": [{"role": "assistant", "content": report}]}

    report = f"# Portfolio Research Report\n\n{body}{sources_section(all_findings)}"
    log_event(
        logger, "portfolio synthesis completed", session_id=state["session_id"],
        tickers=tickers, finding_count=len(all_findings),
    )
    return {"final_report": report}
