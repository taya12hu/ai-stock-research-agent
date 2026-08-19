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

logger = get_logger("app.graph.nodes.synthesis_comparison")


async def synthesis_comparison_node(state: ResearchState) -> dict:
    tickers = state["tickers"]
    per_ticker = state["per_ticker_results"]

    usable = [t for t in tickers if not ticker_all_failed(per_ticker.get(t, {}))]
    failed = [t for t in tickers if t not in usable]

    if len(usable) < 2:
        # A plain reply, not a "Comparison" report card — fewer than two tickers came
        # back usable, so there's no comparison to show.
        reply = (
            "I wasn't able to complete this comparison — usable data was available for "
            f"fewer than two of the requested tickers. Failed: {', '.join(failed) or 'n/a'}."
        )
        log_event(
            logger, "comparison synthesis: insufficient usable tickers",
            session_id=state["session_id"], tickers=tickers,
        )
        return {"followup_answer": reply, "conversation_history": [{"role": "assistant", "content": reply}]}

    ticker_blocks = "\n\n".join(ticker_section_block(t, per_ticker.get(t, {})) for t in tickers)
    all_findings = [f for t in tickers for f in collect_findings(per_ticker.get(t, {}))]
    failed_note = (
        f" Note: no usable data was found for {', '.join(failed)}; excluded from this comparison."
        if failed else ""
    )

    prompt = (
        f"You are writing a structured comparison of {len(usable)} stocks: "
        f"{', '.join(usable)}. Below, for each stock, are findings from three "
        "independent analysts (fundamentals, technical, news/sentiment). Write a genuine "
        "structured comparison — not three separate reports stapled together — covering "
        "relative valuation, relative momentum/technicals, and relative sentiment across "
        f"the stocks. {citation_instruction(all_findings)}{failed_note} End with a "
        "'Verdict' section giving one line per stock in the form 'TICKER: Buy/Sell/Hold — "
        "one-sentence rationale', grounded only in the findings above, followed by a "
        "sentence naming which stock the data favors overall (if any) and why. Note that "
        "this is not personalized financial advice.\n\n"
        f"{ticker_blocks}"
    )

    try:
        llm = get_chat_model(temperature=0.3)
        response = await llm.ainvoke(prompt)
        body = response.content
    except Exception as exc:
        log_event(
            logger, "comparison synthesis LLM call failed", level=logging.ERROR,
            session_id=state["session_id"], tickers=tickers, error=str(exc),
        )
        report = f"# Comparison: {' vs '.join(tickers)}\n\n{ticker_blocks}{sources_section(all_findings)}"
        return {"final_report": report, "conversation_history": [{"role": "assistant", "content": report}]}

    report = f"# Comparison: {' vs '.join(tickers)}\n\n{body}{sources_section(all_findings)}"
    log_event(
        logger, "comparison synthesis completed", session_id=state["session_id"],
        tickers=tickers, finding_count=len(all_findings),
    )
    return {"final_report": report}
