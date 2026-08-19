from __future__ import annotations

import logging

from app.graph.nodes._synthesis_shared import (
    AGENT_LABELS,
    AGENTS,
    citation_instruction,
    collect_findings,
    section_text,
    sources_section,
    ticker_all_failed,
)
from app.graph.state import ResearchState
from app.llm.groq_client import get_chat_model
from app.logging_config import get_logger, log_event

logger = get_logger("app.graph.nodes.synthesis_single")


async def synthesis_single_node(state: ResearchState) -> dict:
    ticker = state["tickers"][0]
    ticker_results = state["per_ticker_results"].get(ticker, {})

    if ticker_all_failed(ticker_results):
        errors = "; ".join(
            f"{AGENT_LABELS[a]}: {ticker_results.get(a, {}).get('error', 'not run')}"  # type: ignore[union-attr]
            for a in AGENTS
        )
        # A plain reply, not a "Research Report" card — every source failed, there's
        # nothing to report.
        reply = f"I wasn't able to complete research for {ticker} — every data source failed. {errors}"
        log_event(logger, "synthesis: all agents failed", session_id=state["session_id"], ticker=ticker)
        return {"followup_answer": reply, "conversation_history": [{"role": "assistant", "content": reply}]}

    sections = "\n\n".join(section_text(agent, ticker_results.get(agent)) for agent in AGENTS)
    all_findings = collect_findings(ticker_results)

    prompt = (
        f"You are writing a concise research report for {ticker} for a retail investor "
        "who wants an informed overview and a clear take. Below are findings from three "
        "independent analysts (fundamentals, technical, news/sentiment). Write a cohesive "
        "3-5 paragraph report combining their perspectives into one narrative. "
        f"{citation_instruction(all_findings)} If a section below says 'Unavailable', "
        "explicitly mention that gap in the report rather than omitting it silently. End "
        "with a line starting exactly 'Verdict: ' followed by Buy, Sell, or Hold and a "
        "one-sentence rationale grounded only in the findings above, plus a confidence "
        "qualifier (low/medium/high) based on how complete the underlying data was. Note "
        "that this is not personalized financial advice.\n\n"
        f"{sections}"
    )

    try:
        llm = get_chat_model(temperature=0.3)
        response = await llm.ainvoke(prompt)
        body = response.content
    except Exception as exc:
        log_event(
            logger, "synthesis LLM call failed", level=logging.ERROR,
            session_id=state["session_id"], ticker=ticker, error=str(exc),
        )
        report = f"# Research Report: {ticker}\n\n{sections}{sources_section(all_findings)}"
        return {"final_report": report, "conversation_history": [{"role": "assistant", "content": report}]}

    report = f"# Research Report: {ticker}\n\n{body}{sources_section(all_findings)}"
    log_event(
        logger, "synthesis completed", session_id=state["session_id"],
        ticker=ticker, finding_count=len(all_findings),
    )
    return {"final_report": report, "conversation_history": [{"role": "assistant", "content": report}]}
