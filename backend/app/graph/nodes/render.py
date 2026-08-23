"""Turns this turn's cells into a report. Replaces `synthesis_single`,
`synthesis_portfolio` and `synthesis_comparison`.

The three shapes genuinely need different prompts, but they shared 80% of their structure
and — crucially — each read `state["tickers"]` and `state["query_type"]` directly. That is
what made A-01 possible: a refresh scoped to one ticker still dispatched off the session's
persisted `query_type` and re-rendered every ticker in the session. Here scope and shape
arrive as arguments taken from `turn`, and there is no session-level field left to read by
accident.

Usability is `status == "ok" AND findings`, not `status == "ok"`. The old
`ticker_all_failed` tested the weaker condition, so a ticker whose fundamentals and
technicals had failed and whose news search legitimately found no articles counted as
usable — and a full report, mandatory verdict line included, was written from two error
strings and a "no news found" (A-10).
"""

from __future__ import annotations

import logging

from app.graph.freshness import is_usable
from app.graph.nodes._synthesis_shared import (
    AGENT_LABELS,
    Cells,
    citation_instruction,
    collect_findings,
    sources_section,
    ticker_section_block,
)
from app.graph.session import AgentName, SessionState, Shape, TurnOutput
from app.llm.groq_client import get_chat_model
from app.logging_config import get_logger, log_event
from app.replies import join_human

logger = get_logger("app.graph.nodes.render")

_NOT_ADVICE = "Note that this is not personalized financial advice."

# How many usable tickers each shape needs before it can be written at all. A comparison
# of one company is not a comparison; a single-stock report of zero is not a report.
_MINIMUM_USABLE: dict[Shape, int] = {"single": 1, "comparison": 2, "portfolio": 1}


def _usable_tickers(scope: list[str], researched: dict, aspects: list[AgentName]) -> list[str]:
    return [
        ticker
        for ticker in scope
        if any(is_usable((researched.get(ticker) or {}).get(agent)) for agent in aspects)
    ]


def _failure_reply(scope: list[str], usable: list[str], researched: dict, shape: Shape) -> str:
    """A plain reply, never a report shell.

    Nothing was actually researched here, so a report-styled card would present a failure
    as a completed deliverable — the same principle the old `_no_tickers_node` applied one
    step earlier in the graph.
    """
    failed = [t for t in scope if t not in usable]
    if shape == "comparison" and usable:
        return (
            "I couldn't complete this comparison — usable data came back for fewer than "
            f"two of the requested companies. No usable data for: {join_human(failed)}."
        )
    details = []
    for ticker in failed:
        cells = researched.get(ticker) or {}
        reasons = "; ".join(
            f"{AGENT_LABELS[agent]}: {cell.get('error') or 'no usable data'}"
            for agent, cell in cells.items()
        )
        details.append(f"{ticker} ({reasons})" if reasons else ticker)
    return (
        "I wasn't able to complete this research — every data source failed for "
        f"{join_human(details) or join_human(failed)}."
    )


def _prompt(shape: Shape, scope: list[str], usable: list[str], blocks: str, findings: list) -> str:
    citations = citation_instruction(findings)
    excluded = [t for t in scope if t not in usable]
    gap_note = (
        f" No usable data was found for {join_human(excluded)}; it is excluded here."
        if excluded
        else ""
    )

    if shape == "single":
        return (
            f"You are writing a concise research report for {scope[0]} for a retail "
            "investor who wants an informed overview and a clear take. Below are findings "
            "from independent analysts. Write a cohesive 3-5 paragraph report combining "
            f"their perspectives into one narrative. {citations}{gap_note} If a section "
            "below says 'Unavailable', mention that gap rather than omitting it silently. "
            "End with a line starting exactly 'Verdict: ' followed by Buy, Sell or Hold, a "
            "one-sentence rationale grounded only in the findings above, and a confidence "
            f"qualifier (low/medium/high) reflecting how complete the data was. {_NOT_ADVICE}\n\n"
            f"{blocks}"
        )

    if shape == "portfolio":
        return (
            f"You are writing a portfolio research report for a holding of {len(usable)} "
            f"stocks: {join_human(usable)}. Below, for each stock, are findings from "
            "independent analysts. Write (1) a brief per-stock summary ending in a "
            "'Verdict: Buy/Sell/Hold' line with a one-sentence rationale grounded only in "
            "those findings, and (2) a portfolio-level section noting any sector "
            "concentration or overlap across the holdings — purely qualitative, do not "
            "compute correlations, optimal weights, or other quantitative portfolio math. "
            f"{citations}{gap_note} If a section says 'Unavailable', mention that gap "
            f"rather than omitting it. {_NOT_ADVICE}\n\n{blocks}"
        )

    return (
        f"You are writing a structured comparison of {len(usable)} stocks: "
        f"{join_human(usable)}. Below, for each stock, are findings from independent "
        "analysts. Write a genuine structured comparison — not separate reports stapled "
        "together — covering relative valuation, relative momentum/technicals and relative "
        f"sentiment across the stocks. {citations}{gap_note} End with a 'Verdict' section "
        "giving one line per stock as 'TICKER: Buy/Sell/Hold — one-sentence rationale', "
        "grounded only in the findings above, then a sentence naming which stock the data "
        f"favours overall (if any) and why. {_NOT_ADVICE}\n\n{blocks}"
    )


def _title(shape: Shape, scope: list[str]) -> str:
    if shape == "single":
        return f"# Research Report: {scope[0]}"
    if shape == "portfolio":
        return "# Portfolio Research Report"
    return f"# Comparison: {' vs '.join(scope)}"


async def render_node(state: SessionState) -> dict:
    turn = state["turn"]
    scope: list[str] = turn["scope"]
    shape: Shape = turn["shape"]
    aspects: list[AgentName] = turn["aspects"]
    researched = state.get("researched") or {}

    usable = _usable_tickers(scope, researched, aspects)
    if len(usable) < _MINIMUM_USABLE[shape]:
        log_event(
            logger, "render: insufficient usable data", session_id=state["session_id"],
            scope=scope, shape=shape, usable=usable,
        )
        reply = _failure_reply(scope, usable, researched, shape)
        return {"turn": {**turn, "output": TurnOutput(kind="answer", text=reply)}}

    cells_by_ticker: dict[str, Cells] = {t: (researched.get(t) or {}) for t in scope}
    blocks = "\n\n".join(
        ticker_section_block(t, cells_by_ticker[t], aspects) for t in scope
    )
    findings = [f for t in scope for f in collect_findings(cells_by_ticker[t], aspects)]

    try:
        llm = get_chat_model(temperature=0.3)
        response = await llm.ainvoke(_prompt(shape, scope, usable, blocks, findings))
        body = response.content
    except Exception as exc:
        # Deterministic fallback: the rendered sections are already a usable, fully-cited
        # document — losing the narrative is far better than losing the research.
        log_event(
            logger, "render LLM call failed", level=logging.ERROR,
            session_id=state["session_id"], scope=scope, error=str(exc),
        )
        body = blocks

    report = f"{_title(shape, scope)}\n\n{body}{sources_section(findings)}"
    log_event(
        logger, "render completed", session_id=state["session_id"],
        scope=scope, shape=shape, finding_count=len(findings),
    )
    return {"turn": {**turn, "output": TurnOutput(kind="report", text=report)}}
