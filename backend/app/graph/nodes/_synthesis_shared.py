"""Helpers shared by the three type-aware synthesis nodes (single/portfolio/comparison).

All three follow the same shape: render each ticker's per-agent sections, collect its
findings, build a citation instruction for the LLM, and append a deterministic sources
list at the end (so every citable id is guaranteed to actually be listed — see
ARCHITECTURE.md §11 on citation integrity).
"""

from __future__ import annotations

from app.graph.state import AGENT_NAMES, AgentResult, Finding, TickerResults

AGENT_LABELS = {
    "fundamentals": "Fundamentals",
    "technical": "Technical",
    "news": "News & Sentiment",
}
AGENTS = AGENT_NAMES


def collect_findings(ticker_results: TickerResults) -> list[Finding]:
    findings: list[Finding] = []
    for agent in AGENTS:
        result = ticker_results.get(agent)
        if result and result["status"] == "ok":
            findings.extend(result["findings"])
    return findings


def section_text(agent: str, result: AgentResult | None) -> str:
    label = AGENT_LABELS[agent]
    if result is None:
        return f"### {label}\n(not run)"
    if result["status"] == "failed":
        return f"### {label}\nUnavailable — {result['error']}"
    lines = [f"### {label}", result["summary"], ""]
    lines.extend(f"- [{f['id']}] {f['claim']} — {f['evidence']}" for f in result["findings"])
    return "\n".join(lines)


def ticker_section_block(ticker: str, ticker_results: TickerResults) -> str:
    body = "\n\n".join(section_text(agent, ticker_results.get(agent)) for agent in AGENTS)
    return f"## {ticker}\n\n{body}"


def ticker_all_failed(ticker_results: TickerResults) -> bool:
    return all(
        ticker_results.get(agent, {}).get("status") != "ok"  # type: ignore[union-attr]
        for agent in AGENTS
    )


def sources_section(findings: list[Finding]) -> str:
    """One line per *unique* source, not per finding. Fundamentals/technical findings
    for a given ticker all cite the same single Yahoo Finance fetch (the node builds one
    `Source` and reuses it across every finding it produces — see fundamentals_node.py/
    technical_node.py), so without deduping this reads as 5 independent fundamentals
    sources and 5 independent technical sources when it's really one of each; only news
    findings genuinely cite distinct URLs. Groups by the source's own identity (label,
    url, as_of) rather than by agent/ticker, since that's what's actually shared or not —
    no assumption needed about which agents happen to reuse a source.
    """
    if not findings:
        return ""
    grouped: dict[tuple[str, str | None, str], list[str]] = {}
    for f in findings:
        src = f["source"]
        key = (src["label"], src.get("url"), src["as_of"])
        grouped.setdefault(key, []).append(f["id"])

    lines = ["", "", "**Sources**"]
    for (label, url, as_of), ids in grouped.items():
        url_part = f" — {url}" if url else ""
        lines.append(f"- [{', '.join(ids)}] {label}{url_part} (as of {as_of})")
    return "\n".join(lines)


def citation_instruction(findings: list[Finding]) -> str:
    if not findings:
        return "No structured findings are available; write from the summaries only."
    finding_ids = ", ".join(f["id"] for f in findings)
    return (
        "When you state a specific fact, cite it inline using its bracket id exactly as "
        f"given (e.g. [{findings[0]['id']}]) — only use ids from this exact list: {finding_ids}. "
        "Square brackets are reserved for these citation ids only — never use them for anything "
        "else (e.g. don't write \"[Unavailable]\" or \"[N/A]\" to mark missing data; say it in "
        "plain prose instead, like \"technical data is unavailable\")."
    )
