"""Rendering helpers shared by the three report shapes.

All three follow the same structure: render each ticker's per-agent sections, collect its
findings, build a citation instruction from the ids that actually exist, and append a
deterministic sources list — so every citable id is guaranteed to be listed, which is what
the eval harness's citation-integrity check verifies mechanically.
"""

from __future__ import annotations

from app.graph.session import AgentName, TickerCell
from app.graph.state import AGENT_NAMES, Finding

AGENT_LABELS: dict[str, str] = {
    "fundamentals": "Fundamentals",
    "technical": "Technical",
    "news": "News & Sentiment",
}
AGENTS = AGENT_NAMES

Cells = dict[AgentName, TickerCell]

# Appended to every prompt whose output reaches the user directly. Models default to em
# dashes for parenthetical asides at a rate that makes reports read as machine-written,
# and there is no post-processing fix: stripping them after the fact either mangles the
# verdict separator (see the frontend's Markdown.tsx) or leaves sentences that no longer
# parse. Asking for the punctuation up front is the only place this can be fixed cleanly.
PROSE_STYLE = (
    " Write in plain prose: use commas, colons and full stops for asides. Do not use em "
    "dashes or en dashes (— or –) anywhere in your answer. Write citation ids in plain "
    "ASCII square brackets like [id]; never use full-width or decorative brackets."
)


def collect_findings(cells: Cells, aspects: list[AgentName] | None = None) -> list[Finding]:
    findings: list[Finding] = []
    for agent in aspects or AGENTS:
        cell = cells.get(agent)
        if cell and cell["status"] == "ok":
            findings.extend(cell["findings"])
    return findings


def section_text(agent: AgentName, cell: TickerCell | None) -> str:
    label = AGENT_LABELS[agent]
    if cell is None:
        return f"### {label}\n(not run)"
    if cell["status"] == "failed":
        return f"### {label}\nUnavailable. {cell['error']}"
    if not cell["findings"]:
        # Ran cleanly, found nothing. Said plainly rather than rendered as an empty
        # section, which reads as though the data were merely omitted.
        return f"### {label}\n{cell['summary']}"
    lines = [f"### {label}", cell["summary"], ""]
    lines.extend(f"- [{f['id']}] {f['claim']} · {f['evidence']}" for f in cell["findings"])
    return "\n".join(lines)


def ticker_section_block(
    ticker: str, cells: Cells, aspects: list[AgentName] | None = None
) -> str:
    """Only the aspects this turn covers.

    A question narrowed to one analysis ("how are Apple's fundamentals?") renders a
    fundamentals answer — not a full report with two "(not run)" sections, which would
    misrepresent a deliberately scoped question as a failed complete one.
    """
    shown = aspects or list(AGENTS)
    body = "\n\n".join(section_text(agent, cells.get(agent)) for agent in shown)
    return f"## {ticker}\n\n{body}"


def sources_section(findings: list[Finding]) -> str:
    """One line per *unique* source, not per finding.

    Fundamentals and technical findings for a given ticker all cite the same single fetch —
    the node builds one `Source` and reuses it across every finding it produces — so
    without deduping this reads as five independent fundamentals sources when it is really
    one. Groups by the source's own identity (label, url, as_of) rather than by agent or
    ticker, since that is what is actually shared; only news findings genuinely cite
    distinct URLs.
    """
    if not findings:
        return ""
    grouped: dict[tuple[str, str | None, str], list[str]] = {}
    for finding in findings:
        source = finding["source"]
        key = (source["label"], source.get("url"), source["as_of"])
        grouped.setdefault(key, []).append(finding["id"])

    lines = ["", "", "**Sources**"]
    for (label, url, as_of), ids in grouped.items():
        url_part = f" · {url}" if url else ""
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
