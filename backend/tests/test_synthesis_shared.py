from __future__ import annotations

from app.graph.nodes._synthesis_shared import sources_section
from app.graph.state import Finding, Source


def _finding(id: str, label: str, url: str | None = None, as_of: str = "2026-08-19T00:00:00Z") -> Finding:
    return Finding(
        id=id, claim="claim", evidence="evidence",
        source=Source(type="market_data", label=label, url=url, as_of=as_of),
    )


def test_sources_section_merges_findings_that_share_the_same_source() -> None:
    """Regression guard for a real observed gap: fundamentals_node/technical_node build
    ONE Source and reuse it across every finding they produce, so without deduping this
    read as 5 independent fundamentals sources and 5 independent technical sources when
    it was really one fetch each."""
    findings = [
        _finding("AAPL-fundamentals-1", "AAPL fundamentals (Finnhub)"),
        _finding("AAPL-fundamentals-2", "AAPL fundamentals (Finnhub)"),
        _finding("AAPL-fundamentals-3", "AAPL fundamentals (Finnhub)"),
        _finding("AAPL-technical-1", "AAPL price history (Twelve Data)", as_of="2026-08-18T00:00:00-04:00"),
        _finding("AAPL-technical-2", "AAPL price history (Twelve Data)", as_of="2026-08-18T00:00:00-04:00"),
    ]

    section = sources_section(findings)
    lines = [line for line in section.splitlines() if line.startswith("-")]

    assert len(lines) == 2  # 5 findings, 2 unique sources
    assert "[AAPL-fundamentals-1, AAPL-fundamentals-2, AAPL-fundamentals-3]" in lines[0]
    assert "[AAPL-technical-1, AAPL-technical-2]" in lines[1]


def test_sources_section_keeps_news_findings_separate_when_urls_differ() -> None:
    """The inverse case: news findings genuinely cite distinct articles (different
    urls), so each one is its own source line, not merged away."""
    findings = [
        _finding("AAPL-news-1", "Reuters", url="https://example.com/a"),
        _finding("AAPL-news-2", "Bloomberg", url="https://example.com/b"),
    ]

    section = sources_section(findings)
    lines = [line for line in section.splitlines() if line.startswith("-")]

    assert len(lines) == 2
    assert "example.com/a" in lines[0]
    assert "example.com/b" in lines[1]


def test_sources_section_empty_for_no_findings() -> None:
    assert sources_section([]) == ""
