from __future__ import annotations

from pydantic import Field

from app.graph.nodes._shared import MAX_FINDINGS_PER_AGENT, LLMFinding, NodeAnalysis, run_structured_analysis
from app.graph.state import Finding, ResearchState, Source, failed_result, ok_result
from app.llm.errors import LLMAnalysisError
from app.logging_config import get_logger, log_event
from app.tools.errors import WebSearchError
from app.tools.web_search import SearchResult, asearch_news

logger = get_logger("app.graph.nodes.news")


class NewsLLMFinding(LLMFinding):
    article_index: int = Field(
        description="1-based index of the source article this finding is grounded in, "
        "matching the numbered article list provided"
    )


class NewsAnalysis(NodeAnalysis):
    findings: list[NewsLLMFinding] = Field(
        description=f"Up to {MAX_FINDINGS_PER_AGENT} findings, each grounded in one specific article"
    )
    overall_sentiment: str = Field(description="One of: positive, negative, neutral, mixed")


def _build_prompt(ticker: str, results: list[SearchResult]) -> str:
    articles = "\n\n".join(
        f"[{i + 1}] {r.title}\n{r.snippet}\n(source: {r.source or 'web'}, date: {r.date or 'unknown'})"
        for i, r in enumerate(results)
    )
    return (
        "You are a market sentiment analyst. Based ONLY on the numbered news articles "
        f"below about {ticker}, write a short summary of recent sentiment/news, an "
        "overall_sentiment (positive/negative/neutral/mixed), and list specific "
        "findings. For each finding, set article_index to the number of the article it "
        "is grounded in. Do not invent facts not present in the articles, and do not "
        "give investment advice.\n\n"
        f"Articles:\n{articles}"
    )


def _finding_source(article: SearchResult | None) -> Source:
    if article is None:
        return Source(type="web", label="web search", url=None, as_of="unknown")
    return Source(
        type="web",
        label=article.source or article.title,
        url=article.url,
        as_of=article.date or "unknown",
    )


async def news_node(state: ResearchState) -> dict:
    ticker = state["tickers"][0]
    # `ticker` may carry a resolved exchange suffix (e.g. "TCS.NS" — see
    # `aresolve_ticker`); strip it for the search query, since a suffix helps Yahoo
    # Finance disambiguate a symbol but only hurts a general web search's relevance.
    search_term = ticker.split(".")[0]

    try:
        results = await asearch_news(f"{search_term} stock news", max_results=6)
    except WebSearchError as exc:
        log_event(logger, "news search failed", session_id=state["session_id"], ticker=ticker, error=str(exc))
        return {"per_ticker_results": {ticker: {"news": failed_result(str(exc))}}}

    if not results:
        summary = f"No recent news articles were found for {ticker}."
        return {"per_ticker_results": {ticker: {"news": ok_result(summary, [])}}}

    try:
        analysis = await run_structured_analysis(_build_prompt(ticker, results), schema=NewsAnalysis)
    except LLMAnalysisError as exc:
        log_event(logger, "news analysis failed", session_id=state["session_id"], ticker=ticker, error=str(exc))
        return {"per_ticker_results": {ticker: {"news": failed_result(str(exc))}}}

    findings: list[Finding] = []
    for i, f in enumerate(analysis.findings[:MAX_FINDINGS_PER_AGENT]):
        article_idx = f.article_index - 1
        article = results[article_idx] if 0 <= article_idx < len(results) else None
        findings.append(
            Finding(
                id=f"{ticker}-news-{i + 1}",
                claim=f.claim,
                evidence=f.evidence,
                source=_finding_source(article),
            )
        )

    summary = f"{analysis.summary} Overall sentiment: {analysis.overall_sentiment}."
    log_event(
        logger, "news node completed", session_id=state["session_id"],
        ticker=ticker, finding_count=len(findings), article_count=len(results),
    )
    return {"per_ticker_results": {ticker: {"news": ok_result(summary, findings)}}}
