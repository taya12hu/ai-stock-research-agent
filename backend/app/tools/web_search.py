"""Web search for recent news, via DuckDuckGo (`ddgs`).

Design notes (see ARCHITECTURE.md §6, §10):
- `DDGS().news(...)` raises `DDGSException("No results found.")` when a query has zero
  matches — that is a legitimate outcome (e.g. a thin-news-coverage company), not a
  failure, so it is caught and treated as an empty result list, distinct from real
  errors (timeouts, rate limits) which are retried and eventually raised as
  `WebSearchError`.
- News results carry a `date` and `source`; if news search comes back empty, we fall
  back to a general text search (title/snippet/url only, no date/source).
- No full-page scraping (see ARCHITECTURE.md §6) — the evidence unit is the search
  result itself (title + snippet + url), which is what the news agent (a later phase)
  grounds citations in.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from cachetools import TTLCache, cached
from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import settings
from app.logging_config import get_logger, trace
from app.tools.errors import WebSearchError

logger = get_logger("app.tools.web_search")

_CACHE_TTL_SECONDS = 300
_search_cache: TTLCache = TTLCache(maxsize=256, ttl=_CACHE_TTL_SECONDS)

_retry_transient = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type((TimeoutException, RatelimitException)),
    reraise=True,
)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    date: str | None
    source: str | None


def _is_no_results(exc: DDGSException) -> bool:
    return "no results" in str(exc).lower()


@_retry_transient
@cached(cache=_search_cache)
def _fetch_news(query: str, max_results: int) -> tuple[dict, ...]:
    return tuple(DDGS().news(query, max_results=max_results, safesearch="moderate"))


@_retry_transient
@cached(cache=_search_cache)
def _fetch_text(query: str, max_results: int) -> tuple[dict, ...]:
    return tuple(DDGS().text(query, max_results=max_results, safesearch="moderate"))


def _parse_news(raw: dict) -> SearchResult:
    return SearchResult(
        title=raw.get("title", ""),
        url=raw.get("url", ""),
        snippet=raw.get("body", ""),
        date=raw.get("date"),
        source=raw.get("source"),
    )


def _parse_text(raw: dict) -> SearchResult:
    return SearchResult(
        title=raw.get("title", ""),
        url=raw.get("href", ""),
        snippet=raw.get("body", ""),
        date=None,
        source=None,
    )


def search_news(query: str, max_results: int = 5) -> list[SearchResult]:
    try:
        raw_results = _fetch_news(query, max_results)
    except DDGSException as exc:
        if not _is_no_results(exc):
            raise WebSearchError(f"news search failed for query '{query}': {exc}") from exc
        raw_results = ()
    except Exception as exc:
        raise WebSearchError(f"news search failed for query '{query}': {exc}") from exc

    if raw_results:
        return [_parse_news(r) for r in raw_results]

    # News search had nothing — fall back to a general text search before giving up.
    try:
        raw_results = _fetch_text(query, max_results)
    except DDGSException as exc:
        if _is_no_results(exc):
            return []
        raise WebSearchError(f"text search fallback failed for query '{query}': {exc}") from exc
    except Exception as exc:
        raise WebSearchError(f"text search fallback failed for query '{query}': {exc}") from exc

    return [_parse_text(r) for r in raw_results]


@trace("app.tools.web_search")
async def asearch_news(query: str, max_results: int = 5) -> list[SearchResult]:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(search_news, query, max_results),
            timeout=settings.request_timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise WebSearchError(
            f"search for '{query}' timed out after {settings.request_timeout_seconds}s"
        ) from exc
