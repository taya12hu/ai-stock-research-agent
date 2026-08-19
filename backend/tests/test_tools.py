from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from app.tools import web_search, yahoo_finance
from app.tools.errors import WebSearchError, YahooFinanceError
from app.tools.web_search import DDGSException


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    """Tool-layer caches are module-level TTLCache instances shared across the whole
    test session (they're built at import time), so tests using the same ticker/query
    strings would otherwise see stale results from a previous test."""
    yahoo_finance._info_cache.clear()
    yahoo_finance._history_cache.clear()
    web_search._search_cache.clear()


# --- yahoo_finance: fundamentals -----------------------------------------------------


class FakeTicker:
    def __init__(self, info: dict, history: pd.DataFrame | None = None) -> None:
        self.info = info
        self._history = history if history is not None else pd.DataFrame()

    def history(self, period: str, interval: str) -> pd.DataFrame:  # noqa: ARG002
        return self._history


VALID_INFO = {
    "symbol": "AAPL",
    "shortName": "Apple Inc.",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "marketCap": 4_400_000_000_000,
    "trailingPE": 34.5,
    "forwardPE": 30.1,
    "priceToBook": 55.2,
    "dividendYield": 0.35,
    "profitMargins": 0.276,
    "revenueGrowth": 0.08,
    "earningsGrowth": 0.1,
    "returnOnEquity": 1.5,
    "totalDebt": 100_000_000_000,
    "totalCash": 60_000_000_000,
    "currentPrice": 302.4,
    "recommendationKey": "buy",
}

# Observed real yfinance behavior for an invalid ticker: a near-empty dict, not an
# exception and not an empty dict.
INVALID_INFO = {"trailingPegRatio": None}


def test_get_fundamentals_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yahoo_finance.yf, "Ticker", lambda ticker: FakeTicker(VALID_INFO))  # noqa: ARG005

    result = yahoo_finance.get_fundamentals("aapl")

    assert result.ticker == "AAPL"
    assert result.name == "Apple Inc."
    assert result.sector == "Technology"
    assert result.current_price == 302.4
    assert result.recommendation == "buy"


def test_get_fundamentals_invalid_ticker_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yahoo_finance.yf, "Ticker", lambda ticker: FakeTicker(INVALID_INFO))  # noqa: ARG005

    with pytest.raises(YahooFinanceError, match="no fundamentals data"):
        yahoo_finance.get_fundamentals("ZZZINVALID")


def test_get_fundamentals_fetch_failure_message_is_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """The raised message is user-facing (surfaces in a failed AgentResult) and must
    never leak the raw underlying exception text — only a fixed, friendly message."""

    def _raise(ticker: str) -> FakeTicker:  # noqa: ARG001
        raise ValueError("obscure internal parsing failure with a stack-trace-like body")

    monkeypatch.setattr(yahoo_finance.yf, "Ticker", _raise)

    with pytest.raises(YahooFinanceError) as exc_info:
        yahoo_finance.get_fundamentals("AAPL")

    assert "Unable to fetch fundamentals data" in str(exc_info.value)
    assert "stack-trace-like" not in str(exc_info.value)


def test_ticker_exists_true_for_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yahoo_finance.yf, "Ticker", lambda ticker: FakeTicker(VALID_INFO))  # noqa: ARG005

    assert yahoo_finance.ticker_exists("AAPL") is True


def test_ticker_exists_false_for_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yahoo_finance.yf, "Ticker", lambda ticker: FakeTicker(INVALID_INFO))  # noqa: ARG005

    assert yahoo_finance.ticker_exists("ZZZINVALID") is False


def test_ticker_exists_false_on_persistent_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(ticker: str) -> FakeTicker:  # noqa: ARG001
        raise ValueError("boom")

    monkeypatch.setattr(yahoo_finance.yf, "Ticker", _raise)

    assert yahoo_finance.ticker_exists("AAPL") is False


def test_get_company_name_returns_short_name_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yahoo_finance.yf, "Ticker", lambda ticker: FakeTicker(VALID_INFO))  # noqa: ARG005

    assert yahoo_finance.get_company_name("AAPL") == "Apple Inc."


def test_get_company_name_none_for_invalid_ticker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yahoo_finance.yf, "Ticker", lambda ticker: FakeTicker(INVALID_INFO))  # noqa: ARG005

    assert yahoo_finance.get_company_name("ZZZINVALID") is None


def test_get_company_name_none_on_fetch_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Best-effort by design (see the function's docstring) — a failure here is a lost
    search-query disambiguation hint, not something worth raising over."""

    def _raise(ticker: str) -> FakeTicker:  # noqa: ARG001
        raise ValueError("boom")

    monkeypatch.setattr(yahoo_finance.yf, "Ticker", _raise)

    assert yahoo_finance.get_company_name("AAPL") is None


async def test_aget_company_name_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yahoo_finance.yf, "Ticker", lambda ticker: FakeTicker(VALID_INFO))  # noqa: ARG005

    assert await yahoo_finance.aget_company_name("AAPL") == "Apple Inc."


def test_fetch_info_retries_on_transient_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def _flaky(ticker: str) -> FakeTicker:  # noqa: ARG001
        calls["count"] += 1
        if calls["count"] < 3:
            raise ConnectionError("transient")
        return FakeTicker(VALID_INFO)

    monkeypatch.setattr(yahoo_finance.yf, "Ticker", _flaky)

    result = yahoo_finance.get_fundamentals("AAPL")

    assert calls["count"] == 3
    assert result.name == "Apple Inc."


# --- yahoo_finance: technicals --------------------------------------------------------


def _synthetic_history(days: int = 300, start_price: float = 100.0) -> pd.DataFrame:
    dates = pd.date_range(end=datetime.now(timezone.utc), periods=days, freq="B")
    # Gentle upward drift with a bit of oscillation so RSI/MACD aren't degenerate.
    prices = [start_price + i * 0.3 + (2 if i % 5 == 0 else 0) for i in range(days)]
    return pd.DataFrame(
        {
            "Open": prices,
            "High": [p + 1 for p in prices],
            "Low": [p - 1 for p in prices],
            "Close": prices,
            "Volume": [1_000_000] * days,
        },
        index=dates,
    )


def test_get_technical_data_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    history = _synthetic_history()
    monkeypatch.setattr(yahoo_finance.yf, "Ticker", lambda ticker: FakeTicker(VALID_INFO, history))  # noqa: ARG005

    result = yahoo_finance.get_technical_data("AAPL")

    assert result.ticker == "AAPL"
    assert result.sma_20 is not None
    assert result.sma_50 is not None
    assert result.sma_200 is not None
    assert result.rsi_14 is not None
    assert 0 <= result.rsi_14 <= 100
    assert result.macd is not None
    assert set(result.macd) == {"macd", "signal", "histogram"}
    assert result.momentum_1m_pct is not None
    assert result.volatility_annualized_pct is not None
    assert result.volatility_annualized_pct >= 0
    assert result.fifty_two_week_high >= result.fifty_two_week_low
    assert result.last_close == round(float(history["Close"].iloc[-1]), 2)


def test_get_technical_data_short_history_returns_partial_indicators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fewer than 200 data points: long-window indicators should be None, not crash."""
    history = _synthetic_history(days=10)
    monkeypatch.setattr(yahoo_finance.yf, "Ticker", lambda ticker: FakeTicker(VALID_INFO, history))  # noqa: ARG005

    result = yahoo_finance.get_technical_data("AAPL")

    assert result.sma_200 is None
    assert result.sma_50 is None
    assert result.rsi_14 is None
    assert result.last_close is not None


def test_get_technical_data_empty_history_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        yahoo_finance.yf, "Ticker", lambda ticker: FakeTicker(INVALID_INFO, pd.DataFrame())  # noqa: ARG005
    )

    with pytest.raises(YahooFinanceError, match="no price history"):
        yahoo_finance.get_technical_data("ZZZINVALID")


def test_get_technical_data_fetch_failure_message_is_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(ticker: str) -> FakeTicker:  # noqa: ARG001
        raise ValueError("obscure internal parsing failure with a stack-trace-like body")

    monkeypatch.setattr(yahoo_finance.yf, "Ticker", _raise)

    with pytest.raises(YahooFinanceError) as exc_info:
        yahoo_finance.get_technical_data("AAPL")

    assert "Unable to fetch price history" in str(exc_info.value)
    assert "stack-trace-like" not in str(exc_info.value)


# --- yahoo_finance: ticker resolution / exchange-suffix fallback ----------------------


def test_resolve_ticker_returns_bare_symbol_when_fully_usable(monkeypatch: pytest.MonkeyPatch) -> None:
    history = _synthetic_history()
    monkeypatch.setattr(yahoo_finance.yf, "Ticker", lambda ticker: FakeTicker(VALID_INFO, history))  # noqa: ARG005

    assert yahoo_finance.resolve_ticker("aapl") == "AAPL"


def test_resolve_ticker_falls_back_to_ns_suffix_when_bare_symbol_is_delisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard for the real observed collision: bare 'TCS' returns real-looking
    `.info` (so `_has_real_data`/`ticker_exists` alone would call it valid) but empty
    price history — the wrong or delisted company. 'TCS.NS' is the actual usable
    listing. No name-to-suffix lookup table involved: both variants are genuinely
    probed against (fake) Yahoo Finance."""
    history = _synthetic_history()
    fakes = {
        "TCS": FakeTicker({"symbol": "TCS", "shortName": "Some Other Co"}, pd.DataFrame()),
        "TCS.NS": FakeTicker({"symbol": "TCS.NS", "shortName": "Tata Consultancy Services"}, history),
    }
    monkeypatch.setattr(yahoo_finance.yf, "Ticker", lambda ticker: fakes[ticker])

    assert yahoo_finance.resolve_ticker("TCS") == "TCS.NS"


def test_resolve_ticker_returns_none_when_no_variant_is_usable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yahoo_finance.yf, "Ticker", lambda ticker: FakeTicker(INVALID_INFO, pd.DataFrame()))  # noqa: ARG005

    assert yahoo_finance.resolve_ticker("ZZZINVALID") is None


async def test_aresolve_ticker_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    history = _synthetic_history()
    monkeypatch.setattr(yahoo_finance.yf, "Ticker", lambda ticker: FakeTicker(VALID_INFO, history))  # noqa: ARG005

    assert await yahoo_finance.aresolve_ticker("AAPL") == "AAPL"


# --- async wrappers --------------------------------------------------------------------


async def test_aget_fundamentals_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yahoo_finance.yf, "Ticker", lambda ticker: FakeTicker(VALID_INFO))  # noqa: ARG005

    result = await yahoo_finance.aget_fundamentals("AAPL")

    assert result.ticker == "AAPL"


async def test_aget_fundamentals_propagates_tool_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yahoo_finance.yf, "Ticker", lambda ticker: FakeTicker(INVALID_INFO))  # noqa: ARG005

    with pytest.raises(YahooFinanceError):
        await yahoo_finance.aget_fundamentals("ZZZINVALID")


# --- web_search --------------------------------------------------------------------


class FakeDDGS:
    def __init__(self, news_result: object, text_result: object) -> None:
        self._news_result = news_result
        self._text_result = text_result

    def news(self, query: str, **kwargs: object) -> list[dict]:  # noqa: ARG002
        if isinstance(self._news_result, Exception):
            raise self._news_result
        return self._news_result

    def text(self, query: str, **kwargs: object) -> list[dict]:  # noqa: ARG002
        if isinstance(self._text_result, Exception):
            raise self._text_result
        return self._text_result


NEWS_ITEM = {
    "date": "2026-08-12T10:00:00+00:00",
    "title": "NVIDIA rounds up AI financing",
    "body": "NVIDIA has secured additional AI infrastructure financing.",
    "url": "https://example.com/nvda-news",
    "source": "Example Wire",
}

TEXT_ITEM = {
    "title": "NVIDIA Corp",
    "href": "https://example.com/nvda-generic",
    "body": "General result about NVIDIA.",
}


def test_search_news_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        web_search, "DDGS", lambda: FakeDDGS(news_result=[NEWS_ITEM], text_result=[])
    )

    results = web_search.search_news("NVIDIA stock news")

    assert len(results) == 1
    assert results[0].title == NEWS_ITEM["title"]
    assert results[0].url == NEWS_ITEM["url"]
    assert results[0].date == NEWS_ITEM["date"]
    assert results[0].source == NEWS_ITEM["source"]


def test_search_news_falls_back_to_text_when_news_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        web_search,
        "DDGS",
        lambda: FakeDDGS(
            news_result=DDGSException("No results found."), text_result=[TEXT_ITEM]
        ),
    )

    results = web_search.search_news("obscure microcap co")

    assert len(results) == 1
    assert results[0].title == TEXT_ITEM["title"]
    assert results[0].url == TEXT_ITEM["href"]
    assert results[0].date is None


def test_search_news_returns_empty_when_nothing_found_anywhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        web_search,
        "DDGS",
        lambda: FakeDDGS(
            news_result=DDGSException("No results found."),
            text_result=DDGSException("No results found."),
        ),
    )

    results = web_search.search_news("truly nonexistent company xyzzy")

    assert results == []


def test_search_news_raises_on_real_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        web_search,
        "DDGS",
        lambda: FakeDDGS(
            news_result=DDGSException("Backend engine crashed"), text_result=[]
        ),
    )

    # The exception's message is user-facing (shown in a failed AgentResult) and must
    # never leak the raw underlying error — that detail is logged instead, not raised.
    with pytest.raises(WebSearchError, match="Unable to search for news") as exc_info:
        web_search.search_news("NVIDIA stock news")
    assert "Backend engine crashed" not in str(exc_info.value)


async def test_asearch_news_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        web_search, "DDGS", lambda: FakeDDGS(news_result=[NEWS_ITEM], text_result=[])
    )

    results = await web_search.asearch_news("NVIDIA stock news")

    assert len(results) == 1
