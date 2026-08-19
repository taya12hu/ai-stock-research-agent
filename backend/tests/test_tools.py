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


# --- yahoo_finance: fundamentals (Finnhub) ---------------------------------------------


FINNHUB_VALID_PROFILE = {
    "ticker": "AAPL",
    "name": "Apple Inc.",
    "finnhubIndustry": "Technology",
    "marketCapitalization": 4_400_000.0,  # millions, per Finnhub's convention
}
FINNHUB_VALID_QUOTE = {"c": 302.4}
FINNHUB_VALID_METRIC = {
    "metric": {
        "peTTM": 34.5,
        "pb": 55.2,
        "dividendYieldIndicatedAnnual": 35.0,  # -> 0.35 after /100 normalization
        "netProfitMarginTTM": 27.6,  # -> 0.276
        "revenueGrowthTTMYoy": 8.0,  # -> 0.08
        "epsGrowthTTMYoy": 10.0,  # -> 0.1
        "roeTTM": 150.0,  # -> 1.5
    }
}

# Observed real Finnhub behavior for an invalid ticker: HTTP 200 with an empty body,
# not an exception (confirmed by live testing — see git history).
FINNHUB_INVALID_PROFILE: dict = {}
FINNHUB_INVALID_QUOTE: dict = {}
FINNHUB_INVALID_METRIC: dict = {"metric": {}}


def _fake_finnhub_get(valid: bool):
    def _get(path: str, params: dict) -> dict:  # noqa: ARG001
        if path == "/stock/profile2":
            return FINNHUB_VALID_PROFILE if valid else FINNHUB_INVALID_PROFILE
        if path == "/quote":
            return FINNHUB_VALID_QUOTE if valid else FINNHUB_INVALID_QUOTE
        if path == "/stock/metric":
            return FINNHUB_VALID_METRIC if valid else FINNHUB_INVALID_METRIC
        raise AssertionError(f"unexpected Finnhub path {path!r}")

    return _get


def test_get_fundamentals_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yahoo_finance, "_finnhub_get", _fake_finnhub_get(valid=True))

    result = yahoo_finance.get_fundamentals("aapl")

    assert result.ticker == "AAPL"
    assert result.name == "Apple Inc."
    assert result.industry == "Technology"
    assert result.market_cap == 4_400_000_000_000
    assert result.trailing_pe == 34.5
    assert result.dividend_yield == 0.35
    assert result.profit_margin == 0.276
    assert result.current_price == 302.4


def test_get_fundamentals_invalid_ticker_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yahoo_finance, "_finnhub_get", _fake_finnhub_get(valid=False))

    with pytest.raises(YahooFinanceError, match="no fundamentals data"):
        yahoo_finance.get_fundamentals("ZZZINVALID")


def test_get_fundamentals_fetch_failure_message_is_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """The raised message is user-facing (surfaces in a failed AgentResult) and must
    never leak the raw underlying exception text — only a fixed, friendly message."""

    def _raise(path: str, params: dict) -> dict:  # noqa: ARG001
        raise ValueError("obscure internal parsing failure with a stack-trace-like body")

    monkeypatch.setattr(yahoo_finance, "_finnhub_get", _raise)

    with pytest.raises(YahooFinanceError) as exc_info:
        yahoo_finance.get_fundamentals("AAPL")

    assert "Unable to fetch fundamentals data" in str(exc_info.value)
    assert "stack-trace-like" not in str(exc_info.value)


def test_ticker_exists_true_for_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yahoo_finance, "_finnhub_get", _fake_finnhub_get(valid=True))

    assert yahoo_finance.ticker_exists("AAPL") is True


def test_ticker_exists_false_for_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yahoo_finance, "_finnhub_get", _fake_finnhub_get(valid=False))

    assert yahoo_finance.ticker_exists("ZZZINVALID") is False


def test_ticker_exists_false_on_persistent_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(path: str, params: dict) -> dict:  # noqa: ARG001
        raise ValueError("boom")

    monkeypatch.setattr(yahoo_finance, "_finnhub_get", _raise)

    assert yahoo_finance.ticker_exists("AAPL") is False


def test_get_company_name_returns_short_name_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yahoo_finance, "_finnhub_get", _fake_finnhub_get(valid=True))

    assert yahoo_finance.get_company_name("AAPL") == "Apple Inc."


def test_get_company_name_none_for_invalid_ticker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yahoo_finance, "_finnhub_get", _fake_finnhub_get(valid=False))

    assert yahoo_finance.get_company_name("ZZZINVALID") is None


def test_get_company_name_none_on_fetch_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Best-effort by design (see the function's docstring) — a failure here is a lost
    search-query disambiguation hint, not something worth raising over."""

    def _raise(path: str, params: dict) -> dict:  # noqa: ARG001
        raise ValueError("boom")

    monkeypatch.setattr(yahoo_finance, "_finnhub_get", _raise)

    assert yahoo_finance.get_company_name("AAPL") is None


async def test_aget_company_name_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yahoo_finance, "_finnhub_get", _fake_finnhub_get(valid=True))

    assert await yahoo_finance.aget_company_name("AAPL") == "Apple Inc."


def test_fetch_info_retries_on_transient_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}
    succeed = _fake_finnhub_get(valid=True)

    def _flaky(path: str, params: dict) -> dict:
        calls["count"] += 1
        if calls["count"] < 3:
            raise ConnectionError("transient")
        return succeed(path, params)

    monkeypatch.setattr(yahoo_finance, "_finnhub_get", _flaky)

    result = yahoo_finance.get_fundamentals("AAPL")

    assert calls["count"] >= 3
    assert result.name == "Apple Inc."


# --- yahoo_finance: technicals (Twelve Data) -------------------------------------------


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


def _twelvedata_body(history: pd.DataFrame) -> dict:
    """Twelve Data's time_series response shape, built from a synthetic history frame.
    Real responses have every numeric field as a string, and order is not guaranteed —
    `_fetch_history` sorts by datetime itself, so this deliberately emits newest-first
    to exercise that."""
    values = [
        {
            "datetime": str(index.date()),
            "open": str(row["Open"]),
            "high": str(row["High"]),
            "low": str(row["Low"]),
            "close": str(row["Close"]),
            "volume": str(int(row["Volume"])),
        }
        for index, row in history.iloc[::-1].iterrows()
    ]
    return {"meta": {"symbol": "AAPL"}, "values": values, "status": "ok"}


class FakeResponse:
    def __init__(self, body: dict, status_code: int = 200, text: str = "") -> None:
        self._body = body
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._body


def test_get_technical_data_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    history = _synthetic_history()
    monkeypatch.setattr(
        yahoo_finance.requests, "get", lambda *a, **kw: FakeResponse(_twelvedata_body(history))  # noqa: ARG005
    )

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
    monkeypatch.setattr(
        yahoo_finance.requests, "get", lambda *a, **kw: FakeResponse(_twelvedata_body(history))  # noqa: ARG005
    )

    result = yahoo_finance.get_technical_data("AAPL")

    assert result.sma_200 is None
    assert result.sma_50 is None
    assert result.rsi_14 is None
    assert result.last_close is not None


def test_get_technical_data_empty_history_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        yahoo_finance.requests,
        "get",
        lambda *a, **kw: FakeResponse({"status": "error", "message": "no data"}),  # noqa: ARG005
    )

    with pytest.raises(YahooFinanceError, match="no price history"):
        yahoo_finance.get_technical_data("ZZZINVALID")


def test_fetch_history_404_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 404 means the symbol genuinely doesn't exist, not a transient failure —
    retrying it would only burn through Twelve Data's tight free-tier rate limit
    (8 requests/minute) on a request that can never succeed."""
    calls = {"count": 0}

    def _fake_get(*args: object, **kwargs: object) -> FakeResponse:
        calls["count"] += 1
        return FakeResponse({}, status_code=404)

    monkeypatch.setattr(yahoo_finance.requests, "get", _fake_get)

    assert yahoo_finance.resolve_ticker("ZZZINVALID").symbol is None
    # 1 request per candidate (bare + 2 suffixes), no retries on any of them.
    assert calls["count"] == 3


def test_get_technical_data_fetch_failure_message_is_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*args: object, **kwargs: object) -> FakeResponse:
        raise ValueError("obscure internal parsing failure with a stack-trace-like body")

    monkeypatch.setattr(yahoo_finance.requests, "get", _raise)

    with pytest.raises(YahooFinanceError) as exc_info:
        yahoo_finance.get_technical_data("AAPL")

    assert "Unable to fetch price history" in str(exc_info.value)
    assert "stack-trace-like" not in str(exc_info.value)


# --- yahoo_finance: ticker resolution / exchange-suffix fallback (Twelve Data) --------


def test_resolve_ticker_returns_bare_symbol_when_fully_usable(monkeypatch: pytest.MonkeyPatch) -> None:
    history = _synthetic_history()
    monkeypatch.setattr(
        yahoo_finance.requests, "get", lambda *a, **kw: FakeResponse(_twelvedata_body(history))  # noqa: ARG005
    )

    result = yahoo_finance.resolve_ticker("aapl")

    assert result.symbol == "AAPL"
    assert result.unsupported_market is False


def test_resolve_ticker_falls_back_to_ns_suffix_when_bare_symbol_is_delisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard for the real observed collision: bare 'TCS' resolves to the
    wrong (or delisted) company on the bare symbol, and only 'TCS.NS' has real price
    history. No name-to-suffix lookup table involved: both variants are genuinely
    probed against (fake) Twelve Data, keyed by the request's `symbol` param."""
    history = _synthetic_history()
    bodies = {
        "TCS": {"status": "error", "message": "symbol not found"},
        "TCS.NS": _twelvedata_body(history),
    }

    def _fake_get(url: str, params: dict, **kwargs: object) -> FakeResponse:  # noqa: ARG001
        return FakeResponse(bodies[params["symbol"]])

    monkeypatch.setattr(yahoo_finance.requests, "get", _fake_get)

    assert yahoo_finance.resolve_ticker("TCS").symbol == "TCS.NS"


def test_resolve_ticker_returns_none_when_no_variant_is_usable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        yahoo_finance.requests,
        "get",
        lambda *a, **kw: FakeResponse({"status": "error", "message": "not found"}),  # noqa: ARG005
    )

    result = yahoo_finance.resolve_ticker("ZZZINVALID")

    assert result.symbol is None
    assert result.unsupported_market is False


def test_resolve_ticker_flags_unsupported_market_for_plan_gated_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard for the real Twelve Data behavior (confirmed live against TCS):
    a symbol that exists but isn't covered by the free plan returns a 404 with
    "available starting with the Grow or Venture plan" in the body — distinct from a
    genuinely invalid symbol's 404, and it should surface as `unsupported_market`, not
    a plain "could not be found"."""

    def _fake_get(url: str, params: dict, **kwargs: object) -> FakeResponse:  # noqa: ARG001
        return FakeResponse(
            {"code": 404, "message": "This symbol is available starting with the Grow or Venture plan."},
            status_code=404,
            text="This symbol is available starting with the Grow or Venture plan.",
        )

    monkeypatch.setattr(yahoo_finance.requests, "get", _fake_get)

    result = yahoo_finance.resolve_ticker("TCS")

    assert result.symbol is None
    assert result.unsupported_market is True


def test_resolve_ticker_plan_gated_symbol_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same reasoning as the 404-not-retried test: a plan-gated symbol will never
    succeed no matter how many times it's retried, so retrying it would only waste the
    free-tier rate limit."""
    calls = {"count": 0}

    def _fake_get(url: str, params: dict, **kwargs: object) -> FakeResponse:  # noqa: ARG001
        calls["count"] += 1
        return FakeResponse({}, status_code=404, text="available starting with the Grow plan")

    monkeypatch.setattr(yahoo_finance.requests, "get", _fake_get)

    yahoo_finance.resolve_ticker("TCS")

    assert calls["count"] == 3  # 1 per candidate (bare + 2 suffixes), no retries


async def test_aresolve_ticker_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    history = _synthetic_history()
    monkeypatch.setattr(
        yahoo_finance.requests, "get", lambda *a, **kw: FakeResponse(_twelvedata_body(history))  # noqa: ARG005
    )

    result = await yahoo_finance.aresolve_ticker("AAPL")

    assert result.symbol == "AAPL"


# --- async wrappers --------------------------------------------------------------------


async def test_aget_fundamentals_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yahoo_finance, "_finnhub_get", _fake_finnhub_get(valid=True))

    result = await yahoo_finance.aget_fundamentals("AAPL")

    assert result.ticker == "AAPL"


async def test_aget_fundamentals_propagates_tool_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yahoo_finance, "_finnhub_get", _fake_finnhub_get(valid=False))

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
