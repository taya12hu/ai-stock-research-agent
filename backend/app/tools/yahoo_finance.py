"""Yahoo Finance data access (fundamentals + technical/price data), via `yfinance`.

Design notes (see ARCHITECTURE.md §6, §10):
- `yfinance` does not raise for an invalid ticker — it silently returns a near-empty
  `info` dict (observed: just `{"trailingPegRatio": None}`) and an empty history
  DataFrame. Validity is therefore checked on the *shape of the returned data*, not on
  exceptions.
- Retries only apply to transient network errors, never to "ticker doesn't exist".
- Every public fetch function is wrapped with a short TTL cache (avoids re-hitting Yahoo
  repeatedly, e.g. during eval runs) and, on the async side, a hard timeout.
- Sync core + `asyncio.to_thread` async wrappers, since `yfinance` itself is sync/blocking
  and agent nodes (a later phase) run in an async LangGraph.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf
from cachetools import TTLCache, cached
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import settings
from app.logging_config import get_logger, log_event, trace
from app.tools.errors import YahooFinanceError

logger = get_logger("app.tools.yahoo_finance")

_CACHE_TTL_SECONDS = 300
_info_cache: TTLCache = TTLCache(maxsize=256, ttl=_CACHE_TTL_SECONDS)
_history_cache: TTLCache = TTLCache(maxsize=256, ttl=_CACHE_TTL_SECONDS)

# yfinance raises plain requests/urllib errors on real network trouble; it does NOT
# raise these for an invalid ticker (see module docstring), so retrying on them is safe.
_RETRYABLE = (ConnectionError, TimeoutError, OSError)

_retry_network = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type(_RETRYABLE),
    reraise=True,
)


@dataclass
class FundamentalsData:
    ticker: str
    name: str | None
    sector: str | None
    industry: str | None
    market_cap: float | None
    trailing_pe: float | None
    forward_pe: float | None
    price_to_book: float | None
    dividend_yield: float | None
    profit_margin: float | None
    revenue_growth: float | None
    earnings_growth: float | None
    return_on_equity: float | None
    total_debt: float | None
    total_cash: float | None
    current_price: float | None
    recommendation: str | None
    as_of: str


@dataclass
class TechnicalData:
    ticker: str
    last_close: float | None
    sma_20: float | None
    sma_50: float | None
    sma_200: float | None
    rsi_14: float | None
    macd: dict[str, float] | None
    momentum_1m_pct: float | None
    volatility_annualized_pct: float | None
    fifty_two_week_high: float | None
    fifty_two_week_low: float | None
    as_of: str


@_retry_network
@cached(cache=_info_cache)
def _fetch_info(ticker: str) -> dict:
    return yf.Ticker(ticker).info or {}


@_retry_network
@cached(cache=_history_cache)
def _fetch_history(ticker: str, period: str) -> pd.DataFrame:
    return yf.Ticker(ticker).history(period=period, interval="1d")


def _has_real_data(info: dict) -> bool:
    return bool(info) and bool(
        info.get("symbol") or info.get("shortName") or info.get("regularMarketPrice")
    )


# --- technical indicator math -------------------------------------------------------


def _sma(close: pd.Series, window: int) -> float | None:
    if len(close) < window:
        return None
    return round(float(close.rolling(window).mean().iloc[-1]), 2)


def _rsi(close: pd.Series, period: int = 14) -> float | None:
    if len(close) < period + 1:
        return None
    delta = close.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean().iloc[-1]
    avg_loss = loss.rolling(period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def _macd(close: pd.Series) -> dict[str, float] | None:
    if len(close) < 26:
        return None
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return {
        "macd": round(float(macd_line.iloc[-1]), 4),
        "signal": round(float(signal_line.iloc[-1]), 4),
        "histogram": round(float(macd_line.iloc[-1] - signal_line.iloc[-1]), 4),
    }


def _momentum_1m(close: pd.Series, window: int = 21) -> float | None:
    if len(close) <= window:
        return None
    return round(float((close.iloc[-1] / close.iloc[-window] - 1) * 100), 2)


def _annualized_volatility(close: pd.Series) -> float | None:
    if len(close) < 2:
        return None
    returns = close.pct_change().dropna()
    if returns.empty:
        return None
    return round(float(returns.std() * (252**0.5) * 100), 2)


def _fifty_two_week_range(history: pd.DataFrame) -> tuple[float, float] | None:
    if history.empty:
        return None
    return round(float(history["High"].max()), 2), round(float(history["Low"].min()), 2)


# --- public sync API -----------------------------------------------------------------


def ticker_exists(ticker: str) -> bool:
    try:
        info = _fetch_info(ticker.upper())
    except Exception as exc:  # noqa: BLE001 - any fetch failure => can't confirm existence
        log_event(
            logger, "ticker_exists check failed", level=logging.WARNING,
            ticker=ticker, error=str(exc),
        )
        return False
    return _has_real_data(info)


def get_fundamentals(ticker: str) -> FundamentalsData:
    ticker = ticker.upper()
    try:
        info = _fetch_info(ticker)
    except Exception as exc:
        raise YahooFinanceError(f"failed to fetch fundamentals for {ticker}: {exc}") from exc

    if not _has_real_data(info):
        raise YahooFinanceError(f"no fundamentals data for {ticker} (ticker may be invalid)")

    return FundamentalsData(
        ticker=ticker,
        name=info.get("shortName") or info.get("longName"),
        sector=info.get("sector"),
        industry=info.get("industry"),
        market_cap=info.get("marketCap"),
        trailing_pe=info.get("trailingPE"),
        forward_pe=info.get("forwardPE"),
        price_to_book=info.get("priceToBook"),
        dividend_yield=info.get("dividendYield"),
        profit_margin=info.get("profitMargins"),
        revenue_growth=info.get("revenueGrowth"),
        earnings_growth=info.get("earningsGrowth"),
        return_on_equity=info.get("returnOnEquity"),
        total_debt=info.get("totalDebt"),
        total_cash=info.get("totalCash"),
        current_price=info.get("currentPrice") or info.get("regularMarketPrice"),
        recommendation=info.get("recommendationKey"),
        as_of=datetime.now(timezone.utc).isoformat(),
    )


def get_technical_data(ticker: str, period: str = "1y") -> TechnicalData:
    ticker = ticker.upper()
    try:
        history = _fetch_history(ticker, period)
    except Exception as exc:
        raise YahooFinanceError(f"failed to fetch price history for {ticker}: {exc}") from exc

    if history is None or history.empty:
        raise YahooFinanceError(f"no price history for {ticker} (ticker may be invalid)")

    close = history["Close"]
    macd = _macd(close)
    fifty_two_week = _fifty_two_week_range(history)

    return TechnicalData(
        ticker=ticker,
        last_close=round(float(close.iloc[-1]), 2),
        sma_20=_sma(close, 20),
        sma_50=_sma(close, 50),
        sma_200=_sma(close, 200),
        rsi_14=_rsi(close, 14),
        macd=macd,
        momentum_1m_pct=_momentum_1m(close),
        volatility_annualized_pct=_annualized_volatility(close),
        fifty_two_week_high=fifty_two_week[0] if fifty_two_week else None,
        fifty_two_week_low=fifty_two_week[1] if fifty_two_week else None,
        as_of=history.index[-1].isoformat(),
    )


# --- public async API (used by agent nodes) -------------------------------------------


@trace("app.tools.yahoo_finance")
async def aticker_exists(ticker: str) -> bool:
    return await asyncio.wait_for(
        asyncio.to_thread(ticker_exists, ticker), timeout=settings.request_timeout_seconds
    )


@trace("app.tools.yahoo_finance")
async def aget_fundamentals(ticker: str) -> FundamentalsData:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(get_fundamentals, ticker), timeout=settings.request_timeout_seconds
        )
    except asyncio.TimeoutError as exc:
        raise YahooFinanceError(
            f"fundamentals fetch for {ticker} timed out after {settings.request_timeout_seconds}s"
        ) from exc


@trace("app.tools.yahoo_finance")
async def aget_technical_data(ticker: str, period: str = "1y") -> TechnicalData:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(get_technical_data, ticker, period),
            timeout=settings.request_timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise YahooFinanceError(
            f"technical data fetch for {ticker} timed out after {settings.request_timeout_seconds}s"
        ) from exc
