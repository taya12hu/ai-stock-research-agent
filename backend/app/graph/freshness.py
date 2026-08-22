"""When a stored (ticker, agent) result still counts as current — one pure predicate,
consulted on every path.

This replaces `followup_router_node._stale_tickers`, which had three defects the audit
found and which are all structural rather than tuning mistakes:

- It checked **timestamp presence**, not **usability**, so a ticker whose three agents had
  all *failed* carried three fresh timestamps and looked perfectly current. A follow-up
  right after a total failure would never retry — it would answer "not covered" from three
  stored error strings (A-03).
- It scoped to the whole session's ticker list rather than the turn's, so a single-ticker
  recall question in a three-ticker session escalated into nine specialist runs (A-04).
- It lived inside the `answer` branch of one classifier decision, so the `refresh` path —
  the one that goes on to render a full report — skipped it entirely (A-05).

Fixing those means freshness stops being a property of a routing decision and becomes a
property of the data, evaluated per cell. Everything here is pure: no I/O, no model, no
network, so it is testable by exact assertion at any simulated clock time.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import Literal
from zoneinfo import ZoneInfo

from app.graph.state import AgentName

# Matches `_CACHE_TTL_SECONDS` in tools/market_data.py and tools/web_search.py. This is
# a hard floor for every TTL below, not a coincidence: the tool layer hands back a cached
# object for this long, so a freshness window shorter than it would refetch and receive
# byte-identical data — work and latency for provably zero new information.
CACHE_TTL_SECONDS = 300

AGENT_TTL_SECONDS: dict[AgentName, int] = {
    # Continuous arrival; nothing to gain from waiting longer than the cache floor.
    "news": CACHE_TTL_SECONDS,
    # Derived from daily OHLC bars. While the session is open the current bar is still
    # forming, so the indicators genuinely move. See `_technical_is_stale` for what
    # happens once it closes — the interesting half.
    "technical": 900,
    # Profile, sector, margins and growth move on filing cadence — quarterly. The
    # price-derived ratios (P/E, market cap) do move intraday, so an hour is the
    # compromise: short enough that a quoted P/E isn't embarrassing, long enough not to
    # re-pull a company profile every few minutes.
    "fundamentals": 3600,
}

# Enforced at import rather than documented in a comment, because the failure it prevents
# is silent: a TTL below the cache floor produces refetches that look like they worked.
for _agent, _ttl in AGENT_TTL_SECONDS.items():
    if _ttl < CACHE_TTL_SECONDS:
        raise ValueError(
            f"AGENT_TTL_SECONDS[{_agent!r}] = {_ttl}s is below the tool-layer cache TTL "
            f"({CACHE_TTL_SECONDS}s); refetching inside the cache window returns identical data."
        )

_MARKET_TZ = ZoneInfo("America/New_York")
_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(16, 0)

FetchReason = Literal["missing", "failed", "empty", "stale"]


def us_market_is_open(now: datetime) -> bool:
    """Regular US session only — 09:30–16:00 ET, Monday to Friday.

    Market holidays are deliberately not modelled. The failure mode is benign and strictly
    better than the status quo: on a holiday this reports "open", so `technical` falls back
    to its ordinary 15-minute TTL and refetches a few times — which is exactly what the
    current code does on *every* day, holiday or not. Modelling the NYSE calendar would
    mean either a dependency or a hardcoded date list that silently rots.
    """
    local = now.astimezone(_MARKET_TZ)
    if local.weekday() >= 5:  # Saturday, Sunday
        return False
    return _MARKET_OPEN <= local.time() < _MARKET_CLOSE


def last_regular_close(now: datetime) -> datetime:
    """The most recent regular-session close at or before `now`, in UTC.

    Walks back a day at a time over weekends. Used only when the market is shut, to answer
    "was this fetched before or after the bar was finalised."
    """
    local = now.astimezone(_MARKET_TZ)
    candidate = local.replace(
        hour=_MARKET_CLOSE.hour, minute=_MARKET_CLOSE.minute, second=0, microsecond=0
    )
    while candidate > local or candidate.weekday() >= 5:
        candidate = (candidate - timedelta(days=1)).replace(
            hour=_MARKET_CLOSE.hour, minute=_MARKET_CLOSE.minute, second=0, microsecond=0
        )
    return candidate.astimezone(timezone.utc)


def _parse_fetched_at(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return None
    # `now_iso()` writes tz-aware UTC, but a hand-written or migrated value might not be.
    # Assuming UTC is right for this codebase and strictly safer than raising.
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _technical_is_stale(fetched_at: datetime, now: datetime) -> bool:
    """Technical indicators come from *daily* bars, so once the session closes the last bar
    is final and SMA/RSI/MACD cannot change until the next open.

    That makes a flat wall-clock TTL wrong in a way that costs real money: a 15-minute
    window refetches all evening, all weekend, and through every holiday to receive data
    that provably has not changed. The correct question when the market is shut is not "how
    old is this" but "was it taken before or after the bar was finalised" — anything
    fetched after the last close is still exactly current, however long ago that was.
    """
    if us_market_is_open(now):
        return (now - fetched_at).total_seconds() > AGENT_TTL_SECONDS["technical"]
    return fetched_at < last_regular_close(now)


def fetch_reason(cell: dict | None, agent: AgentName, now: datetime) -> FetchReason | None:
    """Why this cell needs refetching, or `None` if it is still good.

    Returns the reason rather than a bare bool so callers can log *why* a fetch happened
    and the coverage line can distinguish "we retried a failure" from "this went stale".

    The `empty` case is the subtle one: a cell that ran without error but produced no
    findings is usable as a *fact* ("no recent news exists for this company") and useless
    as *evidence*. Treating it as needing refetch is right — news absent ten minutes ago
    may exist now — and the same "has findings" test is what gates report rendering, which
    is how A-10 is closed: a ticker whose only successful agent found nothing can no longer
    satisfy the usability check and produce a report written from two error strings.
    """
    if cell is None:
        return "missing"
    if cell.get("status") != "ok":
        return "failed"
    if not cell.get("findings"):
        return "empty"

    fetched_at = _parse_fetched_at(cell.get("fetched_at"))
    if fetched_at is None:
        return "stale"

    if agent == "technical":
        return "stale" if _technical_is_stale(fetched_at, now) else None

    ttl = AGENT_TTL_SECONDS[agent]
    return "stale" if (now - fetched_at).total_seconds() > ttl else None


def needs_fetch(cell: dict | None, agent: AgentName, now: datetime) -> bool:
    return fetch_reason(cell, agent, now) is not None


def is_usable(cell: dict | None) -> bool:
    """Whether a cell carries evidence a report can be written from.

    Deliberately stricter than `status == "ok"`: that conflates *the node ran without
    error* with *the node produced something to cite*, and only the second should gate
    rendering. `ticker_all_failed` tested the weaker condition, so a ticker where
    fundamentals and technicals failed and news legitimately found no articles counted as
    usable — and synthesis wrote a full report, verdict line included, from two error
    strings and a "no news found" (A-10).
    """
    return bool(cell) and cell.get("status") == "ok" and bool(cell.get("findings"))
