"""Freshness is the predicate three audit findings hinge on, so it is tested by exact
assertion at simulated clock times rather than through the graph.

Calendar anchors used throughout (verified, not assumed):
    2026-08-19 Wednesday | 2026-08-21 Friday | 2026-08-22 Saturday
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from app.graph.freshness import (
    AGENT_TTL_SECONDS,
    CACHE_TTL_SECONDS,
    fetch_reason,
    is_usable,
    last_regular_close,
    needs_fetch,
    us_market_is_open,
)

ET = ZoneInfo("America/New_York")


def _et(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=ET)


def _cell(status: str = "ok", *, findings: int = 1, fetched_at: datetime | None = None) -> dict:
    stamp = fetched_at or datetime.now(timezone.utc)
    return {
        "status": status,
        "summary": "s",
        "findings": [{"id": f"f{i}"} for i in range(findings)],
        "error": None if status == "ok" else "boom",
        "fetched_at": stamp.isoformat(),
    }


# ─────────────────────────── the four fetch reasons ───────────────────────────


def test_missing_cell_needs_fetch() -> None:
    assert fetch_reason(None, "news", datetime.now(timezone.utc)) == "missing"


def test_failed_cell_is_not_fresh_even_with_a_recent_timestamp() -> None:
    """A-03 regression.

    Specialist nodes stamp `fetched_at` on failure as well as success, so a ticker whose
    three agents all just failed carries three brand-new timestamps. The previous guard
    checked timestamp presence and age only, so it reported that ticker perfectly fresh —
    meaning a follow-up asked seconds later would never retry, and would instead answer
    "not covered" from three stored error strings.
    """
    now = datetime.now(timezone.utc)
    just_failed = _cell("failed", fetched_at=now - timedelta(seconds=5))

    assert fetch_reason(just_failed, "fundamentals", now) == "failed"
    assert needs_fetch(just_failed, "fundamentals", now) is True


def test_successful_but_empty_cell_needs_refetch() -> None:
    """A news search that legitimately found nothing is a usable *fact* and useless as
    *evidence*. News absent ten minutes ago may exist now, so it refetches — bounded by
    the TTL, so at most once per window.
    """
    now = datetime.now(timezone.utc)
    empty = _cell("ok", findings=0, fetched_at=now)

    assert fetch_reason(empty, "news", now) == "empty"


def test_fresh_cell_needs_nothing() -> None:
    now = datetime.now(timezone.utc)
    assert fetch_reason(_cell(fetched_at=now - timedelta(seconds=30)), "news", now) is None


def test_cell_past_its_ttl_is_stale() -> None:
    now = datetime.now(timezone.utc)
    aged = _cell(fetched_at=now - timedelta(seconds=AGENT_TTL_SECONDS["news"] + 1))
    assert fetch_reason(aged, "news", now) == "stale"


def test_unparseable_timestamp_is_treated_as_stale_not_a_crash() -> None:
    cell = _cell()
    cell["fetched_at"] = "not-a-timestamp"
    assert fetch_reason(cell, "news", datetime.now(timezone.utc)) == "stale"


# ─────────────────────────── per-agent TTLs ───────────────────────────


def test_ttls_differ_per_agent_and_respect_the_cache_floor() -> None:
    """No freshness window may sit below the tool layer's own cache TTL — inside it a
    refetch is handed the identical cached object, so it costs latency for provably zero
    new information. Enforced at import; asserted here so the intent is visible.
    """
    assert AGENT_TTL_SECONDS["news"] == CACHE_TTL_SECONDS
    assert AGENT_TTL_SECONDS["technical"] > CACHE_TTL_SECONDS
    assert AGENT_TTL_SECONDS["fundamentals"] > AGENT_TTL_SECONDS["technical"]
    assert all(ttl >= CACHE_TTL_SECONDS for ttl in AGENT_TTL_SECONDS.values())


def test_fundamentals_outlives_news_at_the_same_age() -> None:
    now = datetime.now(timezone.utc)
    ten_minutes_old = now - timedelta(minutes=10)

    assert needs_fetch(_cell(fetched_at=ten_minutes_old), "news", now) is True
    assert needs_fetch(_cell(fetched_at=ten_minutes_old), "fundamentals", now) is False


# ─────────────────────────── market hours ───────────────────────────


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        (_et(2026, 8, 19, 9, 29), False),   # Wednesday, one minute before the bell
        (_et(2026, 8, 19, 9, 30), True),    # Wednesday, the open
        (_et(2026, 8, 19, 12, 0), True),    # Wednesday midday
        (_et(2026, 8, 19, 15, 59), True),   # Wednesday, one minute before the close
        (_et(2026, 8, 19, 16, 0), False),   # Wednesday, the close itself
        (_et(2026, 8, 22, 12, 0), False),   # Saturday midday
    ],
)
def test_us_market_is_open(moment: datetime, expected: bool) -> None:
    assert us_market_is_open(moment) is expected


def test_market_hours_are_evaluated_in_eastern_time_not_utc() -> None:
    """14:00 UTC on a Wednesday is 10:00 ET — open. Naive UTC comparison would call it
    closed on one side of the window and open on the other.
    """
    assert us_market_is_open(datetime(2026, 8, 19, 14, 0, tzinfo=timezone.utc)) is True
    assert us_market_is_open(datetime(2026, 8, 19, 21, 0, tzinfo=timezone.utc)) is False


def test_last_regular_close_walks_back_over_the_weekend() -> None:
    saturday = _et(2026, 8, 22, 12, 0)
    assert last_regular_close(saturday) == _et(2026, 8, 21, 16, 0).astimezone(timezone.utc)


def test_last_regular_close_before_todays_bell_is_yesterdays() -> None:
    wednesday_premarket = _et(2026, 8, 19, 8, 0)
    assert last_regular_close(wednesday_premarket) == _et(2026, 8, 18, 16, 0).astimezone(timezone.utc)


def test_last_regular_close_after_todays_bell_is_today() -> None:
    wednesday_evening = _et(2026, 8, 19, 20, 0)
    assert last_regular_close(wednesday_evening) == _et(2026, 8, 19, 16, 0).astimezone(timezone.utc)


# ─────────────────────────── technical, the interesting one ───────────────────────────


def test_technical_uses_a_wall_clock_ttl_while_the_market_is_open() -> None:
    midday = _et(2026, 8, 19, 12, 0)
    aged = _cell(fetched_at=midday - timedelta(seconds=AGENT_TTL_SECONDS["technical"] + 60))

    assert fetch_reason(aged, "technical", midday) == "stale"


def test_technical_fetched_after_the_close_stays_fresh_all_weekend() -> None:
    """The efficiency bug a flat TTL causes.

    Technical indicators are computed from *daily* bars, so once Friday's session closes
    the last bar is final and SMA/RSI/MACD cannot move until Monday's open. Under a flat
    15-minute window this cell would be refetched every quarter hour for two and a half
    days to receive byte-identical data.
    """
    fetched = _et(2026, 8, 21, 16, 30)          # Friday, half an hour after the close
    cell = _cell(fetched_at=fetched)

    saturday = _et(2026, 8, 22, 12, 0)
    assert fetch_reason(cell, "technical", saturday) is None
    # ...while news at the same age is long past its window.
    assert fetch_reason(cell, "news", saturday) == "stale"


def test_technical_fetched_during_the_session_is_stale_once_it_closes() -> None:
    """The mirror case, and the reason this cannot simply be "closed means fresh": a bar
    fetched mid-session was still forming, so it is not the final bar and must be refetched
    after the close even though minutes-old by wall clock.
    """
    fetched_midsession = _et(2026, 8, 21, 15, 55)   # Friday, five minutes before the bell
    cell = _cell(fetched_at=fetched_midsession)

    friday_evening = _et(2026, 8, 21, 16, 5)
    assert fetch_reason(cell, "technical", friday_evening) == "stale"


# ─────────────────────────── usability ───────────────────────────


def test_is_usable_requires_findings_not_just_a_clean_run() -> None:
    """A-10 regression.

    `ticker_all_failed` tested `status != "ok"`, so a news node that ran cleanly and found
    no articles counted as usable — and synthesis wrote a full report, mandatory verdict
    line included, from two error strings and a "no news found".
    """
    assert is_usable(_cell("ok", findings=2)) is True
    assert is_usable(_cell("ok", findings=0)) is False
    assert is_usable(_cell("failed")) is False
    assert is_usable(None) is False
