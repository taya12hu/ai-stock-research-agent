"""Turns the company *names* the classifier extracted into validated ticker *symbols*.

Deliberately the only impure step between `classify_turn` and `plan_turn`: it makes
network calls, so keeping it separate is what lets every scope and shape decision be
tested against a hand-built `ScopeResolution` with no provider involved.

The model never emits ticker symbols — it reports the name as the user said it ("Amazon",
"Tata Consultancy"), and resolution happens here against `aresolve_ticker`, which is
unchanged and already correct: it demands real price history before accepting a symbol,
which is what caught bare `TCS` resolving to a delisted company when Tata Consultancy
Services is actually `TCS.NS`.
"""

from __future__ import annotations

from app.config import settings
from app.graph.intent import TurnIntent
from app.graph.plan_turn import ScopeResolution
from app.logging_config import get_logger, log_event
from app.tools.yahoo_finance import aresolve_ticker

logger = get_logger("app.graph.resolve_scope")


def _bare(ticker: str) -> str:
    """Strips an exchange suffix ('TCS.NS' -> 'TCS').

    Users say "TCS"; the session stores whatever `aresolve_ticker` settled on, which may
    carry a suffix. Matching on the bare form is what lets a follow-up naming the plain
    symbol find the session's existing entry instead of being treated as a new company.
    """
    return ticker.split(".")[0].strip().upper()


async def resolve_scope(intent: TurnIntent, known_tickers: list[str]) -> ScopeResolution:
    """Resolve the message's companies, reusing symbols this session already holds.

    `known_tickers` short-circuits both a network round-trip and a real correctness risk:
    a session that resolved "TCS" to "TCS.NS" must keep matching later mentions to that
    same symbol rather than re-resolving the bare form and possibly landing somewhere else.
    """
    companies = intent.companies or []
    subject_names = [c.name for c in companies if c.role == "research_subject"]
    unclear_names = [c.name for c in companies if c.role == "unclear"]
    attempted = bool(subject_names or unclear_names)

    notes: list[str] = []
    known_by_bare = {_bare(t): t for t in known_tickers}

    async def resolve_all(names: list[str], *, budget: int) -> list[str]:
        resolved: list[str] = []
        for raw in list(dict.fromkeys(n.strip().upper() for n in names if n.strip())):
            if len(resolved) >= budget:
                notes.append(
                    f"Only the first {settings.max_tickers} companies were analysed "
                    f"(limit reached); skipped: {raw}."
                )
                continue
            existing = known_by_bare.get(_bare(raw))
            if existing:
                resolved.append(existing)
                continue
            found = await aresolve_ticker(raw)
            if found.symbol:
                resolved.append(found.symbol)
            elif found.unsupported_market:
                notes.append(f"'{raw}' isn't a US-listed stock, so it isn't currently supported.")
            else:
                notes.append(f"'{raw}' could not be found and was skipped.")
        return list(dict.fromkeys(resolved))

    # The cap bounds this turn's cost, which is what it exists for: fetches are at most
    # len(scope) x len(aspects). Deliberately NOT applied to the session's accumulated
    # ticker count the way the old `_plan_add_ticker` did — scope is per-turn now, so
    # there is no cost argument for refusing to look at a sixth company on turn twelve,
    # and doing so would make a long session progressively less useful for no reason.
    subjects = await resolve_all(subject_names, budget=settings.max_tickers)
    unclear = await resolve_all(unclear_names, budget=max(settings.max_tickers - len(subjects), 0))

    log_event(
        logger,
        "scope resolved",
        subjects=subjects,
        unclear=unclear,
        attempted=attempted,
        dropped=len(notes),
    )
    return ScopeResolution(
        subjects=subjects, unclear=unclear, notes=notes, attempted=attempted
    )
