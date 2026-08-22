"""Turns the model's observations into the turn's plan. Pure — no I/O, no model, no clock
of its own.

This is where every decision the LLM used to make now lives: scope, shape, which cells
need fetching, and whether the turn researches, recalls, clarifies, or chats. All of it is
a function of `(TurnIntent, ScopeResolution, SessionState, now)`, so the entire decision
surface of the system is testable by exact assertion with no API key and no mocking.

Two rules are load-bearing and worth stating up front, because both encode audit findings:

**A named company always wins.** Rung 1 of the scope ladder is the current message's own
companies, resolved. Nothing is inherited when the user named something — which is what
makes a narrowing follow-up narrow ("how is NVDA doing now?" in a comparison session
scopes to NVDA alone, and shape follows scope, so the answer is a single-stock answer).
A-01 was possible because shape was persisted state that only ratcheted upward and
synthesis dispatched straight off it.

**Falling through is clarifying, not guessing.** When the ladder cannot resolve a scope,
the turn asks. There is no rung that defaults to "the whole session" — that default is
precisely how a single-ticker recall question in a three-ticker session escalated into
nine specialist runs (A-04).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from app import replies
from app.graph.freshness import needs_fetch
from app.graph.intent import TurnIntent, normalized_shape_hint
from app.graph.session import (
    AGENT_NAMES,
    AgentName,
    CellRef,
    SessionState,
    Shape,
    TurnPlan,
    fresh_turn,
    session_tickers,
)


@dataclass(frozen=True)
class ScopeResolution:
    """What ticker resolution made of the companies the model extracted.

    Produced by `resolve_scope` (async, network — it validates symbols against real price
    history) and passed in here so this module stays pure. The split matters for testing:
    every scope and shape decision can be exercised against a hand-built `ScopeResolution`
    without touching a provider.
    """

    # Resolved symbols for companies the model marked `research_subject`.
    subjects: list[str] = field(default_factory=list)
    # Resolved symbols for companies marked `unclear` — candidates, pending the gate below.
    unclear: list[str] = field(default_factory=list)
    # Drop reasons for anything that failed validation, already user-phrased.
    notes: list[str] = field(default_factory=list)
    # Whether the model named any company at all that we tried to resolve. Distinguishes
    # "the user named nothing" from "the user named something and it didn't validate" —
    # two cases that must not share a reply, and must not share a scope.
    attempted: bool = False


def plan_turn(
    *,
    intent: TurnIntent,
    resolution: ScopeResolution,
    state: SessionState,
    now: datetime,
) -> TurnPlan:
    turn = fresh_turn()
    turn["notes"] = list(resolution.notes)
    turn["off_domain_topic"] = intent.off_domain_topic
    turn["aspects"] = _normalize_aspects(intent.aspects)

    tickers = session_tickers(state)
    companies = intent.companies or []
    extracted_names = tuple(c.name for c in companies)

    # ── Screening: the user wants candidates found for them, which this app doesn't do.
    # Checked before anything else so a named company can still override it — a real
    # company beats a fuzzy classification, and `resolution.subjects` being non-empty
    # means one was named and validated.
    if intent.screening_scope and not resolution.subjects:
        return _chat(turn, replies.screening(intent.screening_scope, extracted_names))

    # ── Named companies, none survived validation. This must NOT fall through to the
    # ladder: inheriting `last_scope` here would answer about a company the user didn't
    # ask about, which is worse than admitting we couldn't find the one they did.
    if resolution.attempted and not resolution.subjects and not resolution.unclear:
        return _chat(turn, replies.unresolved_tickers(resolution.notes, tickers))

    candidates = resolution.subjects or resolution.unclear

    # ── Nothing researchable and nothing pointed at → the conversational lane. The stock
    # architecture is never entered; `researched` is not read.
    if not candidates and not intent.refers_to_prior:
        incidental = [c.name for c in companies if c.role == "incidental"]
        reply = (
            replies.off_domain_with_company(incidental[0], tickers)
            if incidental
            else replies.off_domain(intent.off_domain_topic, tickers)
        )
        return _chat(turn, reply)

    # ── Scope ladder.
    scope = _resolve_scope(candidates, intent, state, tickers)
    if scope is None:
        return _clarify(turn, replies.clarify_referent(tickers))

    turn["scope"] = scope
    turn["shape"] = _resolve_shape(scope, intent, state)
    turn["fetch"] = _compute_fetch(scope, turn["aspects"], state, now)

    # ── The asymmetric gate for an unclear subject.
    #
    # Scope was resolved from a company whose role the model could not determine ("How is
    # Amazon doing?"). Whether that uncertainty is worth interrupting the user over depends
    # entirely on what acting on it would cost, and `fetch` is exactly that measure —
    # already computed, one line above.
    #
    # If nothing needs fetching, answering is cheap and reversible: a wrong guess costs one
    # paragraph the user corrects in the next breath, so answer with a hedge. If a fetch
    # would run, a wrong guess spends real API budget and produces a confident, fully-cited
    # report about a company nobody asked about — so ask first. Same uncertainty, opposite
    # action, decided by cost rather than by a self-reported confidence score.
    if resolution.unclear and not resolution.subjects:
        if turn["fetch"]:
            unclear_names = [c.name for c in companies if c.role == "unclear"]
            return _clarify(turn, replies.clarify_intent(unclear_names[0] if unclear_names else None))
        turn["hedged"] = True

    turn["kind"] = "research" if turn["fetch"] else "recall"
    return turn


# ─────────────────────────── ladders ───────────────────────────


def _resolve_scope(
    candidates: list[str],
    intent: TurnIntent,
    state: SessionState,
    tickers: list[str],
) -> list[str] | None:
    """Which tickers this answer covers, or `None` to clarify.

    Rung order is the whole design; see the module docstring.
    """
    # 1. Companies named in THIS message. Always wins — nothing is inherited.
    if candidates:
        return list(dict.fromkeys(candidates))

    # 2. A backward reference resolves against the PREVIOUS TURN's scope, not the
    #    session's full ticker list. A session holding NVDA, AMD and INTC whose last turn
    #    discussed only the first two must read "which one is better?" as those two.
    last_scope = [t for t in (state.get("last_scope") or []) if t in tickers]
    if intent.refers_to_prior and last_scope:
        return last_scope

    # 3. A session with exactly one ticker has no ambiguity to resolve.
    if len(tickers) == 1:
        return list(tickers)

    # 4. Out of rungs. Ask rather than guess.
    return None


def _resolve_shape(scope: list[str], intent: TurnIntent, state: SessionState) -> Shape:
    """Shape describes how many things this answer is about, which is a fact about the
    scope rather than an opinion — so `|scope|` leads and the model's hint only breaks
    ties above one.
    """
    # Unconditional, and deliberately ahead of the hint: one company cannot be compared
    # with itself, and a stale "comparison" must never survive a narrowing turn (A-01).
    if len(scope) == 1:
        return "single"

    hint = normalized_shape_hint(intent.shape_hint)
    if hint in ("comparison", "portfolio"):
        return hint

    # Carry forward a portfolio framing, but only within the same set of holdings —
    # widening the set means the user is doing something new with it.
    last_scope = state.get("last_scope") or []
    if state.get("last_shape") == "portfolio" and set(scope) <= set(last_scope):
        return "portfolio"

    # Safer default of the two: a comparison of holdings still reads sensibly, where a
    # "portfolio" of two unrelated stocks the user never described as holdings does not.
    return "comparison"


def _normalize_aspects(aspects: list[AgentName] | None) -> list[AgentName]:
    """Empty means all three — the absence of a restriction is not a restriction.

    Canonical order regardless of what the model emitted, so `fetch` and every rendered
    section are deterministic for the same inputs.
    """
    requested = {a for a in (aspects or []) if a in AGENT_NAMES}
    if not requested:
        return list(AGENT_NAMES)
    return [a for a in AGENT_NAMES if a in requested]


def _compute_fetch(
    scope: list[str], aspects: list[AgentName], state: SessionState, now: datetime
) -> list[CellRef]:
    """Exactly the cells that fail the freshness check — no more, no less.

    Per (ticker, agent), never per ticker and never per turn, which is what lets a single
    stale technical cell re-run one node instead of escalating the whole session (A-04,
    A-05). Scoped to this turn's tickers, so untouched research stays untouched.
    """
    researched = state.get("researched") or {}
    return [
        CellRef(ticker=ticker, agent=agent)
        for ticker in scope
        for agent in aspects
        if needs_fetch(researched.get(ticker, {}).get(agent), agent, now)
    ]


# ─────────────────────────── terminal plans ───────────────────────────


def _chat(turn: TurnPlan, reply: str) -> TurnPlan:
    turn["kind"] = "chat"
    turn["reply"] = reply
    return turn


def _clarify(turn: TurnPlan, question: str) -> TurnPlan:
    """A clarifying turn carries its question but not the `pending` record — that is
    session state, assembled by the graph node from this plan plus the current attempt
    count. Keeping the counter out of here is what keeps this function pure.
    """
    turn["kind"] = "clarify"
    turn["reply"] = question
    turn["scope"] = []
    turn["fetch"] = []
    return turn
