"""`plan_turn` holds every decision the LLM used to make, so this is the suite that pins
the audit's findings. No mocking and no API key: the whole decision surface is a pure
function of (intent, resolution, state, clock).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.graph.intent import CompanyRef, TurnIntent
from app.graph.plan_turn import ScopeResolution, plan_turn
from app.graph.session import AGENT_NAMES, SessionState, Shape, fresh_turn

NOW = datetime(2026, 8, 19, 14, 0, tzinfo=timezone.utc)  # Wednesday, 10:00 ET — market open


def _cells(*, fresh: bool = True, findings: int = 1, status: str = "ok") -> dict:
    stamp = NOW - (timedelta(seconds=30) if fresh else timedelta(hours=6))
    return {
        agent: {
            "status": status,
            "summary": "s",
            "findings": [{"id": f"{agent}-{i}"} for i in range(findings)],
            "error": None if status == "ok" else "boom",
            "fetched_at": stamp.isoformat(),
        }
        for agent in AGENT_NAMES
    }


def _state(
    *,
    researched: dict | None = None,
    last_scope: list[str] | None = None,
    last_shape: Shape = "single",
    question: str = "q",
) -> SessionState:
    return SessionState(
        session_id="s1",
        user_question=question,
        researched=researched or {},
        conversation=[],
        last_scope=last_scope or [],
        last_shape=last_shape,
        pending=None,
        turn=fresh_turn(),
    )


def _intent(
    *,
    companies: list[tuple[str, str]] | None = None,
    refers_to_prior: bool = False,
    screening_scope: str | None = None,
    shape_hint: str = "none",
    aspects: list[str] | None = None,
    off_domain_topic: str | None = None,
) -> TurnIntent:
    return TurnIntent(
        companies=[
            CompanyRef(name=n, role=r, ticker=n.upper()) for n, r in (companies or [])
        ],
        refers_to_prior=refers_to_prior,
        screening_scope=screening_scope,
        shape_hint=shape_hint,
        aspects=aspects or [],
        off_domain_topic=off_domain_topic,
    )


def _fetch_pairs(plan) -> set[tuple[str, str]]:
    return {(c["ticker"], c["agent"]) for c in plan["fetch"]}


# ─────────────────────── A-01: shape narrows as well as widens ───────────────────────


def test_single_ticker_question_in_a_comparison_session_produces_a_single_answer() -> None:
    """A-01 regression, the bug that started this refactor.

    `_dispatch_synthesis` was `return state["query_type"]`, and `query_type` was persisted
    state that only ratcheted upward. So "how is NVDA doing now?" in an NVDA-vs-AMD session
    re-rendered the entire comparison — including AMD's hours-old findings presented as
    current. Shape now follows scope, and scope comes from the message.
    """
    state = _state(
        researched={"NVDA": _cells(), "AMD": _cells()},
        last_scope=["NVDA", "AMD"],
        last_shape="comparison",
    )
    plan = plan_turn(
        intent=_intent(companies=[("NVDA", "research_subject")]),
        resolution=ScopeResolution(subjects=["NVDA"], attempted=True),
        state=state,
        now=NOW,
    )

    assert plan["scope"] == ["NVDA"]
    assert plan["shape"] == "single"
    assert "AMD" not in {c["ticker"] for c in plan["fetch"]}


def test_a_named_company_always_beats_the_previous_scope() -> None:
    """Rung 1 of the ladder. Nothing is inherited when the user named something — the
    property that makes narrowing possible at all.
    """
    state = _state(researched={"NVDA": _cells()}, last_scope=["NVDA"], last_shape="single")
    plan = plan_turn(
        intent=_intent(companies=[("Intel", "research_subject")], refers_to_prior=True),
        resolution=ScopeResolution(subjects=["INTC"], attempted=True),
        state=state,
        now=NOW,
    )

    assert plan["scope"] == ["INTC"]


def test_shape_hint_cannot_override_a_single_ticker_scope() -> None:
    """One company cannot be compared with itself. `|scope| == 1` is checked before the
    hint precisely so a stale comparison framing cannot survive a narrowing turn.
    """
    state = _state(researched={"NVDA": _cells()}, last_scope=["NVDA"], last_shape="comparison")
    plan = plan_turn(
        intent=_intent(companies=[("NVDA", "research_subject")], shape_hint="comparison"),
        resolution=ScopeResolution(subjects=["NVDA"], attempted=True),
        state=state,
        now=NOW,
    )

    assert plan["shape"] == "single"


# ─────────────────────── backward references (Q7) ───────────────────────


def test_which_one_is_better_resolves_to_the_previous_turns_scope() -> None:
    state = _state(
        researched={"NVDA": _cells(), "AMD": _cells()},
        last_scope=["NVDA", "AMD"],
        last_shape="comparison",
    )
    plan = plan_turn(
        intent=_intent(refers_to_prior=True, shape_hint="comparison"),
        resolution=ScopeResolution(),
        state=state,
        now=NOW,
    )

    assert plan["scope"] == ["NVDA", "AMD"]
    assert plan["shape"] == "comparison"
    assert plan["kind"] == "recall"      # everything fresh — no API calls
    assert plan["fetch"] == []


def test_backward_reference_uses_last_scope_not_the_whole_session() -> None:
    """The distinction that forced `last_scope` into durable state. A session can hold
    three tickers while the last turn discussed two; "which one is better?" means those
    two, and answering about all three would be a different question.
    """
    state = _state(
        researched={"NVDA": _cells(), "AMD": _cells(), "INTC": _cells()},
        last_scope=["NVDA", "AMD"],
        last_shape="comparison",
    )
    plan = plan_turn(
        intent=_intent(refers_to_prior=True),
        resolution=ScopeResolution(),
        state=state,
        now=NOW,
    )

    assert plan["scope"] == ["NVDA", "AMD"]


def test_ambiguous_reference_with_no_previous_scope_clarifies_rather_than_guessing() -> None:
    """A-07 / ladder exhaustion. There is deliberately no rung that defaults to the whole
    session — that default is how a one-ticker question became a nine-node re-research.
    """
    state = _state(researched={"NVDA": _cells(), "AMD": _cells(), "INTC": _cells()})
    plan = plan_turn(
        intent=_intent(refers_to_prior=True),
        resolution=ScopeResolution(),
        state=state,
        now=NOW,
    )

    assert plan["kind"] == "clarify"
    assert plan["scope"] == []
    assert plan["fetch"] == []
    # The choices come from state, so they are always answerable.
    for ticker in ("NVDA", "AMD", "INTC"):
        assert ticker in plan["reply"]


def test_single_ticker_session_resolves_a_reference_without_asking() -> None:
    state = _state(researched={"AAPL": _cells()})
    plan = plan_turn(
        intent=_intent(refers_to_prior=True),
        resolution=ScopeResolution(),
        state=state,
        now=NOW,
    )

    assert plan["scope"] == ["AAPL"]
    assert plan["kind"] == "recall"


# ─────────────────────── A-04 / A-05: fetch is per cell ───────────────────────


def test_recall_question_does_not_re_research_the_rest_of_the_session() -> None:
    """A-04 regression.

    The old guard swept `state["tickers"]` unconditionally, so a single-ticker recall in a
    three-ticker session where everything had aged escalated to nine specialist runs and
    nine LLM calls to answer one question about one company.
    """
    stale = {"NVDA": _cells(fresh=False), "AMD": _cells(fresh=False), "INTC": _cells(fresh=False)}
    state = _state(researched=stale, last_scope=["NVDA", "AMD", "INTC"], last_shape="portfolio")

    plan = plan_turn(
        intent=_intent(companies=[("NVDA", "research_subject")]),
        resolution=ScopeResolution(subjects=["NVDA"], attempted=True),
        state=state,
        now=NOW,
    )

    assert {c["ticker"] for c in plan["fetch"]} == {"NVDA"}


def test_only_the_stale_agent_is_refetched() -> None:
    """Q10. `fetch` is a list of cells, not tickers, so partial staleness costs one node."""
    cells = _cells()
    cells["technical"]["fetched_at"] = (NOW - timedelta(hours=3)).isoformat()
    state = _state(researched={"NVDA": cells}, last_scope=["NVDA"])

    plan = plan_turn(
        intent=_intent(companies=[("NVDA", "research_subject")]),
        resolution=ScopeResolution(subjects=["NVDA"], attempted=True),
        state=state,
        now=NOW,
    )

    assert _fetch_pairs(plan) == {("NVDA", "technical")}
    assert plan["kind"] == "research"


def test_a_ticker_never_researched_fetches_every_aspect() -> None:
    state = _state(researched={"NVDA": _cells()}, last_scope=["NVDA"])
    plan = plan_turn(
        intent=_intent(companies=[("Intel", "research_subject")], shape_hint="comparison"),
        resolution=ScopeResolution(subjects=["INTC"], attempted=True),
        state=state,
        now=NOW,
    )

    assert _fetch_pairs(plan) == {("INTC", a) for a in AGENT_NAMES}


def test_failed_cells_are_retried_on_the_next_turn() -> None:
    """A-03 reaching through `plan_turn`: a total failure leaves three fresh timestamps,
    and the old guard called that fresh, so the retry never happened.
    """
    state = _state(researched={"NVDA": _cells(status="failed")}, last_scope=["NVDA"])
    plan = plan_turn(
        intent=_intent(refers_to_prior=True),
        resolution=ScopeResolution(),
        state=state,
        now=NOW,
    )

    assert _fetch_pairs(plan) == {("NVDA", a) for a in AGENT_NAMES}
    assert plan["kind"] == "research"


# ─────────────────────── Q19: aspects ───────────────────────


def test_a_fundamentals_only_question_fetches_one_cell() -> None:
    state = _state()
    plan = plan_turn(
        intent=_intent(companies=[("Apple", "research_subject")], aspects=["fundamentals"]),
        resolution=ScopeResolution(subjects=["AAPL"], attempted=True),
        state=state,
        now=NOW,
    )

    assert plan["aspects"] == ["fundamentals"]
    assert _fetch_pairs(plan) == {("AAPL", "fundamentals")}


def test_no_named_aspect_means_all_three() -> None:
    plan = plan_turn(
        intent=_intent(companies=[("Apple", "research_subject")]),
        resolution=ScopeResolution(subjects=["AAPL"], attempted=True),
        state=_state(),
        now=NOW,
    )

    assert plan["aspects"] == list(AGENT_NAMES)


def test_aspects_are_returned_in_canonical_order_whatever_the_model_emitted() -> None:
    plan = plan_turn(
        intent=_intent(companies=[("Apple", "research_subject")], aspects=["news", "fundamentals"]),
        resolution=ScopeResolution(subjects=["AAPL"], attempted=True),
        state=_state(),
        now=NOW,
    )

    assert plan["aspects"] == ["fundamentals", "news"]


def test_narrowing_aspects_does_not_discard_the_rest_of_the_session() -> None:
    state = _state(researched={"AAPL": _cells()}, last_scope=["AAPL"])
    plan = plan_turn(
        intent=_intent(companies=[("Apple", "research_subject")], aspects=["news"]),
        resolution=ScopeResolution(subjects=["AAPL"], attempted=True),
        state=state,
        now=NOW,
    )

    assert plan["aspects"] == ["news"]
    assert state["researched"]["AAPL"].keys() == set(AGENT_NAMES)


# ─────────────────────── the unclear gate (Doubt 3) ───────────────────────


def test_unclear_subject_clarifies_when_a_fetch_would_run() -> None:
    """Asymmetric gate, expensive side. A wrong guess here spends API budget and produces
    a confident, fully-cited report about a company nobody asked about.
    """
    plan = plan_turn(
        intent=_intent(companies=[("Amazon", "unclear")]),
        resolution=ScopeResolution(unclear=["AMZN"], attempted=True),
        state=_state(),
        now=NOW,
    )

    assert plan["kind"] == "clarify"
    assert "Amazon" in plan["reply"]


def test_unclear_subject_answers_with_a_hedge_when_nothing_needs_fetching() -> None:
    """Cheap side of the same gate: a wrong guess costs one paragraph the user corrects
    immediately, so interrupting them would be the worse trade.
    """
    state = _state(researched={"AMZN": _cells()}, last_scope=["AMZN"])
    plan = plan_turn(
        intent=_intent(companies=[("Amazon", "unclear")]),
        resolution=ScopeResolution(unclear=["AMZN"], attempted=True),
        state=state,
        now=NOW,
    )

    assert plan["kind"] == "recall"
    assert plan["hedged"] is True
    assert plan["scope"] == ["AMZN"]


def test_a_clear_subject_alongside_an_unclear_one_is_not_hedged() -> None:
    plan = plan_turn(
        intent=_intent(companies=[("NVDA", "research_subject"), ("Amazon", "unclear")]),
        resolution=ScopeResolution(subjects=["NVDA"], unclear=["AMZN"], attempted=True),
        state=_state(),
        now=NOW,
    )

    assert plan["kind"] == "research"
    assert plan["hedged"] is False
    assert plan["scope"] == ["NVDA"]


# ─────────────────────── the conversational lane ───────────────────────


def test_an_incidental_company_never_triggers_research() -> None:
    """'Amazon is not doing well, I might switch jobs'. The company is named; nothing is
    asked of it. The stock lane is never entered.
    """
    plan = plan_turn(
        intent=_intent(
            companies=[("Amazon", "incidental")],
            off_domain_topic="considering a job change",
        ),
        resolution=ScopeResolution(),
        state=_state(),
        now=NOW,
    )

    assert plan["kind"] == "chat"
    assert plan["scope"] == []
    assert plan["fetch"] == []
    assert "Amazon" in plan["reply"]        # offered, not researched


def test_off_domain_with_no_company_stays_in_the_chat_lane() -> None:
    plan = plan_turn(
        intent=_intent(off_domain_topic="writing a resignation email"),
        resolution=ScopeResolution(),
        state=_state(researched={"AAPL": _cells()}),
        now=NOW,
    )

    assert plan["kind"] == "chat"
    assert "writing a resignation email" in plan["reply"]
    # Session-aware offer: mid-session it should point back at the research in play.
    assert "AAPL" in plan["reply"]


def test_screening_request_produces_a_decline_and_invents_no_ticker() -> None:
    plan = plan_turn(
        intent=_intent(screening_scope="Indian stocks"),
        resolution=ScopeResolution(),
        state=_state(),
        now=NOW,
    )

    assert plan["kind"] == "chat"
    assert plan["scope"] == []
    assert "Indian stocks" in plan["reply"]


def test_a_named_company_overrides_a_screening_flag() -> None:
    """Concrete information beats a fuzzy classification — the same backend guard the old
    router had, preserved.
    """
    plan = plan_turn(
        intent=_intent(
            companies=[("NVDA", "research_subject")], screening_scope="chip stocks"
        ),
        resolution=ScopeResolution(subjects=["NVDA"], attempted=True),
        state=_state(),
        now=NOW,
    )

    assert plan["kind"] == "research"
    assert plan["scope"] == ["NVDA"]


def test_named_companies_that_fail_resolution_do_not_inherit_the_previous_scope() -> None:
    """The edge case that would be a silent wrong answer: the user asked about a company
    we could not resolve, so answering about *last* turn's company would be answering a
    question they did not ask.
    """
    state = _state(researched={"NVDA": _cells()}, last_scope=["NVDA"], last_shape="single")
    plan = plan_turn(
        intent=_intent(companies=[("Wakanda Corp", "research_subject")]),
        resolution=ScopeResolution(
            notes=["'WAKANDA' could not be found and was skipped."], attempted=True
        ),
        state=state,
        now=NOW,
    )

    assert plan["kind"] == "chat"
    assert plan["scope"] == []
    assert "could not be found" in plan["reply"]


def test_mixed_message_researches_and_carries_the_off_domain_half() -> None:
    """Q13. Both halves survive: the finance request runs, and the personal half is kept
    on the plan so `emit` can acknowledge it after the report instead of dropping it.
    """
    plan = plan_turn(
        intent=_intent(
            companies=[("Amazon", "research_subject")],
            off_domain_topic="the job decision",
        ),
        resolution=ScopeResolution(subjects=["AMZN"], attempted=True),
        state=_state(),
        now=NOW,
    )

    assert plan["kind"] == "research"
    assert plan["scope"] == ["AMZN"]
    assert plan["off_domain_topic"] == "the job decision"


# ─────────────────────── shape ladder ───────────────────────


def test_multiple_tickers_default_to_comparison() -> None:
    plan = plan_turn(
        intent=_intent(
            companies=[("NVDA", "research_subject"), ("AMD", "research_subject")]
        ),
        resolution=ScopeResolution(subjects=["NVDA", "AMD"], attempted=True),
        state=_state(),
        now=NOW,
    )

    assert plan["shape"] == "comparison"


def test_portfolio_framing_carries_forward_within_the_same_holdings() -> None:
    state = _state(
        researched={"NVDA": _cells(), "AMD": _cells()},
        last_scope=["NVDA", "AMD"],
        last_shape="portfolio",
    )
    plan = plan_turn(
        intent=_intent(refers_to_prior=True),
        resolution=ScopeResolution(),
        state=state,
        now=NOW,
    )

    assert plan["shape"] == "portfolio"


def test_portfolio_framing_does_not_carry_onto_a_widened_set() -> None:
    """Adding a company the user never described as a holding means they are doing
    something new with the set, so the framing is re-derived rather than inherited.
    """
    state = _state(
        researched={"NVDA": _cells(), "AMD": _cells()},
        last_scope=["NVDA", "AMD"],
        last_shape="portfolio",
    )
    plan = plan_turn(
        intent=_intent(
            companies=[
                ("NVDA", "research_subject"),
                ("AMD", "research_subject"),
                ("Intel", "research_subject"),
            ]
        ),
        resolution=ScopeResolution(subjects=["NVDA", "AMD", "INTC"], attempted=True),
        state=state,
        now=NOW,
    )

    assert plan["shape"] == "comparison"


def test_duplicate_companies_collapse_to_one_scope_entry() -> None:
    plan = plan_turn(
        intent=_intent(
            companies=[("Apple", "research_subject"), ("AAPL", "research_subject")]
        ),
        resolution=ScopeResolution(subjects=["AAPL", "AAPL"], attempted=True),
        state=_state(),
        now=NOW,
    )

    assert plan["scope"] == ["AAPL"]
    assert plan["shape"] == "single"


# ─────────────────────── per-turn hygiene ───────────────────────


def test_resolution_notes_land_on_the_turn_not_on_session_state() -> None:
    """A-08: `notes` used to accumulate forever via `operator.add`, and one reply node
    rendered the whole accumulated list — surfacing an earlier turn's aside as part of a
    later turn's answer. They now live on the turn and die with it.
    """
    plan = plan_turn(
        intent=_intent(
            companies=[("NVDA", "research_subject"), ("Wakanda", "research_subject")]
        ),
        resolution=ScopeResolution(
            subjects=["NVDA"],
            notes=["'WAKANDA' could not be found and was skipped."],
            attempted=True,
        ),
        state=_state(),
        now=NOW,
    )

    assert plan["notes"] == ["'WAKANDA' could not be found and was skipped."]
    assert plan["scope"] == ["NVDA"]


def test_clarify_and_chat_turns_never_carry_a_fetch_list() -> None:
    for intent, resolution in (
        (_intent(refers_to_prior=True), ScopeResolution()),
        (_intent(screening_scope="pharma"), ScopeResolution()),
        (_intent(off_domain_topic="a pasta recipe"), ScopeResolution()),
    ):
        plan = plan_turn(
            intent=intent,
            resolution=resolution,
            state=_state(researched={"A": _cells(), "B": _cells()}),
            now=NOW,
        )
        assert plan["fetch"] == []
        assert plan["kind"] in ("chat", "clarify")
