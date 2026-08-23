"""The classify -> resolve -> plan pipeline: the projection the model sees, symbol
resolution, and the state assembly `plan_node` does around the pure planner.

Provider and LLM calls are mocked; what's under test is the wiring and the decisions made
around them, not the model's judgment (that belongs in the classifier eval suite).
"""

from __future__ import annotations

import pytest

import app.graph.nodes.plan_node as plan_node_mod
import app.graph.resolve_scope as resolve_mod
from app.graph.intent import CompanyRef, TurnIntent
from app.graph.nodes.classify_turn import _projection
from app.graph.nodes.plan_node import MAX_CLARIFY_ATTEMPTS, plan_node
from app.graph.resolve_scope import resolve_scope
from app.graph.session import AGENT_NAMES, SessionState, fresh_turn
from app.llm.errors import RATE_LIMIT_MESSAGE, LLMAnalysisError
from app.tools.market_data import ResolvedTicker


def _state(
    *,
    question: str = "q",
    researched: dict | None = None,
    conversation: list | None = None,
    last_scope: list[str] | None = None,
    pending: dict | None = None,
) -> SessionState:
    return SessionState(
        session_id="s1",
        user_question=question,
        researched=researched or {},
        conversation=conversation or [],
        last_scope=last_scope or [],
        last_shape="single",
        pending=pending,
        turn=fresh_turn(),
    )


def _cells() -> dict:
    return {
        agent: {
            "status": "ok",
            "summary": "s",
            "findings": [{"id": "f1"}],
            "error": None,
            "fetched_at": "2026-08-19T14:00:00+00:00",
        }
        for agent in AGENT_NAMES
    }


def _intent(**kwargs) -> TurnIntent:
    companies = [
        CompanyRef(name=n, role=r, ticker=n.upper()) for n, r in kwargs.pop("companies", [])
    ]
    return TurnIntent(companies=companies, **kwargs)


# ─────────────────────── the projection ───────────────────────


def test_projection_shows_user_turns_verbatim_and_assistant_turns_as_gists() -> None:
    """The report body is withheld, not truncated.

    A comparison report mentions competitors, suppliers and analyst firms in passing, and
    the classifier's main job is pulling company names out of text — so an Intel mention
    inside an NVDA report is a live risk of INTC entering `companies[]` on a turn where
    the user never said it. Truncating to N characters only changes which paragraph leaks.
    """
    conversation = [
        {"role": "user", "content": "compare nvidia and amd"},
        {
            "role": "assistant",
            "content": "NVDA leads on momentum while Intel and Broadcom lag the sector...",
            "gist": "(comparison on NVDA, AMD)",
        },
        {"role": "user", "content": "which one is better?"},
    ]
    text = _projection(_state(conversation=conversation, researched={"NVDA": _cells()}))

    assert "compare nvidia and amd" in text
    assert "which one is better?" in text
    assert "(comparison on NVDA, AMD)" in text
    assert "Broadcom" not in text
    assert "Intel" not in text


def test_projection_states_the_pending_question_outright() -> None:
    """What makes the dedicated clarification-resolver nodes unnecessary: the reply is
    classified as an ordinary message with the question it answers in plain view.
    """
    state = _state(
        question="the second one",
        pending={
            "question": "This session has NVDA, AMD and INTC — which did you mean?",
            "original_question": "how's the other one doing?",
            "attempts": 1,
        },
    )
    text = _projection(state)

    assert "which did you mean?" in text
    assert "how's the other one doing?" in text


def test_projection_reports_an_empty_session_plainly() -> None:
    assert "Nothing has been researched" in _projection(_state())


# ─────────────────────── symbol resolution ───────────────────────


async def test_resolution_splits_subjects_from_unclear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        resolve_mod, "aresolve_ticker", _fake_resolver({"NVDA": "NVDA", "AMAZON": "AMZN"})
    )
    intent = _intent(companies=[("NVDA", "research_subject"), ("Amazon", "unclear")])

    resolution = await resolve_scope(intent, [])

    assert resolution.subjects == ["NVDA"]
    assert resolution.unclear == ["AMZN"]
    assert resolution.attempted is True


async def test_incidental_companies_are_never_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    """An incidental mention must not cost a provider round-trip, and must not be able to
    enter scope by accident.
    """
    calls: list[str] = []

    async def _tracking(ticker: str) -> ResolvedTicker:
        calls.append(ticker)
        return ResolvedTicker("AMZN")

    monkeypatch.setattr(resolve_mod, "aresolve_ticker", _tracking)
    intent = _intent(companies=[("Amazon", "incidental")])

    resolution = await resolve_scope(intent, [])

    assert calls == []
    assert resolution.attempted is False
    assert resolution.subjects == []


async def test_a_known_session_ticker_is_reused_without_a_network_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session that resolved 'TCS' to 'TCS.NS' must keep matching later bare mentions to
    that same symbol — re-resolving could land somewhere else, and bare 'TCS' is exactly
    the symbol observed resolving to a delisted company.
    """
    calls: list[str] = []

    async def _tracking(ticker: str) -> ResolvedTicker:
        calls.append(ticker)
        return ResolvedTicker("SOMETHING-ELSE")

    monkeypatch.setattr(resolve_mod, "aresolve_ticker", _tracking)
    intent = _intent(companies=[("TCS", "research_subject")])

    resolution = await resolve_scope(intent, ["TCS.NS"])

    assert resolution.subjects == ["TCS.NS"]
    assert calls == []


async def test_unresolvable_and_unsupported_tickers_get_distinct_notes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _resolver(ticker: str) -> ResolvedTicker:
        if ticker == "TCS":
            return ResolvedTicker(None, unsupported_market=True)
        return ResolvedTicker(None)

    monkeypatch.setattr(resolve_mod, "aresolve_ticker", _resolver)
    intent = _intent(
        companies=[("TCS", "research_subject"), ("Wakanda", "research_subject")]
    )

    resolution = await resolve_scope(intent, [])

    assert resolution.subjects == []
    assert any("isn't currently supported" in n for n in resolution.notes)
    assert any("could not be found" in n for n in resolution.notes)


async def test_scope_is_capped_at_max_tickers(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import settings

    monkeypatch.setattr(
        resolve_mod, "aresolve_ticker", _fake_resolver({}, default_symbol_is_input=True)
    )
    names = [f"C{i}" for i in range(settings.max_tickers + 2)]
    intent = _intent(companies=[(n, "research_subject") for n in names])

    resolution = await resolve_scope(intent, [])

    assert len(resolution.subjects) == settings.max_tickers
    assert any("limit reached" in n for n in resolution.notes)


def _fake_resolver(mapping: dict[str, str], *, default_symbol_is_input: bool = False):
    async def _resolve(ticker: str) -> ResolvedTicker:
        if ticker in mapping:
            return ResolvedTicker(mapping[ticker])
        return ResolvedTicker(ticker if default_symbol_is_input else None)

    return _resolve


# ─────────────────────── plan_node state assembly ───────────────────────


def _patch_pipeline(monkeypatch: pytest.MonkeyPatch, intent: TurnIntent, mapping: dict) -> None:
    async def _classify(state):  # noqa: ARG001
        return intent

    monkeypatch.setattr(plan_node_mod, "classify_turn", _classify)
    monkeypatch.setattr(resolve_mod, "aresolve_ticker", _fake_resolver(mapping))


async def test_plan_node_records_the_user_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_pipeline(monkeypatch, _intent(companies=[("NVDA", "research_subject")]), {"NVDA": "NVDA"})

    update = await plan_node(_state(question="how is nvda?"))

    assert update["conversation"] == [{"role": "user", "content": "how is nvda?"}]


async def test_classification_failure_says_so_instead_of_answering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A-06 regression.

    The old handler returned `followup_path="answer"` directly, bypassing
    `apply_followup_decision` — and therefore the freshness guard — to answer from
    arbitrarily old context, with nothing telling the user classification had broken.
    """

    async def _boom(state):  # noqa: ARG001
        raise LLMAnalysisError("The analysis service is temporarily unavailable.")

    monkeypatch.setattr(plan_node_mod, "classify_turn", _boom)

    update = await plan_node(_state(researched={"NVDA": _cells()}))

    assert update["turn"]["kind"] == "chat"
    assert update["turn"]["fetch"] == []
    assert "rephrase" in update["turn"]["reply"]


async def test_rate_limited_classification_gets_the_specific_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _boom(state):  # noqa: ARG001
        raise LLMAnalysisError(RATE_LIMIT_MESSAGE)

    monkeypatch.setattr(plan_node_mod, "classify_turn", _boom)

    update = await plan_node(_state())

    assert "rate-limited" in update["turn"]["reply"]


async def test_clarifying_opens_a_pending_record(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_pipeline(monkeypatch, _intent(refers_to_prior=True), {})
    state = _state(
        question="how's the other one?",
        researched={"NVDA": _cells(), "AMD": _cells(), "INTC": _cells()},
    )

    update = await plan_node(state)

    assert update["turn"]["kind"] == "clarify"
    assert update["pending"]["attempts"] == 1
    assert update["pending"]["original_question"] == "how's the other one?"


async def test_repeated_vagueness_stops_asking_after_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A-07 regression.

    Both old resolvers could re-arm `awaiting_clarification` from inside their own
    resolution path with no counter anywhere, so a user replying vaguely could be asked a
    freshly-worded question forever.
    """
    _patch_pipeline(monkeypatch, _intent(refers_to_prior=True), {})
    state = _state(
        question="you know, the good one",
        researched={"NVDA": _cells(), "AMD": _cells(), "INTC": _cells()},
        pending={
            "question": "which did you mean?",
            "original_question": "how's the other one?",
            "attempts": MAX_CLARIFY_ATTEMPTS,
        },
    )

    update = await plan_node(state)

    assert update["turn"]["kind"] == "chat"
    assert update["pending"] is None
    assert "?" not in update["turn"]["reply"]
    assert "NVDA" in update["turn"]["reply"]


async def test_the_original_question_survives_repeated_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_pipeline(monkeypatch, _intent(refers_to_prior=True), {})
    state = _state(
        question="the good one",
        researched={"NVDA": _cells(), "AMD": _cells(), "INTC": _cells()},
        pending={
            "question": "which did you mean?",
            "original_question": "how's the other one doing?",
            "attempts": 1,
        },
    )

    update = await plan_node(state)

    assert update["pending"]["attempts"] == 2
    assert update["pending"]["original_question"] == "how's the other one doing?"


async def test_a_resolved_clarification_clears_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_pipeline(monkeypatch, _intent(companies=[("NVDA", "research_subject")]), {"NVDA": "NVDA"})
    state = _state(
        question="nvda",
        researched={"NVDA": _cells(), "AMD": _cells()},
        pending={"question": "which?", "original_question": "the other one?", "attempts": 1},
    )

    update = await plan_node(state)

    assert update["pending"] is None
    assert update["turn"]["scope"] == ["NVDA"]


async def test_a_company_name_is_resolved_via_the_models_ticker_not_the_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for a refactor bug that broke every natural-language question.

    `aresolve_ticker` *validates and corrects* symbols — it has no name lookup. Asking it
    to resolve "NVIDIA" fails, because no symbol is spelled that way. So when the intent
    schema stopped carrying a ticker and only reported the name as spoken, "Analyze NVIDIA"
    started answering "'NVIDIA' could not be found and was skipped" about a company that
    obviously exists.

    It survived a whole refactor because the eval stub was keyed on company names, making
    the double able to do something the real resolver cannot.
    """
    seen: list[str] = []

    async def _only_symbols(ticker: str) -> ResolvedTicker:
        seen.append(ticker)
        return ResolvedTicker("NVDA" if ticker == "NVDA" else None)

    monkeypatch.setattr(resolve_mod, "aresolve_ticker", _only_symbols)
    intent = TurnIntent(
        companies=[CompanyRef(name="NVIDIA", role="research_subject", ticker="NVDA")]
    )

    resolution = await resolve_scope(intent, [])

    assert resolution.subjects == ["NVDA"]
    assert seen == ["NVDA"], "the symbol must be looked up, not the spoken name"


async def test_a_company_the_model_cannot_ticker_falls_back_to_the_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It will not resolve, but dropping it silently would be worse: the attempt is
    recorded, so the user is told we couldn't find it rather than having it vanish.
    """
    monkeypatch.setattr(resolve_mod, "aresolve_ticker", _fake_resolver({}))
    intent = TurnIntent(
        companies=[CompanyRef(name="Some Private Startup", role="research_subject", ticker="STARTUP")]
    )

    resolution = await resolve_scope(intent, [])

    assert resolution.subjects == []
    assert resolution.attempted is True
    assert any("could not be found" in n for n in resolution.notes)
