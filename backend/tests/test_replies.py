"""The copy in `app/replies.py` makes claims about the product and about session state, so
its guarantees are asserted mechanically rather than trusted to a prompt.
"""

from __future__ import annotations

import pytest

from app import replies
from app.replies import CAPABILITIES, is_valid_slot


# ─────────────────────── A-12: capability facts have one home ───────────────────────


def test_no_reply_claims_a_data_source_we_do_not_use() -> None:
    """A-12 regression.

    `router_node`/`followup_router_node` hardcoded "this app researches stocks using Yahoo
    Finance data" into prompt strings, and it stayed there for four commits after
    fundamentals moved to Finnhub and price data to Twelve Data — because provider facts
    were duplicated into prompts instead of derived from one constant.
    """
    surfaces = [
        replies.off_domain("a pasta recipe", []),
        replies.off_domain(None, ["NVDA"]),
        replies.off_domain_with_company("Amazon", []),
        replies.screening("Indian stocks"),
        replies.clarify_intent("Amazon"),
        replies.clarify_referent(["NVDA", "AMD"]),
        replies.classification_failed(rate_limited=False),
        replies.unresolved_tickers(["'X' could not be found and was skipped."], []),
    ]
    for text in surfaces:
        assert "yahoo" not in text.lower(), text


def test_capabilities_name_the_providers_actually_in_use() -> None:
    joined = " ".join(CAPABILITIES.data_sources).lower()
    assert "finnhub" in joined
    assert "twelve data" in joined
    assert "yahoo" not in joined


# ─────────────────────── slot validation ───────────────────────


@pytest.mark.parametrize(
    "slot",
    [
        None,
        "",
        "   ",
        "I can't help you with that, but I'd be happy to look at something else",  # a sentence
        "I'm sorry",
        "sorry about that",
        "we can look at this",
        "one two three four five six seven eight nine",  # over the word cap
        "data from Yahoo Finance",  # a provider we don't use
    ],
)
def test_invalid_slots_are_rejected(slot: str | None) -> None:
    assert is_valid_slot(slot) is False


@pytest.mark.parametrize(
    "slot",
    ["Indian stocks", "drafting a resignation email", "the job decision", "healthcare"],
)
def test_valid_slots_are_accepted(slot: str) -> None:
    assert is_valid_slot(slot) is True


def test_a_rejected_slot_degrades_to_the_slotless_frame_not_an_error() -> None:
    text = replies.off_domain("I'm sorry, I really cannot help you with any of this", [])
    assert "stock research assistant" in text
    assert "I'm sorry" not in text


# ─────────────────────── screening must never name a company ───────────────────────


def test_screening_rejects_a_slot_echoing_an_extracted_company() -> None:
    """The constraint was previously a prompt instruction ("do NOT name any specific
    company in it — not even as an example"). Naming one immediately after declining to
    pick stocks *is* the recommendation the sentence just disclaimed, and this repo has
    already shipped a bug of exactly that shape: `citation_instruction()` asked for
    brackets around citation ids only and got `[Unavailable]` anyway.
    """
    text = replies.screening("Apple", extracted_companies=("Apple",))

    assert "Apple" not in text
    assert "I don't screen for candidates like that" in text


def test_screening_rejects_a_lone_proper_noun_even_when_not_extracted() -> None:
    """Backstop for when the classifier put a company in the scope slot without also
    listing it under `companies`. A screening scope is categorical by nature.
    """
    text = replies.screening("Nvidia")
    assert "Nvidia" not in text


def test_screening_keeps_a_genuine_scope_phrase() -> None:
    text = replies.screening("Indian stocks")
    assert "Indian stocks" in text
    assert "Name a few you're considering" in text


# ─────────────────────── clarification offers real choices ───────────────────────


def test_clarify_referent_lists_only_the_sessions_own_tickers() -> None:
    """The finding from the reply spec: `FollowUpDecision.clarifying_question` was
    free-form model text instructed to name the session's tickers, but never checked — so
    a session holding NVDA and AMD could ask "Did you mean AAPL or MSFT?", a question the
    user cannot answer correctly, inside what was already an unbounded retry loop.
    """
    text = replies.clarify_referent(["NVDA", "AMD"])

    assert "NVDA" in text and "AMD" in text
    assert "AAPL" not in text and "MSFT" not in text


def test_clarify_referent_handles_one_and_zero_tickers() -> None:
    assert "NVDA" in replies.clarify_referent(["NVDA"])
    assert replies.clarify_referent([])  # non-empty fallback, no crash


def test_clarify_exhausted_ends_the_loop_with_a_concrete_example() -> None:
    """A-07: after the attempt cap this must stop asking and hand the user a way out,
    rather than producing a third differently-worded question.
    """
    text = replies.clarify_exhausted(["NVDA", "AMD"])
    assert "NVDA" in text
    assert "?" not in text


# ─────────────────────── session awareness ───────────────────────


def test_off_domain_offer_is_session_aware() -> None:
    """The previous `DEFAULT_OFF_TOPIC_REPLY` was one static string suggesting "Should I
    buy TCS right now?" whether it was the user's first message or their tenth in an
    active session — which reads as though the app forgot the conversation.
    """
    fresh = replies.off_domain("a pasta recipe", [])
    mid_session = replies.off_domain("a pasta recipe", ["NVDA", "AMD"])

    assert "Ask me about a company" in fresh
    assert "NVDA and AMD" in mid_session
    assert "Ask me about a company" not in mid_session


def test_off_domain_with_company_offers_the_adjacent_research() -> None:
    text = replies.off_domain_with_company("Amazon", [])
    assert "Amazon" in text
    assert CAPABILITIES.aspects_phrase in text


def test_mixed_acknowledgment_is_empty_when_there_is_nothing_to_acknowledge() -> None:
    assert replies.mixed_acknowledgment(None) == ""
    assert replies.mixed_acknowledgment("the job decision").startswith("\n\n—")


# ─────────────────────── unresolved tickers ───────────────────────


def test_unsupported_market_does_not_imply_a_typo() -> None:
    """These resolved fine — they are simply outside coverage — so the generic "check the
    spelling" closer would be actively misleading.
    """
    text = replies.unresolved_tickers(
        ["'TCS' isn't a US-listed stock, so it isn't currently supported."], []
    )

    assert "double-check" not in text
    assert "US-listed company" in text


def test_a_genuinely_unknown_ticker_does_suggest_checking_it() -> None:
    text = replies.unresolved_tickers(["'WAKANDA' could not be found and was skipped."], [])
    assert "double-check" in text


# ─────────────────────── formatting helpers ───────────────────────


@pytest.mark.parametrize(
    ("items", "expected"),
    [([], ""), (["A"], "A"), (["A", "B"], "A and B"), (["A", "B", "C"], "A, B and C")],
)
def test_join_human(items: list[str], expected: str) -> None:
    assert replies.join_human(items) == expected


def test_scope_echo_is_empty_without_a_scope() -> None:
    assert replies.scope_echo([]) == ""
    assert "NVDA" in replies.scope_echo(["NVDA"])
