"""Every user-facing string this app produces on a turn that generates no research.

**Authorship rule**: the model fills *slots*, it never writes *sentences*. A slot is a
short noun phrase echoing the user's own words ("drafting a resignation email", "Indian
stocks"); the frame around it — the boundary statement, the capability claim, the offer,
the list of choices — is built here, in code, because those are claims about the product
and about session state. Anything asserting what this app is or can do has exactly one
home, so it cannot drift out of sync with the code the way a prompt string silently did:
`router_node`/`followup_router_node` told users the app runs on Yahoo Finance for four
commits after fundamentals moved to Finnhub and price data to Twelve Data.

Two consequences worth stating, because both were previously enforced by asking an LLM
politely rather than by code:

1. **A screening decline must never name a company.** Naming one immediately after saying
   "I don't pick stocks for you" *is* the recommendation the sentence just disclaimed.
   `screening()` rejects a slot that collides with a company the classifier extracted, and
   falls back to a slotless phrasing — a mechanical check, not a prompt instruction. The
   precedent for not trusting the prompt here is `citation_instruction()`, which asked the
   model to use brackets only for citation ids and got `[Unavailable]` anyway.
2. **A clarifying question must only offer choices that exist.** `clarify_referent()`
   formats the session's real tickers. Previously `FollowUpDecision.clarifying_question`
   was free-form model text — instructed to name the session's tickers, but never checked,
   so a session holding NVDA and AMD could ask "Did you mean AAPL or MSFT?", a question
   the user cannot answer correctly.

Every function here is pure: state in, string out. No I/O, no model, no network — so the
copy is unit-testable by exact match rather than graded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Capabilities:
    """What this app actually does, in one place. Interpolated into every reply that
    makes a claim about the product, so a provider change is a single edit here rather
    than a search across prompt strings (see the module docstring for why that matters).
    """

    data_sources: tuple[str, ...]
    can: tuple[str, ...]
    cannot: tuple[str, ...]
    # How the three specialist agents are described to a user. Deliberately in the user's
    # vocabulary, not the codebase's ("technical" alone means nothing to a retail user).
    aspects_phrase: str


CAPABILITIES = Capabilities(
    data_sources=(
        "Finnhub for fundamentals",
        "Twelve Data for price history and technical indicators",
        "web search for recent news",
    ),
    can=(
        "analyse a company you name",
        "compare companies",
        "review a list of holdings",
    ),
    cannot=(
        "screen or rank a market or sector to find candidates",
        "give personalised investment advice",
        "cover non-US listings",
    ),
    aspects_phrase="fundamentals, price trend, and recent news",
)

# Provider names that, if they appear in a model-written slot, must be ones we actually
# use. Catches A-12 recurring: the model reintroducing "Yahoo Finance" (or inventing
# Bloomberg) into a sentence describing this app. Cheap, and precise about the one thing
# it guards.
_KNOWN_PROVIDER_WORDS = frozenset(
    {
        "yahoo",
        "bloomberg",
        "reuters",
        "morningstar",
        "polygon",
        "iex",
        "alpha vantage",
        "finnhub",
        "twelve data",
    }
)

_OUR_PROVIDER_WORDS = frozenset(
    word for word in _KNOWN_PROVIDER_WORDS
    if any(word in source.lower() for source in CAPABILITIES.data_sources)
)

MAX_SLOT_WORDS = 8

# A slot is a noun phrase, not a clause. These openings mean the model wrote a sentence
# instead of filling a slot, and interpolating one produces "I can't help with I can't
# help you with that — I'm a stock research assistant."
_SENTENCE_OPENERS = ("i ", "i'm", "im ", "you ", "we ", "sorry", "unfortunately", "that's")


def is_valid_slot(slot: str | None, *, forbid: tuple[str, ...] = ()) -> bool:
    """Whether a model-written slot is safe to interpolate into a frame below.

    `forbid` is for the screening case: the company names the classifier extracted from
    this same message. If the slot echoes one of them, the model has put a company into a
    sentence whose entire point is that this app doesn't pick companies.
    """
    if slot is None:
        return False
    cleaned = slot.strip()
    if not cleaned:
        return False
    if len(cleaned.split()) > MAX_SLOT_WORDS:
        return False

    lowered = cleaned.lower()
    if lowered.startswith(_SENTENCE_OPENERS):
        return False
    for word in _KNOWN_PROVIDER_WORDS - _OUR_PROVIDER_WORDS:
        if word in lowered:
            return False
    for banned in forbid:
        if banned.strip() and banned.strip().lower() in lowered:
            return False
    return True


def _clean(slot: str) -> str:
    return re.sub(r"\s+", " ", slot.strip()).rstrip(".!?")


def join_human(items: list[str] | tuple[str, ...]) -> str:
    """'A', 'A and B', 'A, B and C' — used wherever a reply lists tickers or choices."""
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _continue_offer(session_tickers: list[str]) -> str:
    """The door left open at the end of every decline. Session-aware on purpose: offering
    a worked example ("try asking about a specific stock") to someone ten turns into an
    active NVDA session reads as though the app forgot the conversation, which is worse
    than offering nothing. The previous single static constant did exactly that.
    """
    if session_tickers:
        return f"Happy to keep going on {join_human(session_tickers)}, or look at another company."
    return f"Ask me about a company and I'll pull its {CAPABILITIES.aspects_phrase}."


# ─────────────────────────── pleasantries ───────────────────────────


def greeting(session_tickers: list[str]) -> str:
    """"hi" with nothing attached.

    Distinct from `off_domain` on purpose. A greeting is not a request that has to be
    declined, and answering it with "That's not something I can help with" was both
    inaccurate and cold: nobody had asked for anything yet. Same job as the decline
    replies — say what this is, offer the obvious next step — without the refusal frame.
    """
    if session_tickers:
        return (
            f"Hello. We're on {join_human(session_tickers)} at the moment. Ask me anything "
            "about them, or name another company."
        )
    return (
        "Hello. I research stocks: name a company and I'll pull its "
        f"{CAPABILITIES.aspects_phrase}."
    )


def acknowledgement(session_tickers: list[str]) -> str:
    """"thanks", "ok got it", "interesting" — a reaction, not a question.

    These used to get the off-domain decline, which meant thanking the assistant was
    answered with "That's not something I can help with". Worth its own reply for the same
    reason the greeting is: the user asked for nothing, so there is nothing to refuse.
    """
    if session_tickers:
        return f"Glad it helped. Happy to keep going on {join_human(session_tickers)}, or look at another company."
    return "Glad it helped. Name another company whenever you want to dig into one."


# ─────────────────────────── off-domain ───────────────────────────


def off_domain(topic: str | None, session_tickers: list[str]) -> str:
    """No company in the message at all — a greeting, a thank-you, or a request this app
    has nothing to do with ("how do I write a resignation email?").
    """
    if is_valid_slot(topic):
        opening = f"I can't help with {_clean(topic)}. I'm a stock research assistant."
    else:
        opening = "That's not something I can help with. I'm a stock research assistant."
    return f"{opening} {_continue_offer(session_tickers)}"


def off_domain_with_company(company: str | None, session_tickers: list[str]) -> str:
    """A real company is named, but only as background for something else — "Amazon is not
    doing well, I might switch jobs". The offer names the adjacent thing this app *does*
    do, which converts a dead end into a usable next turn; it is an offer, not an action,
    so no research runs and no ticker enters scope from this path.
    """
    if is_valid_slot(company):
        return (
            "That's not something I can help you weigh up. I only do stock research. "
            f"If it's useful, I can look at how {_clean(company)} itself is doing: "
            f"{CAPABILITIES.aspects_phrase}."
        )
    return off_domain(None, session_tickers)


def mixed_acknowledgment(topic: str | None) -> str:
    """Appended by `emit` *after* a report when the message also carried an off-domain
    half ("Amazon stock is falling and I'm thinking of leaving Amazon"). Placement is
    deliberate: leading with a decline buries the report the user actually asked for
    behind a caveat about the part they didn't.
    """
    if not is_valid_slot(topic):
        return ""
    return (
        f"\n\nOn {_clean(topic)}: that's outside what I can help with, but the research "
        "above may be useful input."
    )


# ─────────────────────────── screening ───────────────────────────


def screening(scope: str | None, extracted_companies: tuple[str, ...] = ()) -> str:
    """The user wants candidates found for them, which this app doesn't do. Tightest leash
    of any reply here — see the module docstring for why the no-company rule is enforced
    mechanically rather than requested in a prompt.
    """
    if is_valid_slot(scope, forbid=extracted_companies) and not _looks_like_proper_noun(scope):
        boundary = f"I don't screen {_clean(scope)} to find candidates"
    else:
        boundary = "I don't screen for candidates like that"
    return (
        f"I can research and compare specific companies, but {boundary}. I analyse the "
        "stocks you give me. Name a few you're considering and I'll look into them."
    )


def _looks_like_proper_noun(slot: str) -> bool:
    """A screening scope is categorical by nature ("Indian stocks", "healthcare"). A lone
    capitalised token is far more likely a company name that slipped into the slot, which
    is the one thing this reply must never contain. Backstop to the `forbid` check, for
    when the classifier didn't also list it under `companies`.
    """
    tokens = slot.strip().split()
    return len(tokens) == 1 and tokens[0][:1].isupper()


# ─────────────────────────── clarification ───────────────────────────


def clarify_intent(company: str | None) -> str:
    """A company is named but we can't tell whether its *stock* is the subject — "How is
    Amazon doing?" with nothing in the session to disambiguate it.
    """
    if is_valid_slot(company):
        return (
            f"Just to check: do you want me to look at {_clean(company)} as a stock? "
            f"I'd pull {CAPABILITIES.aspects_phrase}."
        )
    return "Just to check: is this a question about a company's stock?"


def clarify_referent(session_tickers: list[str]) -> str:
    """A backward reference we can't resolve — "how's the other one doing?" with several
    tickers in play. The choices come from state, never from the model.
    """
    if not session_tickers:
        return "Which company or stock ticker would you like me to look at?"
    if len(session_tickers) == 1:
        return f"Did you mean {session_tickers[0]}?"
    return f"This session has {join_human(session_tickers)}. Which did you mean?"


def clarify_exhausted(session_tickers: list[str]) -> str:
    """After the attempt limit. Ends the loop deterministically instead of asking a third
    differently-worded question (A-07: the previous design bounded nothing, so a vaguely
    replying user could be asked indefinitely).
    """
    if session_tickers:
        return (
            "I'm still not sure which one you mean. Name the ticker directly, "
            f"for example {session_tickers[0]}, and I'll take it from there."
        )
    return (
        "I'm still not sure which company you mean. Name it directly, for example "
        "'AAPL' or 'Apple', and I'll take it from there."
    )


def hedge_prefix() -> str:
    """Prepended to a recall answer when the subject was unclear but every relevant cell
    was already fresh. The asymmetric gate: a wrong guess here costs one cheap paragraph
    the user corrects immediately, where the same uncertainty on a fetching turn would
    spend API budget and produce a confident, fully-cited report about the wrong company.
    """
    return "Taking that as a question about the stock.\n\n"


# ─────────────────────────── failures ───────────────────────────


def classification_failed(*, rate_limited: bool) -> str:
    """Necessarily code-owned: the component that would draft this is the one that just
    failed. Replaces the previous fallback, which silently degraded to answering from
    arbitrarily old context without telling the user classification had broken (A-06).
    """
    if rate_limited:
        return "I'm being rate-limited by the AI provider right now. Try again in a moment."
    return "I couldn't read that one. Could you rephrase it?"


def unresolved_tickers(reasons: list[str], session_tickers: list[str]) -> str:
    """The user named companies, but none survived validation. Distinct from a
    clarification: they *did* name something, so asking "which company?" would read as
    though we hadn't listened. The reasons carry the real explanation (not found vs.
    outside US coverage), so this frame stays out of their way.
    """
    detail = " ".join(r for r in reasons if r).strip()
    if not detail:
        return f"I couldn't find that company. {_continue_offer(session_tickers)}"
    if all("isn't currently supported" in r for r in reasons if r):
        # Nothing to double-check — these resolved fine, they're just outside coverage,
        # so the generic "check the spelling" closer would wrongly imply a typo.
        return f"{detail} Happy to help with a US-listed company instead."
    return f"{detail} Could you double-check the name or ticker, or try a different company?"


def scope_echo(scope: list[str]) -> str:
    """Prefix for recall answers, which otherwise carry no header at all. Reports already
    state their scope in the title ("# Comparison: NVDA vs AMD"); a recall answer doesn't,
    and it is also where a wrong scope is hardest to notice, because there's no report
    structure to look wrong. Cheapest available mitigation for a misclassified scope: it
    turns a silently wrong answer into an obviously wrong one, correctable in a turn.
    """
    if not scope:
        return ""
    return f"**{join_human(scope)}** · "
