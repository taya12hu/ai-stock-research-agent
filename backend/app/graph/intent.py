"""The contract between the model and the code — what `classify_turn` is allowed to say.

Every field here is an **observation about the message text** that a careful human reader
could verify from the message alone, without knowing anything about this system: which
companies appear and in what role, does the message point backwards, does it name a scope
to pick candidates from, which analyses does it ask for. Nothing here requires the model
to know what the app can do, what has already been fetched, or what should happen next —
those are decisions, and they belong in `plan_turn`.

What is deliberately *absent* is the point. There is no `is_stock_related`, no `path`, no
`refresh` versus `add_ticker`, no `needs_clarification`. Each of those was a fused decision
the model used to make, and each is now a consequence code derives:

| Old decision           | Now derived from                                              |
|------------------------|---------------------------------------------------------------|
| is_stock_related       | any company with role `research_subject`, or a resolvable back-reference |
| refresh vs add_ticker  | set membership against `researched` — never a judgment call    |
| answer vs refresh      | `fetch == []` — a computation over timestamps and statuses     |
| needs_clarification    | the scope ladder falling through every rung                    |
| discovery              | `screening_scope` set with no research subject                 |
| unrelated              | no subject, no back-reference, no screening scope              |

That leaves exactly one genuine judgment crossing this boundary: **is this company the
subject of a request, or background?** It is a much narrower failure surface than picking
one of six paths, and — unlike a path enum — it is per-company, which is what makes a
mixed message ("Amazon stock is falling and I'm thinking of leaving Amazon") expressible
at all rather than forced into one bucket.

Every list field is typed `list[X] | None` with `default_factory=list`. That is not
defensive style, it is a lesson this repo has paid for twice: Groq's structured-output
enforcement generates a strict `array` schema from a plain `list[X]`, the model reliably
emits `null` for a list that doesn't apply, and the server-side validator rejects the call
before it reaches client code — deterministically, so retries never help. Read every one
of these as `field or []`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.graph.session import CompanyRole, Shape

ShapeHint = Literal["single", "comparison", "portfolio", "none"]


class CompanyRef(BaseModel):
    name: str = Field(
        description=(
            "The company or ticker exactly as the user referred to it — 'Amazon', 'AMZN', "
            "'Tata Consultancy'. Do not convert it to a ticker symbol and do not correct "
            "the spelling; symbol resolution happens elsewhere."
        )
    )
    role: CompanyRole = Field(
        description=(
            "How this company appears in the message.\n\n"
            "'research_subject' — the message asks a question about this company, or "
            "requests an assessment of it, as a business or an investment. Something is "
            "being asked OF the company.\n\n"
            "'incidental' — the company is named, but the actual subject of the sentence "
            "is the user's own situation: their job, a purchase they made, somewhere they "
            "shop. Nothing is being asked of the company itself. Use this no matter how "
            "negative or financial the surrounding words sound — 'doing badly', 'falling', "
            "'laying people off' describe a company, they do not by themselves ask "
            "anything about it. 'Amazon is not doing well, I might switch jobs' is "
            "incidental: the request is about the job.\n\n"
            "'unclear' — a company is named and it is genuinely impossible to tell from "
            "this message which of the two above applies, e.g. a bare 'How is Amazon "
            "doing?' with nothing in the conversation to settle it. Use this honestly "
            "rather than guessing; it is handled gracefully and costs the user nothing "
            "when the answer was already available."
        )
    )


class TurnIntent(BaseModel):
    companies: list[CompanyRef] | None = Field(
        default_factory=list,
        description=(
            "Every company named or unambiguously referred to in this message, each with "
            "its role. Include companies mentioned incidentally too — marking them "
            "'incidental' is how the reply can acknowledge them naturally instead of "
            "ignoring them. NEVER add a company the user did not name: not one you infer "
            "from a sector, market, theme or criterion, and not one that merely appears in "
            "an earlier report. 'Healthcare stocks', 'Indian stocks', 'the best IT stocks' "
            "name a scope, not a company — leave this empty and set screening_scope."
        ),
    )
    refers_to_prior: bool = Field(
        default=False,
        description=(
            "True when the message points back at something already discussed instead of "
            "naming it — 'which one is better?', 'why?', 'what about the other one?', "
            "'is it still a good entry?'. Set this even when you cannot tell which "
            "specific company is meant; resolving the reference is not your job."
        ),
    )
    screening_scope: str | None = Field(
        default=None,
        description=(
            "When the user wants candidates FOUND for them, the scope they named to pick "
            "from — 'Indian stocks', 'healthcare', 'undervalued companies', 'the best IT "
            "stocks'. Copy their own words, a short noun phrase, never a company name.\n\n"
            "Only set this when the message names a scope to select FROM. 'Which stock has "
            "stronger momentum?' names no scope at all — the user most likely has specific "
            "companies in mind and simply didn't say them, so leave this null and let the "
            "back-reference or clarification path handle it. Must be null whenever any "
            "company was named above."
        ),
    )
    shape_hint: ShapeHint = Field(
        default="none",
        description=(
            "What form of answer the wording asks for, if it says. 'comparison' for "
            "'compare X and Y', 'which is better', 'X vs Y'. 'portfolio' when the user "
            "describes the companies as things they hold together — 'my portfolio', 'my "
            "holdings'. 'single' for one company. 'none' when the wording doesn't say — "
            "that is the common case and is handled fine; do not guess."
        ),
    )
    aspects: list[Literal["fundamentals", "technical", "news"]] | None = Field(
        default_factory=list,
        description=(
            "Which analyses the message specifically asks for: 'fundamentals' for "
            "financials, valuation, margins, growth, debt; 'technical' for price, chart, "
            "momentum, RSI, moving averages, entry points; 'news' for recent events, "
            "sentiment, headlines.\n\n"
            "Leave EMPTY when the message doesn't restrict itself — 'analyse Apple', 'how "
            "is NVDA doing' — which means all three. Only list aspects when the user "
            "genuinely narrowed the question, e.g. 'how are Apple's fundamentals' or "
            "'what's the RSI on NVDA'. An empty list is not a failure to answer; it is the "
            "correct answer most of the time."
        ),
    )
    off_domain_topic: str | None = Field(
        default=None,
        description=(
            "If the message asks for something this stock research app cannot do, a SHORT "
            "noun phrase naming it, in the user's own words and at most eight words — "
            "'drafting a resignation email', 'the job decision', 'a pasta recipe'. Used to "
            "acknowledge that part specifically rather than giving a generic refusal.\n\n"
            "Set it both when the whole message is off-topic and when only part of it is "
            "alongside a real research request. Write a noun phrase, not a sentence: no "
            "leading 'I', no verb-first phrasing. Null when the message is entirely about "
            "researching companies."
        ),
    )


def normalized_shape_hint(hint: ShapeHint | None) -> Shape | None:
    """`"none"` and `None` both mean "the wording didn't say" — collapse them so callers
    test one thing.
    """
    return hint if hint in ("single", "comparison", "portfolio") else None
