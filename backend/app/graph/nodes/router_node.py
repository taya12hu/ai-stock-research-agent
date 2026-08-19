"""Classifies a free-text request into a query type + validated ticker list.

Runs first in the graph. If the caller already supplied `tickers` (tests, or a later
follow-up flow that already knows which tickers to (re)run), classification is skipped
entirely — this node is a no-op passthrough in that case.

Classification is a single structured-output LLM call (`RouterDecision`) that decides,
in order of precedence: (1) is this actually about a company's stock/business as an
investment at all — `is_stock_related` — so an incidental mention ("I work at Google")
doesn't trigger research; (2) if not, a ready-to-show `off_topic_reply`; (3) does it name
specific company/companies, or is it asking the app to *find/select/rank* candidates from
some scope (a sector, market, theme, "best", "undervalued", "any stock I can buy") that
this app doesn't have a screening capability for — `is_discovery_request` — in which case
a ready-to-show `discovery_reply` explains the limitation instead of either fabricating
candidates or interrogating the user for investor preferences; (4) if it's a real,
specific-stock(s) request, the query type + tickers, or a `clarifying_question` when a
company is clearly implied but not named. `apply_router_decision` turns that single
decision into a state update, and is reused as-is by `clarification_response_node` when a
clarification reply turns out to be abandoning the original question rather than
answering it — see that module's docstring.

The discovery/clarification boundary (see `RouterDecision.is_discovery_request`) is the
one part of this file worth calling out: "which stock has stronger sentiment?" and
"which Indian stocks should I buy?" both name zero companies, but only the second is
discovery — it names a *selection scope* (a market) the user expects the app to search,
where the first has no scope at all and most likely means the user has specific companies
in mind and simply didn't name them. Getting this distinction right is the entire point
of the field's prompt description; getting it wrong either interrogates the user for
irrelevant investor-preference details (the bug this was built to fix) or, worse, invents
candidate companies the app never actually researched.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.config import settings
from app.graph.nodes._shared import run_structured_analysis
from app.graph.state import QueryType, ResearchState
from app.llm.errors import LLMAnalysisError
from app.logging_config import get_logger, log_event
from app.tools.yahoo_finance import aresolve_ticker

logger = get_logger("app.graph.nodes.router")

DEFAULT_OFF_TOPIC_REPLY = (
    "I'm focused on stock research — I can look into a company's fundamentals, price "
    "trends, and recent news. Try asking about a specific stock, e.g. \"Should I buy "
    "TCS right now?\""
)

DEFAULT_DISCOVERY_REPLY = (
    "I can research and compare specific stocks, but I don't currently screen the "
    "market to find candidates for you. If you give me a few companies you're "
    "considering, I can look into them."
)


class TickerCandidate(BaseModel):
    ticker: str = Field(description="Stock ticker symbol in caps, e.g. AAPL, NVDA")


class RouterDecision(BaseModel):
    is_stock_related: bool = Field(
        default=True,
        description=(
            "False if the request isn't about a company's stock, financial performance, "
            "or business as an investment/research subject at all — including when a "
            "company is only mentioned as incidental context (e.g. 'I work at Google', "
            "'I bought this at Target') without asking about its stock. True otherwise."
        )
    )
    query_type: QueryType = Field(
        default="single",
        description=(
            "'single' if exactly one company is mentioned; 'portfolio' if the user "
            "describes multiple stocks they hold together as a portfolio/holdings; "
            "'comparison' if the user wants two or more companies compared, or asks "
            "which is better / one 'vs' another. When multiple tickers are mentioned "
            "with no portfolio framing, default to 'comparison'. Irrelevant (ignored, "
            "no need to set it) when is_stock_related is false."
        )
    )
    tickers: list[TickerCandidate] = Field(
        default_factory=list,
        description=(
            "Every distinct publicly-traded company/stock ticker the user is actually "
            "asking about as an investment or research subject. Leave empty if "
            "is_stock_related is false, a company is only mentioned incidentally, or "
            "this is a discovery request (see is_discovery_request). 'Implied' means "
            "ONLY an unambiguous conversational reference back to a company already "
            "established in this exchange (a pronoun, 'that stock', a short form of a "
            "name already used) — NEVER a company you infer from a category, sector, "
            "market, theme, or investment criterion. 'Healthcare stocks', 'Indian "
            "stocks', 'undervalued companies', 'the best IT stocks' name a scope, not a "
            "company — do not fill this field with companies you know would fit that "
            "scope, no matter how well-known or obviously representative they are."
        )
    )
    is_discovery_request: bool = Field(
        default=False,
        description=(
            "True when the user wants the app to FIND, SELECT, RANK, or RECOMMEND "
            "candidate stocks from some scope — a market ('Indian stocks'), a sector "
            "('healthcare', 'IT', 'banking', 'pharma'), a theme, or a criterion "
            "('best', 'top', 'undervalued', 'strong fundamentals', 'most affected by "
            "news', 'good dividend stocks') — with NO specific company named. This app "
            "analyzes/compares stocks the user provides; it does not screen a market or "
            "sector for candidates. Must be false whenever any ticker was found above "
            "(a real company already named always takes priority over this flag).\n\n"
            "Contrast with needs_clarification: 'which stock has stronger sentiment?' "
            "or 'which is better?' name NO scope at all — the user most likely already "
            "has specific companies in mind and simply didn't name them, so that's "
            "needs_clarification (ask which companies), not discovery. 'how is my "
            "portfolio doing?' with no holdings listed is the same — the user has "
            "specific holdings in mind, just didn't list them; not discovery. Only set "
            "this true when the message itself names a scope/criterion to select "
            "*from*, e.g. 'which Indian stocks should I buy', 'find me the best IT "
            "stocks', 'which pharma stocks are falling on bad news', 'give me 5 stocks "
            "with strong fundamentals', 'which banking stocks should I consider', "
            "'find undervalued stocks'. If a message is genuinely ambiguous between the "
            "two readings, prefer needs_clarification with a clarifying_question that "
            "asks which one is meant (see that field) rather than guessing."
        )
    )
    needs_clarification: bool = Field(
        default=False,
        description=(
            "True only when is_stock_related is true, is_discovery_request is false, no "
            "tickers were found above, AND the request is clearly about researching, "
            "analyzing, or comparing specific stocks/companies the user has in mind "
            "without naming which one(s) (e.g. 'which stock has stronger momentum?', "
            "'how is my portfolio doing?' with no holdings listed). False otherwise."
        )
    )
    clarifying_question: str | None = Field(
        default=None,
        description=(
            "If needs_clarification is true: a short question tailored to what was "
            "actually asked — for a comparison ask which companies to compare, for a "
            "portfolio question ask which stocks are in it, for a single vague question "
            "ask which company. Write it fresh each time; never reuse a stock template "
            "phrase regardless of what this request was about. If the message is "
            "genuinely ambiguous between 'the user has specific companies in mind' and "
            "'the user wants the app to find candidates' (which this app can't do), ask "
            "a neutral question that resolves the ambiguity itself, e.g. 'Are you "
            "looking for me to find stocks matching that, or do you already have "
            "specific ones you'd like me to look at?' — don't assume either reading. "
            "Otherwise null."
        ),
    )
    off_topic_reply: str | None = Field(
        default=None,
        description=(
            "If is_stock_related is false: a brief, friendly reply. If the user asked "
            "something about this app itself (e.g. what data it uses), answer directly "
            "using: this app researches stocks using Yahoo Finance data (fundamentals "
            "and price/technical indicators) and web search for recent news. Otherwise, "
            "politely note that you focus on stock research and give one example "
            "question, e.g. 'Should I buy TCS right now?'. Otherwise null."
        ),
    )
    discovery_reply: str | None = Field(
        default=None,
        description=(
            "If is_discovery_request is true: a brief, natural reply explaining that "
            "this app analyzes and compares specific stocks the user provides, but "
            "doesn't currently screen a market/sector to find candidates — then invite "
            "them to name a few specific companies you can look into. Tailor it to what "
            "was actually asked (mention the market/sector/criterion they named), write "
            "it fresh each time, and do NOT name any specific company in it — not even "
            "as an example of what fits their criteria; naming one would defeat the "
            "point of explaining the app doesn't select candidates. Otherwise null."
        ),
    )
    unaddressed_note: str | None = Field(
        default=None,
        description=(
            "If part of this message can't be addressed by this app while the rest "
            "names specific companies that CAN be researched, one short sentence noting "
            "what's being skipped and why — covering BOTH: (a) an open-ended discovery "
            "tail, e.g. 'TCS, Infosys, and a few other good IT stocks' (TCS/Infosys can "
            "be researched, 'a few other good ones' can't be discovered), and (b) a "
            "side-request unrelated to stocks entirely mixed into the same message, e.g. "
            "'how do I make pasta, and how's TCS doing?' (TCS can be researched; the "
            "pasta question can't be answered by a stock research app and should be "
            "named, not silently dropped). Otherwise null — do not use this for a PURE "
            "discovery request or a PURE off-topic message with no named company at "
            "all, that's what is_discovery_request/discovery_reply and "
            "is_stock_related/off_topic_reply are for respectively."
        ),
    )
    resolves_pending_clarification: bool = Field(
        default=True,
        description=(
            "Only meaningful when classifying a reply to an earlier clarifying "
            "question (see that prompt): true if this reply answers it, false if the "
            "user has abandoned it for something else. Ignored otherwise."
        ),
    )


_CLASSIFICATION_EXAMPLES = (
    "Examples of the discovery vs. clarification boundary (apply the underlying "
    "principle to any sector, market, or theme — these are illustrations, not a "
    "checklist to match against):\n"
    "- 'Which stock has stronger sentiment?' -> no scope named -> needs_clarification "
    "(ask which stocks).\n"
    "- 'Which Indian stocks should I buy?' -> scope = Indian market, no company -> "
    "is_discovery_request.\n"
    "- 'Should I buy HDFC Bank?' -> specific company -> normal single-stock request.\n"
    "- 'Compare HDFC Bank and ICICI Bank' -> specific companies -> normal comparison.\n"
    "- 'Which Indian bank should I buy?' -> scope = Indian banks, no company -> "
    "is_discovery_request.\n"
    "- 'I'm considering HDFC Bank and ICICI Bank, which is better?' -> specific "
    "companies given -> normal comparison, NOT discovery.\n"
    "- 'Find me the best IT stocks' / 'Which pharma stocks are falling on bad news?' / "
    "'Give me 5 Indian stocks with strong fundamentals' / 'Find undervalued stocks' / "
    "'Which banking stocks should I consider?' -> all name a sector or selection "
    "criterion with no company -> is_discovery_request.\n"
    "- 'How is my portfolio doing?' with no holdings listed -> the user has specific "
    "holdings in mind, not a market to screen -> needs_clarification (ask what's in "
    "it), NOT discovery."
)


def _build_prompt(user_question: str) -> str:
    return (
        "Classify this chat message for a stock research assistant. This app analyzes "
        "and compares specific stocks the user provides — it does NOT screen a market "
        "or sector to find candidates.\n\n"
        "First decide whether it's actually about a company's stock, financial "
        "performance, or business as an investment/research subject — set "
        "is_stock_related. A company mentioned only as incidental context (e.g. 'I work "
        "at Google', 'I bought this at Target') does NOT make a request stock-related.\n\n"
        "If is_stock_related is false: leave tickers empty and write off_topic_reply (see "
        "its description).\n\n"
        "If is_stock_related is true: identify every publicly-traded company the user is "
        "actually asking about (see the tickers field's description for what counts), "
        "convert each to its stock ticker symbol, and classify the request type.\n\n"
        "If no company was named: decide whether this is is_discovery_request (the "
        "message names a scope/criterion to select stocks from) or needs_clarification "
        "(the message implies the user has specific companies in mind but didn't name "
        "them) — see both fields' descriptions for the exact test, and "
        f"{_CLASSIFICATION_EXAMPLES}\n\n"
        "If some companies ARE named alongside either an open-ended, undiscoverable "
        "request for more, or a completely unrelated side-question (e.g. a recipe), set "
        "unaddressed_note (see its description) rather than silently dropping that part "
        "or treating the whole message as discovery/off-topic.\n\n"
        f'Message: "{user_question}"'
    )


async def resolve_tickers(
    raw_tickers: list[str], query_type: QueryType
) -> tuple[list[str], QueryType, list[str]]:
    """Validates, dedupes, and caps a ticker list an LLM extracted from user text, and
    downgrades `query_type` to 'single' if only one ticker survives. Validation goes
    through `aresolve_ticker`, not a bare existence check — a symbol that comes back
    from this may differ from what was passed in (e.g. 'TCS' -> 'TCS.NS') when the bare
    symbol collides with an unrelated or delisted listing; the *resolved* symbol is what
    gets stored and used for every downstream fetch. Shared by every call site that
    resolves a `RouterDecision`-shaped result into tickers — the initial router, a
    resolved clarification reply, and (indirectly) a clarification's abandon-and-
    reclassify path — so the limits/validity rules are applied identically everywhere a
    ticker list can originate.
    """
    deduped = list(dict.fromkeys(t.strip().upper() for t in raw_tickers if t.strip()))

    notes: list[str] = []
    if len(deduped) > settings.max_tickers:
        dropped, deduped = deduped[settings.max_tickers :], deduped[: settings.max_tickers]
        notes.append(
            f"Only the first {settings.max_tickers} tickers were analyzed (limit reached); "
            f"dropped: {', '.join(dropped)}."
        )

    valid_tickers: list[str] = []
    for ticker in deduped:
        resolved = await aresolve_ticker(ticker)
        if resolved:
            valid_tickers.append(resolved)
        else:
            notes.append(f"'{ticker}' could not be found and was skipped.")
    valid_tickers = list(dict.fromkeys(valid_tickers))  # a bare symbol and its resolved
    # (suffixed) form could theoretically both have been named and collapse to one

    if len(valid_tickers) <= 1 and query_type != "single":
        # e.g. a requested comparison where one side turned out to be invalid
        query_type = "single"

    return valid_tickers, query_type, notes


async def apply_router_decision(decision: RouterDecision, user_question: str, session_id: str) -> dict:
    """Turns a `RouterDecision` into a state update for a fresh (non-clarification-
    reply) message. Used by `router_node` directly, and reused by
    `clarification_response_node` when a reply abandons the pending clarification —
    in that case the same LLM call that noticed the abandonment already classified the
    reply as if it were a brand-new message, so there's no second LLM round-trip.
    """
    user_turn = [{"role": "user", "content": user_question}]

    if not decision.is_stock_related:
        log_event(logger, "router: off-topic message", session_id=session_id)
        return {
            "tickers": [],
            "query_type": "single",
            "notes": [],
            "conversation_history": user_turn,
            "awaiting_clarification": False,
            "clarification_question": None,
            "pending_question": None,
            "pending_intent": None,
            "off_topic_reply": decision.off_topic_reply or DEFAULT_OFF_TOPIC_REPLY,
            "clarification_origin": None,
        }

    raw_tickers = [c.ticker for c in decision.tickers]

    # Backend guard, not just a prompt instruction: a real named company always beats
    # the discovery flag, regardless of what the model set it to — concrete information
    # is never overridden by a fuzzy classification. Checked against the model's raw
    # (pre-validation) ticker list, not `valid_tickers` below, so this doesn't do a
    # round of ticker-resolution network calls just to discard them for a discovery case.
    if decision.is_discovery_request and not raw_tickers:
        log_event(logger, "router: discovery request (unsupported)", session_id=session_id)
        return {
            "tickers": [],
            "query_type": "single",
            "notes": [],
            "conversation_history": user_turn,
            "awaiting_clarification": False,
            "clarification_question": None,
            "pending_question": None,
            "pending_intent": None,
            "off_topic_reply": decision.discovery_reply or DEFAULT_DISCOVERY_REPLY,
            "clarification_origin": None,
        }

    valid_tickers, query_type, notes = await resolve_tickers(raw_tickers, decision.query_type)
    if decision.unaddressed_note:
        notes = [*notes, decision.unaddressed_note]

    if not valid_tickers and decision.needs_clarification and decision.clarifying_question:
        log_event(logger, "router needs clarification", session_id=session_id, question=user_question)
        # Use the model's original `decision.query_type`, not the `query_type` computed
        # above — `resolve_tickers`'s "downgrade to single" rule exists for when research
        # is about to run with however many tickers survived validation. It doesn't apply
        # here: there are zero tickers because none have been named yet, not because a
        # comparison collapsed down to one company.
        return {
            "tickers": [],
            "query_type": decision.query_type,
            "notes": notes,
            "conversation_history": user_turn,
            "awaiting_clarification": True,
            "clarification_question": decision.clarifying_question,
            "pending_question": user_question,
            "pending_intent": decision.query_type,
            "off_topic_reply": None,
            "clarification_origin": "router",
        }

    log_event(
        logger, "router completed", session_id=session_id,
        query_type=query_type, tickers=valid_tickers, dropped_count=len(notes),
    )
    return {
        "tickers": valid_tickers,
        "query_type": query_type,
        "notes": notes,
        "conversation_history": user_turn,
        "awaiting_clarification": False,
        "clarification_question": None,
        "pending_question": None,
        "pending_intent": None,
        "off_topic_reply": None,
        "clarification_origin": None,
    }


async def router_node(state: ResearchState) -> dict:
    if state.get("tickers"):
        return {}

    try:
        decision = await run_structured_analysis(
            _build_prompt(state["user_question"]), schema=RouterDecision
        )
    except LLMAnalysisError as exc:
        log_event(
            logger, "router classification failed", level=logging.ERROR,
            session_id=state["session_id"], error=str(exc),
        )
        # `str(exc)` is guaranteed clean here — every raise site in run_structured_analysis
        # produces a fixed, human-safe message (see _shared.py) — so it's safe to show
        # directly as the reply, through the same plain-reply channel as off-topic/
        # discovery, rather than burying it in `notes` alongside a misleading dead end.
        return {
            "tickers": [],
            "query_type": "single",
            "notes": [],
            "conversation_history": [{"role": "user", "content": state["user_question"]}],
            "awaiting_clarification": False,
            "clarification_question": None,
            "pending_question": None,
            "pending_intent": None,
            "off_topic_reply": str(exc),
            "clarification_origin": None,
        }

    return await apply_router_decision(decision, state["user_question"], state["session_id"])
