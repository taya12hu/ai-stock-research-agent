"""One LLM call per turn, producing observations about the message and nothing else.

Replaces `router_node`, `followup_router_node`, and both clarification resolvers' own
classification calls. Those asked the model to pick a path — six values on the follow-up
side — which fused several independent judgments into one enum, so a mistake anywhere
invalidated everything beneath it. Here the model reports what it sees and `plan_turn`
decides what to do, which means a wrong `shape_hint` cannot corrupt `companies` the way a
wrong `path` used to invalidate every field under it.

What the model is given is a deliberately lossy **projection** of the session, not the
transcript: the ticker list, what the last answer covered, recent user turns verbatim, and
one-line gists for assistant turns. See `ConversationMessage.gist` for why the report body
is withheld rather than truncated.
"""

from __future__ import annotations

from app.graph.intent import TurnIntent
from app.graph.nodes._shared import run_structured_analysis
from app.graph.session import SessionState, session_tickers
from app.logging_config import get_logger, log_event
from app.replies import CAPABILITIES, join_human

logger = get_logger("app.graph.nodes.classify_turn")

# Three exchanges. Enough for every backward reference this system supports, because
# `refers_to_prior` resolves against `last_scope` — a field — rather than something the
# model has to reconstruct from transcript. Deeper history buys nothing and costs
# contamination risk.
_HISTORY_MESSAGES = 6
_MAX_USER_CHARS = 400

_BOUNDARY_EXAMPLES = (
    "Distinguishing a request from a mention (apply the principle, these are "
    "illustrations rather than a list to match against):\n"
    "- 'Amazon is not doing well, I might switch jobs' -> Amazon is incidental; the "
    "request is about the job. off_domain_topic = 'considering a job change'.\n"
    "- 'How is Amazon stock doing?' -> Amazon is a research_subject.\n"
    "- 'Is Amazon a good buy right now?' -> research_subject.\n"
    "- 'I work at Amazon' -> incidental, and nothing is being requested at all.\n"
    "- 'Amazon stock is falling and I'm thinking of leaving Amazon' -> one company, "
    "research_subject: the request half wins. off_domain_topic = 'the job decision', so "
    "that half can be acknowledged rather than dropped.\n"
    "- 'How is Amazon doing?' with nothing in the conversation to settle it -> unclear.\n\n"
    "Distinguishing a scope from a company:\n"
    "- 'Which Indian stocks should I buy?' -> screening_scope = 'Indian stocks', no "
    "company. Never fill in companies you think would fit.\n"
    "- 'Which stock has stronger momentum?' -> no scope named at all, so this is NOT "
    "screening: the user has specific companies in mind and didn't say them. Leave "
    "screening_scope null and set refers_to_prior if it points at this conversation.\n"
    "- 'How is my portfolio doing?' with no holdings listed -> same; not screening.\n"
    "- 'I'm considering HDFC Bank and ICICI Bank, which is better?' -> two "
    "research_subjects, shape_hint 'comparison'. Companies were named, so not screening."
)


def _projection(state: SessionState) -> str:
    """The compact session view the classifier is allowed to see."""
    lines: list[str] = []

    tickers = session_tickers(state)
    if tickers:
        lines.append(f"Companies researched in this session: {join_human(tickers)}.")
    else:
        lines.append("Nothing has been researched in this session yet.")

    last_scope = state.get("last_scope") or []
    if last_scope:
        lines.append(
            f"The last answer covered {join_human(last_scope)} "
            f"({state.get('last_shape', 'single')})."
        )

    history = (state.get("conversation") or [])[-_HISTORY_MESSAGES:]
    if history:
        lines.append("\nRecent conversation:")
        for message in history:
            if message["role"] == "user":
                lines.append(f"  user: {message['content'][:_MAX_USER_CHARS]}")
            else:
                gist = message.get("gist") or "(replied)"
                lines.append(f"  assistant: {gist}")

    # Stated outright rather than left for the model to infer from the transcript. This is
    # what makes the two dedicated clarification-resolver nodes unnecessary: the reply is
    # classified as an ordinary message, with the question it answers in plain view.
    pending = state.get("pending")
    if pending:
        lines.append(
            f'\nYou just asked them to clarify: "{pending["question"]}"\n'
            f'The question they originally asked was: "{pending["original_question"]}"\n'
            "This new message is most likely their answer — read it that way. If it names "
            "companies, those are the answer to that question, and the original question "
            "tells you what form of answer they wanted. If instead they have moved on to "
            "something else, classify it as whatever it actually is."
        )

    return "\n".join(lines)


def _build_prompt(state: SessionState) -> str:
    return (
        "You are classifying one message in a conversation with a stock research "
        "assistant. Report what the message contains. Do not decide what should happen "
        "next, what needs re-fetching, or whether anything is already covered — other "
        "parts of the system own those decisions and have information you don't.\n\n"
        f"The assistant can: {join_human(list(CAPABILITIES.can))}. It cannot: "
        f"{join_human(list(CAPABILITIES.cannot))}.\n\n"
        f"{_projection(state)}\n\n"
        f'New message: "{state["user_question"]}"\n\n'
        "For each company named, the judgment that matters is whether something is being "
        "asked OF it, or whether it is background for a different subject. Financial "
        "vocabulary is not the test: 'doing badly', 'falling', 'laying people off' "
        "describe a company without asking anything about it.\n\n"
        f"{_BOUNDARY_EXAMPLES}\n\n"
        "Never introduce a company the user did not name — not one inferred from a "
        "sector, market or criterion, and not one that merely appeared in an earlier "
        "answer. If a message points back at something already discussed without naming "
        "it, set refers_to_prior and leave companies empty; resolving which one it means "
        "is not your job."
    )


async def classify_turn(state: SessionState) -> TurnIntent:
    """Raises `LLMAnalysisError` on failure — deliberately not swallowed here.

    The previous design's error path returned `followup_path="answer"`, which bypassed the
    freshness guard entirely and answered from arbitrarily old context without telling the
    user classification had broken (A-06). The caller now turns a failure into an explicit
    reply instead.
    """
    intent = await run_structured_analysis(_build_prompt(state), schema=TurnIntent)
    log_event(
        logger,
        "turn classified",
        session_id=state["session_id"],
        companies=[(c.name, c.role) for c in intent.companies or []],
        refers_to_prior=intent.refers_to_prior,
        screening=bool(intent.screening_scope),
        aspects=intent.aspects or [],
    )
    return intent
