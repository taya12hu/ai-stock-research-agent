from __future__ import annotations

import groq


class LLMAnalysisError(Exception):
    """Raised when an LLM analysis call fails after retries."""


# Shown verbatim to the user (see run_structured_analysis / answer_from_context_node) —
# distinct from the generic "temporarily unavailable" fallback because this specific
# cause has a specific, honest thing to say: it's not broken, it's out of quota, and it
# will recover on its own. Worth naming separately rather than folding into the generic
# message once you've actually seen it happen (a day's Groq token quota is easy to hit
# during heavy testing) — a vague "unavailable" reads as broken; this reads as temporary.
RATE_LIMIT_MESSAGE = (
    "I'm temporarily rate-limited by the AI provider and can't process this right now. "
    "Please try again in a few minutes."
)


def is_rate_limited(exc: BaseException) -> bool:
    return isinstance(exc, groq.RateLimitError)
