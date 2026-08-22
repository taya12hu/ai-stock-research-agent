"""Value types shared across the graph.

These outlived the state-schema refactor unchanged because they were always correct:
`Finding` is the atomic citable unit every report traces claims back to, and `Source` is
where it came from. Session and per-turn state live in `session.py`; how a turn is planned
lives in `plan_turn.py`.
"""

from __future__ import annotations

from typing import Literal, TypedDict

AgentName = Literal["fundamentals", "technical", "news"]
AGENT_NAMES: tuple[AgentName, ...] = ("fundamentals", "technical", "news")


class Source(TypedDict):
    type: Literal["market_data", "web"]
    label: str
    url: str | None
    # What this source's own data is dated to. Deliberately distinct from a cell's
    # `fetched_at`: this means different things per agent — a fetch timestamp for
    # fundamentals, the last trading day for technical, an article's publish date for news
    # — and so cannot answer "how long ago did *we* last fetch this", which is what the
    # freshness predicate needs.
    as_of: str


class Finding(TypedDict):
    id: str
    claim: str
    evidence: str
    source: Source
