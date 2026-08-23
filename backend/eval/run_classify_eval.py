"""Classification eval — the cheap suite.

Separate from `run_eval.py`, which runs full research (three providers plus several model
calls per case) and therefore can't reasonably run on every prompt edit. This one asserts
on *decisions*: what the classifier observed, and what the planner did with it.

Two kinds of case, and the split is the point:

- **plan** cases feed a hand-written `TurnIntent` straight into `plan_turn`. No model, no
  network, no API key — they run in milliseconds and are exactly reproducible. Most of the
  system's decision surface lives here, so most cases are this kind.
- **classify** cases send a real message to the real model and check what it observed, then
  push that through the planner. One model call each; ticker resolution is stubbed from the
  case's own `resolve` map, so no market-data calls either. That map is keyed on SYMBOLS,
  matching what the real resolver accepts — keying it on company names made the stub able
  to do something the real code cannot, and hid a regression for a full refactor.

Run everything:      .venv/Scripts/python.exe -m eval.run_classify_eval
No API key needed:   .venv/Scripts/python.exe -m eval.run_classify_eval --plan-only
One case:            .venv/Scripts/python.exe -m eval.run_classify_eval --only narrow_after_comparison
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

import app.graph.resolve_scope as resolve_mod
from app.graph.intent import CompanyRef, TurnIntent
from app.graph.nodes.classify_turn import classify_turn
from app.graph.plan_turn import ScopeResolution, plan_turn
from app.graph.session import AGENT_NAMES, SessionState, TurnPlan, fresh_turn
from app.llm.errors import LLMAnalysisError
from app.tools.market_data import ResolvedTicker

DATASET_PATH = Path(__file__).resolve().parent / "classify_dataset.yaml"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# A fixed clock so freshness decisions are reproducible: Wednesday 2026-08-19, 14:00 UTC,
# which is 10:00 in New York — the US market is open. Cases that care about the closed-
# market branch set `now` themselves.
DEFAULT_NOW = datetime(2026, 8, 19, 14, 0, tzinfo=timezone.utc)

# How old each named cell state is, relative to the case's clock.
_AGE = {"fresh": timedelta(seconds=30), "stale": timedelta(hours=6)}


def _cell(state: str, now: datetime) -> dict | None:
    """Build one (ticker, agent) cell from a shorthand name."""
    if state == "missing":
        return None
    status = "failed" if state == "failed" else "ok"
    findings = [] if state in ("empty", "failed") else [{"id": "f1"}]
    age = _AGE.get(state, _AGE["fresh"])
    return {
        "status": status,
        "summary": "s",
        "findings": findings,
        "error": "boom" if status == "failed" else None,
        "fetched_at": (now - age).isoformat(),
    }


def _researched(spec: dict[str, Any] | None, now: datetime) -> dict:
    """`{NVDA: fresh}` or `{NVDA: {technical: stale, news: fresh}}`."""
    out: dict[str, dict] = {}
    for ticker, value in (spec or {}).items():
        per_agent = value if isinstance(value, dict) else {a: value for a in AGENT_NAMES}
        cells = {}
        for agent in AGENT_NAMES:
            cell = _cell(per_agent.get(agent, "missing"), now)
            if cell is not None:
                cells[agent] = cell
        out[ticker] = cells
    return out


def _state(case: dict[str, Any], now: datetime) -> SessionState:
    session = case.get("session") or {}
    return SessionState(
        session_id=f"eval-{case['id']}",
        user_question=case["question"],
        researched=_researched(session.get("researched"), now),
        conversation=session.get("conversation") or [],
        last_scope=session.get("last_scope") or [],
        last_shape=session.get("last_shape") or "single",
        pending=session.get("pending"),
        turn=fresh_turn(),
    )


def _intent_from_spec(spec: dict[str, Any]) -> TurnIntent:
    return TurnIntent(
        companies=[
            CompanyRef(
                name=c["name"], role=c["role"], ticker=c.get("ticker", c["name"]).upper()
            )
            for c in (spec.get("companies") or [])
        ],
        refers_to_prior=spec.get("refers_to_prior", False),
        screening_scope=spec.get("screening_scope"),
        shape_hint=spec.get("shape_hint", "none"),
        aspects=spec.get("aspects") or [],
        off_domain_topic=spec.get("off_domain_topic"),
    )


def _stub_resolver(resolve_map: dict[str, str]):
    async def _resolve(ticker: str) -> ResolvedTicker:
        return ResolvedTicker(resolve_map.get(ticker.upper()))

    return _resolve


# ─────────────────────────── assertions ───────────────────────────


def _check_plan(plan: TurnPlan, expect: dict[str, Any]) -> list[str]:
    failures: list[str] = []

    def cmp(name: str, actual: Any, wanted: Any) -> None:
        if actual != wanted:
            failures.append(f"{name}: expected {wanted!r}, got {actual!r}")

    if "kind" in expect:
        cmp("kind", plan["kind"], expect["kind"])
    if "scope" in expect:
        cmp("scope", sorted(plan["scope"]), sorted(expect["scope"]))
    if "shape" in expect:
        cmp("shape", plan["shape"], expect["shape"])
    if "aspects" in expect:
        cmp("aspects", plan["aspects"], expect["aspects"])
    if "hedged" in expect:
        cmp("hedged", plan["hedged"], expect["hedged"])
    if "fetch" in expect:
        actual = sorted(f"{c['ticker']}:{c['agent']}" for c in plan["fetch"])
        cmp("fetch", actual, sorted(expect["fetch"]))
    if "fetch_count" in expect:
        cmp("fetch_count", len(plan["fetch"]), expect["fetch_count"])
    if "off_domain_topic_set" in expect:
        cmp("off_domain_topic_set", bool(plan["off_domain_topic"]), expect["off_domain_topic_set"])
    for phrase in expect.get("reply_contains") or []:
        if phrase.lower() not in (plan["reply"] or "").lower():
            failures.append(f"reply_contains: {phrase!r} missing from {plan['reply']!r}")
    for phrase in expect.get("reply_excludes") or []:
        if phrase.lower() in (plan["reply"] or "").lower():
            failures.append(f"reply_excludes: {phrase!r} present in {plan['reply']!r}")

    return failures


def _check_intent(intent: TurnIntent, expect: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    roles = {c.name.strip().lower(): c.role for c in intent.companies or []}

    for name, wanted in (expect.get("roles") or {}).items():
        actual = roles.get(name.strip().lower())
        if actual != wanted:
            failures.append(f"role[{name}]: expected {wanted!r}, got {actual!r}")

    if "company_count" in expect and len(intent.companies or []) != expect["company_count"]:
        failures.append(
            f"company_count: expected {expect['company_count']}, got {len(intent.companies or [])}"
        )
    if "refers_to_prior" in expect and intent.refers_to_prior != expect["refers_to_prior"]:
        failures.append(
            f"refers_to_prior: expected {expect['refers_to_prior']}, got {intent.refers_to_prior}"
        )
    if "screening_scope_set" in expect:
        actual = bool(intent.screening_scope)
        if actual != expect["screening_scope_set"]:
            failures.append(
                f"screening_scope_set: expected {expect['screening_scope_set']}, got {actual} "
                f"({intent.screening_scope!r})"
            )
    if "aspects" in expect and sorted(intent.aspects or []) != sorted(expect["aspects"]):
        failures.append(f"aspects: expected {expect['aspects']}, got {intent.aspects}")
    if "off_domain_topic_set" in expect:
        actual = bool(intent.off_domain_topic)
        if actual != expect["off_domain_topic_set"]:
            failures.append(
                f"off_domain_topic_set: expected {expect['off_domain_topic_set']}, got {actual}"
            )
    return failures


# ─────────────────────────── runner ───────────────────────────


async def run_case(case: dict[str, Any]) -> dict[str, Any]:
    now = (
        datetime.fromisoformat(case["now"]) if case.get("now") else DEFAULT_NOW
    )
    state = _state(case, now)
    failures: list[str] = []
    observed: dict[str, Any] = {}

    if case["kind"] == "plan":
        intent = _intent_from_spec(case.get("intent") or {})
        spec = case.get("resolved") or {}
        resolution = ScopeResolution(
            subjects=spec.get("subjects") or [],
            unclear=spec.get("unclear") or [],
            notes=spec.get("notes") or [],
            attempted=spec.get("attempted", bool(spec.get("subjects") or spec.get("unclear"))),
        )
    else:
        intent = await classify_turn(state)
        observed["intent"] = intent.model_dump()
        failures += _check_intent(intent, case.get("expect_intent") or {})

        resolve_mod.aresolve_ticker = _stub_resolver(case.get("resolve") or {})
        resolution = await resolve_mod.resolve_scope(intent, list(state["researched"]))

    plan = plan_turn(intent=intent, resolution=resolution, state=state, now=now)
    observed["plan"] = {
        "kind": plan["kind"],
        "scope": plan["scope"],
        "shape": plan["shape"],
        "aspects": plan["aspects"],
        "hedged": plan["hedged"],
        "fetch": [f"{c['ticker']}:{c['agent']}" for c in plan["fetch"]],
        "reply": plan["reply"],
    }
    failures += _check_plan(plan, case.get("expect") or {})

    return {
        "id": case["id"],
        "kind": case["kind"],
        "question": case["question"],
        "why": case.get("why"),
        "passed": not failures,
        "failures": failures,
        "observed": observed,
    }


async def main(plan_only: bool, only: list[str] | None) -> None:
    cases = yaml.safe_load(DATASET_PATH.read_text(encoding="utf-8"))
    if only:
        cases = [c for c in cases if c["id"] in only]
    if plan_only:
        cases = [c for c in cases if c["kind"] == "plan"]

    results = []
    for case in cases:
        try:
            result = await run_case(case)
        except LLMAnalysisError as exc:
            # Almost always a daily token quota wall. A suite meant to run on every prompt
            # edit must not die halfway through with a traceback and no results file —
            # record the case as errored, keep going, and let the summary say plainly that
            # the run was incomplete. An errored case is not a passing case and not a
            # failing one; conflating it with either would misreport the run.
            result = {
                "id": case["id"], "kind": case["kind"], "question": case["question"],
                "why": case.get("why"), "passed": False, "errored": True,
                "failures": [f"could not run: {exc}"], "observed": {},
            }
        results.append(result)
        mark = "ERR " if result.get("errored") else ("PASS" if result["passed"] else "FAIL")
        print(f"  {mark}  [{case['kind']:8}] {case['id']}")
        for failure in result["failures"]:
            print(f"          {failure}")

    passed = sum(1 for r in results if r["passed"])
    errored = [r for r in results if r.get("errored")]
    scored = [r for r in results if not r.get("errored")]

    by_kind: dict[str, list[bool]] = {}
    for r in scored:
        by_kind.setdefault(r["kind"], []).append(r["passed"])

    print(f"\n{passed}/{len(scored)} passed" + (f"  ({len(errored)} could not run)" if errored else ""))
    for kind, marks in sorted(by_kind.items()):
        print(f"  {kind}: {sum(marks)}/{len(marks)}")
    if errored:
        print(
            "\nIncomplete run — some cases never executed, so this is not a result to "
            "compare against a previous one."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"classify-{stamp}.json"
    out.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "passed": passed,
                "total": len(results),
                "scored": len(scored),
                "errored": len(errored),
                "complete": not errored,
                "cases": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out}")

    if passed != len(results):
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan-only", action="store_true",
        help="Skip cases that call the model — runs with no API key.",
    )
    parser.add_argument("--only", type=str, help="Comma-separated case ids.")
    args = parser.parse_args()
    asyncio.run(main(args.plan_only, args.only.split(",") if args.only else None))
