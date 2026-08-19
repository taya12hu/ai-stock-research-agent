"""Deterministic, mechanically-checkable pass/fail gates (ARCHITECTURE.md §11) — these
catch real bugs (fabricated citations, malformed output, silently-dropped failures) that
an LLM judge might rate fine anyway. `hard=True` checks gate a case's pass/fail; `hard=
False` checks are informational (heuristic or LLM-behavior-dependent, not a strict
contract the code guarantees).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    hard: bool = True


def _collect_all_finding_ids(per_ticker_results: dict[str, Any] | None) -> set[str]:
    ids: set[str] = set()
    for ticker_results in (per_ticker_results or {}).values():
        for agent_result in ticker_results.values():
            for finding in agent_result.get("findings", []):
                ids.add(finding["id"])
    return ids


def _all_agent_results(per_ticker_results: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [r for agents in (per_ticker_results or {}).values() for r in agents.values()]


def check_schema(snapshot: dict[str, Any]) -> CheckResult:
    # A turn produces EITHER a research report OR a plain reply (clarification question,
    # off-topic/discovery-limitation explanation, follow-up answer) — never neither.
    has_report = isinstance(snapshot.get("final_report"), str) and bool(snapshot["final_report"])
    has_reply = isinstance(snapshot.get("followup_answer"), str) and bool(snapshot["followup_answer"])
    ok = (
        (has_report or has_reply)
        and snapshot.get("query_type") in {"single", "portfolio", "comparison"}
        and isinstance(snapshot.get("tickers"), list)
    )
    detail = "final_report/followup_answer, query_type, tickers present and well-typed" if ok else f"malformed snapshot: {snapshot}"
    return CheckResult("schema", ok, detail)


def check_no_fabricated_discovery_tickers(turn: dict[str, Any]) -> CheckResult | None:
    """For a turn tagged `expected_discovery: true` in the dataset — a request this app
    can't fulfill because it would require screening a market/sector for candidates —
    the only hard safety property that matters is that no ticker got fabricated to fill
    the gap. Whether the model produced a discovery-limitation reply or fell back to one
    clarifying question, both are safe; inventing a candidate company is the one
    outcome this must never allow, mirroring what `check_citation_integrity` guards
    against on the report side."""
    if not turn.get("expected_discovery"):
        return None
    tickers = turn["result_snapshot"].get("tickers") or []
    ok = not tickers
    detail = "no ticker fabricated" if ok else f"fabricated ticker(s) for an unsupported discovery request: {tickers}"
    return CheckResult("no_fabricated_discovery_tickers", ok, detail)


def check_citation_integrity(snapshot: dict[str, Any]) -> CheckResult:
    """Every [id] marker in the report body must resolve to a real Finding.id — catches
    fabricated/hallucinated citations mechanically."""
    report = snapshot.get("final_report") or ""
    body = report.split("**Sources**")[0]
    cited_ids = set(re.findall(r"\[([\w-]+)\]", body))
    all_ids = _collect_all_finding_ids(snapshot.get("per_ticker_results"))
    fabricated = cited_ids - all_ids
    ok = not fabricated
    detail = "all citations resolve to real findings" if ok else f"fabricated citation ids: {sorted(fabricated)}"
    return CheckResult("citation_integrity", ok, detail)


def check_findings_well_formed(snapshot: dict[str, Any]) -> CheckResult:
    problems: list[str] = []
    for ticker, agent_results in (snapshot.get("per_ticker_results") or {}).items():
        for agent, result in agent_results.items():
            if result["status"] != "ok":
                continue
            findings = result.get("findings", [])
            for f in findings:
                if not f.get("claim") or not f.get("evidence") or not f.get("id"):
                    problems.append(f"{ticker}/{agent}: finding with a blank field")
            if len(findings) > 5:
                problems.append(f"{ticker}/{agent}: {len(findings)} findings exceeds the cap of 5")
    ok = not problems
    return CheckResult("findings_well_formed", ok, "ok" if ok else "; ".join(problems))


def check_total_failure_explicit(snapshot: dict[str, Any]) -> CheckResult:
    """If literally every agent failed, the deterministic no-LLM-call fallback path
    guarantees a specific message — this is a hard contract, not LLM-dependent prose.
    It's a plain reply (`followup_answer`), not a "Research Report" card — there's no
    real content to put in a report when every source failed — so this checks either
    field, whichever the synthesis node actually used."""
    all_results = _all_agent_results(snapshot.get("per_ticker_results"))
    all_failed = bool(all_results) and all(r["status"] == "failed" for r in all_results)
    if not all_failed:
        return CheckResult("total_failure_explicit", True, "not applicable (not a total-failure case)")
    text = ((snapshot.get("final_report") or "") + (snapshot.get("followup_answer") or "")).lower()
    ok = "wasn't able to complete" in text or "unable to complete" in text
    detail = "deterministic failure message present" if ok else "total failure but no message says so"
    return CheckResult("total_failure_explicit", ok, detail)


def check_partial_failure_mentioned(snapshot: dict[str, Any]) -> CheckResult:
    """Softer sibling of the above: when only SOME agents failed, the synthesis LLM is
    instructed to mention the gap, but that's LLM behavior, not a guaranteed string."""
    all_results = _all_agent_results(snapshot.get("per_ticker_results"))
    any_failed = any(r["status"] == "failed" for r in all_results)
    if not any_failed:
        return CheckResult("partial_failure_mentioned", True, "not applicable (no failures)", hard=False)
    report = (snapshot.get("final_report") or "").lower()
    keywords = ("unavailable", "could not", "failed", "no data", "not available")
    ok = any(kw in report for kw in keywords)
    detail = "report acknowledges the gap" if ok else "report may be silently omitting a failed source"
    return CheckResult("partial_failure_mentioned", ok, detail, hard=False)


def check_followup_path(turn: dict[str, Any]) -> CheckResult | None:
    expected = turn.get("expected_path")
    if not expected:
        return None
    actual = turn["result_snapshot"].get("followup_path")
    ok = actual == expected
    return CheckResult("followup_path", ok, f"expected {expected!r}, got {actual!r}", hard=False)


def check_latency(elapsed_seconds: float, ticker_count: int) -> CheckResult:
    budget = 30 + 20 * max(ticker_count, 1)
    ok = elapsed_seconds <= budget
    return CheckResult("latency", ok, f"{elapsed_seconds:.1f}s (budget {budget}s)", hard=False)


def run_all_checks(turns: list[dict[str, Any]]) -> list[CheckResult]:
    final_snapshot = turns[-1]["result_snapshot"]
    checks = [
        check_schema(final_snapshot),
        check_citation_integrity(final_snapshot),
        check_findings_well_formed(final_snapshot),
        check_total_failure_explicit(final_snapshot),
        check_partial_failure_mentioned(final_snapshot),
        check_latency(turns[-1]["elapsed_seconds"], len(final_snapshot.get("tickers") or [])),
    ]
    for turn in turns:
        discovery_check = check_no_fabricated_discovery_tickers(turn)
        if discovery_check:
            checks.append(discovery_check)
    for turn in turns[1:]:
        followup_check = check_followup_path(turn)
        if followup_check:
            checks.append(followup_check)
    return checks
