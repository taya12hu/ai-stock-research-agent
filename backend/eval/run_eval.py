"""Evaluation harness (ARCHITECTURE.md §11) — runs the dataset against the graph
in-process (bypassing HTTP for speed), scores each case with objective checks + an LLM
judge, and writes a timestamped JSON report to eval/results/. Re-run after a change and
diff the two JSON files to see whether it actually helped quality/reliability/latency.

Usage (run from backend/ — needs `-m` so the `eval` package's own imports resolve):
    .venv/Scripts/python.exe -m eval.run_eval
    .venv/Scripts/python.exe -m eval.run_eval --limit 3       # quick smoke run
    .venv/Scripts/python.exe -m eval.run_eval --only single_aapl_direct,invalid_ticker_only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from langgraph.checkpoint.memory import InMemorySaver

# Judge rationales and report text can contain characters (em-dashes, curly quotes) outside
# Windows' default console codepage (cp1252) — printing them without this crashes the run
# outright with UnicodeEncodeError partway through case 1, before any results are ever
# written. UTF-8 is safe everywhere this actually runs (a terminal or a redirected file).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from app.graph.build_graph import build_research_graph
from app.graph.session import new_session_state
from eval.judge import judge_report
from eval.objective_checks import CheckResult, run_all_checks

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DATASET_PATH = Path(__file__).resolve().parent / "dataset.yaml"


def _snapshot(result: dict[str, Any]) -> dict[str, Any]:
    """Flatten a graph result into the shape the checks read.

    The snapshot is the harness's adapter layer, so the turn-based state is translated
    here rather than by teaching every check about `turn`. The mapping is one-to-one:
    `shape` is what `query_type` always meant, `scope` is the tickers this answer covers,
    and `researched` has the same nested {ticker: {agent: ...}} shape `per_ticker_results`
    had, with cells carrying the same `status`/`findings` keys.

    The one real improvement is that report-versus-plain-reply is now read off
    `output.kind` rather than inferred from which field happened to be populated — a turn
    can no longer appear to have both a fresh answer and a stale report (A-09).
    """
    turn = result.get("turn") or {}
    output = turn.get("output") or {}
    is_report = output.get("kind") == "report"
    return {
        "query_type": turn.get("shape"),
        "tickers": turn.get("scope"),
        "notes": turn.get("notes"),
        "final_report": output.get("text") if is_report else None,
        "per_ticker_results": result.get("researched"),
        "followup_path": turn.get("kind"),
        # Any plain-reply turn (clarification, off-domain, recall answer) rather than a
        # research report — needed so check_schema and the discovery-fabrication check can
        # see turns that never produce a report at all.
        "followup_answer": None if is_report else output.get("text"),
    }


async def run_case(graph: Any, case: dict[str, Any]) -> dict[str, Any]:
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    start = time.perf_counter()
    state = new_session_state(user_question=case["question"], session_id=session_id)
    result = await graph.ainvoke(state, config=config)
    turns: list[dict[str, Any]] = [
        {
            "question": case["question"],
            "result_snapshot": _snapshot(result),
            "elapsed_seconds": time.perf_counter() - start,
            "expected_discovery": case.get("expected_discovery"),
            "expected_query_type": case.get("expected_query_type"),
            "expected_tickers": case.get("expected_tickers"),
        }
    ]

    for followup in case.get("followups", []):
        f_start = time.perf_counter()
        result = await graph.ainvoke({"user_question": followup["question"]}, config=config)
        turns.append(
            {
                "question": followup["question"],
                "result_snapshot": _snapshot(result),
                "elapsed_seconds": time.perf_counter() - f_start,
                "expected_path": followup.get("expected_path"),
                "expected_discovery": followup.get("expected_discovery"),
                "expected_query_type": followup.get("expected_query_type"),
                "expected_tickers": followup.get("expected_tickers"),
            }
        )

    checks = run_all_checks(turns)
    final = turns[-1]["result_snapshot"]
    if final.get("final_report"):
        try:
            judge_scores = await judge_report(
                case["question"], final["query_type"], final["final_report"]
            )
            judge_dict: dict[str, Any] = judge_scores.model_dump()
        except Exception as exc:  # eval-harness robustness: one bad judge call shouldn't kill the run
            judge_dict = {"error": str(exc)}
    else:
        # The last turn is a plain reply (clarification/off-topic/discovery), not a
        # research report — scoring an empty report against report-quality dimensions
        # (grounding, structure, ...) would be meaningless, not just uninformative.
        judge_dict = {"skipped": "no final_report produced by the last turn (non-research reply)"}

    return {
        "id": case["id"],
        "question": case["question"],
        "turns": turns,
        "checks": [c.__dict__ for c in checks],
        "judge": judge_dict,
        "passed_hard_checks": all(c.passed for c in checks if c.hard),
    }


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summarize(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    judge_dims = ["grounding", "relevance", "completeness", "structure"]
    avg_judge = {
        dim: round(_avg([c["judge"][dim] for c in case_results if dim in c["judge"]]), 2)
        for dim in judge_dims
    }
    avg_latency = round(
        _avg([turn["elapsed_seconds"] for c in case_results for turn in c["turns"]]), 1
    )
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "case_count": len(case_results),
        "hard_checks_pass_rate": round(_avg([float(c["passed_hard_checks"]) for c in case_results]), 2),
        "avg_judge_scores": avg_judge,
        "avg_latency_seconds": avg_latency,
        "cases": case_results,
    }


async def main(limit: int | None, only: list[str] | None) -> None:
    cases = yaml.safe_load(DATASET_PATH.read_text())
    if only:
        cases = [c for c in cases if c["id"] in only]
    if limit:
        cases = cases[:limit]

    checkpointer = InMemorySaver()
    graph = build_research_graph(checkpointer=checkpointer)

    case_results = []
    for case in cases:
        print(f"Running case: {case['id']}...", flush=True)
        case_result = await run_case(graph, case)
        case_results.append(case_result)
        failed_hard = [c["name"] for c in case_result["checks"] if c["hard"] and not c["passed"]]
        status = "PASS" if case_result["passed_hard_checks"] else f"FAIL ({', '.join(failed_hard)})"
        print(f"  {status} | judge: {case_result['judge']}")

    summary = _summarize(case_results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))

    print(f"\nWrote results to {out_path}")
    print(f"Hard-check pass rate: {summary['hard_checks_pass_rate']:.0%}")
    print(f"Avg judge scores: {summary['avg_judge_scores']}")
    print(f"Avg latency: {summary['avg_latency_seconds']}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N cases")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated case ids to run")
    args = parser.parse_args()
    asyncio.run(main(args.limit, args.only.split(",") if args.only else None))
