"""Manual/dev verification script for the full research graph (router + dynamic
multi-ticker fan-out + rendering) — runs against real market-data providers,
DuckDuckGo, and Groq calls (not mocked; see tests/test_graph.py for the mocked/
automated version).

Usage:
    .venv/Scripts/python.exe scripts/verify_graph.py "Analyze NVIDIA"
    .venv/Scripts/python.exe scripts/verify_graph.py "Compare NVIDIA and AMD"
    .venv/Scripts/python.exe scripts/verify_graph.py "Analyze my portfolio of NVIDIA, Apple and Microsoft"
    .venv/Scripts/python.exe scripts/verify_graph.py "Analyze ZZZINVALIDTICKERXYZ"   # forced-failure case
"""

from __future__ import annotations

import asyncio
import sys
import uuid

from app.graph.build_graph import build_research_graph
from app.graph.session import new_session_state


async def main(question: str) -> None:
    graph = build_research_graph()
    state = new_session_state(user_question=question, session_id=str(uuid.uuid4()))
    result = await graph.ainvoke(state)

    print("=" * 80)
    print(f"query_type: {result['query_type']}  tickers: {result['tickers']}")
    if result.get("notes"):
        print(f"notes: {result['notes']}")
    for ticker, ticker_results in result["researched"].items():
        print(f"-- {ticker} --")
        for agent, agent_result in ticker_results.items():
            if agent_result["status"] == "failed":
                print(f"  {agent}: FAILED — {agent_result['error']}")
            else:
                print(f"  {agent}: ok — {len(agent_result['findings'])} findings")
    print("=" * 80)
    print()
    print(result["final_report"])


if __name__ == "__main__":
    question_arg = sys.argv[1] if len(sys.argv) > 1 else "Analyze AAPL"
    asyncio.run(main(question_arg))
