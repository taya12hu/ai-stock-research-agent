"""Manual/dev verification script for the follow-up conversation flow — exercises all
three follow-up paths (answer / refresh / add_ticker) against real Yahoo Finance,
DuckDuckGo, and Groq calls, using a real checkpointer so state persists across turns
exactly as the API layer does.

Usage:
    .venv/Scripts/python.exe scripts/verify_followup.py
"""

from __future__ import annotations

import asyncio
import uuid

from app.graph.build_graph import build_research_graph
from app.graph.session import new_session_state
from app.memory.checkpointer import get_checkpointer


async def _ask(graph, config: dict, question: str) -> dict:
    print(f"\n{'=' * 80}\nQ: {question}\n{'=' * 80}")
    result = await graph.ainvoke({"user_question": question}, config=config)
    print(f"followup_path: {result.get('followup_path')}")
    print(f"notes: {result.get('notes')}")
    if result.get("followup_answer"):
        print(f"answer: {result['followup_answer']}")
    print(f"tickers now: {result['tickers']}  query_type: {result['query_type']}")
    return result


async def main() -> None:
    async with get_checkpointer() as checkpointer:
        graph = build_research_graph(checkpointer=checkpointer)
        session_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": session_id}}

        state = new_session_state(user_question="Compare NVIDIA and AMD", session_id=session_id)
        result = await graph.ainvoke(state, config=config)
        print(f"TURN 1 (initial): tickers={result['tickers']} query_type={result['query_type']}")
        print(result["final_report"][:400], "...\n")

        await _ask(graph, config, "Which of the two has the higher profit margin?")
        r3 = await _ask(graph, config, "Any fresh news on NVIDIA today?")
        print("report changed:", bool(r3.get("final_report")))
        r4 = await _ask(graph, config, "Now also add Intel to this comparison")
        print("\nFinal report tail:\n", r4["final_report"][-1200:])

        print("\nconversation turns:", len(r4["conversation"]))
        for m in r4["conversation"]:
            print(f"  [{m['role']}] {m['content'][:80]}")


if __name__ == "__main__":
    asyncio.run(main())
