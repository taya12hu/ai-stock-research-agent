"""POST /research starts a run in the background; GET /research/{id}/stream serves its
progress as SSE, replaying persisted events on reconnect via `Last-Event-ID` (see
ARCHITECTURE.md §9 and `app/streaming/session_bus.py`). POST /research/{id}/ask handles
a follow-up turn in the same session (ARCHITECTURE.md §8) — the session's conversation
persists across both endpoints via the LangGraph checkpointer, keyed by
`thread_id` = `session_id` (see `app/memory/checkpointer.py`).

Graph execution is observed via `astream(state, stream_mode="updates")` and translated
into the event shapes in `app/streaming/events.py` — no node code is touched to make
this work, so the (already-tested) node functions stay free of streaming concerns.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from app.graph.build_graph import SPECIALIST_NODES
from app.graph.state import new_state
from app.logging_config import get_logger, log_event
from app.streaming import events as ev
from app.streaming import session_bus

logger = get_logger("app.api.research")
router = APIRouter()

# Keeps references to fire-and-forget run tasks so they aren't garbage-collected mid-run.
_running_tasks: dict[str, asyncio.Task] = {}


class ResearchRequest(BaseModel):
    question: str


class FollowUpRequest(BaseModel):
    question: str


class ResearchResponse(BaseModel):
    session_id: str


def _thread_config(session_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": session_id}}


@router.post("/research", response_model=ResearchResponse)
async def start_research(payload: ResearchRequest, request: Request) -> ResearchResponse:
    session_id = str(uuid.uuid4())
    graph = request.app.state.graph
    task = asyncio.create_task(_run_and_publish(graph, session_id, payload.question))
    _running_tasks[session_id] = task
    log_event(logger, "research run started", session_id=session_id)
    return ResearchResponse(session_id=session_id)


@router.post("/research/{session_id}/ask", response_model=ResearchResponse)
async def ask_followup(session_id: str, payload: FollowUpRequest, request: Request) -> ResearchResponse:
    graph = request.app.state.graph
    existing = await graph.aget_state(_thread_config(session_id))
    if not existing.values.get("per_ticker_results"):
        raise HTTPException(status_code=404, detail="No prior research session found for this session_id")

    task = asyncio.create_task(_run_followup_and_publish(graph, session_id, payload.question))
    _running_tasks[session_id] = task
    log_event(logger, "followup run started", session_id=session_id)
    return ResearchResponse(session_id=session_id)


async def _run_and_publish(graph: CompiledStateGraph, session_id: str, user_question: str) -> None:
    state = new_state(user_question=user_question, session_id=session_id)
    await _stream_run(graph, session_id, state)


async def _run_followup_and_publish(graph: CompiledStateGraph, session_id: str, question: str) -> None:
    await _stream_run(graph, session_id, {"user_question": question})


async def _stream_run(graph: CompiledStateGraph, session_id: str, run_input: dict[str, Any]) -> None:
    await session_bus.publish(session_id, ev.run_started())
    try:
        async for chunk in graph.astream(run_input, config=_thread_config(session_id), stream_mode="updates"):
            for node_name, update in chunk.items():
                await _publish_for_node(session_id, node_name, update)
        await session_bus.publish(session_id, ev.run_completed())
    except Exception as exc:
        log_event(
            logger, "research run failed", level=logging.ERROR,
            session_id=session_id, error=str(exc),
        )
        await session_bus.publish(session_id, ev.run_failed(str(exc)))
    finally:
        _running_tasks.pop(session_id, None)


async def _publish_for_node(session_id: str, node_name: str, update: dict[str, Any] | None) -> None:
    if not update:
        return

    if node_name == "router":
        tickers: list[str] = update.get("tickers", [])
        await session_bus.publish(
            session_id, ev.router_completed(update.get("query_type", ""), tickers, update.get("notes", []))
        )
        for ticker in tickers:
            for agent in SPECIALIST_NODES:
                await session_bus.publish(session_id, ev.agent_started(ticker, agent))
        return

    if node_name == "followup_router":
        path = update.get("followup_path", "answer")
        targets = update.get("followup_targets", [])
        target_tickers = [t["ticker"] for t in targets]
        await session_bus.publish(session_id, ev.followup_classified(path, target_tickers, update.get("notes", [])))
        for target in targets:
            for agent in target["agents"]:
                await session_bus.publish(session_id, ev.agent_started(target["ticker"], agent))
        return

    if node_name == "answer_from_context" and "followup_answer" in update:
        await session_bus.publish(session_id, ev.followup_answer_ready(update["followup_answer"]))
        return

    if "per_ticker_results" in update:
        for ticker, agent_updates in update["per_ticker_results"].items():
            for agent, result in agent_updates.items():
                await session_bus.publish(session_id, ev.agent_completed(ticker, agent, result))
        return

    if "final_report" in update:
        await session_bus.publish(session_id, ev.report_ready(update["final_report"]))


def _format_sse(event_id: int, event: dict[str, Any]) -> str:
    return f"id: {event_id}\nevent: {event['type']}\ndata: {json.dumps(event)}\n\n"


@router.get("/research/{session_id}/stream")
async def stream_research(session_id: str, request: Request) -> StreamingResponse:
    last_event_id = request.headers.get("last-event-id")
    after_id = int(last_event_id) if last_event_id else 0

    async def event_generator() -> AsyncIterator[str]:
        queue = await session_bus.register_live_queue(session_id)
        try:
            last_sent_id = after_id
            for event_id, event in await session_bus.get_events_after(session_id, after_id):
                last_sent_id = event_id
                yield _format_sse(event_id, event)

            if await session_bus.is_run_finished(session_id):
                return

            while True:
                if await request.is_disconnected():
                    break
                item = await queue.get()
                if item is None:
                    break
                event_id, event = item
                if event_id <= last_sent_id:
                    continue  # already sent during replay (race with live registration)
                last_sent_id = event_id
                yield _format_sse(event_id, event)
        finally:
            await session_bus.unregister_live_queue(session_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
