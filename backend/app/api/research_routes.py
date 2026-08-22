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

from app.graph.session import new_session_state
from app.logging_config import get_logger, log_event
from app.streaming import events as ev
from app.streaming import session_bus

logger = get_logger("app.api.research")
router = APIRouter()

# Keeps references to fire-and-forget run tasks so they aren't garbage-collected mid-run,
# and doubles as the "is this session busy" registry.
_running_tasks: dict[str, asyncio.Task] = {}

# Guards the check-and-claim below. The previous version was correct, but only by accident
# of statement ordering: there was no `await` between reading `_running_tasks` and writing
# it back, so asyncio's single-threaded scheduling happened to make the pair atomic.
# Any future `await` inserted into that window would have silently reintroduced a race in
# which two concurrent runs drive the same checkpointer thread. Making the critical section
# explicit means the correctness no longer depends on nobody adding a line.
_registry_lock = asyncio.Lock()


async def _claim_session(session_id: str, coro: Any) -> bool:
    """Start `coro` as this session's run, unless one is already in flight.

    `.done()` rather than plain membership: `_stream_run`'s `finally` pops the entry, but
    only once the task has actually finished, so a membership check would report "busy" for
    one extra tick at completion.
    """
    async with _registry_lock:
        running = _running_tasks.get(session_id)
        if running is not None and not running.done():
            coro.close()  # never awaited — close it so Python doesn't warn about it
            return False
        _running_tasks[session_id] = asyncio.create_task(coro)
        return True


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
    # A fresh uuid can't collide, so this claim always succeeds — routed through the same
    # helper anyway so there is exactly one place that registers a run.
    await _claim_session(session_id, _run_and_publish(graph, session_id, payload.question))
    log_event(logger, "research run started", session_id=session_id)
    return ResearchResponse(session_id=session_id)


@router.post("/research/{session_id}/ask", response_model=ResearchResponse)
async def ask_followup(session_id: str, payload: FollowUpRequest, request: Request) -> ResearchResponse:
    graph = request.app.state.graph
    existing = await graph.aget_state(_thread_config(session_id))
    # Any completed turn — research, a clarification question asked, or an off-topic
    # reply — means the session exists and the conversation can continue. Checking
    # `conversation` (written by `emit` on every path) rather than `researched` avoids
    # 404ing a session whose first message was off-topic, or one whose only clarification
    # attempt dead-ended.
    has_prior_turn = bool(existing.values.get("conversation"))
    if not has_prior_turn:
        raise HTTPException(status_code=404, detail="No prior research session found for this session_id")

    # The frontend already disables its input while `status === "running"` (App.tsx), which
    # covers a single tab — this guards the same session against a second tab, a client
    # retry, or any other caller racing a `POST .../ask` against a run that is still
    # writing to the same LangGraph checkpointer thread.
    claimed = await _claim_session(
        session_id, _run_followup_and_publish(graph, session_id, payload.question)
    )
    if not claimed:
        raise HTTPException(status_code=409, detail="A run is already in progress for this session")

    log_event(logger, "followup run started", session_id=session_id)
    return ResearchResponse(session_id=session_id)


async def _run_and_publish(graph: CompiledStateGraph, session_id: str, user_question: str) -> None:
    state = new_session_state(user_question=user_question, session_id=session_id)
    await _stream_run(graph, session_id, state)


async def _run_followup_and_publish(graph: CompiledStateGraph, session_id: str, question: str) -> None:
    await _stream_run(graph, session_id, {"user_question": question})


async def _stream_run(graph: CompiledStateGraph, session_id: str, run_input: dict[str, Any]) -> None:
    try:
        await session_bus.publish(session_id, ev.run_started())
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
    """Derive progress events from the plan, not from which state field a node happened to
    populate.

    The previous version sniffed for `followup_answer` / `per_ticker_results` /
    `final_report` in the update, which meant the event a turn produced depended on which
    node ran last rather than on what the turn was (A-09). `turn.kind` and
    `turn.output.kind` say it directly.
    """
    if not update:
        return

    if node_name == "plan":
        turn = update.get("turn") or {}
        # `shape` and `scope` map onto the existing wire fields: the frontend's "query
        # type" header and per-ticker agent cards mean exactly this, so the format is
        # unchanged even though the state behind it is entirely different.
        await session_bus.publish(
            session_id, ev.router_completed(turn.get("shape", ""), turn.get("scope", []), turn.get("notes", []))
        )
        for cell in turn.get("fetch", []):
            await session_bus.publish(session_id, ev.agent_started(cell["ticker"], cell["agent"]))
        return

    if node_name == "emit":
        output = (update.get("turn") or {}).get("output") or {}
        if output.get("kind") == "report":
            await session_bus.publish(session_id, ev.report_ready(output["text"]))
        else:
            await session_bus.publish(session_id, ev.followup_answer_ready(output.get("text", "")))
        return

    if "researched" in update:
        for ticker, cells in update["researched"].items():
            for agent, cell in cells.items():
                await session_bus.publish(session_id, ev.agent_completed(ticker, agent, cell))


def _format_sse(event_id: int, event: dict[str, Any]) -> str:
    return f"id: {event_id}\nevent: {event['type']}\ndata: {json.dumps(event)}\n\n"


@router.get("/research/{session_id}/stream")
async def stream_research(session_id: str, request: Request) -> StreamingResponse:
    # A browser-native EventSource auto-reconnect (same connection, dropped mid-run) sends
    # `Last-Event-ID` itself. But our own code opens a brand-new EventSource at the start of
    # every turn (see useResearchStream.ts's `openStream`), which never carries that header —
    # so the client passes how far it's already consumed via `after_id` instead, covering the
    # whole session rather than just the current run. Without this, every follow-up would
    # replay the entire session's event history (including prior turns' answers) from
    # scratch, misapplying old events to the new turn's transcript entry.
    header_last_id = request.headers.get("last-event-id")
    query_after_id = request.query_params.get("after_id")
    after_id = max(
        int(header_last_id) if header_last_id else 0,
        int(query_after_id) if query_after_id else 0,
    )

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
