from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api.research_routes import FollowUpRequest, _running_tasks, ask_followup
from app.streaming import session_bus


@pytest.fixture(autouse=True)
async def _temp_db(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session_bus, "DB_PATH", tmp_path / "test_sessions.db")
    await session_bus.init_db()


@pytest.fixture(autouse=True)
def _clean_running_tasks():
    _running_tasks.clear()
    yield
    _running_tasks.clear()


class _FakeState:
    def __init__(self, conversation: list[dict]) -> None:
        self.values = {"conversation": conversation}


class _FakeGraph:
    """Enough of the compiled graph's surface for `ask_followup`'s own logic — the
    404/409 guards and background-task bookkeeping — without exercising real LangGraph
    execution, which is already covered in test_graph.py."""

    def __init__(self, conversation: list[dict]) -> None:
        self._state = _FakeState(conversation)

    async def aget_state(self, config: dict) -> _FakeState:  # noqa: ARG002
        return self._state

    async def astream(self, *args, **kwargs):  # noqa: ARG002
        return
        yield  # pragma: no cover - makes this an async generator that yields nothing


def _fake_request(conversation: list[dict]) -> SimpleNamespace:
    graph = _FakeGraph(conversation)
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(graph=graph)))


async def test_ask_followup_404s_when_no_prior_turn_exists() -> None:
    session_id = str(uuid.uuid4())
    request = _fake_request([])

    with pytest.raises(HTTPException) as exc_info:
        await ask_followup(session_id, FollowUpRequest(question="hi"), request)

    assert exc_info.value.status_code == 404


async def test_ask_followup_rejects_a_second_call_while_a_run_is_still_in_progress() -> None:
    """Regression guard: nothing previously stopped two `POST .../ask` calls for the same
    session_id from racing each other against the same LangGraph checkpointer thread — a
    second tab, or a client retry, could both be mid-run at once. `_running_tasks` existed
    already but was only ever used for GC-avoidance, never actually checked."""
    session_id = str(uuid.uuid4())
    request = _fake_request([{"role": "user", "content": "hi"}])

    async def _never_finishes() -> None:
        await asyncio.Event().wait()

    in_flight = asyncio.create_task(_never_finishes())
    _running_tasks[session_id] = in_flight

    with pytest.raises(HTTPException) as exc_info:
        await ask_followup(session_id, FollowUpRequest(question="another one"), request)

    assert exc_info.value.status_code == 409
    assert _running_tasks[session_id] is in_flight  # untouched — the guard fired first

    in_flight.cancel()
    with pytest.raises(asyncio.CancelledError):
        await in_flight


async def test_ask_followup_allows_a_new_call_once_the_previous_run_finished() -> None:
    session_id = str(uuid.uuid4())
    request = _fake_request([{"role": "user", "content": "hi"}])

    finished = asyncio.create_task(asyncio.sleep(0))
    await finished
    _running_tasks[session_id] = finished

    response = await ask_followup(session_id, FollowUpRequest(question="another one"), request)

    assert response.session_id == session_id
    await _running_tasks[session_id]  # let the newly-spawned background run finish cleanly
