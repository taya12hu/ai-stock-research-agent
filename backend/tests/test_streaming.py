from __future__ import annotations

import asyncio
import uuid

import pytest

from app.streaming import session_bus


@pytest.fixture(autouse=True)
async def _temp_db(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session_bus, "DB_PATH", tmp_path / "test_sessions.db")
    await session_bus.init_db()


async def test_publish_and_replay_roundtrip() -> None:
    session_id = str(uuid.uuid4())
    id1 = await session_bus.publish(session_id, {"type": "run_started"})
    id2 = await session_bus.publish(session_id, {"type": "run_completed"})

    events = await session_bus.get_events_after(session_id, 0)

    assert [e for _, e in events] == [{"type": "run_started"}, {"type": "run_completed"}]
    assert [i for i, _ in events] == [id1, id2]


async def test_get_events_after_only_returns_events_past_the_given_id() -> None:
    session_id = str(uuid.uuid4())
    await session_bus.publish(session_id, {"type": "run_started"})
    id2 = await session_bus.publish(session_id, {"type": "agent_started", "ticker": "AAPL", "agent": "news"})

    events = await session_bus.get_events_after(session_id, id2 - 1)

    assert len(events) == 1
    assert events[0][0] == id2


async def test_events_are_scoped_per_session() -> None:
    session_a, session_b = str(uuid.uuid4()), str(uuid.uuid4())
    await session_bus.publish(session_a, {"type": "run_started"})
    await session_bus.publish(session_b, {"type": "run_started"})

    events_a = await session_bus.get_events_after(session_a, 0)

    assert len(events_a) == 1


async def test_is_run_finished_reflects_terminal_event() -> None:
    session_id = str(uuid.uuid4())
    await session_bus.publish(session_id, {"type": "run_started"})
    assert await session_bus.is_run_finished(session_id) is False

    await session_bus.publish(session_id, {"type": "run_completed"})
    assert await session_bus.is_run_finished(session_id) is True


async def test_live_queue_receives_events_and_sentinel_on_terminal_event() -> None:
    session_id = str(uuid.uuid4())
    queue = await session_bus.register_live_queue(session_id)

    await session_bus.publish(session_id, {"type": "run_started"})
    await session_bus.publish(session_id, {"type": "run_completed"})

    item1 = await asyncio.wait_for(queue.get(), timeout=2)
    item2 = await asyncio.wait_for(queue.get(), timeout=2)
    sentinel = await asyncio.wait_for(queue.get(), timeout=2)

    assert item1[1] == {"type": "run_started"}
    assert item2[1] == {"type": "run_completed"}
    assert sentinel is None

    await session_bus.unregister_live_queue(session_id, queue)


async def test_multiple_subscribers_each_get_a_full_copy_of_events() -> None:
    """Multi-tab / overlapping-reconnect safety: two live subscribers to the same
    session must each receive every event, not split them between each other."""
    session_id = str(uuid.uuid4())
    queue_a = await session_bus.register_live_queue(session_id)
    queue_b = await session_bus.register_live_queue(session_id)

    await session_bus.publish(session_id, {"type": "run_started"})

    item_a = await asyncio.wait_for(queue_a.get(), timeout=2)
    item_b = await asyncio.wait_for(queue_b.get(), timeout=2)

    assert item_a[1] == item_b[1] == {"type": "run_started"}

    await session_bus.unregister_live_queue(session_id, queue_a)
    await session_bus.unregister_live_queue(session_id, queue_b)


async def test_unregister_removes_only_that_queue() -> None:
    session_id = str(uuid.uuid4())
    queue_a = await session_bus.register_live_queue(session_id)
    queue_b = await session_bus.register_live_queue(session_id)

    await session_bus.unregister_live_queue(session_id, queue_a)
    await session_bus.publish(session_id, {"type": "run_started"})

    item_b = await asyncio.wait_for(queue_b.get(), timeout=2)
    assert item_b[1] == {"type": "run_started"}
    assert queue_a.empty()

    await session_bus.unregister_live_queue(session_id, queue_b)
