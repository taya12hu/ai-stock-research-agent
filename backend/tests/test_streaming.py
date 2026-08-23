from __future__ import annotations

import asyncio
import uuid

import pytest

from app.api.research_routes import _publish_for_node
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


# --- _publish_for_node: notes only broadcast when they're not about to be redundant ---


async def test_plan_event_carries_this_turns_scope_shape_and_notes() -> None:
    """Progress events are derived from the plan, not from which state field a node
    happened to populate. `shape`/`scope` map onto the existing wire fields — the
    frontend's query-type header and per-ticker agent cards mean exactly this — so the
    format is unchanged even though the state behind it is entirely different."""
    session_id = str(uuid.uuid4())
    await _publish_for_node(
        session_id, "plan",
        {"turn": {
            "kind": "research", "shape": "single", "scope": ["AAPL"],
            "aspects": ["news"],
            "notes": ["'ZZZ' could not be found and was skipped."],
            "fetch": [{"ticker": "AAPL", "agent": "news"}],
        }},
    )

    events = [e for _, e in await session_bus.get_events_after(session_id, 0)]

    # `aspects` is on the plan but deliberately not on the wire: which agents are running
    # is carried by the `agent_started` events derived from `fetch`, which is the set that
    # is actually dispatched.
    assert events[0] == {
        "type": "router_completed", "query_type": "single", "tickers": ["AAPL"],
        "notes": ["'ZZZ' could not be found and was skipped."],
    }
    assert events[1]["type"] == "agent_started"
    assert events[1]["agent"] == "news"


async def test_plan_event_starts_only_the_agents_actually_dispatched() -> None:
    """One `agent_started` per cell in `fetch`, not per (ticker x agent) pair. A turn
    narrowed to one aspect must not light up three agent cards that will never run."""
    session_id = str(uuid.uuid4())
    await _publish_for_node(
        session_id, "plan",
        {"turn": {
            "kind": "research", "shape": "single", "scope": ["AAPL"], "notes": [],
            "fetch": [{"ticker": "AAPL", "agent": "fundamentals"}],
        }},
    )

    events = [e for _, e in await session_bus.get_events_after(session_id, 0)]
    started = [e for e in events if e["type"] == "agent_started"]

    assert len(started) == 1
    assert started[0]["agent"] == "fundamentals"


async def test_notes_folded_into_a_chat_reply_are_not_also_broadcast() -> None:
    """The note field has two audiences: an aside alongside a real result, and the raw
    material for the reply itself when there is no result. `plan_turn` clears them once
    they have been folded into the prose, so a user is never told the same thing twice —
    handled in the plan rather than re-derived by every downstream consumer."""
    session_id = str(uuid.uuid4())
    await _publish_for_node(
        session_id, "plan",
        {"turn": {
            "kind": "chat", "shape": "single", "scope": [], "aspects": [], "notes": [],
            "fetch": [],
        }},
    )

    events = [e for _, e in await session_bus.get_events_after(session_id, 0)]

    assert events == [
        {"type": "router_completed", "query_type": "single", "tickers": [], "notes": []}
    ]


async def test_emit_event_kind_follows_the_turns_output_not_the_node_that_ran() -> None:
    """A-09: previously the event a turn produced depended on which state field the last
    node happened to set, so a stale `final_report` from two turns earlier could surface
    as the current answer. `turn.output.kind` says it directly."""
    report_session = str(uuid.uuid4())
    await _publish_for_node(
        report_session, "emit",
        {"turn": {"output": {"kind": "report", "text": "# Research Report: AAPL"}}},
    )
    answer_session = str(uuid.uuid4())
    await _publish_for_node(
        answer_session, "emit",
        {"turn": {"output": {"kind": "answer", "text": "Its P/E is 30."}}},
    )

    report_events = [e for _, e in await session_bus.get_events_after(report_session, 0)]
    answer_events = [e for _, e in await session_bus.get_events_after(answer_session, 0)]

    assert report_events[0]["type"] == "report_ready"
    assert answer_events[0]["type"] == "followup_answer_ready"


async def test_specialist_results_broadcast_one_completion_per_cell() -> None:
    session_id = str(uuid.uuid4())
    await _publish_for_node(
        session_id, "news",
        {"researched": {"AAPL": {"news": {
            "status": "ok", "summary": "s", "findings": [], "error": None,
            "fetched_at": "2026-08-19T14:00:00+00:00",
        }}}},
    )

    events = [e for _, e in await session_bus.get_events_after(session_id, 0)]

    assert len(events) == 1
    assert events[0]["type"] == "agent_completed"
    assert events[0]["ticker"] == "AAPL"
    assert events[0]["agent"] == "news"
