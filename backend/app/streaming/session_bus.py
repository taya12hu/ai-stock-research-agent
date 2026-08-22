"""Per-session event log + live pub/sub (ARCHITECTURE.md §10).

Events are durably appended to SQLite first, then fanned out to any live subscribers.
This ordering is what makes reconnection correct: a client's `Last-Event-ID` replay
always reads from the same durable store live subscribers are fed from, so nothing
published before a client (re)connects can be missed, and nothing is double-delivered
(the SSE route dedupes by event id — see `app/api/research_routes.py`).

Single-process scope: live fan-out is an in-memory `dict[session_id, list[Queue]]`, so
this does not span multiple worker processes. Fine for this project's scope (a single
dev/demo backend process); a multi-worker deployment would need a shared pub/sub layer
(e.g. Redis) instead.
"""

from __future__ import annotations

import asyncio
import json
import random
from typing import Any

import aiosqlite

from app.config import settings
from app.logging_config import get_logger, log_event
from app.streaming.events import TERMINAL_EVENT_TYPES

logger = get_logger("app.streaming.session_bus")

DB_PATH = settings.data_dir / "sessions.db"

_live_queues: dict[str, list[asyncio.Queue]] = {}
_live_queues_lock = asyncio.Lock()

_CLEANUP_SAMPLE_RATE = 0.02  # opportunistic TTL eviction on ~2% of publishes


async def init_db() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, event_id)"
        )
        await conn.commit()


async def publish(session_id: str, event: dict[str, Any]) -> int:
    async with aiosqlite.connect(DB_PATH) as conn:
        cursor = await conn.execute(
            "INSERT INTO events (session_id, event_type, payload) VALUES (?, ?, ?)",
            (session_id, event["type"], json.dumps(event)),
        )
        await conn.commit()
        event_id = cursor.lastrowid

    async with _live_queues_lock:
        queues = list(_live_queues.get(session_id, []))
    for queue in queues:
        await queue.put((event_id, event))
        if event["type"] in TERMINAL_EVENT_TYPES:
            await queue.put(None)  # sentinel: tells the SSE route no more events are coming

    if random.random() < _CLEANUP_SAMPLE_RATE:
        await _cleanup_expired_sessions()

    return event_id


async def get_events_after(session_id: str, after_id: int) -> list[tuple[int, dict[str, Any]]]:
    async with aiosqlite.connect(DB_PATH) as conn:
        cursor = await conn.execute(
            "SELECT event_id, payload FROM events WHERE session_id = ? AND event_id > ? ORDER BY event_id",
            (session_id, after_id),
        )
        rows = await cursor.fetchall()
    return [(row[0], json.loads(row[1])) for row in rows]


async def is_run_finished(session_id: str) -> bool:
    async with aiosqlite.connect(DB_PATH) as conn:
        cursor = await conn.execute(
            "SELECT event_type FROM events WHERE session_id = ? ORDER BY event_id DESC LIMIT 1",
            (session_id,),
        )
        row = await cursor.fetchone()
    return bool(row) and row[0] in TERMINAL_EVENT_TYPES


async def register_live_queue(session_id: str) -> asyncio.Queue:
    queue: asyncio.Queue = asyncio.Queue()
    async with _live_queues_lock:
        _live_queues.setdefault(session_id, []).append(queue)
    return queue


async def unregister_live_queue(session_id: str, queue: asyncio.Queue) -> None:
    async with _live_queues_lock:
        queues = _live_queues.get(session_id)
        if queues and queue in queues:
            queues.remove(queue)
        if queues is not None and not queues:
            _live_queues.pop(session_id, None)


async def _cleanup_expired_sessions() -> None:
    cutoff = f"-{settings.session_ttl_seconds} seconds"
    async with aiosqlite.connect(DB_PATH) as conn:
        cursor = await conn.execute(
            """
            DELETE FROM events WHERE session_id IN (
                SELECT session_id FROM events
                GROUP BY session_id
                HAVING MAX(
                    CASE WHEN event_type IN ('run_completed', 'run_failed') THEN created_at END
                ) < datetime('now', ?)
            )
            """,
            (cutoff,),
        )
        await conn.commit()
        if cursor.rowcount:
            log_event(logger, "evicted expired session events", rows_deleted=cursor.rowcount)
