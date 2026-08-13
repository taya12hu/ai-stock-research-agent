"""LangGraph checkpointer lifecycle — persists conversation state per session
(`thread_id` = `session_id`) so follow-ups resume from where the last turn left off.

Reuses the same SQLite file as the SSE event log (`app/streaming/session_bus.py`) —
ARCHITECTURE.md §9 calls for "the same SQLite store", and `AsyncSqliteSaver` just adds
its own tables to that file alongside the `events` table.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.config import settings

CHECKPOINT_DB_PATH = settings.data_dir / "sessions.db"


@asynccontextmanager
async def get_checkpointer() -> AsyncIterator[AsyncSqliteSaver]:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB_PATH)) as saver:
        await saver.setup()
        yield saver
