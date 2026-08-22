"""Central logging setup.

Every node, tool call, and API request should log through a logger obtained from
`get_logger(component)`, and wrap non-trivial functions/flows with `@trace(component)`
so entry/exit/duration/failure show up in `logs/dump.log` without hand-written
boilerplate at every call site. This is the developer-facing observability layer,
distinct from the user-facing SSE event stream (see ARCHITECTURE.md §10).
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import logging.handlers
import time
from typing import Any, Callable, TypeVar

from app.config import settings

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S"

_configured = False

F = TypeVar("F", bound=Callable[..., Any])


def configure_logging() -> None:
    """Idempotent: safe to call from multiple entrypoints (app startup, tests, eval scripts)."""
    global _configured
    if _configured:
        return

    settings.log_dir.mkdir(parents=True, exist_ok=True)
    log_file = settings.log_dir / "dump.log"

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(settings.log_level)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    _configured = True


def get_logger(component: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(component)


def log_event(
    logger: logging.Logger,
    message: str,
    *,
    level: int = logging.INFO,
    session_id: str | None = None,
    **fields: Any,
) -> None:
    """Structured log line: human-readable message + a JSON tail of extra fields."""
    payload = {"session_id": session_id, **fields}
    logger.log(level, "%s | %s", message, json.dumps(payload, default=str))


def trace(component: str) -> Callable[[F], F]:
    """Decorator: logs start/completed/failed + duration_ms for a function or flow.

    Works on both sync and async functions (agent nodes and tool calls are async;
    small helpers may be sync).
    """

    def decorator(func: F) -> F:
        logger = get_logger(component)
        name = func.__qualname__

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            log_event(logger, f"{name} started")
            try:
                result = await func(*args, **kwargs)
            except Exception as exc:
                log_event(
                    logger,
                    f"{name} failed",
                    level=logging.ERROR,
                    duration_ms=round((time.perf_counter() - start) * 1000, 1),
                    error=str(exc),
                )
                raise
            log_event(
                logger,
                f"{name} completed",
                duration_ms=round((time.perf_counter() - start) * 1000, 1),
            )
            return result

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            log_event(logger, f"{name} started")
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                log_event(
                    logger,
                    f"{name} failed",
                    level=logging.ERROR,
                    duration_ms=round((time.perf_counter() - start) * 1000, 1),
                    error=str(exc),
                )
                raise
            log_event(
                logger,
                f"{name} completed",
                duration_ms=round((time.perf_counter() - start) * 1000, 1),
            )
            return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper  # type: ignore[return-value]

    return decorator
