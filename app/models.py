"""
In-memory analysis log. No SQLAlchemy, no database file.
Keeps last 500 results in RAM. Zero disk writes.
Termux-safe: no file locking, no migrations.

Swap this module for SQLite persistence if needed later.
"""
from __future__ import annotations

from collections import deque
from typing import Any

_LOG: deque[dict[str, Any]] = deque(maxlen=500)


def log_result(result: dict[str, Any]) -> None:
    _LOG.append(result)


def get_log() -> list[dict[str, Any]]:
    return list(_LOG)


def clear_log() -> None:
    _LOG.clear()
