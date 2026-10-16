"""Small verbosity helpers for collection-time logging."""

from __future__ import annotations

from typing import Any

LOG_LEVELS = {"quiet": 0, "normal": 1, "debug": 2}


def normalize_log_level(value: Any) -> str:
    value = str(value or "quiet").lower()
    return value if value in LOG_LEVELS else "quiet"


def should_log(current: Any, required: str) -> bool:
    current_level = LOG_LEVELS[normalize_log_level(current)]
    required_level = LOG_LEVELS[normalize_log_level(required)]
    return current_level >= required_level

