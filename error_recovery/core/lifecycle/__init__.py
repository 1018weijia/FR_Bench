"""Compatibility-friendly lifecycle helpers for TaskWrapper."""

from .attempt_state import (
    backfill_fake_error_detected,
    close_open_attempt,
    finalize_current_error_attempt,
    get_latest_error_attempt,
    get_precise_error_state,
    is_error_phase,
    mark_error_detected,
    mark_error_onset,
    mark_recovery_start,
    return_to_normal,
    start_error_attempt,
)

__all__ = [
    "backfill_fake_error_detected",
    "close_open_attempt",
    "finalize_current_error_attempt",
    "get_latest_error_attempt",
    "get_precise_error_state",
    "is_error_phase",
    "mark_error_detected",
    "mark_error_onset",
    "mark_recovery_start",
    "return_to_normal",
    "start_error_attempt",
]
