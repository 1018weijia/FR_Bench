"""Error attempt lifecycle helpers for TaskWrapper."""

from __future__ import annotations

import pickle
from typing import Any, Dict, Optional


def is_error_phase(phase: Any) -> bool:
    """Coarse phase counts as an error frame, including recovery sub-phases."""
    if not isinstance(phase, str):
        return False
    return phase in {"error_attempt", "error_detected"} or phase.startswith("recovery")


def get_latest_error_attempt(wrapper: Any) -> Optional[Dict[str, Any]]:
    if not wrapper.error_metadata:
        return None
    last = wrapper.error_metadata[-1]
    return last if isinstance(last, dict) else None


def start_error_attempt(wrapper: Any, injector: Any) -> None:
    wrapper.error_attempt_counter += 1
    attempt_id = wrapper.error_attempt_counter
    wrapper.current_error_attempt_id = attempt_id

    error_info = {
        "attempt_id": attempt_id,
        "injector": injector.name,
        "phase": "error_attempt",
        "inject_decision_frame": wrapper.frame_count,
        "error_onset_frame": None,
        "error_detected_frame": None,
        "error_detected_is_fake": False,
        "recovery_start_frame": None,
        "recovery_end_frame": None,
        "error_end_frame": None,
        "end_status": None,
        "onset_reason": None,
        "metadata": injector.get_metadata(),
    }
    wrapper.error_metadata.append(error_info)


def mark_error_onset(wrapper: Any, reason: str, logger: Any) -> None:
    attempt = get_latest_error_attempt(wrapper)
    if attempt is None or attempt.get("error_onset_frame") is not None:
        return
    attempt["error_onset_frame"] = wrapper.frame_count
    attempt["onset_reason"] = reason
    attempt["phase"] = "error_active"
    logger.debug(f"[TaskWrapper] Fine label: error_onset @ frame={wrapper.frame_count}, reason={reason}")


def mark_error_detected(wrapper: Any) -> None:
    attempt = get_latest_error_attempt(wrapper)
    if attempt is None:
        return
    if attempt.get("error_onset_frame") is None:
        attempt["error_onset_frame"] = attempt.get("inject_decision_frame", wrapper.frame_count)
        attempt["onset_reason"] = attempt.get("onset_reason") or "auto_backfill_before_detected"
    if attempt.get("error_detected_frame") is None:
        attempt["error_detected_frame"] = wrapper.frame_count
    attempt["error_detected_is_fake"] = False
    attempt["phase"] = "error_detected"


def backfill_fake_error_detected(wrapper: Any) -> bool:
    """Backfill a fake detected frame for failed attempts lacking a real detect point."""
    attempt = get_latest_error_attempt(wrapper)
    if attempt is None:
        return False
    if attempt.get("error_detected_frame") is not None:
        return False

    fallback_frame = next(
        (
            frame
            for frame in (
                attempt.get("error_end_frame"),
                attempt.get("error_onset_frame"),
                attempt.get("inject_decision_frame"),
            )
            if _frame_number(frame) >= 1
        ),
        None,
    )
    if fallback_frame is None:
        return False

    attempt["error_detected_frame"] = int(fallback_frame)
    attempt["error_detected_is_fake"] = True
    return True


def _frame_number(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def close_open_attempt(wrapper: Any) -> None:
    """Finish an attempt that is still open when the episode ends."""
    attempt = get_latest_error_attempt(wrapper)
    if attempt is None or attempt.get("end_status"):
        return
    if _frame_number(attempt.get("error_detected_frame")) >= 1:
        finalize_current_error_attempt(wrapper, "recovery_failed")
    else:
        finalize_current_error_attempt(wrapper, "no_failure")
    return_to_normal(wrapper)


def mark_recovery_start(wrapper: Any) -> None:
    attempt = get_latest_error_attempt(wrapper)
    if attempt is None:
        return
    if attempt.get("recovery_start_frame") is None:
        attempt["recovery_start_frame"] = wrapper.frame_count
    attempt["phase"] = "recovery"


def finalize_current_error_attempt(wrapper: Any, status: str) -> None:
    attempt = get_latest_error_attempt(wrapper)
    if attempt is None:
        return
    attempt["end_status"] = status
    attempt["phase"] = status
    if status in ("recovered", "recovery_failed"):
        if attempt.get("recovery_end_frame") is None:
            attempt["recovery_end_frame"] = wrapper.frame_count
        attempt["error_end_frame"] = attempt.get("recovery_end_frame")
    else:
        attempt["error_end_frame"] = wrapper.frame_count
    wrapper.current_error_attempt_id = None


def return_to_normal(wrapper: Any) -> None:
    """Close the live phase and make the already saved end frame match."""
    wrapper.current_phase = "normal"
    _relabel_saved_end_frame(wrapper)


def _relabel_saved_end_frame(wrapper: Any) -> None:
    task = getattr(wrapper, "wrapped_task", None)
    folder = getattr(task, "folder_path", None)
    frame_idx = getattr(task, "FRAME_IDX", None)
    if not isinstance(folder, dict) or not frame_idx:
        return
    cache = folder.get("cache")
    if not cache:
        return
    path = f"{cache}{int(frame_idx) - 1}.pkl"
    try:
        with open(path, "rb") as handle:
            frame = pickle.load(handle)
    except (OSError, pickle.UnpicklingError, EOFError):
        return
    recovery = frame.get("error_recovery") if isinstance(frame, dict) else None
    if not isinstance(recovery, dict) or recovery.get("frame_count") != wrapper.frame_count:
        return
    recovery["phase"] = wrapper.current_phase
    recovery["is_error_frame"] = is_error_phase(wrapper.current_phase)
    recovery.update(get_precise_error_state(wrapper))
    attempt = get_latest_error_attempt(wrapper)
    saved_attempt = recovery.get("error_metadata")
    if isinstance(attempt, dict) and isinstance(saved_attempt, dict):
        saved_attempt["phase"] = attempt.get("phase")
        saved_attempt["end_status"] = attempt.get("end_status")
    try:
        with open(path, "wb") as handle:
            pickle.dump(frame, handle)
    except OSError:
        return


def get_precise_error_state(wrapper: Any) -> Dict[str, Any]:
    """Build a per-frame fine stage label from the current frame's position on the attempt timeline"""
    attempt = get_latest_error_attempt(wrapper)
    frame = wrapper.frame_count
    default = {
        "error_stage_precise": "normal",
        "is_error_frame_precise": False,
    }
    if attempt is None:
        return default

    inject_decision = attempt.get("inject_decision_frame")
    onset = attempt.get("error_onset_frame")
    detected = attempt.get("error_detected_frame")
    recovery_start = attempt.get("recovery_start_frame")
    error_end = attempt.get("error_end_frame")
    recovery_end = attempt.get("recovery_end_frame")
    end_status = attempt.get("end_status")

    # Not yet injected, or a frame after this attempt has ended
    if inject_decision is None or frame < inject_decision:
        return default
    if error_end is not None and frame > error_end:
        return default

    # Injected, but the error action has not started (onset is marked by on_before_move_action)
    if onset is None or frame < onset:
        return {"error_stage_precise": "pre_error", "is_error_frame_precise": False}

    # Recovery segment: the last frame can be refined to recovered / recovery_failed
    if recovery_start is not None and frame >= recovery_start:
        if recovery_end is not None and frame == recovery_end and end_status in ("recovered", "recovery_failed"):
            return {"error_stage_precise": end_status, "is_error_frame_precise": True}
        return {"error_stage_precise": "recovery", "is_error_frame_precise": True}

    # Failure decided, recovery not started yet (after onset, before recovery_start)
    if detected is not None and frame >= detected:
        return {"error_stage_precise": "error_detected", "is_error_frame_precise": True}

    # After onset and before detected: the error behavior is in progress
    return {"error_stage_precise": "error_active", "is_error_frame_precise": True}
