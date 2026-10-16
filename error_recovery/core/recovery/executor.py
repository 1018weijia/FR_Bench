"""Recovery flow orchestration helpers."""

from __future__ import annotations

from typing import Any, Dict, List

from ..lifecycle import finalize_current_error_attempt, mark_error_detected, mark_recovery_start
from ..logging_utils import should_log
from .action_ops import execute_single_recovery_action
from .judge import finalize_recovery_outcome, infer_recovery_injector_name, maybe_mark_recovery_success


def execute_recovery_actions(wrapper: Any, recovery_actions: List[Dict[str, Any]], logger: Any) -> None:
    """Execute recovery actions while preserving TaskWrapper semantics."""
    if should_log(getattr(wrapper, "error_recovery_log_level", "quiet"), "normal"):
        logger.info("[Recovery] Starting recovery actions, count=%d", len(recovery_actions))

    wrapper.current_phase = "recovery"
    mark_recovery_start(wrapper)
    recovery_succeeded = False
    infer_recovery_injector_name(wrapper, recovery_actions)

    for index, action in enumerate(recovery_actions):
        action_type = action.get("action_type", "unknown")
        try:
            execute_single_recovery_action(wrapper, action, logger)
            if "phase" in action:
                wrapper.current_phase = action["phase"]
            if maybe_mark_recovery_success(wrapper, recovery_actions, index, action, logger):
                recovery_succeeded = True
                break
        except Exception as exc:
            wrapper.last_recovery_error = {
                "step": index + 1,
                "total": len(recovery_actions),
                "action_type": action_type,
                "error": str(exc),
            }
            if should_log(getattr(wrapper, "error_recovery_log_level", "quiet"), "normal"):
                logger.warning(
                    "[Recovery] Action failed step=%d/%d type=%s err=%s",
                    index + 1,
                    len(recovery_actions),
                    action_type,
                    exc,
                )
            if action_type in ["grasp_actor", "place_actor"] and should_log(
                getattr(wrapper, "error_recovery_log_level", "quiet"), "debug"
            ):
                logger.error("[Recovery] Possible cause: path planning failed")
            wrapper.current_phase = "recovery_failed"
            finalize_current_error_attempt(wrapper, "recovery_failed")
            break

    finalize_recovery_outcome(wrapper, recovery_actions, recovery_succeeded, logger)
    wrapper.current_recovery_injector_name = None


def handle_detected_failure(
    wrapper: Any,
    injector: Any,
    context: Dict[str, Any],
    logger: Any,
    *,
    recovery_attempt: int = 1,
    set_recovery_injector: bool = True,
    completion_log: str = "",
    no_actions_phase: str = "",
    no_actions_log: str = "",
) -> None:
    """Mark a detected failure and run the injector recovery plan."""
    wrapper.current_phase = "error_detected"
    mark_error_detected(wrapper)
    if set_recovery_injector:
        wrapper.current_recovery_injector_name = getattr(injector, "name", None)

    try:
        recovery_actions = injector.plan_recovery(context, recovery_attempt=recovery_attempt)
        if recovery_actions:
            execute_recovery_actions(wrapper, recovery_actions, logger)
            if completion_log:
                logger.info(completion_log)
        else:
            if no_actions_log:
                logger.warning(no_actions_log)
            if no_actions_phase:
                wrapper.current_phase = no_actions_phase
            finalize_current_error_attempt(wrapper, "recovery_failed")
    finally:
        if set_recovery_injector:
            wrapper.current_recovery_injector_name = None
