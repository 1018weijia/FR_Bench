"""TaskWrapper deferred detection helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterator

from ..injectors import ErrorInjector
from ..lifecycle import finalize_current_error_attempt, return_to_normal
from ..logging_utils import should_log
from ..recovery import handle_detected_failure


def reset_deferred_error_tracking(wrapper: Any) -> None:
    clear_grasp_slip_pending(wrapper)
    for state_key, _ in iter_deferred_detection_configs(wrapper):
        clear_deferred_detection_state(wrapper, state_key)


def register_grasp_slip_pending(wrapper: Any, result: Any, logger: Any) -> None:
    if isinstance(result, tuple) and len(result) == 2:
        arm_tag, _ = result
        wrapper.grasp_slip_injected_arm = arm_tag
        wrapper.grasp_slip_actions = result[1]
    wrapper._pending_grasp_slip_check = True


def clear_grasp_slip_pending(wrapper: Any) -> None:
    wrapper.grasp_slip_injected_arm = None
    wrapper.grasp_slip_actions = None
    wrapper._pending_grasp_slip_check = False


def set_deferred_detection_state(
    wrapper: Any,
    state_key: str,
    arm_tag: Any,
    context: Dict[str, Any],
    injector: ErrorInjector,
) -> None:
    setattr(wrapper, f"{state_key}_injected_arm", arm_tag)
    setattr(wrapper, f"{state_key}_context", context)
    setattr(wrapper, f"{state_key}_injector", injector)


def clear_deferred_detection_state(wrapper: Any, state_key: str) -> None:
    setattr(wrapper, f"{state_key}_injected_arm", None)
    setattr(wrapper, f"{state_key}_context", None)
    setattr(wrapper, f"{state_key}_injector", None)


def maybe_defer_failure_detection(
    wrapper: Any,
    injector: ErrorInjector,
    modified_context: Dict[str, Any],
    result: Any,
    logger: Any,
) -> bool:
    injector_name = getattr(injector, "name", "")
    method_name = modified_context.get("method_name")

    if injector_name == "grasp_slip" and method_name == "move_by_displacement":
        register_grasp_slip_pending(wrapper, result, logger)
        wrapper.current_phase = "normal"
        return True

    if method_name != "grasp_actor":
        return False

    deferred_config = wrapper._DEFERRED_GRASP_DETECTION_CONFIGS.get(injector_name)
    if deferred_config is None:
        return False

    if isinstance(result, tuple) and len(result) == 2:
        arm_tag, _ = result
        state_key = deferred_config["state_key"]
        set_deferred_detection_state(wrapper, state_key, arm_tag, modified_context, injector)

    wrapper.current_phase = "normal"
    return True


def iter_deferred_detection_configs(wrapper: Any) -> Iterator[tuple[str, int]]:
    yielded = set()
    for config in wrapper._DEFERRED_GRASP_DETECTION_CONFIGS.values():
        state_key = config["state_key"]
        if state_key in yielded:
            continue
        yielded.add(state_key)
        yield state_key, config["wait_steps"]


def run_deferred_detection(wrapper: Any, state_key: str, wait_steps: int, logger: Any) -> None:
    """Run deferred failure detection after move completes."""
    arm_attr = f"{state_key}_injected_arm"
    ctx_attr = f"{state_key}_context"
    inj_attr = f"{state_key}_injector"
    if getattr(wrapper, arm_attr) is None:
        return

    injector = getattr(wrapper, inj_attr)
    if injector is None:
        return

    error_type = getattr(injector, "name", "unknown")
    try:
        ctx = getattr(wrapper, ctx_attr)
        wrapper._safe_scene_step(wrapper.wrapped_task, wait_steps)
        failed = injector.detect_failure(ctx)
        injector.is_active = False

        clear_deferred_detection_state(wrapper, state_key)

        if failed:
            if should_log(getattr(wrapper, "error_recovery_log_level", "quiet"), "normal"):
                logger.info("[DeferredDetection:%s] Failure detected, entering recovery", error_type)
            try:
                handle_detected_failure(
                    wrapper,
                    injector,
                    ctx,
                    logger,
                    completion_log=f"[DeferredDetection:{error_type}] Recovery flow finished"
                    if should_log(getattr(wrapper, "error_recovery_log_level", "quiet"), "normal")
                    else "",
                    no_actions_phase="recovery_failed",
                    no_actions_log=f"[DeferredDetection:{error_type}] Recovery actions are empty",
                )
            except Exception as recovery_error:
                logger.exception("[DeferredDetection:%s] Recovery flow error: %s", error_type, recovery_error)
                wrapper.current_phase = "recovery_failed"
                finalize_current_error_attempt(wrapper, "recovery_failed")
                return_to_normal(wrapper)
        else:
            wrapper.current_phase = "normal"
            finalize_current_error_attempt(wrapper, "no_failure")
    except Exception as exc:
        logger.exception("[DeferredDetection:%s] Detection error: %s", error_type, exc)
        wrapper.current_phase = "normal"
        clear_deferred_detection_state(wrapper, state_key)
