"""Execution flow and move-plan helpers for TaskWrapper."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..action_compat import make_action
from ..error_injectors import ErrorInjector
from ..lifecycle import finalize_current_error_attempt, start_error_attempt
from ..logging_utils import should_log
from ..recovery import handle_detected_failure
from ..runtime import clear_grasp_slip_pending, iter_deferred_detection_configs, maybe_defer_failure_detection, run_deferred_detection
from . import logger
from .action_mutation import execute_modified_action

_RECOVERY_PHASES = frozenset(
    {
        "recovery",
        "recovery_grasp",
        "recovery_reposition",
        "recovery_open_gripper",
        "recovery_failed",
        "error_detected",
        "error_attempt",
    }
)


def check_error_injection(wrapper: Any, context: Dict[str, Any]) -> Optional[ErrorInjector]:
    if wrapper.current_phase in _RECOVERY_PHASES:
        return None
    injection_limit = min(int(getattr(wrapper, "max_errors_per_episode", 1)), 1)
    if wrapper.error_injection_count >= injection_limit:
        return None
    for injector in wrapper.error_injectors:
        if getattr(injector, "has_injected_this_episode", False):
            continue
        if injector.should_inject(context):
            return injector
    return None


def execute_with_error(wrapper: Any, original_method: Any, context: Dict[str, Any], injector: ErrorInjector):
    wrapper.error_injection_count += 1
    if should_log(getattr(wrapper, "error_recovery_log_level", "quiet"), "normal"):
        logger.info(
            "[execution] Injecting %s (episode attempt %d/%d)",
            injector.name,
            wrapper.error_injection_count,
            wrapper.max_errors_per_episode,
        )

    modified_context = injector.inject_error(context)
    wrapper.current_phase = "error_attempt"
    start_error_attempt(wrapper, injector)

    try:
        result = execute_modified_action(wrapper, original_method, modified_context)
        if maybe_defer_failure_detection(wrapper, injector, modified_context, result, logger):
            return result

        wrapper._safe_scene_step(wrapper.wrapped_task, 50)
        failed = injector.detect_failure(modified_context)
        injector.is_active = False
        if failed:
            handle_detected_failure(wrapper, injector, modified_context, logger)
        else:
            wrapper.current_phase = "normal"
            finalize_current_error_attempt(wrapper, "no_failure")
        return result
    except Exception:
        logger.exception("[execution] Error injection execution failed (%s)", injector.name)
        injector.is_active = False
        handle_detected_failure(wrapper, injector, modified_context, logger)
        raise


def execute_normal(wrapper: Any, original_method: Any, context: Dict[str, Any]):
    args = None
    if context["method_name"] == "move":
        args = list(context["args"])
        if args and isinstance(args[0], tuple) and len(args[0]) == 2:
            _, actions = args[0]
            if isinstance(actions, list) and len(actions) == 0 and wrapper.current_phase in _RECOVERY_PHASES:
                return True

    wrapper.action_history.append(
        {
            "method_name": context["method_name"],
            "phase": wrapper.current_phase,
            "frame_count": wrapper.frame_count,
            "timestamp": context["timestamp"],
        }
    )

    if context["method_name"] == "move":
        if args is None:
            args = list(context["args"])
        args = _preprocess_move_args(wrapper, args)
        valid = [
            item
            for item in args
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], list) and len(item[1]) > 0
        ]
        context["args"] = tuple(valid if valid else args)

    result = original_method(*context["args"], **context["kwargs"])

    if context["method_name"] == "move":
        _run_post_move_checks(wrapper)

    if context["method_name"] == "grasp_actor":
        try:
            actor = context.get("actor")
            if isinstance(result, tuple) and len(result) == 2:
                arm_tag_result, _ = result
                if actor is not None and arm_tag_result is not None:
                    wrapper.last_grasp_actor_by_arm[arm_tag_result] = actor
        except Exception:
            pass

    if context["method_name"] == "move_by_displacement":
        result = _sync_lift_action_segments(wrapper, result)

    return result


def _preprocess_move_args(wrapper: Any, args: list) -> list:
    """Normalize dual-arm arguments for task.move; split a single lift when needed to match the grasp_slip injected arm.

    move executes step by step via zip(left_actions, right_actions). The grasp_slip injected arm is usually
    [move, open, move] (three steps). If the other arm has only one move, split it into two segments so the arm step counts do not diverge badly.
    """
    # If planning failed or the caller passed a bad value, keep move from receiving None
    if not args or args[0] is None:
        return [("left", [])]

    for index, arg in enumerate(args):
        if isinstance(arg, tuple) and len(arg) == 2 and arg[0] is None:
            args[index] = ("left", [])

    # Single-arm move and similar cases do not need dual-arm sync
    if len(args) < 2 or not all(isinstance(item, tuple) and len(item) == 2 for item in args):
        return args

    left_arm, left_actions = args[0][0], (args[0][1] if isinstance(args[0][1], list) else [])
    right_arm, right_actions = args[1][0], (args[1][1] if isinstance(args[1][1], list) else [])

    def _has_slip(actions: list) -> bool:
        # Matches the three-step plan returned by action_mutation._execute_grasp_slip_lift_action
        return len(actions) == 3 and any(getattr(action, "action", None) == "open" for action in actions)

    # When only one side is the slip trio and the other side has a single move, split that move into two segments (segments=2)
    if _has_slip(left_actions) and not _has_slip(right_actions) and len(right_actions) == 1:
        args[1] = (right_arm, _split_lift_action(wrapper, right_arm, right_actions[0], segments=2))
    elif _has_slip(right_actions) and not _has_slip(left_actions) and len(left_actions) == 1:
        args[0] = (left_arm, _split_lift_action(wrapper, left_arm, left_actions[0], segments=2))
    return args


def _split_lift_action(wrapper: Any, arm_tag: Any, action: Any, segments: int = 2) -> list:
    """Split one vertical move into several segments so its timing matches the other arm's grasp_slip lift.

    segments=2: [move(small lift), move(to target height)]
    segments=3: [move, None, move]; the middle None is a placeholder so zip stays aligned (see _sync_lift_action_segments)
    """
    if getattr(action, "action", None) != "move" or getattr(action, "target_pose", None) is None:
        return [action]
    try:
        target_pose = action.target_pose
        if not (isinstance(target_pose, (list, np.ndarray)) and len(target_pose) >= 3):
            return [action]
        current_pose = np.array(wrapper.wrapped_task.get_arm_pose(arm_tag), dtype=np.float64)
        z_total = target_pose[2] - current_pose[2]
        if z_total <= 0.001:
            return [action]
        # The first segment lifts only a small part of the total height, matching grasp_slip's small lift before open
        z_first = min(0.01, z_total * 0.1)
        pose_first = current_pose.copy()
        pose_first[2] += z_first
        first = make_action(arm_tag, "move", target_pose=pose_first.tolist())
        last = make_action(arm_tag, "move", target_pose=target_pose)
        if segments >= 3:
            return [first, None, last]
        return [first, last]
    except Exception:
        return [action]


def _run_post_move_checks(wrapper: Any) -> None:
    if wrapper._pending_grasp_slip_check:
        try:
            for injector in wrapper.error_injectors:
                if getattr(injector, "name", "") != "grasp_slip" or not getattr(injector, "pending_check", False):
                    continue
                failure_context = {
                    "task": wrapper.wrapped_task,
                    "actor": getattr(injector, "target_actor", None),
                    "arm_tag": getattr(injector, "target_arm", None),
                }
                wrapper._safe_scene_step(wrapper.wrapped_task, 5)
                failed = injector.detect_failure(failure_context)
                injector.is_active = False
                injector.pending_check = False
                if failed:
                    handle_detected_failure(
                        wrapper,
                        injector,
                        failure_context,
                        logger,
                        set_recovery_injector=False,
                    )
                    clear_grasp_slip_pending(wrapper)
                else:
                    finalize_current_error_attempt(wrapper, "no_failure")
            wrapper._pending_grasp_slip_check = False
        except Exception:
            wrapper._pending_grasp_slip_check = True
            logger.exception("[execution] grasp_slip deferred detection error")

    for state_key, wait_steps in iter_deferred_detection_configs(wrapper):
        run_deferred_detection(wrapper, state_key, wait_steps, logger)


def _sync_lift_action_segments(wrapper: Any, result: Any):
    if not (isinstance(result, tuple) and len(result) == 2):
        return result
    arm_tag, actions = result
    if (
        wrapper.grasp_slip_injected_arm is not None
        and wrapper.grasp_slip_injected_arm != arm_tag
        and isinstance(actions, list)
        and len(actions) == 1
        and actions[0] is not None
        and getattr(actions[0], "action", None) is not None
    ):
        new_actions = _split_lift_action(wrapper, arm_tag, actions[0], segments=3)
        if len(new_actions) > 1:
            return (arm_tag, new_actions)
    return result
