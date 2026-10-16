"""Recovery success and retry judgement helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..lifecycle import finalize_current_error_attempt, return_to_normal
from ..logging_utils import should_log


def infer_recovery_injector_name(wrapper: Any, recovery_actions: List[Dict[str, Any]]) -> None:
    if wrapper.current_recovery_injector_name:
        return
    for action in recovery_actions:
        if action.get("action_type") == "grasp_actor" and action.get("recovery_attempt") is not None:
            if any(getattr(inj, "name", None) == "grasp_slip" for inj in wrapper.error_injectors):
                wrapper.current_recovery_injector_name = "grasp_slip"
            break


def _get_robot_gripper_val(task: Any, arm_tag: Any) -> float:
    if str(arm_tag) == "left":
        return task.robot.get_left_gripper_val()
    return task.robot.get_right_gripper_val()


def _check_recovery_grasp_success(
    task: Any,
    actor: Any,
    arm_tag: Any,
    *,
    require_lift: bool,
    is_grasp_slip: bool,
):
    if is_grasp_slip:
        distance_threshold = min(task.get_dynamic_distance_threshold(actor, base_threshold=0.15), 0.25)
        gripper_threshold = task.get_dynamic_gripper_threshold(actor)
        success = task.check_grasp_success(
            actor,
            arm_tag,
            require_lift=require_lift,
            distance_threshold=distance_threshold,
            gripper_val_threshold=gripper_threshold,
            use_dynamic_threshold=False,
        )
        return success, distance_threshold, gripper_threshold

    distance_threshold = task.get_dynamic_distance_threshold(actor, base_threshold=0.12)
    success = task.check_grasp_success(
        actor,
        arm_tag,
        require_lift=require_lift,
        distance_threshold=distance_threshold,
        use_dynamic_threshold=True,
    )
    return success, distance_threshold, None


def maybe_mark_recovery_success(
    wrapper: Any,
    recovery_actions: List[Dict[str, Any]],
    action_index: int,
    action: Dict[str, Any],
    logger: Any,
) -> bool:
    task = wrapper.wrapped_task
    action_type = action.get("action_type")
    if action_type == "grasp_actor":
        return _check_grasp_actor_recovery(wrapper, recovery_actions, action_index, action, task, logger)
    if action_type == "move_by_displacement" and action.get("z", 0) > 0:
        return _check_lift_recovery(wrapper, recovery_actions, action_index, task, logger)
    return False


def finalize_recovery_outcome(wrapper: Any, recovery_actions: List[Dict[str, Any]], recovery_succeeded: bool, logger: Any) -> None:
    task = wrapper.wrapped_task
    if wrapper.current_phase != "recovery_failed" and recovery_succeeded:
        wrapper.current_phase = "recovered"
        finalize_current_error_attempt(wrapper, "recovered")
        if should_log(getattr(wrapper, "error_recovery_log_level", "quiet"), "normal"):
            logger.info("[Recovery] Error recovery succeeded")
        return_to_normal(wrapper)
        return

    if wrapper.current_phase == "recovery_failed":
        return_to_normal(wrapper)
        return

    actor, arm_tag = _find_recovery_grasp_target(recovery_actions)
    injector = _find_active_grasp_slip_injector(wrapper)
    if actor and arm_tag and injector:
        if _maybe_run_second_recovery(wrapper, injector, actor, arm_tag, task, logger):
            return

    if wrapper.current_phase != "recovered":
        wrapper.current_phase = "recovery_failed"
        finalize_current_error_attempt(wrapper, "recovery_failed")
        if should_log(getattr(wrapper, "error_recovery_log_level", "quiet"), "normal"):
            logger.warning("[Recovery] Recovery failed")
        return_to_normal(wrapper)


def _check_grasp_actor_recovery(
    wrapper: Any,
    recovery_actions: List[Dict[str, Any]],
    action_index: int,
    action: Dict[str, Any],
    task: Any,
    logger: Any,
) -> bool:
    actor = action.get("actor")
    arm_tag = action.get("arm_tag")
    if not (actor and arm_tag and hasattr(task, "check_grasp_success")):
        return False

    try:
        if hasattr(task, "actor_initial_poses"):
            task.actor_initial_poses[actor] = actor.get_pose()
    except Exception:
        pass

    wrapper._safe_scene_step(task, 200)
    is_grasp_slip = wrapper.current_recovery_injector_name == "grasp_slip"
    success1, _, _ = _check_recovery_grasp_success(
        task,
        actor,
        arm_tag,
        require_lift=False,
        is_grasp_slip=is_grasp_slip,
    )

    if success1:
        if is_grasp_slip:
            return False
        if should_log(getattr(wrapper, "error_recovery_log_level", "quiet"), "normal"):
            logger.info("[Recovery] Recovery grasp succeeded, skip_remaining=%d", len(recovery_actions) - action_index - 1)
        wrapper.current_phase = "recovered"
        return True

    wrapper._safe_scene_step(task, 200)
    success2, _, _ = _check_recovery_grasp_success(
        task,
        actor,
        arm_tag,
        require_lift=False,
        is_grasp_slip=is_grasp_slip,
    )

    if success2:
        if is_grasp_slip:
            return False
        if should_log(getattr(wrapper, "error_recovery_log_level", "quiet"), "normal"):
            logger.info("[Recovery] Recovery grasp succeeded (second check), skip_remaining=%d", len(recovery_actions) - action_index - 1)
        wrapper.current_phase = "recovered"
        return True

    return _backup_grasp_actor_check(
        wrapper,
        recovery_actions,
        action_index,
        actor,
        arm_tag,
        task,
        is_grasp_slip,
        logger,
    )


def _backup_grasp_actor_check(
    wrapper: Any,
    recovery_actions: List[Dict[str, Any]],
    action_index: int,
    actor: Any,
    arm_tag: Any,
    task: Any,
    is_grasp_slip: bool,
    logger: Any,
) -> bool:
    del is_grasp_slip
    try:
        gripper_pose = task.get_arm_pose(arm_tag)
        actor_pos = actor.get_pose().p
        distance = np.linalg.norm(gripper_pose[:3] - actor_pos)
        gripper_val = _get_robot_gripper_val(task, arm_tag)

        backup_distance_threshold = min(task.get_dynamic_distance_threshold(actor, base_threshold=0.18), 0.28)
        backup_gripper_threshold = task.get_dynamic_gripper_threshold(actor, base_threshold=0.3)
        if should_log(getattr(wrapper, "error_recovery_log_level", "quiet"), "normal"):
            logger.warning(
                "[Recovery] Recovery grasp failed, distance=%.3f/%.3f gripper=%.2f/%.2f",
                distance,
                backup_distance_threshold,
                gripper_val,
                backup_gripper_threshold,
            )
    except Exception as exc:
        if should_log(getattr(wrapper, "error_recovery_log_level", "quiet"), "normal"):
            logger.warning("[Recovery] Recovery grasp backup check error: %s", exc)
    return False


def _check_lift_recovery(
    wrapper: Any,
    recovery_actions: List[Dict[str, Any]],
    action_index: int,
    task: Any,
    logger: Any,
) -> bool:
    actor, arm_tag = _find_recovery_grasp_target(recovery_actions[: action_index + 1])
    if not (actor and arm_tag and hasattr(task, "check_grasp_success")):
        return False

    wrapper._safe_scene_step(task, 40)
    is_grasp_slip = wrapper.current_recovery_injector_name == "grasp_slip"
    success1, _, _ = _check_recovery_grasp_success(
        task,
        actor,
        arm_tag,
        require_lift=True,
        is_grasp_slip=is_grasp_slip,
    )

    wrapper._safe_scene_step(task, 20)
    success2, _, _ = _check_recovery_grasp_success(
        task,
        actor,
        arm_tag,
        require_lift=True,
        is_grasp_slip=is_grasp_slip,
    )

    if success1 or success2:
        if should_log(getattr(wrapper, "error_recovery_log_level", "quiet"), "normal"):
            logger.info("[Recovery] Recovery grasp succeeded (including lift), skip_remaining=%d", len(recovery_actions) - action_index - 1)
        wrapper.current_phase = "recovered"
        return True

    if is_grasp_slip:
        return _backup_lift_check(wrapper, actor, arm_tag, task, logger)
    return False


def _backup_lift_check(wrapper: Any, actor: Any, arm_tag: Any, task: Any, logger: Any) -> bool:
    try:
        gripper_pose = task.get_arm_pose(arm_tag)
        actor_pos = actor.get_pose().p
        distance = np.linalg.norm(gripper_pose[:3] - actor_pos)
        gripper_val = _get_robot_gripper_val(task, arm_tag)

        backup_dist_th = min(task.get_dynamic_distance_threshold(actor, base_threshold=0.18), 0.28)
        backup_grip_th = task.get_dynamic_gripper_threshold(actor)
        current_pose = actor.get_pose()
        initial_pose = task.actor_initial_poses.get(actor)
        if initial_pose is not None:
            dz = current_pose.p[2] - initial_pose.p[2]
            is_lifted = dz > 0.02
            if is_lifted and gripper_val < backup_grip_th and distance < backup_dist_th:
                if should_log(getattr(wrapper, "error_recovery_log_level", "quiet"), "normal"):
                    logger.info("[Recovery] Recovery grasp succeeded (lift backup check), dz=%.3f g=%.2f d=%.3f", dz, gripper_val, distance)
                wrapper.current_phase = "recovered"
                return True
            if should_log(getattr(wrapper, "error_recovery_log_level", "quiet"), "normal"):
                logger.warning("[Recovery] Lift backup check failed, dz=%.3f g=%.2f d=%.3f", dz, gripper_val, distance)
    except Exception as exc:
        if should_log(getattr(wrapper, "error_recovery_log_level", "quiet"), "normal"):
            logger.warning("[Recovery] Lift backup check error: %s", exc)
    return False


def _find_recovery_grasp_target(recovery_actions: List[Dict[str, Any]]) -> Tuple[Optional[Any], Optional[Any]]:
    for action in recovery_actions:
        if action.get("action_type") == "grasp_actor":
            return action.get("actor"), action.get("arm_tag")
    return None, None


def _find_active_grasp_slip_injector(wrapper: Any) -> Optional[Any]:
    for injector in wrapper.error_injectors:
        if injector.name == "grasp_slip" and injector.is_active:
            return injector
    return None


def _maybe_run_second_recovery(
    wrapper: Any,
    injector: Any,
    actor: Any,
    arm_tag: Any,
    task: Any,
    logger: Any,
) -> bool:
    try:
        actor_pos = actor.get_pose().p
        gripper_pose = task.get_arm_pose(arm_tag)
        distance = np.linalg.norm(gripper_pose[:3] - actor_pos)
        if not (distance < 0.3 and 0.7 < actor_pos[2] < 1.0):
            if should_log(getattr(wrapper, "error_recovery_log_level", "quiet"), "normal"):
                logger.warning("[Recovery] Second recovery not started, distance=%.3fm height=%.3fm", distance, actor_pos[2])
            return False

        if should_log(getattr(wrapper, "error_recovery_log_level", "quiet"), "normal"):
            logger.info("[Recovery] Starting second recovery, distance=%.3fm", distance)
        second_recovery_actions = injector.plan_recovery(
            {"actor": actor, "arm_tag": arm_tag, "task": task},
            recovery_attempt=2,
        )
        if not second_recovery_actions:
            if should_log(getattr(wrapper, "error_recovery_log_level", "quiet"), "normal"):
                logger.warning("[Recovery] Second recovery produced no actions")
            return False

        from .executor import execute_recovery_actions

        execute_recovery_actions(wrapper, second_recovery_actions, logger)
        if wrapper.current_phase == "recovered":
            if should_log(getattr(wrapper, "error_recovery_log_level", "quiet"), "normal"):
                logger.info("[Recovery] Second recovery succeeded")
            return True

        if should_log(getattr(wrapper, "error_recovery_log_level", "quiet"), "normal"):
            logger.warning("[Recovery] Second recovery failed")
    except Exception as exc:
        if should_log(getattr(wrapper, "error_recovery_log_level", "quiet"), "normal"):
            logger.warning("[Recovery] Second recovery condition check error: %s", exc)
    return False
