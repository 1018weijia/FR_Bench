"""Single recovery action execution helpers."""

from __future__ import annotations

import time
from typing import Any, Dict

from ..action_compat import make_action


def execute_single_recovery_action(wrapper: Any, action: Dict[str, Any], logger: Any) -> None:
    """Execute one recovery action while preserving the existing flow."""
    action_type = action["action_type"]
    task = wrapper.wrapped_task

    handler = _RECOVERY_ACTION_HANDLERS.get(action_type)
    if handler is not None:
        handler(wrapper, action, logger)

    settle_required = _resolve_settle_steps(action)
    if settle_required:
        wrapper._safe_scene_step(task, settle_required)


def _resolve_settle_steps(action: Dict[str, Any]) -> int:
    settle_required = action.get("settle_steps", 0)
    if settle_required:
        return settle_required
    if action.get("action_type") in {"place_actor", "move_by_displacement", "move", "close_gripper"}:
        return 20
    return 0


def _handle_open_gripper(wrapper: Any, action: Dict[str, Any], logger: Any) -> None:
    arm_tag = action["arm_tag"]
    task = wrapper.wrapped_task
    task.move((arm_tag, [make_action(arm_tag, "open", target_gripper_pos=1.0)]))
    wrapper._safe_scene_step(task, 50)


def _handle_close_gripper(wrapper: Any, action: Dict[str, Any], logger: Any) -> None:
    arm_tag = action["arm_tag"]
    task = wrapper.wrapped_task
    task.move((arm_tag, [make_action(arm_tag, "close", target_gripper_pos=0.0)]))


def _handle_perceive_actor(wrapper: Any, action: Dict[str, Any], logger: Any) -> None:
    return


def _handle_move_to_actor(wrapper: Any, action: Dict[str, Any], logger: Any) -> None:
    _execute_move_to_actor(wrapper, action, logger)


def _handle_grasp_actor(wrapper: Any, action: Dict[str, Any], logger: Any) -> None:
    _execute_grasp_actor(wrapper, action, logger)


def _handle_place_actor(wrapper: Any, action: Dict[str, Any], logger: Any) -> None:
    task = wrapper.wrapped_task
    actor = action["actor"]
    arm_tag = action["arm_tag"]
    target_pose = action["target_pose"]
    place_action = task.place_actor(actor, arm_tag=arm_tag, target_pose=target_pose)
    if place_action is None or (isinstance(place_action, tuple) and place_action[0] is None):
        raise ValueError("place_actor returned an invalid action; planning may have failed")
    task.move(place_action)


def _handle_move(wrapper: Any, action: Dict[str, Any], logger: Any) -> None:
    if "target_pose" not in action:
        return
    task = wrapper.wrapped_task
    arm_tag = action["arm_tag"]
    target_pose = action["target_pose"]
    task.move((arm_tag, [make_action(arm_tag, "move", target_pose=target_pose)]))


def _handle_move_by_displacement(wrapper: Any, action: Dict[str, Any], logger: Any) -> None:
    task = wrapper.wrapped_task
    arm_tag = action["arm_tag"]
    displacement_kwargs = {k: action[k] for k in ["x", "y", "z", "rx", "ry", "rz"] if k in action}
    move_action = task.move_by_displacement(arm_tag=arm_tag, **displacement_kwargs)
    task.move(move_action)


def _handle_wait(wrapper: Any, action: Dict[str, Any], logger: Any) -> None:
    task = wrapper.wrapped_task
    duration = action.get("duration", 0.5)
    time.sleep(duration)
    wrapper._safe_scene_step(task, int(duration * 60))


_RECOVERY_ACTION_HANDLERS = {
    "open": _handle_open_gripper,
    "open_gripper": _handle_open_gripper,
    "close": _handle_close_gripper,
    "close_gripper": _handle_close_gripper,
    "perceive_actor": _handle_perceive_actor,
    "move_to_actor": _handle_move_to_actor,
    "grasp_actor": _handle_grasp_actor,
    "place_actor": _handle_place_actor,
    "move": _handle_move,
    "move_by_displacement": _handle_move_by_displacement,
    "wait": _handle_wait,
}


def _execute_move_to_actor(wrapper: Any, action: Dict[str, Any], logger: Any) -> None:
    task = wrapper.wrapped_task
    actor = action["actor"]
    arm_tag = action["arm_tag"]
    pre_grasp_dis = action.get("pre_grasp_dis", 0.1)
    stable_ok = wrapper._wait_stable_before_grasp_if_enabled(actor, stage="move_to_actor(recovery)", in_recovery=True)
    if (
        getattr(task, "wait_stable_before_grasp_enable", False)
        and getattr(task, "wait_stable_require_success_in_recovery", False)
        and not stable_ok
    ):
        raise ValueError("Stability check timed out before move_to_actor, and recovery is configured to require stability before execution")

    plan_attempts = int(getattr(task, "plan_retry_max_attempts", 4)) if getattr(task, "plan_retry_enable", True) else 1
    pre_grasp_pose = None
    for attempt in range(max(1, plan_attempts)):
        pre_dis_try = pre_grasp_dis + 0.02 * attempt
        try:
            pre_grasp_pose, _ = task.choose_grasp_pose(
                actor,
                arm_tag=arm_tag,
                pre_dis=pre_dis_try,
                target_dis=0.0,
            )
        except Exception as exc:
            logger.warning(f"[Recovery] move_to_actor planning attempt {attempt + 1} error: {exc}")
            pre_grasp_pose = None
        if pre_grasp_pose is not None:
            break

    if pre_grasp_pose is None:
        raise ValueError(
            f"move_to_actor path planning failed: could not compute the pre-grasp pose (retried {max(1, plan_attempts)} times; the object pose may be unreachable or in collision)"
        )

    task.move((arm_tag, [make_action(arm_tag, "move", target_pose=pre_grasp_pose)]))
    wrapper._safe_scene_step(task, 30)


def _execute_grasp_actor(wrapper: Any, action: Dict[str, Any], logger: Any) -> None:
    task = wrapper.wrapped_task
    actor = action["actor"]
    arm_tag = action["arm_tag"]
    pre_grasp_dis = action.get("pre_grasp_dis", 0.1)
    grasp_dis = action.get("grasp_dis", 0.0)
    gripper_pos = action.get("gripper_pos", None)
    force_adjustment = action.get("force_adjustment", None)
    stable_ok = wrapper._wait_stable_before_grasp_if_enabled(actor, stage="grasp_actor(recovery)", in_recovery=True)
    if (
        getattr(task, "wait_stable_before_grasp_enable", False)
        and getattr(task, "wait_stable_require_success_in_recovery", False)
        and not stable_ok
    ):
        raise ValueError("Stability check timed out before grasp_actor, and recovery is configured to require stability before execution")

    grasp_kwargs = {
        "pre_grasp_dis": pre_grasp_dis,
        "grasp_dis": grasp_dis,
    }
    if gripper_pos is not None:
        grasp_kwargs["gripper_pos"] = gripper_pos

    grasp_action = task.grasp_actor(actor, arm_tag=arm_tag, **grasp_kwargs)
    if grasp_action is None or (isinstance(grasp_action, tuple) and grasp_action[0] is None):
        raise ValueError("grasp_actor returned an invalid action; planning may have failed")

    grasp_action = _ensure_grasp_close_action(grasp_action, grasp_kwargs, logger)
    if force_adjustment == "increase":
        grasp_action = _increase_grasp_force(arm_tag, grasp_action, logger)

    task.move(grasp_action)
    wrapper._safe_scene_step(task, 150)


def _ensure_grasp_close_action(grasp_action: Any, grasp_kwargs: Dict[str, Any], logger: Any) -> Any:
    if not (isinstance(grasp_action, tuple) and len(grasp_action) == 2):
        return grasp_action

    arm_tag_result, actions = grasp_action
    has_close = any(
        (
            getattr(act, "action", None) == "gripper"
            and float(getattr(act, "target_gripper_pos", 1.0)) <= 0.3
        )
        for act in (actions or [])
        if act is not None
    )
    if has_close:
        return grasp_action

    target_pos = grasp_kwargs.get("gripper_pos", 0.0)
    actions = list(actions) if isinstance(actions, list) else []
    actions.append(make_action(arm_tag_result, "close", target_gripper_pos=target_pos))
    logger.warning(f"[Recovery] grasp_actor actions were missing close; added close(target_gripper_pos={target_pos:.2f})")
    return (arm_tag_result, actions)


def _increase_grasp_force(arm_tag: Any, grasp_action: Any, logger: Any) -> Any:
    if not (isinstance(grasp_action, tuple) and len(grasp_action) == 2):
        return grasp_action

    arm_tag_result, actions = grasp_action
    modified_actions = []
    for act in actions:
        if (
            getattr(act, "action", None) == "gripper"
            and float(getattr(act, "target_gripper_pos", 1.0)) <= 0.5
        ):
            original_pos = getattr(act, "target_gripper_pos", 0.0)
            new_pos = max(0.0, original_pos - 0.15)
            modified_actions.append(make_action(arm_tag, "close", target_gripper_pos=new_pos))
        else:
            modified_actions.append(act)
    return (arm_tag_result, modified_actions)
