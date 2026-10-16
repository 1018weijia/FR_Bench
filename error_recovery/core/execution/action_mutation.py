"""Action mutation helpers for injected error execution."""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from ..action_compat import make_action
from . import logger

_ONSET_MARKER_BY_TYPE = {
    "position_offset": "grasp_position_offset",
    "grasp_orientation_mismatch": "grasp_orientation_mismatch",
}


def execute_modified_action(wrapper: Any, original_method: Any, context: Dict[str, Any]):
    """Execute an action after an injector has modified the context."""
    method_name = context["method_name"]
    error_injection = context.get("error_injection", {})
    error_type = error_injection.get("type")

    if error_type == "premature_close" and method_name == "grasp_actor":
        return _execute_premature_close_grasp(original_method, context)
    if error_type == "grasp_slip" and method_name == "move_by_displacement":
        return _execute_grasp_slip_lift_action(original_method, context)

    args = list(context["args"])
    kwargs = dict(context["kwargs"])
    if "target_pose" in context and "target_pose" in kwargs:
        kwargs["target_pose"] = context["target_pose"]

    result = original_method(*args, **kwargs)
    if method_name == "grasp_actor":
        return _postprocess_grasp_actor_result(wrapper, result, error_injection)
    return result


def _postprocess_grasp_actor_result(wrapper: Any, result: Any, error_injection: Dict[str, Any]):
    error_type = error_injection.get("type")
    if error_type not in _ONSET_MARKER_BY_TYPE:
        return result
    if not (isinstance(result, tuple) and len(result) == 2):
        return result
    try:
        arm_tag, actions = result
        if not actions:
            return result
        return arm_tag, _modify_actions_by_injection(wrapper, actions, error_injection)
    except Exception:
        logger.exception("[mutation] Error applying %s to grasp_actor", error_type)
        return result


def _execute_premature_close_grasp(original_method: Any, context: Dict[str, Any]):
    error_injection = context["error_injection"]
    close_ratio = error_injection.get("close_ratio", 0.5)
    actor = context["actor"]
    arm_tag = context["arm_tag"]
    task = context["task"]
    pre_grasp_dis = context.get("pre_grasp_dis", 0.1)
    grasp_dis = context.get("grasp_dis", 0.0)

    try:
        pre_grasp_pose, grasp_pose = task.choose_grasp_pose(
            actor,
            arm_tag=arm_tag,
            pre_dis=pre_grasp_dis,
            target_dis=grasp_dis,
        )
        if pre_grasp_pose is None or grasp_pose is None:
            raise ValueError("choose_grasp_pose returned no grasp pose")
        if isinstance(pre_grasp_pose, list):
            pre_pos = np.array(pre_grasp_pose[:3])
            intermediate_pose = list(pre_grasp_pose)
        else:
            pre_pos = np.array(pre_grasp_pose.p)
            intermediate_pose = [*pre_grasp_pose.p, *pre_grasp_pose.q]

        grasp_pos = np.array(grasp_pose[:3]) if isinstance(grasp_pose, list) else np.array(grasp_pose.p)
        intermediate_pose[:3] = (pre_pos + (grasp_pos - pre_pos) * close_ratio).tolist()
        actions = [
            make_action(arm_tag, "move", target_pose=pre_grasp_pose),
            make_action(arm_tag, "move", target_pose=intermediate_pose),
            make_action(arm_tag, "close", target_gripper_pos=0.0, error_onset_marker="premature_close"),
        ]
        return arm_tag, actions
    except Exception:
        logger.exception("[mutation] premature_close planning failed, falling back to original grasp_actor")
        return original_method(*context["args"], **context["kwargs"])


def _modify_actions_by_injection(wrapper: Any, actions: list, error_injection: Dict[str, Any]) -> list:
    if not isinstance(actions, list) or not actions:
        return actions

    error_type = error_injection.get("type")
    if error_type not in _ONSET_MARKER_BY_TYPE:
        return actions

    onset_marker = _ONSET_MARKER_BY_TYPE[error_type]
    onset_marked = False
    new_actions = []

    for action in actions:
        if action is None:
            new_actions.append(action)
            continue

        if getattr(action, "action", None) == "move" and getattr(action, "target_pose", None) is not None:
            target_pose = list(action.target_pose)
            if error_type == "position_offset":
                offset = error_injection.get("offset")
                if isinstance(offset, (list, tuple)) and len(offset) >= 3:
                    for index in range(3):
                        target_pose[index] = float(target_pose[index]) + float(offset[index])
            else:
                target_pose = _apply_z_rotation(target_pose, error_injection.get("rotation_deg", 0.0))
            action.target_pose = target_pose
        elif getattr(action, "action", None) == "gripper" and not onset_marked:
            wrapper._set_action_onset_marker(action, onset_marker)
            onset_marked = True

        new_actions.append(action)

    if not onset_marked:
        logger.warning(
            "[mutation] %s found no grasp-close action; error_onset cannot be labeled as expected",
            error_type,
        )
    return new_actions


def _apply_z_rotation(pose: list, degree: float) -> list:
    if pose is None or len(pose) < 7:
        return pose
    rad = math.radians(float(degree))
    half = rad / 2.0
    quat_z = [math.cos(half), 0.0, 0.0, math.sin(half)]
    w1, x1, y1, z1 = quat_z
    w2, x2, y2, z2 = float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])
    new_pose = list(pose)
    new_pose[3:7] = [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]
    return new_pose


def _execute_grasp_slip_lift_action(original_method: Any, context: Dict[str, Any]):
    error_injection = context.get("error_injection", {})
    force_reduction = float(error_injection.get("force_reduction", 0.8))
    task = context.get("task")
    arm_tag = context.get("arm_tag")
    if task is None or arm_tag is None:
        return original_method(*context["args"], **context["kwargs"])

    z_total = float(context.get("z", 0.0))
    if z_total <= 0.0:
        return original_method(*context["args"], **context["kwargs"])

    try:
        origin_pose = np.array(task.get_arm_pose(arm_tag), dtype=np.float64)
    except Exception:
        return original_method(*context["args"], **context["kwargs"])

    final_gripper_pos = max(0.0, min(1.0, 1.0 - force_reduction))
    z_first = min(0.01, z_total * 0.1)
    z_second = z_total - z_first
    pose_first = origin_pose.copy()
    pose_first[2] += z_first
    pose_final = origin_pose.copy()
    pose_final[2] += z_total

    actions = [
        make_action(arm_tag, "move", target_pose=pose_first.tolist()),
        make_action(arm_tag, "open", target_gripper_pos=final_gripper_pos, error_onset_marker="grasp_slip"),
    ]
    if z_second > 0.001:
        actions.append(make_action(arm_tag, "move", target_pose=pose_final.tolist()))
    return arm_tag, actions
