"""TaskWrapper runtime context construction helpers."""

from __future__ import annotations

import time
from typing import Any, Dict


def _get_positional_or_kwarg(args: tuple, kwargs: dict, index: int, key: str) -> Any:
    if len(args) > index:
        return args[index]
    return kwargs.get(key)


def build_context(wrapper: Any, method_name: str, args: tuple, kwargs: dict) -> Dict[str, Any]:
    """Build the execution context while preserving the existing flow."""
    context = {
        "method_name": method_name,
        "args": args,
        "kwargs": kwargs,
        "task": wrapper.wrapped_task,
        "frame_count": wrapper.frame_count,
        "current_phase": wrapper.current_phase,
        "timestamp": time.time(),
    }

    if method_name == "grasp_actor":
        context.update(
            {
                "action_type": "grasp_actor",
                "actor": _get_positional_or_kwarg(args, kwargs, 0, "actor"),
                "arm_tag": _get_positional_or_kwarg(args, kwargs, 1, "arm_tag"),
                "pre_grasp_dis": kwargs.get("pre_grasp_dis", 0.1),
                "grasp_dis": kwargs.get("grasp_dis", 0.0),
            }
        )
    elif method_name == "place_actor":
        context.update(
            {
                "action_type": "place_actor",
                "actor": _get_positional_or_kwarg(args, kwargs, 0, "actor"),
                "arm_tag": _get_positional_or_kwarg(args, kwargs, 1, "arm_tag"),
                "target_pose": kwargs.get("target_pose"),
            }
        )
    elif method_name == "move":
        context.update(
            {
                "action_type": "move",
                "actions": args[0] if args else None,
            }
        )
    elif method_name == "move_by_displacement":
        arm_tag = _get_positional_or_kwarg(args, kwargs, 0, "arm_tag")
        context.update(
            {
                "action_type": "move_by_displacement",
                "arm_tag": arm_tag,
                "x": kwargs.get("x", 0.0),
                "y": kwargs.get("y", 0.0),
                "z": kwargs.get("z", 0.0),
                "actor": wrapper.last_grasp_actor_by_arm.get(arm_tag),
            }
        )
    elif method_name in ["open_gripper", "close_gripper"]:
        context.update(
            {
                "action_type": method_name,
                "arm_tag": _get_positional_or_kwarg(args, kwargs, 0, "arm_tag"),
            }
        )

    return context
