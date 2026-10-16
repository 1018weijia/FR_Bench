"""Compatibility helpers for creating Action objects lazily."""

from __future__ import annotations

from typing import Any


class _FallbackAction:
    def __init__(
        self,
        arm_tag: Any,
        action: Any,
        target_pose: Any = None,
        target_gripper_pos: Any = None,
        **kwargs,
    ):
        self.arm_tag = arm_tag
        self.action = action
        self.target_pose = target_pose
        self.target_gripper_pos = target_gripper_pos
        self.args = kwargs


def make_action(
    arm_tag: Any,
    action: Any,
    target_pose: Any = None,
    target_gripper_pos: Any = None,
    **kwargs,
):
    """Delay envs.utils import until an Action is actually needed."""
    try:
        from envs.utils import Action as EnvAction
    except Exception:
        EnvAction = _FallbackAction
    return EnvAction(
        arm_tag,
        action,
        target_pose=target_pose,
        target_gripper_pos=target_gripper_pos,
        **kwargs,
    )


def safe_scene_step(task: Any, num_steps: int = 1, logger: Any = None, prefix: str = "ErrorRecovery") -> None:
    """Safely advance simulation and keep render state synchronized."""
    if task is None or not hasattr(task, "scene"):
        return
    scene = getattr(task, "scene", None)
    step = getattr(scene, "step", None) if scene is not None else None
    if step is None:
        return
    update_render = getattr(task, "_update_render", None) or getattr(scene, "update_render", None)
    for _ in range(max(1, num_steps)):
        try:
            step()
            if update_render:
                update_render()
        except Exception as exc:
            if logger is not None:
                logger.warning(f"[{prefix}] safe_scene_step failed: {exc}")
            break
