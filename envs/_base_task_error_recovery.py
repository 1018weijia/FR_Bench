"""Task-side helpers required by the universal error recovery collector.

Most benchmark tasks inherit this class. The actual injection/detection/recovery
flow lives in ``error_recovery/core`` and is driven by ``TaskWrapper`` plus
registered injectors. This class intentionally stays small: it only provides
actor-size heuristics, adaptive grasp thresholds, and a grasp-success predicate
that injectors and recovery judges can call.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ._base_task import Base_Task

logger = logging.getLogger(__name__)


class Base_Task_With_Error_Recovery(Base_Task):
    """Base task extension for recovery-aware tasks.

    Do not put task-side injection flows here. New error behavior should be
    implemented as an injector under ``error_recovery/core/injectors`` so it can
    be collected consistently by ``collect_universal.py``.
    """

    def __init__(self):
        super().__init__()
        self.actor_initial_poses = {}
        self.enable_error_recovery = False
        self.error_recovery_rate = 0.3

    def _init_task_env_(self, enable_error_recovery: bool = False, error_rate: float = 0.3, **kwargs):
        """Initialize the task and reset recovery helper state.

        ``enable_error_recovery`` and ``error_rate`` are accepted for backward
        compatibility with old demo tasks. Universal collection ignores them and
        decides injection through ``TaskWrapper``.
        """
        self.actor_initial_poses = {}
        self.enable_error_recovery = bool(enable_error_recovery)
        self.error_recovery_rate = float(error_rate)
        super()._init_task_env_(**kwargs)

    def get_actor_size(self, actor: Any) -> float:
        """Estimate actor maximum size in meters for adaptive thresholds."""
        try:
            if hasattr(actor, "config") and "extents" in actor.config:
                extents = np.array(actor.config["extents"], dtype=float)
                scale = float(actor.config.get("scale", 1.0))
                return float(np.max(extents) * scale * 2.0)

            entity = getattr(actor, "actor", None)
            if entity is not None:
                max_size = 0.0
                for component in entity.get_components():
                    get_shapes = getattr(component, "get_collision_shapes", None)
                    if get_shapes is None:
                        continue
                    for shape in get_shapes():
                        half_size = getattr(shape, "half_size", None)
                        if half_size is not None:
                            max_size = max(max_size, float(np.max(half_size) * 2.0))
                if max_size > 0.0:
                    return max_size

            name = ""
            if hasattr(actor, "get_name"):
                name = str(actor.get_name()).lower()
            elif hasattr(actor, "name"):
                name = str(actor.name).lower()
            if "bottle" in name:
                return 0.10
            if "block" in name or "cube" in name:
                return 0.05
            if "shoe" in name:
                return 0.15
            if "laptop" in name:
                return 0.20
        except Exception:
            logger.debug("actor size estimation failed", exc_info=True)
        return 0.08

    def get_dynamic_distance_threshold(self, actor: Any, base_threshold: float = 0.08) -> float:
        """Return an end-effector-to-actor distance threshold."""
        threshold = float(base_threshold) + self.get_actor_size(actor) * 0.5
        return max(0.05, min(threshold, 0.25))

    def get_dynamic_gripper_threshold(self, actor: Any, base_threshold: float = 0.3) -> float:
        """Return a gripper-closed threshold adjusted for object size."""
        actor_size = self.get_actor_size(actor)
        if actor_size > 0.10:
            return 0.5
        if actor_size > 0.05:
            return float(base_threshold)
        return 0.1

    def check_grasp_success(
        self,
        actor: Any,
        arm_tag: Any,
        lift_threshold: float = 0.02,
        distance_threshold: float | None = None,
        gripper_val_threshold: float | None = None,
        require_lift: bool = True,
        use_dynamic_threshold: bool = True,
    ) -> bool:
        """Check whether an actor is likely grasped by one arm.

        The check combines optional lift, end-effector distance, and gripper
        closedness. Recovery code may call it with ``require_lift=False`` before
        a lift action has been executed.
        """
        if actor is None or arm_tag is None or not hasattr(self, "robot"):
            return False

        if actor not in self.actor_initial_poses:
            try:
                self.actor_initial_poses[actor] = actor.get_pose()
            except Exception:
                return False

        if use_dynamic_threshold:
            if distance_threshold is None:
                distance_threshold = self.get_dynamic_distance_threshold(actor, base_threshold=0.08)
            if gripper_val_threshold is None:
                gripper_val_threshold = self.get_dynamic_gripper_threshold(actor, base_threshold=0.3)
        else:
            distance_threshold = 0.08 if distance_threshold is None else distance_threshold
            gripper_val_threshold = 0.3 if gripper_val_threshold is None else gripper_val_threshold

        is_lifted = True
        dz = 0.0
        if require_lift:
            try:
                current_z = float(actor.get_pose().p[2])
                initial_z = float(self.actor_initial_poses[actor].p[2])
                dz = current_z - initial_z
                is_lifted = dz > lift_threshold
            except Exception:
                is_lifted = False

        try:
            gripper_pose = self.get_arm_pose(arm_tag)
            actor_pos = actor.get_pose().p
            distance = float(np.linalg.norm(gripper_pose[:3] - actor_pos))
            is_near = distance < float(distance_threshold)
        except Exception:
            distance = float("inf")
            is_near = False

        try:
            if str(arm_tag) == "left":
                gripper_val = float(self.robot.get_left_gripper_val())
            else:
                gripper_val = float(self.robot.get_right_gripper_val())
            is_closed = gripper_val < float(gripper_val_threshold)
        except Exception:
            gripper_val = 1.0
            is_closed = False

        success = bool(is_lifted and is_near and is_closed)
        if self._should_log_error_recovery("debug"):
            lift_info = f"lifted={is_lifted} (dz={dz:.3f}m)" if require_lift else "lifted=N/A"
            print(
                f"[Base_Task_With_Error_Recovery.check_grasp_success] arm={arm_tag} "
                f"{lift_info}, near={is_near} (d={distance:.3f}m), "
                f"closed={is_closed} (g={gripper_val:.2f}) "
                f"d_thresh={float(distance_threshold):.3f}m, "
                f"g_thresh={float(gripper_val_threshold):.2f} => {'SUCCESS' if success else 'FAILED'}"
            )
        return success

    def _safe_scene_step(self, num_steps: int = 1) -> None:
        """Advance simulation and refresh render state when available."""
        if not hasattr(self, "scene") or self.scene is None:
            return
        update_render = getattr(self, "_update_render", None)
        for _ in range(max(1, int(num_steps))):
            try:
                self.scene.step()
                if update_render:
                    update_render()
                elif hasattr(self.scene, "update_render"):
                    self.scene.update_render()
            except Exception as exc:
                logger.warning("[Base_Task_With_Error_Recovery] _safe_scene_step failed: %s", exc)
                break
