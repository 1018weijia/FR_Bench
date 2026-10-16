from typing import Any, Dict, List, Tuple

import numpy as np

from ..base import ErrorInjector
from ..common import build_regrasp_recovery_actions, _np_rand, _np_uniform, logger


class GraspPositionOffsetInjector(ErrorInjector):
    """Grasp position offset - gripper pose is wrong"""

    def __init__(self, probability: float = 0.5, offset_range: Tuple[float, float] = (0.025, 0.050)):
        super().__init__("grasp_position_offset", probability)
        self.offset_range = offset_range
        self.injected_offset = None
        self.has_injected_this_episode = False

    def should_inject(self, context: Dict[str, Any]) -> bool:
        action_type = context.get('action_type')
        if action_type != 'grasp_actor':
            return False
        if self.has_injected_this_episode:
            return False
        return _np_rand() < self.probability

    def reset_for_new_episode(self):
        self.has_injected_this_episode = False
        self.is_active = False
        self.injected_offset = None

    def inject_error(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.is_active = True
        self.has_injected_this_episode = True

        angle = _np_uniform(0, 2 * np.pi)
        magnitude = _np_uniform(self.offset_range[0], self.offset_range[1])
        offset = np.array([
            magnitude * np.cos(angle),
            magnitude * np.sin(angle),
            0.0
        ])
        self.injected_offset = offset

        context['error_injection'] = {
            'type': 'position_offset',
            'offset': offset.tolist(),
            'injector': self
        }

        offset_magnitude_mm = np.linalg.norm(offset) * 1000
        self.metadata.update({
            'offset_magnitude_mm': offset_magnitude_mm,
            'offset_x_mm': offset[0] * 1000,
            'offset_y_mm': offset[1] * 1000,
            'offset_z_mm': offset[2] * 1000,
            'offset_range': self.offset_range
        })

        logger.debug(
            "[Injector:grasp_position_offset] Injecting position offset, magnitude=%.1fmm range=(%.1f, %.1f)mm",
            offset_magnitude_mm,
            self.offset_range[0] * 1000,
            self.offset_range[1] * 1000,
        )
        return context

    def detect_failure(self, context: Dict[str, Any]) -> bool:
        if not self.is_active:
            return False

        actor = context.get('actor')
        arm_tag = context.get('arm_tag')
        task = context.get('task')

        if actor and arm_tag and task and hasattr(task, 'check_grasp_success'):
            dynamic_distance_threshold = task.get_dynamic_distance_threshold(actor, base_threshold=0.12)
            dynamic_gripper_threshold = task.get_dynamic_gripper_threshold(actor)

            self._safe_scene_step(task, 40)
            success1 = task.check_grasp_success(
                actor, arm_tag,
                require_lift=True,
                distance_threshold=dynamic_distance_threshold,
                gripper_val_threshold=dynamic_gripper_threshold,
                use_dynamic_threshold=False
            )
            if success1:
                return False

            self._safe_scene_step(task, 30)
            success2 = task.check_grasp_success(
                actor, arm_tag,
                require_lift=True,
                distance_threshold=dynamic_distance_threshold,
                gripper_val_threshold=dynamic_gripper_threshold,
                use_dynamic_threshold=False
            )
            if success2:
                return False

            self._safe_scene_step(task, 20)
            backup_distance_threshold = task.get_dynamic_distance_threshold(actor, base_threshold=0.15)
            backup_distance_threshold = min(backup_distance_threshold, 0.25)
            success3 = task.check_grasp_success(
                actor, arm_tag,
                require_lift=True,
                distance_threshold=backup_distance_threshold,
                gripper_val_threshold=dynamic_gripper_threshold,
                use_dynamic_threshold=False
            )
            if success3:
                return False

            self._safe_scene_step(task, 20)
            final_check = task.check_grasp_success(
                actor, arm_tag,
                require_lift=False,
                distance_threshold=backup_distance_threshold,
                gripper_val_threshold=dynamic_gripper_threshold,
                use_dynamic_threshold=False
            )
            if final_check:
                micro_lift_success = task.check_grasp_success(
                    actor, arm_tag,
                    require_lift=True,
                    lift_threshold=0.01,
                    distance_threshold=backup_distance_threshold,
                    gripper_val_threshold=dynamic_gripper_threshold,
                    use_dynamic_threshold=False
                )
                if micro_lift_success:
                    return False
                try:
                    gripper_val = task.robot.get_left_gripper_val() if str(arm_tag) == "left" else task.robot.get_right_gripper_val()
                    if gripper_val < 0.05 and backup_distance_threshold is not None:
                        self._safe_scene_step(task, 30)
                        micro_lift_success2 = task.check_grasp_success(
                            actor, arm_tag,
                            require_lift=True,
                            lift_threshold=0.005,
                            distance_threshold=backup_distance_threshold,
                            gripper_val_threshold=dynamic_gripper_threshold,
                            use_dynamic_threshold=False
                        )
                        if micro_lift_success2:
                            return False
                except Exception:
                    pass
                logger.info("[Injector:grasp_position_offset] Grasp position offset caused failure: lift check failed")
                return True

            logger.info("[Injector:grasp_position_offset] Grasp position offset caused failure")
            return True

        logger.warning("[Injector:grasp_position_offset] Grasp detection interface missing; treating as success")
        return False

    def plan_recovery(self, context: Dict[str, Any], recovery_attempt: int = 1) -> List[Dict[str, Any]]:
        actor = context.get('actor')
        arm_tag = context.get('arm_tag')

        return build_regrasp_recovery_actions(
            actor,
            arm_tag,
            pre_grasp_dis=0.08,
            open_phase='recovery_open',
            perceive_phase='recovery_perceive',
            move_phase='recovery_reposition',
            grasp_phase='recovery_grasp',
            grasp_kwargs={'gripper_pos': 0.0},
            lift_z=0.12,
            lift_phase='recovery_relift',
        )
