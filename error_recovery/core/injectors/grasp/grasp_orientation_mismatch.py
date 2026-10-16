from typing import Any, Dict, List, Tuple

import numpy as np

from ..base import ErrorInjector
from ..common import R, _np_rand, _np_uniform, logger


class GraspOrientationMismatchInjector(ErrorInjector):
    """Grasp orientation mismatch - grasp the object at the wrong angle"""

    def __init__(self, probability: float = 0.5, rotation_range: Tuple[float, float] = (45, 90)):
        super().__init__("grasp_orientation_mismatch", probability)
        self.rotation_range = rotation_range
        self.injected_rotation = None
        self._non_symmetric_keywords = ('shoe', 'laptop', 'hammer', 'drill', 'brush', 'knife')

    def should_inject(self, context: Dict[str, Any]) -> bool:
        action_type = context.get('action_type')
        if action_type != 'grasp_actor':
            return False

        actor = context.get('actor')
        if not actor:
            return False

        actor_name = actor.get_name().lower()
        if not any(obj in actor_name for obj in self._non_symmetric_keywords):
            return False

        return _np_rand() < self.probability

    def inject_error(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.is_active = True

        rotation_deg = _np_uniform(self.rotation_range[0], self.rotation_range[1])
        self.injected_rotation = rotation_deg

        if R and 'target_pose' in context:
            original_pose = np.array(context['target_pose'])
            if len(original_pose) >= 7:
                rotation_rad = np.radians(rotation_deg)
                additional_rotation = R.from_euler('z', rotation_rad)

                original_quat = original_pose[3:7]
                original_rotation = R.from_quat([original_quat[1], original_quat[2], original_quat[3], original_quat[0]])

                new_rotation = additional_rotation * original_rotation
                new_quat = new_rotation.as_quat()

                modified_pose = original_pose.copy()
                modified_pose[3:7] = [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]
                context['target_pose'] = modified_pose.tolist()
        context['error_injection'] = {
            'type': 'grasp_orientation_mismatch',
            'rotation_deg': rotation_deg,
            'injector': self
        }

        self.metadata.update({
            'rotation_deg': rotation_deg
        })

        logger.debug("[Injector:grasp_orientation_mismatch] Injecting rotation error, rotation=%.0fdeg", rotation_deg)
        return context

    def detect_failure(self, context: Dict[str, Any]) -> bool:
        if not self.is_active or self.injected_rotation is None:
            return False

        actor = context.get('actor')
        task = context.get('task')
        arm_tag = context.get('arm_tag')

        if actor and task and arm_tag and hasattr(task, 'check_grasp_success'):
            self._safe_scene_step(task, 40)
            success1 = task.check_grasp_success(actor, arm_tag, require_lift=True)
            if success1:
                return False

            self._safe_scene_step(task, 30)
            success2 = task.check_grasp_success(actor, arm_tag, require_lift=True)
            if success2:
                return False

            self._safe_scene_step(task, 20)
            success3 = task.check_grasp_success(actor, arm_tag, require_lift=False)
            if success3:
                logger.info("[Injector:grasp_orientation_mismatch] Orientation error caused grasp failure: object not stably grasped")
                return True

            logger.info("[Injector:grasp_orientation_mismatch] Orientation error caused grasp failure")
            return True

        logger.warning("[Injector:grasp_orientation_mismatch] Detection interface missing; treating as not failed")
        return False

    def plan_recovery(self, context: Dict[str, Any], recovery_attempt: int = 1) -> List[Dict[str, Any]]:
        actor = context.get('actor')
        arm_tag = context.get('arm_tag')

        return [
            {'action_type': 'open_gripper', 'arm_tag': arm_tag,
             'phase': 'recovery_open'},
            {'action_type': 'perceive_actor', 'actor': actor, 'arm_tag': arm_tag,
             'phase': 'recovery_perceive'},
            {'action_type': 'grasp_actor', 'actor': actor, 'arm_tag': arm_tag,
             'phase': 'recovery_grasp_correct_orientation'}
        ]
