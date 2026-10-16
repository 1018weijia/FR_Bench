from typing import Any, Dict, List

from ..base import ErrorInjector
from ..common import build_regrasp_recovery_actions, _np_rand, logger


class GraspSlipInjector(ErrorInjector):
    """Grasp slip - the object slips out of the gripper"""

    def __init__(self, probability: float = 0.5, force_reduction: float = 0.8):
        super().__init__("grasp_slip", probability)
        self.force_reduction = force_reduction
        self.has_injected_this_episode: bool = False
        self.target_arm = None
        self.target_actor = None
        self.pending_check: bool = False
        self.injection_attempts: int = 0

    def reset_for_new_episode(self):
        self.is_active = False
        self.metadata = {}
        self.has_injected_this_episode = False
        self.target_arm = None
        self.target_actor = None
        self.pending_check = False
        self.injection_attempts = 0

    def should_inject(self, context: Dict[str, Any]) -> bool:
        if self.has_injected_this_episode:
            return False

        action_type = context.get('action_type')
        if action_type != 'move_by_displacement':
            return False

        z = float(context.get('z', 0.0))
        if z <= 0.0:
            return False

        actor = context.get('actor')
        arm_tag = context.get('arm_tag')
        if actor is None or arm_tag is None:
            return False

        self.injection_attempts += 1
        rand_val = _np_rand()

        task = context.get('task')
        other_errors_injected = False
        if task:
            wrapper = getattr(task, '_error_recovery_wrapper', None)
            if wrapper is not None and getattr(wrapper, 'error_injection_count', 0) > 0:
                other_errors_injected = True

        effective_probability = self.probability
        if other_errors_injected:
            effective_probability = max(0.9, self.probability)
        elif self.injection_attempts > 1:
            effective_probability = min(1.0, self.probability + 0.3 * (self.injection_attempts - 1))

        if rand_val < effective_probability:
            self.has_injected_this_episode = True
            self.target_arm = arm_tag
            self.target_actor = actor
            logger.debug(
                "[Injector:grasp_slip] Injecting slip, attempt=%d prob=%.3f arm=%s",
                self.injection_attempts,
                effective_probability,
                arm_tag,
            )
            return True

        return False

    def inject_error(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.is_active = True
        self.pending_check = True

        if self.target_actor is not None:
            context['actor'] = self.target_actor
        if self.target_arm is not None:
            context['arm_tag'] = self.target_arm

        context['error_injection'] = {
            'type': 'grasp_slip',
            'force_reduction': self.force_reduction,
            'injector': self
        }

        self.metadata.update({
            'force_reduction_percent': self.force_reduction * 100,
            'target_arm': str(self.target_arm),
            'target_actor_name': self.target_actor.get_name() if self.target_actor else None,
        })

        logger.debug("[Injector:grasp_slip] force_reduction=%.0f%%", self.force_reduction * 100)
        return context

    def detect_failure(self, context: Dict[str, Any]) -> bool:
        if not self.is_active:
            return False

        actor = self.target_actor or context.get('actor')
        arm_tag = self.target_arm or context.get('arm_tag')
        task = context.get('task')

        if actor and arm_tag and task and hasattr(task, 'check_grasp_success'):
            self._safe_scene_step(task, 2)
            success1 = task.check_grasp_success(actor, arm_tag, require_lift=True)
            self._safe_scene_step(task, 2)
            success2 = task.check_grasp_success(actor, arm_tag, require_lift=True)
            if success1 or success2:
                return False

            self._safe_scene_step(task, 2)
            half_success = task.check_grasp_success(actor, arm_tag, require_lift=False)
            failed = not half_success
            if failed:
                logger.info("[Injector:grasp_slip] Lift-phase slip failure detected")
            return failed

        logger.warning("[Injector:grasp_slip] Cannot check grasp result; treating as not failed")
        return False

    def plan_recovery(self, context: Dict[str, Any], recovery_attempt: int = 1) -> List[Dict[str, Any]]:
        actor = self.target_actor or context.get('actor')
        arm_tag = self.target_arm or context.get('arm_tag')
        pre_grasp_dis = 0.0 if recovery_attempt == 1 else 0.05

        return build_regrasp_recovery_actions(
            actor,
            arm_tag,
            pre_grasp_dis=pre_grasp_dis,
            open_phase='recovery_open',
            perceive_phase='recovery_perceive',
            move_phase='recovery_reposition',
            grasp_phase='recovery_grasp_with_more_force',
            grasp_kwargs={
                'pre_grasp_dis': pre_grasp_dis,
                'grasp_dis': 0.0,
                'force_adjustment': 'increase',
            },
            recovery_attempt=recovery_attempt,
            lift_z=0.10,
            lift_phase='recovery_lift_verify',
        )
