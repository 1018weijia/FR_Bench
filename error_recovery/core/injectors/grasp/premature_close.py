from typing import Any, Dict, List

from ..base import ErrorInjector
from ..common import _np_rand, logger


class PrematureCloseInjector(ErrorInjector):
    """Premature gripper close mid-motion - requires a regrasp"""

    def __init__(self, probability: float = 0.5, close_ratio: float = 0.5):
        super().__init__("premature_close", probability)
        self.close_ratio = close_ratio
        self.original_grasp_plan = None
        self.premature_close_executed = False

    def should_inject(self, context: Dict[str, Any]) -> bool:
        return context.get('action_type') == 'grasp_actor' and _np_rand() < self.probability

    def inject_error(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.is_active = True
        self.premature_close_executed = False
        self.original_grasp_plan = context.copy()

        context['error_injection'] = {
            'type': 'premature_close',
            'close_ratio': self.close_ratio,
            'injector': self
        }

        actor = context.get('actor')
        self.metadata.update({
            'close_ratio': self.close_ratio,
            'target_actor_name': actor.get_name() if actor else None,
            'arm_tag': str(context.get('arm_tag'))
        })

        logger.debug("[Injector:premature_close] Injecting premature close, close_ratio=%.2f", self.close_ratio)
        return context

    def detect_failure(self, context: Dict[str, Any]) -> bool:
        if not self.is_active:
            return False

        actor = context.get('actor')
        arm_tag = context.get('arm_tag')
        task = context.get('task')

        if actor and arm_tag and task:
            if hasattr(task, 'check_grasp_success'):
                success = task.check_grasp_success(actor, arm_tag)
                return not success

        return self._default_grasp_failure_check(context)

    def plan_recovery(self, context: Dict[str, Any], recovery_attempt: int = 1) -> List[Dict[str, Any]]:
        recovery_actions = []
        actor = context.get('actor')
        arm_tag = context.get('arm_tag')

        recovery_actions.append({
            'action_type': 'open_gripper',
            'arm_tag': arm_tag,
            'phase': 'recovery_open_gripper'
        })

        if actor:
            recovery_actions.append({
                'action_type': 'perceive_actor',
                'actor': actor,
                'arm_tag': arm_tag,
                'phase': 'recovery_perceive'
            })

            pre_grasp_dis = 0.1 if recovery_attempt == 1 else 0.12

            recovery_actions.append({
                'action_type': 'move_to_actor',
                'actor': actor,
                'arm_tag': arm_tag,
                'pre_grasp_dis': pre_grasp_dis,
                'phase': 'recovery_reposition'
            })

            recovery_actions.append({
                'action_type': 'grasp_actor',
                'actor': actor,
                'arm_tag': arm_tag,
                'pre_grasp_dis': 0.05,
                'grasp_dis': 0.0,
                'phase': 'recovery_grasp',
                'recovery_attempt': recovery_attempt
            })

        return recovery_actions
