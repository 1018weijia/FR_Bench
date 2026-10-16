import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np

from ..action_compat import safe_scene_step


logger = logging.getLogger(__name__)


class ErrorInjector(ABC):
    """Error injector base class"""

    def __init__(self, name: str, probability: float = 0.5):
        self.name = name
        self.probability = probability
        self.is_active = False
        self.metadata = {}

    @abstractmethod
    def should_inject(self, context: Dict[str, Any]) -> bool:
        """Decide whether an error should be injected"""
        pass

    @abstractmethod
    def inject_error(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Inject an error and return the modified context"""
        pass

    @abstractmethod
    def detect_failure(self, context: Dict[str, Any]) -> bool:
        """Detect whether the error caused a failure"""
        pass

    @abstractmethod
    def plan_recovery(self, context: Dict[str, Any], recovery_attempt: int = 1) -> List[Dict[str, Any]]:
        """Plan the recovery action sequence"""
        pass

    @staticmethod
    def _default_grasp_failure_check(context: Dict[str, Any]) -> bool:
        """Default grasp-failure detection method"""
        actor = context.get('actor')
        arm_tag = context.get('arm_tag')
        task = context.get('task')

        if not (actor and arm_tag and task):
            return True

        try:
            ee_pose = task.get_arm_pose(arm_tag)
            ee_pos = np.array(ee_pose[:3], dtype=float)
            actor_pose = actor.get_pose()
            actor_pos = np.array(actor_pose.p, dtype=float)
            distance = np.linalg.norm(ee_pos - actor_pos)
            return distance > 0.05
        except Exception:
            return True

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata for the injected error"""
        return {
            'probability': self.probability,
            **self.metadata
        }

    def _safe_scene_step(self, task: Any, num_steps: int = 1):
        """Unified safe physics step; sync the render after every step to avoid OIDN errors"""
        safe_scene_step(task, num_steps, logger, "ErrorInjector")
