import logging
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    R = None


logger = logging.getLogger(__name__)

# Micro-optimizations for hot paths (does not change behavior/results).
_np_rand = np.random.rand
_np_uniform = np.random.uniform


def build_regrasp_recovery_actions(
    actor: Any,
    arm_tag: Any,
    *,
    pre_grasp_dis: float,
    open_phase: str,
    perceive_phase: str,
    move_phase: str,
    grasp_phase: str,
    grasp_kwargs: Optional[Dict[str, Any]] = None,
    recovery_attempt: Optional[int] = None,
    lift_z: Optional[float] = None,
    lift_phase: Optional[str] = None,
) -> List[Dict[str, Any]]:
    actions = [
        {'action_type': 'open_gripper', 'arm_tag': arm_tag, 'phase': open_phase},
        {'action_type': 'perceive_actor', 'actor': actor, 'arm_tag': arm_tag,
         'phase': perceive_phase},
        {'action_type': 'move_to_actor', 'actor': actor, 'arm_tag': arm_tag,
         'pre_grasp_dis': pre_grasp_dis, 'phase': move_phase},
    ]

    grasp_action = {'action_type': 'grasp_actor', 'actor': actor, 'arm_tag': arm_tag}
    if grasp_kwargs:
        grasp_action.update(grasp_kwargs)
    grasp_action['phase'] = grasp_phase
    if recovery_attempt is not None:
        grasp_action['recovery_attempt'] = recovery_attempt
    actions.append(grasp_action)

    if lift_z is not None:
        actions.append({'action_type': 'move_by_displacement', 'arm_tag': arm_tag,
                        'z': lift_z, 'phase': lift_phase})
    return actions


__all__ = ["R", "_np_rand", "_np_uniform", "build_regrasp_recovery_actions", "logger"]
