#!/usr/bin/env python3
"""Custom error injector template for the current version."""

import os
import sys
from typing import Any, Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from error_recovery.core.injectors.base import ErrorInjector


class CustomErrorTemplateInjector(ErrorInjector):
    """Custom error template.

    Suggested usage:
    1. Copy this file and rename it
    2. Change `name`, the injection condition, and the recovery strategy
    3. Register the new class in `error_recovery/core/injectors/registry.py`
    4. Verify with `collect_universal.py --list_error_types` and a small sample collection
    """

    def __init__(self, probability: float = 0.5, magnitude: float = 0.02):
        super().__init__("custom_error_template", probability)
        self.magnitude = magnitude

    def should_inject(self, context: Dict[str, Any]) -> bool:
        """Decide whether to inject an error on the current action."""
        action_type = context.get("action_type")
        return action_type == "grasp_actor" and self.probability > 0.0

    def inject_error(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Write the error info and return the modified context."""
        self.is_active = True
        actor = context.get("actor")
        context["error_injection"] = {
            "type": self.name,
            "magnitude": self.magnitude,
            "injector": self,
        }
        self.metadata.update(
            {
                "magnitude": self.magnitude,
                "arm_tag": str(context.get("arm_tag")),
                "target_actor_name": actor.get_name() if actor else None,
            }
        )
        return context

    def detect_failure(self, context: Dict[str, Any]) -> bool:
        """Judge whether the injected action actually caused a failure."""
        task = context.get("task")
        actor = context.get("actor")
        arm_tag = context.get("arm_tag")

        if not self.is_active or task is None or actor is None or arm_tag is None:
            return False

        if hasattr(task, "check_grasp_success"):
            return not task.check_grasp_success(actor, arm_tag)
        return self._default_grasp_failure_check(context)

    def plan_recovery(self, context: Dict[str, Any], recovery_attempt: int = 1) -> List[Dict[str, Any]]:
        """Return a minimal recovery action sequence."""
        actor = context.get("actor")
        arm_tag = context.get("arm_tag")
        if actor is None or arm_tag is None:
            return []

        pre_grasp_dis = 0.10 if recovery_attempt == 1 else 0.12
        return [
            {
                "action_type": "open",
                "arm_tag": arm_tag,
                "phase": "recovery_open_gripper",
            },
            {
                "action_type": "perceive_actor",
                "actor": actor,
                "arm_tag": arm_tag,
                "phase": "recovery_perceive",
            },
            {
                "action_type": "move_to_actor",
                "actor": actor,
                "arm_tag": arm_tag,
                "pre_grasp_dis": pre_grasp_dis,
                "phase": "recovery_reposition",
            },
            {
                "action_type": "grasp_actor",
                "actor": actor,
                "arm_tag": arm_tag,
                "pre_grasp_dis": 0.05,
                "grasp_dis": 0.0,
                "phase": "recovery_grasp",
            },
        ]


def main():
    print("CustomErrorTemplateInjector")
    print("1. Copy this file and rename it")
    print("2. Edit should_inject / inject_error / detect_failure / plan_recovery")
    print("3. Register it in error_recovery/core/injectors/registry.py")
    print("4. Document the parameter format in DATA_FORMAT.md")
    print("5. Verify with collect_universal.py --list_error_types and a small sample collection")


if __name__ == "__main__":
    main()
