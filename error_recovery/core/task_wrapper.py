"""
Task wrapper - automatically intercept and modify task execution to inject errors

Provides a generic task wrapper that injects errors at the right time and collects recovery data.
"""

from typing import Dict, List, Any, Type, Callable
import functools
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the path so the envs module can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .action_compat import safe_scene_step
from .error_injectors import ErrorInjector
from .runtime import (
    build_context,
    initialize_runtime_state,
    reset_episode_runtime_state,
)
from .execution import (
    check_error_injection,
    execute_normal,
    execute_with_error,
)
from .lifecycle import (
    mark_error_onset,
)
from .serialization import (
    get_error_recovery_data,
    sanitize_error_metadata_item,
    sanitize_error_recovery_data,
)


class TaskWrapper:
    """
    Task wrapper class
    
    Wraps an existing task class, automatically injects errors, and collects recovery data
    """
    
    _DEFERRED_GRASP_DETECTION_CONFIGS = {
        "grasp_position_offset": {
            "state_key": "grasp_position_offset",
            "wait_steps": 130,
        },
        "grasp_orientation_mismatch": {
            "state_key": "grasp_orientation_mismatch",
            "wait_steps": 130,
        },
        "premature_close": {
            "state_key": "premature_close",
            "wait_steps": 80,
        },
    }
    _WRAPPED_METHODS = (
        "move",
        "move_by_displacement",
        "grasp_actor",
        "place_actor",
        "open_gripper",
        "close_gripper",
        "_take_picture",
        "play_once",
    )

    def __init__(self, task_class: Type, error_injectors: List[ErrorInjector], max_errors_per_episode: int = 1):
        """
        Args:
            task_class: Task class to wrap
            error_injectors: List of error injectors
            max_errors_per_episode: Injection cap for a single episode; clamped to 1 at execution time
        """
        self.task_class = task_class
        self.error_injectors = error_injectors
        self.wrapped_task = None
        self.max_errors_per_episode = max_errors_per_episode
        initialize_runtime_state(self)
        
    def create_wrapped_task(self, **kwargs) -> Any:
        """Create a wrapped task instance"""
        # Create the original task instance
        original_task = self.task_class()
        
        # Wrap the key methods
        self._wrap_task_methods(original_task)
        
        # Set the wrapper reference
        original_task._error_recovery_wrapper = self
        
        self.wrapped_task = original_task
        return original_task

    def detach(self, task: Any) -> None:
        """Drop wrapper closures so the episode's scene can be freed."""
        if task is None:
            return
        for method_name in self._WRAPPED_METHODS:
            if method_name in getattr(task, "__dict__", {}):
                try:
                    delattr(task, method_name)
                except Exception:
                    pass
        for attribute in ("get_obs", "_error_recovery_wrapper"):
            if attribute in getattr(task, "__dict__", {}):
                try:
                    delattr(task, attribute)
                except Exception:
                    pass
        self.wrapped_task = None

    def _wrap_task_methods(self, task):
        """Wrap the task's key methods"""
        for method_name in self._WRAPPED_METHODS:
            self._wrap_method(task, method_name)
    
    def _wrap_method(self, task, method_name: str):
        """Wrap a single method"""
        if not hasattr(task, method_name):
            return
        
        original_method = getattr(task, method_name)
        
        @functools.wraps(original_method)
        def wrapped_method(*args, **kwargs):
            return self._execute_with_error_injection(
                original_method, method_name, *args, **kwargs
            )
        
        setattr(task, method_name, wrapped_method)

    def _safe_scene_step(self, task: Any, num_steps: int = 1):
        """Step the simulation safely to avoid OIDN errors from a long stretch without update_render"""
        safe_scene_step(task, num_steps, logger, "TaskWrapper")

    def _wait_stable_before_grasp_if_enabled(self, actor: Any, stage: str, in_recovery: bool = False) -> bool:
        """Optional: wait for the target object to settle before grasping. Returns whether it is stable."""
        task = self.wrapped_task
        if task is None or actor is None:
            if actor is None:
                logger.warning(f"[TaskWrapper] Stability check before {stage} skipped: actor=None")
            return False
        if not getattr(task, "wait_stable_before_grasp_enable", False):
            return True
        if getattr(task, "wait_stable_only_in_recovery", True) and not in_recovery:
            return True
        if not hasattr(task, "wait_actor_stable"):
            logger.warning("[TaskWrapper] wait_stable_before_grasp is enabled, but the task does not implement wait_actor_stable")
            return False
        try:
            stable_ok, stable_stats = task.wait_actor_stable(
                actor,
                log_prefix=f"[TaskWrapper][StabilityWait:{stage}]",
            )
            logger.log(
                logging.DEBUG if stable_ok else logging.WARNING,
                "[TaskWrapper] Stability check before %s %s: %s",
                stage,
                "passed" if stable_ok else "timed out, continuing",
                stable_stats,
            )
            return bool(stable_ok)
        except Exception as exc:
            logger.warning("[TaskWrapper] Stability check before %s raised an error, continuing: %s", stage, exc)
            return False

    def _set_action_onset_marker(self, action: Any, marker: str):
        if action is None:
            return
        args = getattr(action, 'args', None)
        if not isinstance(args, dict):
            try:
                action.args = {}
                args = action.args
            except Exception as exc:
                logger.debug("Failed to set the action onset marker", exc_info=exc)
                return
        args['error_onset_marker'] = marker

    def on_before_move_action(self, left_action: Any, right_action: Any):
        """Called back by Base_Task.move: fires before each sub-action actually runs, for fine-grained onset marking."""
        for action in (left_action, right_action):
            if action is None:
                continue
            args = getattr(action, 'args', None)
            if not isinstance(args, dict):
                continue
            marker = args.pop('error_onset_marker', None)
            if marker:
                mark_error_onset(self, str(marker), logger)

    def _execute_with_error_injection(self, original_method: Callable, method_name: str, *args, **kwargs):
        """Run the method and possibly inject an error"""
        context = build_context(self, method_name, args, kwargs)
        active_injector = check_error_injection(self, context)

        if active_injector:
            return execute_with_error(self, original_method, context, active_injector)
        return execute_normal(self, original_method, context)

    def get_error_recovery_data(self) -> Dict[str, Any]:
        """Get error-recovery data"""
        return get_error_recovery_data(self)
    
    def reset_for_new_episode(self):
        """Reset state for a new episode"""
        reset_episode_runtime_state(self)
        
        # Reset all injectors
        for injector in self.error_injectors:
            injector.is_active = False
            injector.metadata = {}
            # Allow an injector to implement its own per-episode reset logic
            if hasattr(injector, "reset_for_new_episode"):
                try:
                    injector.reset_for_new_episode()
                except Exception as exc:
                    logger.debug("Failed to reset injector %s", getattr(injector, 'name', type(injector).__name__), exc_info=exc)
        # Clear the current recovery error-type marker
        self.current_recovery_injector_name = None
    
    def update_frame_count(self):
        """Update the frame count"""
        self.frame_count += 1
    
    def sanitize_error_recovery_data(self, error_recovery_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean error_recovery data and drop objects that cannot be serialized
        """
        return sanitize_error_recovery_data(self, error_recovery_data)

    def sanitize_error_metadata_item(self, item: Any) -> Any:
        """Sanitize a single metadata item without changing the output schema."""
        return sanitize_error_metadata_item(item)
