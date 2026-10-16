"""TaskWrapper runtime state helpers."""

from __future__ import annotations

from typing import Any

from .deferred_detection import reset_deferred_error_tracking


def initialize_runtime_state(wrapper: Any) -> None:
    wrapper.error_injection_count = 0
    wrapper.current_phase = "normal"
    wrapper.error_metadata = []
    wrapper.action_history = []
    wrapper.recovery_actions = []
    wrapper.frame_count = 0
    wrapper.last_grasp_actor_by_arm = {}
    wrapper.current_recovery_injector_name = None
    wrapper.last_recovery_error = None
    wrapper.error_attempt_counter = 0
    wrapper.current_error_attempt_id = None
    reset_deferred_error_tracking(wrapper)


def reset_episode_runtime_state(wrapper: Any) -> None:
    """Compatibility wrapper for per-episode state reset."""
    initialize_runtime_state(wrapper)
