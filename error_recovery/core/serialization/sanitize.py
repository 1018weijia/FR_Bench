"""Serialization helpers for error recovery metadata."""

from __future__ import annotations

from typing import Any, Dict

from ..lifecycle import get_precise_error_state


def get_error_recovery_data(wrapper: Any) -> Dict[str, Any]:
    precise = get_precise_error_state(wrapper)
    return {
        "current_phase": wrapper.current_phase,
        "error_stage_precise": precise["error_stage_precise"],
        "is_error_frame_precise": precise["is_error_frame_precise"],
        "error_metadata": wrapper.error_metadata,
        "action_history": wrapper.action_history,
        "recovery_actions": wrapper.recovery_actions,
        "frame_count": wrapper.frame_count,
        "active_injectors": [injector.name for injector in wrapper.error_injectors if injector.is_active],
    }


def sanitize_error_recovery_data(wrapper: Any, error_recovery_data: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = {}
    for key, value in error_recovery_data.items():
        if key == "error_metadata" and isinstance(value, list):
            sanitized_metadata_list = []
            for item in value:
                sanitized_item = sanitize_error_metadata_item(item)
                if sanitized_item is None:
                    continue
                sanitized_metadata_list.append(sanitized_item)
            sanitized[key] = sanitized_metadata_list
        elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
            sanitized[key] = value
        else:
            try:
                sanitized[key] = str(value)
            except Exception:
                continue
    return sanitized


def sanitize_error_metadata_item(item: Any) -> Any:
    """Sanitize a single metadata item without changing the output schema."""
    if isinstance(item, dict):
        sanitized_item = {}
        for meta_key, meta_value in item.items():
            if meta_key in ["target_actor", "actor", "task"]:
                continue
            if hasattr(meta_value, "__name__"):
                sanitized_item[meta_key] = str(meta_value)
            elif isinstance(meta_value, (str, int, float, bool, list, dict, type(None))):
                sanitized_item[meta_key] = meta_value
            else:
                try:
                    sanitized_item[meta_key] = str(meta_value)
                except Exception:
                    continue
        return sanitized_item

    try:
        return str(item)
    except Exception:
        return None
