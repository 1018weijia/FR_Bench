"""Compatibility-friendly serialization helpers for TaskWrapper."""

from .sanitize import get_error_recovery_data, sanitize_error_metadata_item, sanitize_error_recovery_data

__all__ = [
    "get_error_recovery_data",
    "sanitize_error_metadata_item",
    "sanitize_error_recovery_data",
]
