"""Compatibility-friendly execution helpers for TaskWrapper."""

import logging

logger = logging.getLogger("error_recovery.execution")

from .action_mutation import execute_modified_action
from .flow import check_error_injection, execute_normal, execute_with_error

__all__ = [
    "check_error_injection",
    "execute_modified_action",
    "execute_normal",
    "execute_with_error",
    "logger",
]
