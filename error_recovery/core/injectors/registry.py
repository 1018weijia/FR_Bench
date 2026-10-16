from typing import List

from .base import ErrorInjector
from .grasp.grasp_orientation_mismatch import GraspOrientationMismatchInjector
from .grasp.grasp_position_offset import GraspPositionOffsetInjector
from .grasp.grasp_slip import GraspSlipInjector
from .grasp.premature_close import PrematureCloseInjector


ERROR_INJECTORS = {
    'premature_close': PrematureCloseInjector,
    'grasp_slip': GraspSlipInjector,
    'grasp_position_offset': GraspPositionOffsetInjector,
    'grasp_orientation_mismatch': GraspOrientationMismatchInjector,
}


def create_error_injector(error_type: str, **kwargs) -> ErrorInjector:
    """Factory that creates an error injector"""
    if error_type not in ERROR_INJECTORS:
        raise ValueError(
            f"Unknown error type: {error_type}.\n"
            f"Available: {list(ERROR_INJECTORS.keys())}"
        )

    return ERROR_INJECTORS[error_type](**kwargs)


def list_available_errors() -> List[str]:
    """List all available error types"""
    return list(ERROR_INJECTORS.keys())
