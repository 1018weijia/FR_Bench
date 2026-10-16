"""Lightweight utility entry points for error_recovery."""

__all__ = [
    "ErrorRecoveryDataAnalyzer",
    "print_summary",
    "print_frame_data",
    "find_error_events",
]


def __getattr__(name: str):
    if name == "ErrorRecoveryDataAnalyzer":
        from .data_analyzer import ErrorRecoveryDataAnalyzer

        return ErrorRecoveryDataAnalyzer
    if name in {"print_summary", "print_frame_data", "find_error_events"}:
        from . import parse_hdf5

        return getattr(parse_hdf5, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
