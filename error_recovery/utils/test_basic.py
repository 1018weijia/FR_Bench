#!/usr/bin/env python3
"""Smoke checks for the error_recovery package without launching simulation."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _check(name: str, fn) -> bool:
    try:
        fn()
        print(f"ok - {name}")
        return True
    except Exception as exc:
        print(f"not ok - {name}: {exc}")
        return False


def check_imports() -> None:
    from error_recovery.core.error_injectors import ERROR_INJECTORS, create_error_injector
    from error_recovery.core.framework import ErrorRecoveryFramework

    if "premature_close" not in ERROR_INJECTORS:
        raise AssertionError("premature_close injector missing")
    injector = create_error_injector("premature_close", probability=0.5)
    if injector.name != "premature_close":
        raise AssertionError("injector creation failed")
    if "premature_close" not in ErrorRecoveryFramework().list_supported_error_types():
        raise AssertionError("framework supported error list is inconsistent")


def check_env_tasks() -> None:
    tasks = [path for path in (_repo_root() / "envs").glob("*.py") if not path.name.startswith("_")]
    if not tasks:
        raise AssertionError("no env task files found")


def check_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, str(_repo_root() / "error_recovery" / "collect_universal.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr.strip() or f"returncode={result.returncode}")


def main() -> bool:
    logging.getLogger().setLevel(logging.WARNING)
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)

    checks = [
        ("imports", check_imports),
        ("env task discovery", check_env_tasks),
        ("collect_universal --help", check_cli_help),
    ]
    passed = sum(_check(name, fn) for name, fn in checks)
    print(f"result: {passed}/{len(checks)} passed")
    return passed == len(checks)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
