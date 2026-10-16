"""Per-worker runtime helpers for the failure-recovery collector.

Imported lazily from the collection loop so importing the package does not
depend on this module's optional heavy libraries.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

EXIT_RESTART = 75

_CUDA_FATAL_MARKERS = (
    "cuda error",
    "cudaerror",
    "illegal instruction",
    "illegal memory",
    "an illegal",
    "unspecified launch failure",
    "device-side assert",
)

_PLANNER_MARKERS = ("warp", "curobo", "motion_gen")


def apply_thread_limits() -> None:
    raw = os.environ.get("FRBENCH_NUM_THREADS", "")
    if not raw:
        return
    try:
        limit = int(raw)
    except ValueError:
        return
    if limit <= 0:
        return
    try:
        import torch

        torch.set_num_threads(limit)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except Exception:
        pass
    try:
        import cv2

        cv2.setNumThreads(limit)
    except Exception:
        pass


def is_cuda_fatal(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _CUDA_FATAL_MARKERS)


def is_planner_failure(exc: BaseException) -> bool:
    text = str(exc)
    if type(exc).__name__ == "UnboundLocalError" and "local variable 'i' referenced before assignment" in text:
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in _PLANNER_MARKERS)


def read_rss_mb(pid: Optional[int] = None) -> Optional[float]:
    path = "/proc/self/status" if pid is None else f"/proc/{pid}/status"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


def read_vram_mb(pid: Optional[int] = None) -> Optional[float]:
    target = os.getpid() if pid is None else pid
    try:
        import subprocess

        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        output = ""
    total = 0.0
    found = False
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            if int(parts[0]) == target:
                total += float(parts[1])
                found = True
        except ValueError:
            continue
    if found:
        return total
    # In containers nvidia-smi often reports host PIDs, so the process cannot
    # find itself. The visible device's used memory is the right figure when
    # this worker is alone on that GPU.
    visible = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0].strip()
    if not visible:
        return None
    try:
        import subprocess

        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={visible}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
            stderr=subprocess.DEVNULL,
        )
        return float(output.strip().splitlines()[0])
    except Exception:
        return None


def detect_planner_backend(task: Any) -> Optional[str]:
    robot = getattr(task, "robot", None)
    planner = getattr(robot, "left_planner", None) if robot is not None else None
    if planner is None:
        return None
    name = type(planner).__name__.lower()
    if "curobo" in name:
        return "curobo"
    if "mplib" in name:
        return "mplib"
    return type(planner).__name__


def append_event(path: Optional[str], event: Dict[str, Any]) -> None:
    if not path:
        return
    event.setdefault("ts", time.time())
    from error_recovery.parallel.io_util import append_jsonl

    append_jsonl(path, event)
