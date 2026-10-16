"""Process and machine resource readings. Standard library only."""

from __future__ import annotations

import os
import subprocess
from typing import Dict, List, Optional


THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "FFMPEG_THREADS",
    "FRBENCH_NUM_THREADS",
)


def read_rss_mb(pid: int) -> Optional[float]:
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


def read_thread_count(pid: int) -> Optional[int]:
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("Threads:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def read_mem_available_mb() -> Optional[float]:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


def read_load_average() -> Optional[float]:
    try:
        return os.getloadavg()[0]
    except OSError:
        return None


def gpu_used_mb_by_index() -> Dict[str, float]:
    """Device-level used memory. Works when per-process PIDs are not visible."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    usage: Dict[str, float] = {}
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            usage[parts[0]] = float(parts[1])
        except ValueError:
            continue
    return usage


def gpu_memory_by_pid() -> Dict[int, float]:
    """Map pid -> MiB from nvidia-smi. Empty when the tool is unavailable."""
    try:
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
    except (OSError, subprocess.SubprocessError):
        return {}
    usage: Dict[int, float] = {}
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            used = float(parts[1])
        except ValueError:
            continue
        usage[pid] = usage.get(pid, 0.0) + used
    return usage


def assign_cores(
    n_workers: int,
    reserve: int = 8,
    cpu_count: Optional[int] = None,
    offset: int = 0,
    length: Optional[int] = None,
) -> List[List[int]]:
    """Contiguous, disjoint core sets. Leaves `reserve` cores for the OS when possible.

    `offset` and `length` pin this supervisor to one slice so several supervisors
    can run at once without sharing the same cores.
    """
    if n_workers <= 0:
        return []
    total = os.cpu_count() if cpu_count is None else cpu_count
    total = max(1, int(total or 1))
    if length is not None and length > 0:
        start = min(max(0, offset), total - 1)
        usable = list(range(start, min(total, start + length)))
    else:
        usable = list(range(total))
        if total - max(0, reserve) >= n_workers:
            usable = list(range(total - max(0, reserve)))
    base, remainder = divmod(len(usable), n_workers)
    chunks: List[List[int]] = []
    cursor = 0
    for index in range(n_workers):
        count = base + (1 if index < remainder else 0)
        chunks.append(usable[cursor:cursor + count])
        cursor += count
    return chunks


def thread_env(n_threads: int) -> Dict[str, str]:
    value = str(max(1, int(n_threads)))
    env = {name: value for name in THREAD_ENV_VARS}
    # Passive waiting keeps idle BLAS threads from pinning every core.
    env["OMP_WAIT_POLICY"] = "PASSIVE"
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return env
