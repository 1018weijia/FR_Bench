#!/usr/bin/env python3
"""Launch isolated failure-recovery collection workers.

Each worker is a fresh interpreter pinned to one GPU and a private set of CPU
cores. This process itself never imports the simulator, so it cannot fork a
live CUDA context.

Example:
    python error_recovery/collect_parallel.py \
        --task move_can_pot --episodes 100 --gpus 0,1 --workers_per_gpu 2 \
        --error_types premature_close
"""

from __future__ import annotations

import argparse
import os
import sys
import types


def _install_lightweight_package() -> None:
    """Import error_recovery.parallel without executing error_recovery/__init__.py.

    The package __init__ pulls in the simulator. The supervisor must stay a
    plain Python process so its children start with a clean CUDA state.
    """
    if "error_recovery.parallel" in sys.modules:
        return
    package_dir = os.path.dirname(os.path.abspath(__file__))
    if "error_recovery" not in sys.modules:
        package = types.ModuleType("error_recovery")
        package.__path__ = [package_dir]
        package.__package__ = "error_recovery"
        sys.modules["error_recovery"] = package


def _drop_option(argv, name: str):
    kept = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg == name:
            skip = True
            continue
        if arg.startswith(name + "="):
            continue
        kept.append(arg)
    return kept


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parallel failure-recovery data collection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids, e.g. 0,1")
    parser.add_argument(
        "--workers_per_gpu",
        type=int,
        default=2,
        help="Workers sharing one GPU. A measured worker uses about 7GB VRAM and 6GB RAM",
    )
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--max_episodes_per_process", type=int, default=50)
    parser.add_argument("--max_restarts", type=int, default=5)
    parser.add_argument("--episode_timeout", type=float, default=1200.0, help="Seconds before a stuck attempt is killed")
    parser.add_argument("--max_rss_gb", type=float, default=16.0, help="Kill a worker above this RSS. The measured plateau is about 6GB")
    parser.add_argument(
        "--max_vram_gb",
        type=float,
        default=16.0,
        help="Per-worker VRAM budget. The GPU is capped at this value times the workers on it",
    )
    parser.add_argument("--min_free_gb", type=float, default=32.0)
    parser.add_argument("--reserve_cores", type=int, default=8)
    parser.add_argument("--core_offset", type=int, default=0, help="First CPU core this supervisor may use")
    parser.add_argument("--core_count", type=int, default=0, help="How many cores this supervisor may use; 0 means the rest of the machine")
    parser.add_argument("--nice", type=int, default=10)
    parser.add_argument("--poll_s", type=float, default=2.0)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args(argv)
    # These belong to the supervisor. Passing them through would split workers
    # onto the wrong GPU or restart them with a different seed block.
    for owned in ("--gpu", "--seed_start", "--seed_end", "--episode_offset", "--worker_id", "--events_path"):
        passthrough = _drop_option(passthrough, owned)

    _install_lightweight_package()
    from error_recovery.parallel.supervisor import Supervisor, SupervisorConfig

    gpus = [part.strip() for part in args.gpus.split(",") if part.strip()]
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")
    if args.workers_per_gpu < 1:
        parser.error("--workers_per_gpu must be >= 1")
    if not gpus:
        parser.error("--gpus must list at least one GPU id")

    config = SupervisorConfig(
        task=args.task,
        episodes=args.episodes,
        gpus=gpus,
        workers_per_gpu=args.workers_per_gpu,
        seed_start=args.seed_start,
        save_path=args.save_path or os.path.join(".", "error_recovery", "data", args.task),
        max_episodes_per_process=args.max_episodes_per_process,
        max_restarts=args.max_restarts,
        episode_timeout_s=args.episode_timeout,
        max_rss_gb=args.max_rss_gb,
        max_vram_gb=args.max_vram_gb,
        min_free_gb=args.min_free_gb,
        reserve_cores=args.reserve_cores,
        core_offset=args.core_offset,
        core_count=args.core_count,
        nice=args.nice,
        poll_s=args.poll_s,
        passthrough=passthrough,
    )
    return Supervisor(config).run()


if __name__ == "__main__":
    sys.exit(main())
