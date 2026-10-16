#!/usr/bin/env python3
"""Batch-collect error-recovery data for multiple tasks.

Each task starts one collect_parallel.py process. This process does not load the simulator.
"""

import argparse
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_COLLECT = os.path.join(_ROOT, "error_recovery", "collect_parallel.py")


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-collect error-recovery data for multiple tasks")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["pick_diverse_bottles", "place_bread_basket", "handover_block", "stack_blocks_two"],
        help="Task list (defaults to the four tasks above)",
    )
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per task, default 50")
    parser.add_argument("--gpus", default="0", help="GPU list passed to collect_parallel.py")
    parser.add_argument("--workers_per_gpu", type=int, default=2, help="Processes per GPU")
    parser.add_argument(
        "--error_types",
        nargs="+",
        default=["premature_close"],
        help="Error type list; accepts `TYPE[:prob[:param...]]` directly",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="./error_recovery/data",
        help="Save root; each task is written to a subdirectory of the same name",
    )
    parser.add_argument("--render", action="store_true", help="Enable rendering (for debugging)")
    parser.add_argument(
        "--log_level",
        choices=["quiet", "normal", "debug"],
        default="quiet",
        help="Collection log verbosity: quiet=default brief, normal=recovery summary, debug=detailed diagnostics",
    )
    parser.add_argument(
        "--data_types",
        nargs="+",
        default=None,
        help="Override the default data_type, for example: --data_types rgb qpos endpose",
    )
    parser.add_argument("--no_depth", action="store_true", help="Do not save depth images")
    parser.add_argument("--pointcloud", action="store_true", help="Save point clouds")
    parser.add_argument(
        "--trim_failure_trajectory_on_save",
        action="store_true",
        default=True,
        help="Trim the official failed-trajectory output to before the first failure-detection frame (on by default)",
    )
    parser.add_argument(
        "--no_trim_failure_trajectory_on_save",
        dest="trim_failure_trajectory_on_save",
        action="store_false",
        help="Disable failed-trajectory trimming and keep the full failed trajectory (for debugging)",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")
    if args.workers_per_gpu < 1:
        parser.error("--workers_per_gpu must be >= 1")

    failed = []
    for task_name in args.tasks:
        save_path = os.path.join(args.save_root, task_name)
        cmd = [
            sys.executable,
            _COLLECT,
            "--task",
            task_name,
            "--episodes",
            str(args.episodes),
            "--gpus",
            args.gpus,
            "--workers_per_gpu",
            str(args.workers_per_gpu),
            "--save_path",
            save_path,
            "--error_types",
            *args.error_types,
            "--log_level",
            args.log_level,
        ]
        if args.render:
            cmd.append("--render")
        if args.data_types:
            cmd.extend(["--data_types", *args.data_types])
        if args.no_depth:
            cmd.append("--no_depth")
        if args.pointcloud:
            cmd.append("--pointcloud")
        if not args.trim_failure_trajectory_on_save:
            cmd.append("--no_trim_failure_trajectory_on_save")

        print(f"\n{'=' * 60}")
        print(f"Collecting task: {task_name}")
        print(f"{'=' * 60}")
        result = subprocess.run(cmd, cwd=_ROOT)
        if result.returncode != 0:
            failed.append(task_name)
            print(f"❌ Failed: {task_name} exit code {result.returncode}")
        else:
            print(f"✅ Done: {task_name}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
