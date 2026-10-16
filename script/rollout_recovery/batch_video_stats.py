#!/usr/bin/env python3
from __future__ import annotations

"""
Batch analysis of success video frame counts.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from script.rollout_recovery.video_stats import analyze_eval_dir, write_task_threshold


def discover_latest_eval_dirs(eval_root: str, task_names: Iterable[str] | None = None) -> list[dict[str, str]]:
    eval_root_path = Path(eval_root)
    if not eval_root_path.exists():
        return []

    allowed_tasks = set(task_names) if task_names else None
    latest: dict[str, tuple[float, Path]] = {}

    for result_file in eval_root_path.rglob("_result.txt"):
        try:
            rel = result_file.relative_to(eval_root_path)
        except ValueError:
            continue
        if len(rel.parts) < 2:
            continue

        task_name = rel.parts[0]
        if allowed_tasks is not None and task_name not in allowed_tasks:
            continue

        eval_dir = result_file.parent
        score = result_file.stat().st_mtime
        current = latest.get(task_name)
        if current is None or score > current[0]:
            latest[task_name] = (score, eval_dir)

    return [
        {"task_name": task_name, "eval_dir": str(eval_dir)}
        for task_name, (_, eval_dir) in sorted(latest.items())
    ]


def analyze_task(eval_dir: str, task_name: str, task_lim_dir: str | None = None) -> bool:
    print(f"\n{'=' * 80}")
    print(f"Analyzing task: {task_name}")
    print(f"Eval dir: {eval_dir}")
    print(f"{'=' * 80}")

    stats = analyze_eval_dir(eval_dir, task_name)
    if stats is None:
        print(f"\033[91mError: Failed to analyze {task_name}\033[0m")
        return False

    output_file = Path(eval_dir) / "success_video_stats.json"
    with open(output_file, "w") as f:
        json.dump(stats, f, indent=2)

    if task_lim_dir:
        threshold_file = write_task_threshold(stats, task_lim_dir)
        print(f"\033[92m✓ Threshold saved: {threshold_file}\033[0m")

    print(f"\033[92m✓ Successfully analyzed {task_name}\033[0m")
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_root",
        type=str,
        default=os.environ.get("ROBOTWIN_EVAL_ROOT", "./eval_result"),
        help="Root directory that contains per-task eval result folders",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional task name filter. If omitted, discover all tasks under eval_root.",
    )
    parser.add_argument(
        "--task_lim_dir",
        type=str,
        default=os.environ.get("ROBOTWIN_TASK_LIM_DIR", "/data2/liangxiwen/zkd/cry/RoboTwin/task_lim"),
    )
    args = parser.parse_args(argv)

    print("\n" + "=" * 80)
    print("Batch Analysis of Success Videos")
    print("=" * 80)

    task_entries = discover_latest_eval_dirs(args.eval_root, args.tasks)
    if not task_entries:
        print(f"\033[91mError: No eval results found under {args.eval_root}\033[0m")
        return

    print(f"\nEval root: {args.eval_root}")
    print(f"Task threshold directory: {args.task_lim_dir}")

    results = []
    for task in task_entries:
        success = analyze_task(task["eval_dir"], task["task_name"], task_lim_dir=args.task_lim_dir)
        results.append({"task_name": task["task_name"], "eval_dir": task["eval_dir"], "success": success})

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    print(f"\n\033[92mSuccessful: {len(successful)}/{len(results)}\033[0m")
    for r in successful:
        print(f"  ✓ {r['task_name']}")
    if failed:
        print(f"\n\033[91mFailed: {len(failed)}/{len(results)}\033[0m")
        for r in failed:
            print(f"  ✗ {r['task_name']}: {r['eval_dir']}")

    print("\n" + "=" * 80)
    print("Analysis complete. Thresholds saved in:", args.task_lim_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()
