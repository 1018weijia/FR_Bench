#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rollout_recovery")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_recover = sub.add_parser("recover", help="Run rollout + auto recovery evaluation")
    p_recover.add_argument("--config", type=str, required=True)
    p_recover.add_argument("--overrides", nargs=argparse.REMAINDER)

    p_stats = sub.add_parser("video-stats", help="Analyze success videos for frame thresholds")
    p_stats.add_argument("--eval_dir", type=str, required=True)
    p_stats.add_argument("--task_name", type=str, required=True)
    p_stats.add_argument("--output", type=str, default=None)
    p_stats.add_argument("--task_lim_dir", type=str, default=None)

    p_batch = sub.add_parser("batch-video-stats", help="Batch analyze success videos for many tasks")
    p_batch.add_argument("--eval_root", type=str, default=None)
    p_batch.add_argument("--tasks", nargs="*", default=None)
    p_batch.add_argument("--task_lim_dir", type=str, default=None)

    args = parser.parse_args(argv)

    if args.cmd == "recover":
        os.chdir(str(_repo_root()))
        from script.rollout_recovery.config import parse_args_and_config
        from script.rollout_recovery.rollout_recover import main as rollout_main

        cfg = parse_args_and_config(
            ["--config", args.config] + (["--overrides"] + args.overrides if args.overrides else [])
        )
        rollout_main(cfg)
        return 0

    if args.cmd == "video-stats":
        from script.rollout_recovery.video_stats import main as stats_main

        stats_main(
            [
                "--eval_dir",
                args.eval_dir,
                "--task_name",
                args.task_name,
                *([] if args.output is None else ["--output", args.output]),
                *([] if args.task_lim_dir is None else ["--task_lim_dir", args.task_lim_dir]),
            ]
        )
        return 0

    if args.cmd == "batch-video-stats":
        from script.rollout_recovery.batch_video_stats import main as batch_main

        batch_argv: list[str] = []
        if args.eval_root is not None:
            batch_argv += ["--eval_root", args.eval_root]
        if args.tasks is not None:
            batch_argv += ["--tasks", *args.tasks]
        if args.task_lim_dir is not None:
            batch_argv += ["--task_lim_dir", args.task_lim_dir]
        batch_main(batch_argv)
        return 0

    raise AssertionError(f"unknown cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
