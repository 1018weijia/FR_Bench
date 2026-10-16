#!/usr/bin/env python3
"""Stand-in for collect_universal.py. Used by the supervisor unit tests."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _append(path: str, event: dict) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    event.setdefault("ts", time.time())
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _next_invocation(state_file: str) -> int:
    if not state_file:
        return 0
    directory = os.path.dirname(state_file)
    if directory:
        os.makedirs(directory, exist_ok=True)
    previous = 0
    if os.path.exists(state_file):
        try:
            previous = int(open(state_file, "r", encoding="utf-8").read().strip() or "0")
        except ValueError:
            previous = 0
    with open(state_file, "w", encoding="utf-8") as handle:
        handle.write(str(previous + 1))
    return previous


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=os.environ.get("FRBENCH_FAKE_MODE", "ok"))
    parser.add_argument("--fail_times", type=int, default=1)
    parser.add_argument("--state_file", default="")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--seed_end", type=int, default=10**9)
    parser.add_argument("--episode_offset", type=int, default=0)
    parser.add_argument("--worker_id", default="w0")
    parser.add_argument("--events_path", required=True)
    parser.add_argument("--max_episodes_per_process", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    args, _unknown = parser.parse_known_args()

    invocation = _next_invocation(args.state_file)
    mode = args.mode
    if mode == "flaky" and invocation >= args.fail_times:
        mode = "ok"

    def start(episode_id: int, seed: int) -> None:
        _append(
            args.events_path,
            {
                "type": "start",
                "worker_id": args.worker_id,
                "gpu": args.gpu,
                "pid": os.getpid(),
                "episode_id": episode_id,
                "seed": seed,
            },
        )

    if mode in ("crash", "hang"):
        start(args.episode_offset, args.seed_start)
        if mode == "crash":
            return 75
        time.sleep(3600)
        return 0

    produced = 0
    seed = args.seed_start
    while produced < args.episodes:
        if args.max_episodes_per_process and produced >= args.max_episodes_per_process:
            break
        if seed >= args.seed_end:
            break
        episode_id = args.episode_offset + produced
        start(episode_id, seed)
        _append(
            args.events_path,
            {
                "type": "attempt",
                "worker_id": args.worker_id,
                "gpu": args.gpu,
                "pid": os.getpid(),
                "episode_id": episode_id,
                "seed": seed,
                "outcome": "saved",
                "success": True,
                "error_types": ["premature_close"],
                "recovery": "success",
                "frames": 1,
            },
        )
        produced += 1
        seed += 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
