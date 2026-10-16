#!/usr/bin/env python3
from __future__ import annotations

"""
Analyze frame-count distribution of success videos for rollout recovery threshold.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np


def get_video_frame_count(video_path: str) -> Optional[int]:
    """Get video frame count via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_packets",
                "-show_entries",
                "stream=nb_read_packets",
                "-of",
                "csv=p=0",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception as e:
        print(f"Warning: Failed to get frame count for {video_path}: {e}")

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate,duration",
                "-of",
                "csv=p=0",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 2:
                fps_str = parts[0]
                duration = float(parts[1])
                if "/" in fps_str:
                    num, den = map(int, fps_str.split("/"))
                    fps = num / den
                else:
                    fps = float(fps_str)
                return int(fps * duration)
    except Exception as e:
        print(f"Warning: Failed to estimate frame count for {video_path}: {e}")

    return None


def analyze_eval_dir(eval_dir: str, task_name: str) -> Optional[dict]:
    """Compute frame-count stats over successful videos in an eval dir."""
    eval_dir_p = Path(eval_dir)

    result_file = eval_dir_p / "_result.txt"
    if not result_file.exists():
        print(f"Warning: {result_file} not found")
        return None

    with open(result_file, "r") as f:
        lines = f.readlines()

    success_rate = None
    for line in lines:
        try:
            rate = float(line.strip())
            if 0 <= rate <= 1:
                success_rate = rate
        except Exception:
            pass

    video_dir = eval_dir_p / "video"
    if not video_dir.exists():
        video_dir = eval_dir_p

    video_files = sorted(video_dir.glob("episode*.mp4"))
    if len(video_files) == 0:
        print(f"Warning: No video files found in {video_dir}")
        return None

    all_frame_counts = []
    video_frame_map = {}
    for video_file in video_files:
        frame_count = get_video_frame_count(str(video_file))
        if frame_count is not None:
            episode_id = video_file.stem.replace("episode", "")
            all_frame_counts.append(frame_count)
            video_frame_map[episode_id] = frame_count

    if len(all_frame_counts) == 0:
        print(f"Warning: No valid videos in {video_dir}")
        return None

    all_frame_counts = np.array(all_frame_counts)
    max_frames = np.max(all_frame_counts)
    frame_threshold = max_frames * 0.95
    successful_frame_counts = all_frame_counts[all_frame_counts < frame_threshold]
    failed_count = int(np.sum(all_frame_counts >= frame_threshold))
    success_count = int(len(successful_frame_counts))

    print(f"\n[Analysis] Total videos: {len(all_frame_counts)}")
    print(f"[Analysis] Max frames (failed): {max_frames}")
    print(f"[Analysis] Failed: {failed_count}, Successful: {success_count}")

    if len(successful_frame_counts) == 0:
        print(f"Warning: No successful videos (all have {max_frames} frames)")
        return None

    frame_counts = successful_frame_counts
    stats = {
        "task_name": task_name,
        "eval_dir": str(eval_dir_p),
        "total_videos": int(len(all_frame_counts)),
        "successful_videos": int(len(successful_frame_counts)),
        "failed_videos": int(failed_count),
        "max_frames": int(max_frames),
        "success_rate": (success_count / len(all_frame_counts)) if len(all_frame_counts) > 0 else 0.0,
        "frame_counts": frame_counts.tolist(),
        "mean": float(np.mean(frame_counts)),
        "median": float(np.median(frame_counts)),
        "std": float(np.std(frame_counts)),
        "min": int(np.min(frame_counts)),
        "max": int(np.max(frame_counts)),
        "p75": int(np.percentile(frame_counts, 75)),
        "p90": int(np.percentile(frame_counts, 90)),
        "p95": int(np.percentile(frame_counts, 95)),
        "p99": int(np.percentile(frame_counts, 99)),
    }
    if success_rate is not None:
        stats["reported_success_rate"] = success_rate
    stats["video_frame_map"] = video_frame_map
    return stats


def build_threshold_data(stats: dict) -> dict:
    return {
        "task_name": stats["task_name"],
        "thresholds": {
            "max": stats["max"],
            "p99": stats["p99"],
            "p95": stats["p95"],
            "p90": stats["p90"],
            "mean": int(stats["mean"]),
        },
        "statistics": {
            "total_videos": stats["total_videos"],
            "successful_videos": stats["successful_videos"],
            "failed_videos": stats["failed_videos"],
            "success_rate": stats["success_rate"],
            "mean": stats["mean"],
            "median": stats["median"],
            "std": stats["std"],
            "min": stats["min"],
            "max": stats["max"],
        },
        "source_eval_dir": stats["eval_dir"],
    }


def write_task_threshold(stats: dict, task_lim_dir: str) -> str:
    task_lim_dir_p = Path(task_lim_dir)
    task_lim_dir_p.mkdir(parents=True, exist_ok=True)
    threshold_file = task_lim_dir_p / f"{stats['task_name']}.json"
    with open(threshold_file, "w") as f:
        json.dump(build_threshold_data(stats), f, indent=2)
    return str(threshold_file)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Analyze success video frame distribution")
    parser.add_argument("--eval_dir", type=str, required=True, help="Eval result directory")
    parser.add_argument("--task_name", type=str, required=True, help="Task name")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path")
    parser.add_argument("--task_lim_dir", type=str, default=None, help="Unified task threshold directory")
    args = parser.parse_args(argv)

    stats = analyze_eval_dir(args.eval_dir, args.task_name)
    if stats is None:
        print("Failed to analyze eval directory")
        return None

    print(f"\n{'=' * 60}")
    print(f"Task: {stats['task_name']}")
    print(f"Eval Dir: {stats['eval_dir']}")
    print(f"{'=' * 60}")
    print(
        f"Total: {stats['total_videos']}, Successful: {stats['successful_videos']}, Failed: {stats['failed_videos']}"
    )
    print(f"Success Rate: {stats['success_rate']:.2%}")
    print(
        f"\nFrame stats (successful only): Mean {stats['mean']:.1f}, Median {stats['median']:.1f}, "
        f"Min {stats['min']}, Max {stats['max']}, P90 {stats['p90']}, P95 {stats['p95']}, P99 {stats['p99']}"
    )

    output_file = Path(args.eval_dir) / "success_video_stats.json"
    with open(output_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to {output_file}")

    if args.task_lim_dir:
        threshold_file = write_task_threshold(stats, args.task_lim_dir)
        print(f"Threshold file saved to {threshold_file}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Also saved to {args.output}")

    return stats


if __name__ == "__main__":
    main()
