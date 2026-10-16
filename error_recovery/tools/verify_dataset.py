#!/usr/bin/env python3
"""Check a failure-recovery collection directory, optionally against a golden run."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple


EPISODE_RE = re.compile(r"episode(\d+)\.hdf5$")
SUMMARY_KEYS = (
    "episode_id",
    "task_name",
    "success",
    "seed",
    "final_phase",
    "total_frames",
    "error_types",
)


def _tmp_files(root: str) -> List[str]:
    found = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if ".tmp" in name or name.endswith(".tmp"):
                found.append(os.path.join(dirpath, name))
    return found


def _episode_ids(root: str) -> List[int]:
    data_dir = os.path.join(root, "data")
    if not os.path.isdir(data_dir):
        return []
    ids = []
    for name in os.listdir(data_dir):
        match = EPISODE_RE.match(name)
        if match:
            ids.append(int(match.group(1)))
    return sorted(ids)


def _video_ok(path: str) -> bool:
    if not os.path.exists(path) or os.path.getsize(path) <= 0:
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-f", "null", "-"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return os.path.getsize(path) > 0
    return result.returncode == 0


def _hdf5_schema(path: str) -> List[Dict[str, Any]]:
    import h5py

    rows = []

    def walk(group, prefix: str) -> None:
        for key, value in group.items():
            item_path = f"{prefix}/{key}"
            if isinstance(value, h5py.Dataset):
                rows.append(
                    {
                        "path": item_path,
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                    }
                )
            else:
                walk(value, item_path)

    with h5py.File(path, "r") as handle:
        walk(handle, "")
    return rows


def _load_summary(root: str, episode_id: int) -> Dict[str, Any]:
    path = os.path.join(root, "metadata", f"episode{episode_id}_metadata.json")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    summary = payload.get("episode_summary")
    if not isinstance(summary, dict):
        raise ValueError(f"episode {episode_id} metadata has no episode_summary")
    return summary


def _frame_count(schema: List[Dict[str, Any]]) -> Optional[int]:
    for row in schema:
        if row["path"].endswith("/joint_action/vector") and row["shape"]:
            return int(row["shape"][0])
    return None


def _compare_schema(
    golden: List[Dict[str, Any]],
    new: List[Dict[str, Any]],
    frame_tolerance: float,
) -> List[str]:
    errors = []
    new_by_path = {row["path"]: row for row in new}
    golden_frames = _frame_count(golden)
    new_frames = _frame_count(new)
    if golden_frames and new_frames is not None:
        allowed = max(30, int(round(golden_frames * frame_tolerance)))
        if abs(new_frames - golden_frames) > allowed:
            errors.append(
                f"frame count {new_frames} differs from golden {golden_frames} by more than {allowed}"
            )
    for row in golden:
        other = new_by_path.get(row["path"])
        if other is None:
            errors.append(f"missing dataset {row['path']}")
            continue
        if len(row["shape"]) != len(other["shape"]):
            errors.append(f"rank mismatch {row['path']}: {row['shape']} vs {other['shape']}")
            continue
        for index, (left, right) in enumerate(zip(row["shape"], other["shape"])):
            if index == 0 or str(row["dtype"]).startswith("|S") or str(row["dtype"]).startswith("object"):
                continue
            if left != right:
                errors.append(f"shape mismatch {row['path']}: {row['shape']} vs {other['shape']}")
                break
    return errors


def verify(root: str, expect: Optional[int], golden: Optional[str], frame_tolerance: float) -> List[str]:
    errors: List[str] = []
    if not os.path.isdir(root):
        return [f"missing directory {root}"]
    tmp_files = _tmp_files(root)
    if tmp_files:
        errors.append(f"partial files left behind: {tmp_files[:5]}")
    episode_ids = _episode_ids(root)
    if expect is not None and len(episode_ids) != expect:
        errors.append(f"expected {expect} episodes, found {len(episode_ids)}: {episode_ids}")
    seeds = []
    for episode_id in episode_ids:
        hdf5_path = os.path.join(root, "data", f"episode{episode_id}.hdf5")
        video_path = os.path.join(root, "video", f"episode{episode_id}.mp4")
        meta_path = os.path.join(root, "metadata", f"episode{episode_id}_metadata.json")
        if os.path.getsize(hdf5_path) <= 0:
            errors.append(f"empty hdf5 for episode {episode_id}")
            continue
        if not os.path.exists(meta_path):
            errors.append(f"missing metadata for episode {episode_id}")
            continue
        if not _video_ok(video_path):
            errors.append(f"video for episode {episode_id} is missing or unreadable")
        try:
            schema = _hdf5_schema(hdf5_path)
        except Exception as exc:
            errors.append(f"hdf5 episode {episode_id} cannot be read: {exc}")
            continue
        paths = {row["path"] for row in schema}
        for required in ("/observation/head_camera/rgb", "/joint_action/vector"):
            if required not in paths:
                errors.append(f"episode {episode_id} missing {required}")
        try:
            summary = _load_summary(root, episode_id)
        except Exception as exc:
            errors.append(str(exc))
            continue
        for key in SUMMARY_KEYS:
            if key not in summary:
                errors.append(f"episode {episode_id} summary missing {key}")
        if summary.get("seed") is not None:
            seeds.append(summary.get("seed"))
        if golden:
            golden_hdf5 = os.path.join(golden, "data", "episode0.hdf5")
            golden_meta = os.path.join(golden, "metadata", "episode0_metadata.json")
            if os.path.exists(golden_hdf5):
                errors.extend(
                    _compare_schema(_hdf5_schema(golden_hdf5), schema, frame_tolerance)
                )
            if os.path.exists(golden_meta):
                golden_summary = _load_summary(golden, 0)
                missing = [key for key in golden_summary.keys() if key not in summary]
                if missing:
                    errors.append(f"episode {episode_id} dropped metadata fields {missing}")
    if len(seeds) != len(set(seeds)):
        errors.append(f"duplicate seeds: {seeds}")
    return errors


def dump_schema(root: str, dest: str) -> None:
    episode_ids = _episode_ids(root)
    payload = {"episodes": episode_ids, "schemas": {}}
    for episode_id in episode_ids:
        hdf5_path = os.path.join(root, "data", f"episode{episode_id}.hdf5")
        payload["schemas"][str(episode_id)] = _hdf5_schema(hdf5_path)
        payload["summary"] = _load_summary(root, episode_id)
    with open(dest, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a collected failure-recovery dataset")
    parser.add_argument("--path", required=True)
    parser.add_argument("--expect", type=int, default=None)
    parser.add_argument("--golden", default=None)
    parser.add_argument("--frame-tolerance", type=float, default=0.25)
    parser.add_argument("--dump-schema", default=None)
    args = parser.parse_args(argv)
    if args.dump_schema:
        dump_schema(args.path, args.dump_schema)
        print(f"wrote {args.dump_schema}")
    errors = verify(args.path, args.expect, args.golden, args.frame_tolerance)
    if errors:
        for error in errors:
            print(f"FAIL {error}")
        return 1
    print(f"OK {args.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
