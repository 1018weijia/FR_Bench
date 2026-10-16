#!/usr/bin/env python3
"""Inspect error_recovery episode HDF5 files."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from typing import Any, Optional

import h5py
import numpy as np


def decode_string(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [decode_string(item) for item in value.tolist()] if value.dtype.kind == "S" else value.tolist()
    return value


def _dataset_len(group: h5py.Group) -> Optional[int]:
    for value in group.values():
        if isinstance(value, h5py.Dataset) and value.shape:
            return int(value.shape[0])
    return None


def _print_datasets(group: h5py.Group, prefix: str = "") -> None:
    for key, value in group.items():
        path = f"{prefix}{key}"
        if isinstance(value, h5py.Dataset):
            print(f"  {path}: shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, h5py.Group):
            _print_datasets(value, f"{path}/")


def print_summary(hdf5_path: str) -> None:
    with h5py.File(hdf5_path, "r") as f:
        print(f"HDF5: {hdf5_path}")
        print(f"top-level: {', '.join(f.keys())}")
        er = f.get("error_recovery")
        if isinstance(er, h5py.Group):
            n = _dataset_len(er)
            print(f"frames: {n if n is not None else 'unknown'}")
            if "phase" in er:
                phases = [decode_string(item) for item in er["phase"][:]]
                print("phase_counts:")
                for phase, count in Counter(phases).most_common():
                    print(f"  {phase}: {count}")
            if "is_error_frame" in er:
                print(f"error_frames: {int(np.count_nonzero(er['is_error_frame'][:]))}")
            if "timestamp" in er and len(er["timestamp"]) > 1:
                start = float(er["timestamp"][0])
                end = float(er["timestamp"][-1])
                print(f"duration: {end - start:.2f}s ({datetime.fromtimestamp(start)} -> {datetime.fromtimestamp(end)})")
        else:
            print("error_recovery: missing")
        print("datasets:")
        _print_datasets(f)


def print_frame_data(hdf5_path: str, frame_idx: int) -> None:
    with h5py.File(hdf5_path, "r") as f:
        er = f.get("error_recovery")
        if not isinstance(er, h5py.Group):
            print("error_recovery: missing")
            return
        n = _dataset_len(er)
        if n is not None and not 0 <= frame_idx < n:
            raise IndexError(f"frame {frame_idx} out of range [0, {n})")
        print(f"frame: {frame_idx}")
        for key in ("enabled", "phase", "error_stage_precise", "frame_count", "timestamp", "is_error_frame",
                    "is_error_frame_precise", "active_injectors", "error_metadata"):
            if key not in er:
                continue
            value = decode_string(er[key][frame_idx])
            if key == "timestamp":
                value = f"{float(value):.6f} ({datetime.fromtimestamp(float(value))})"
            print(f"{key}: {value}")


def find_error_events(hdf5_path: str) -> None:
    interesting = {"error_attempt", "error_detected", "recovery", "recovered", "recovery_failed"}
    with h5py.File(hdf5_path, "r") as f:
        er = f.get("error_recovery")
        if not isinstance(er, h5py.Group) or "phase" not in er:
            print("error_recovery.phase: missing")
            return
        phases = [decode_string(item) for item in er["phase"][:]]
        frame_counts = er["frame_count"][:] if "frame_count" in er else None
        timestamps = er["timestamp"][:] if "timestamp" in er else None
        last_phase = None
        for idx, phase in enumerate(phases):
            if phase == last_phase or phase not in interesting:
                last_phase = phase
                continue
            frame_count = int(frame_counts[idx]) if frame_counts is not None else idx
            suffix = f" | {datetime.fromtimestamp(float(timestamps[idx]))}" if timestamps is not None else ""
            print(f"frame={idx} frame_count={frame_count} phase={phase}{suffix}")
            last_phase = phase


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect an error_recovery HDF5 episode")
    parser.add_argument("hdf5_path", nargs="?", help="HDF5 path")
    parser.add_argument("--hdf5_path", dest="hdf5_path_opt", help="HDF5 path, kept for backward compatibility")
    parser.add_argument("--summary", action="store_true", help="Print file summary")
    parser.add_argument("--frame", type=int, help="Print one frame's error_recovery fields")
    parser.add_argument("--events", action="store_true", help="Print phase transition events")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    hdf5_path = args.hdf5_path_opt or args.hdf5_path
    if not hdf5_path:
        raise SystemExit("missing hdf5_path")
    if args.frame is not None:
        print_frame_data(hdf5_path, args.frame)
    elif args.events:
        find_error_events(hdf5_path)
    else:
        print_summary(hdf5_path)


if __name__ == "__main__":
    main()
