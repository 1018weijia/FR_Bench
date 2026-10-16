#!/usr/bin/env python3
"""Full failure-recovery matrix: collect one cell at a time, then check labels.

Run from the repo root with the RoboTwin interpreter:

    python error_recovery/tests/full_matrix.py run
    python error_recovery/tests/full_matrix.py report
    python error_recovery/tests/full_matrix.py check --path DIR --expect 2
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ERROR_TYPES = (
    "premature_close",
    "grasp_slip",
    "grasp_position_offset",
    "grasp_orientation_mismatch",
)
FRAME_KEYS = (
    "inject_decision_frame",
    "error_onset_frame",
    "error_detected_frame",
    "recovery_start_frame",
    "recovery_end_frame",
    "error_end_frame",
)
PRECISE_OK = {"normal", "pre_error"}
CELL_TIMEOUT_S = 45 * 60
DONE = {"pass", "crash", "label_error", "insufficient", "timeout"}


def _is_error_phase():
    path = os.path.join(ROOT, "error_recovery", "core", "lifecycle", "attempt_state.py")
    spec = importlib.util.spec_from_file_location("fr_attempt_state", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.is_error_phase


IS_ERROR_PHASE = _is_error_phase()


def tasks() -> List[str]:
    env_dir = os.path.join(ROOT, "envs")
    names = []
    for name in sorted(os.listdir(env_dir)):
        if name.startswith("_") or not name.endswith(".py"):
            continue
        names.append(name[:-3])
    return names


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if hasattr(value, "item"):
        try:
            return _text(value.item())
        except Exception:
            pass
    return "" if value is None else str(value)


def _video_frames(path: str) -> Optional[int]:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    counted = 0
    while True:
        ok, _frame = cap.read()
        if not ok:
            break
        counted += 1
    cap.release()
    return counted


def _load_frames(hdf5_path: str) -> Dict[str, Any]:
    import h5py

    with h5py.File(hdf5_path, "r") as handle:
        er = handle["error_recovery"]
        vector = handle["joint_action/vector"]
        return {
            "vector_frames": int(vector.shape[0]),
            "phase": [_text(item) for item in er["phase"][:]],
            "precise": [_text(item) for item in er["error_stage_precise"][:]],
            "frame_count": [int(item) for item in er["frame_count"][:]],
            "is_error": [bool(item) for item in er["is_error_frame"][:]],
            "is_precise": [bool(item) for item in er["is_error_frame_precise"][:]],
        }


def _episode_ids(root: str) -> List[int]:
    data_dir = os.path.join(root, "data")
    if not os.path.isdir(data_dir):
        return []
    ids = []
    for name in os.listdir(data_dir):
        if name.startswith("episode") and name.endswith(".hdf5"):
            ids.append(int(name[len("episode"):-len(".hdf5")]))
    return sorted(ids)


def check_episode(root: str, episode_id: int) -> Tuple[List[str], Dict[str, Any]]:
    errors: List[str] = []
    info = {"episode_id": episode_id, "kind": "unknown"}
    hdf5_path = os.path.join(root, "data", f"episode{episode_id}.hdf5")
    meta_path = os.path.join(root, "metadata", f"episode{episode_id}_metadata.json")
    video_path = os.path.join(root, "video", f"episode{episode_id}.mp4")
    try:
        frames = _load_frames(hdf5_path)
    except Exception as exc:
        return [f"episode {episode_id} hdf5 unreadable: {exc}"], info
    try:
        with open(meta_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
    except Exception as exc:
        return [f"episode {episode_id} metadata unreadable: {exc}"], info

    summary = metadata.get("episode_summary") or {}
    recovery = metadata.get("error_recovery_data") or {}
    attempts = [item for item in (recovery.get("error_metadata") or []) if isinstance(item, dict)]
    total = summary.get("total_frames")
    n = len(frames["phase"])
    if frames["vector_frames"] != n or total != n:
        errors.append(
            f"episode {episode_id} frame counts differ: hdf5={n} vector={frames['vector_frames']} metadata={total}"
        )
    if frames["frame_count"] != list(range(1, n + 1)):
        errors.append(f"episode {episode_id} frame_count is not 1..{n}")
    video_n = _video_frames(video_path)
    if video_n != n:
        errors.append(f"episode {episode_id} video frames {video_n} != hdf5 {n}")
    if "recovered" in frames["phase"]:
        errors.append(f"episode {episode_id} has per-frame phase recovered")
    for index, phase in enumerate(frames["phase"]):
        if frames["is_error"][index] != IS_ERROR_PHASE(phase):
            errors.append(f"episode {episode_id} frame {index} is_error_frame disagrees with phase {phase}")
            break
    for index, stage in enumerate(frames["precise"]):
        expected = stage not in PRECISE_OK
        if frames["is_precise"][index] != expected:
            errors.append(f"episode {episode_id} frame {index} precise flag disagrees with {stage}")
            break

    injected = bool(summary.get("error_injected"))
    end_status = attempts[-1].get("end_status") if attempts else None
    attempted = bool(summary.get("error_recovery_attempted"))
    if not injected:
        info["kind"] = "no_injection"
        if any(phase != "normal" for phase in frames["phase"]) or attempted:
            errors.append(f"episode {episode_id} has no injection but is not a normal trajectory")
    else:
        info["kind"] = end_status or "injected"
        if summary.get("final_phase") != end_status:
            errors.append(
                f"episode {episode_id} final_phase {summary.get('final_phase')} != end_status {end_status}"
            )
        if attempted != (end_status in {"recovered", "recovery_failed"}):
            errors.append(f"episode {episode_id} error_recovery_attempted does not match {end_status}")

    success = bool(summary.get("success"))
    if success:
        if summary.get("trim_detected_frame") is not None or summary.get("trimmed_total_frames") is not None:
            errors.append(f"episode {episode_id} succeeded but was trimmed")
        recovery_end = attempts[-1].get("recovery_end_frame") if attempts else None
        try:
            recovery_end_frame = int(recovery_end) if recovery_end is not None else None
        except (TypeError, ValueError):
            recovery_end_frame = None
        if recovery_end_frame is not None:
            for index, phase in enumerate(frames["phase"]):
                if frames["frame_count"][index] >= recovery_end_frame and phase != "normal":
                    errors.append(f"episode {episode_id} frame {frames['frame_count'][index]} phase is {phase} after recovery")
                    break
        elif frames["phase"] and frames["phase"][-1] != "normal":
            errors.append(f"episode {episode_id} success tail phase is {frames['phase'][-1]}")
    else:
        if summary.get("trim_detected_frame") != total:
            errors.append(
                f"episode {episode_id} failed trim {summary.get('trim_detected_frame')} != total {total}"
            )
        for attempt in attempts:
            for key in FRAME_KEYS:
                value = attempt.get(key)
                if value is None:
                    continue
                try:
                    frame = int(value)
                except (TypeError, ValueError):
                    errors.append(f"episode {episode_id} {key} is not an int")
                    continue
                if total is not None and frame > int(total):
                    errors.append(f"episode {episode_id} {key}={frame} past kept frames {total}")
    return errors, info


def _leftovers(root: str) -> List[str]:
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        base = os.path.basename(dirpath)
        if base in {".cache", ".trimmed_save_cache"} or base.endswith(".cache"):
            if any(os.listdir(dirpath)):
                found.append(dirpath)
        for name in filenames:
            if ".tmp" in name:
                found.append(os.path.join(dirpath, name))
    return found


def _saved_event_ids(root: str) -> Optional[List[int]]:
    events_dir = os.path.join(root, "events")
    if not os.path.isdir(events_dir):
        return None
    saved = set()
    for name in os.listdir(events_dir):
        if not name.endswith(".jsonl"):
            continue
        with open(os.path.join(events_dir, name), encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "attempt" and event.get("outcome") == "saved":
                    saved.add(int(event["episode_id"]))
    return sorted(saved)


def check_directory(root: str, expect: Optional[int], require_parallel_files: bool = True) -> List[str]:
    errors: List[str] = []
    py = sys.executable
    command = [py, os.path.join(ROOT, "error_recovery", "tools", "verify_dataset.py"), "--path", root]
    if expect is not None:
        command.extend(["--expect", str(expect)])
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        errors.append((result.stdout or result.stderr or "verify_dataset failed").strip())
    leftovers = _leftovers(root)
    if leftovers:
        errors.append(f"leftover files: {leftovers[:5]}")
    ids = _episode_ids(root)
    if not require_parallel_files:
        return errors

    stats_path = os.path.join(root, "collection_stats.json")
    if not os.path.exists(stats_path):
        errors.append("missing collection_stats.json")
        return errors
    with open(stats_path, encoding="utf-8") as handle:
        stats = json.load(handle)
    stat_ids = sorted(int(item["episode_id"]) for item in stats.get("episodes") or [] if isinstance(item, dict))
    if stat_ids != ids:
        errors.append(f"stats episodes {stat_ids} != files {ids}")
    summaries = []
    for episode_id in ids:
        path = os.path.join(root, "metadata", f"episode{episode_id}_metadata.json")
        with open(path, encoding="utf-8") as handle:
            summaries.append(json.load(handle).get("episode_summary") or {})
    total = len(summaries)
    successful = sum(1 for item in summaries if item.get("success"))
    injected = sum(1 for item in summaries if item.get("error_injected"))
    recovered = sum(1 for item in summaries if item.get("final_phase") == "recovered")
    expected_rates = {
        "total_episodes": total,
        "successful_episodes": successful,
        "error_injected_episodes": injected,
        "error_recovery_episodes": recovered,
        "success_rate": (successful / total) if total else 0.0,
        "error_recovery_rate": (recovered / injected) if injected else 0.0,
    }
    for key, expected in expected_rates.items():
        actual = stats.get(key)
        if isinstance(expected, float):
            if actual is None or not math.isclose(float(actual), expected, rel_tol=1e-6, abs_tol=1e-6):
                errors.append(f"stats {key}={actual} != {expected}")
        elif actual != expected:
            errors.append(f"stats {key}={actual} != {expected}")
    saved_ids = _saved_event_ids(root)
    if saved_ids is None:
        errors.append("missing events/")
    elif saved_ids != ids:
        errors.append(f"saved events {saved_ids} != files {ids}")
    return errors


def check_tree(root: str, expect: Optional[int], require_parallel_files: bool = True) -> Tuple[List[str], Dict[str, int]]:
    errors: List[str] = []
    counts = {"recovered": 0, "recovery_failed": 0, "no_injection": 0, "other": 0}
    for episode_id in _episode_ids(root):
        episode_errors, info = check_episode(root, episode_id)
        errors.extend(episode_errors)
        kind = info["kind"]
        if kind in counts:
            counts[kind] += 1
        else:
            counts["other"] += 1
    errors.extend(check_directory(root, expect, require_parallel_files))
    return errors, counts


def _crash_markers(root: str, supervisor_log: str) -> List[str]:
    markers = ("Segmentation fault", "core dumped", "code=75", "Traceback (most recent call last)")
    hits = []
    paths = []
    log_dir = os.path.join(root, "logs")
    if os.path.isdir(log_dir):
        paths.extend(os.path.join(log_dir, name) for name in os.listdir(log_dir))
    if os.path.exists(supervisor_log):
        paths.append(supervisor_log)
    for path in paths:
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for marker in markers:
            if marker in text and marker not in hits:
                # A caught episode error is logged and the worker keeps going.
                # Count a traceback as a crash only when the supervisor saw exit 75
                # or a segmentation fault. Plain tracebacks are recorded separately.
                if marker == "Traceback (most recent call last)":
                    continue
                hits.append(marker)
        if "Traceback (most recent call last)" in text and ("code=75" in text or "Segmentation fault" in text):
            if "traceback-with-crash" not in hits:
                hits.append("traceback-with-crash")
    return hits


def _kill_matching(token: str) -> None:
    needle = token.encode()
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            cmdline = open(f"/proc/{name}/cmdline", "rb").read()
        except OSError:
            continue
        if needle in cmdline:
            try:
                os.kill(int(name), signal.SIGKILL)
            except OSError:
                pass


def _kill_tree(pid: int) -> None:
    children = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            stat = open(f"/proc/{name}/stat", encoding="utf-8").read()
            ppid = int(stat.rsplit(")", 1)[-1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        if ppid == pid:
            children.append(int(name))
    for child in children:
        _kill_tree(child)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def _load_done(path: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("status") in DONE:
                done[(row["error"], row["task"])] = row
    return done


def _append(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _free_gb() -> float:
    stat = os.statvfs(ROOT)
    return stat.f_bavail * stat.f_frsize / (1024 ** 3)


def run_cell(
    task: str,
    error: str,
    save_path: str,
    error_types: str,
    episodes: int,
    gpus: str,
    timeout: int,
    core_offset: int = 0,
    core_count: int = 0,
) -> Dict[str, Any]:
    os.makedirs(save_path, exist_ok=True)
    log_path = os.path.join(save_path, "supervisor.log")
    command = [
        sys.executable,
        os.path.join(ROOT, "error_recovery", "collect_parallel.py"),
        "--task", task,
        "--episodes", str(episodes),
        "--gpus", gpus,
        "--workers_per_gpu", "2",
        "--seed_start", "0",
        "--save_path", save_path,
        "--error_types", error_types,
    ]
    if core_count > 0:
        command.extend(["--core_offset", str(core_offset), "--core_count", str(core_count)])
    started = time.time()
    with open(log_path, "ab", buffering=0) as log:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            exit_code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(proc.pid)
            _kill_matching(save_path)
            try:
                exit_code = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                exit_code = -9
    elapsed = round(time.time() - started, 1)
    return {"exit_code": exit_code, "elapsed_s": elapsed, "timed_out": timed_out, "log": log_path}


def classify(save_path: str, collect: Dict[str, Any], expect: int) -> Dict[str, Any]:
    errors, counts = check_tree(save_path, expect=None, require_parallel_files=False)
    ids = _episode_ids(save_path)
    crashes = _crash_markers(save_path, collect["log"])
    row = {
        "saved": len(ids),
        "elapsed_s": collect["elapsed_s"],
        "exit_code": collect["exit_code"],
        "counts": counts,
        "errors": errors[:8],
        "crashes": crashes,
    }
    if collect["timed_out"] or crashes or (collect["exit_code"] not in (0, None) and len(ids) >= expect):
        row["status"] = "timeout" if collect["timed_out"] else "crash"
        return row
    if len(ids) < expect:
        row["status"] = "insufficient"
        return row
    dir_errors = check_directory(save_path, expect, require_parallel_files=True)
    row["errors"] = (errors + dir_errors)[:12]
    row["status"] = "label_error" if row["errors"] else "pass"
    return row


def cmd_run(args: argparse.Namespace) -> int:
    free = _free_gb()
    if free < 40:
        print(f"disk free {free:.1f} GB is below 40 GB", file=sys.stderr)
        return 1
    matrix = os.path.join(args.root, "matrix")
    os.makedirs(matrix, exist_ok=True)
    jsonl = os.path.join(args.root, "matrix.jsonl")
    done = _load_done(jsonl)
    pending = [(error, task) for error in ERROR_TYPES for task in tasks()]
    gpu_list = [part.strip() for part in args.gpus.split(",") if part.strip()]
    usable_cores = max(len(gpu_list), (os.cpu_count() or 1) - 8)
    per_gpu = max(2, usable_cores // len(gpu_list))
    print(
        f"cells {len(pending)} done {len(done)} disk_free_gb {free:.0f} "
        f"gpus {','.join(gpu_list)} cores_per_gpu {per_gpu}",
        flush=True,
    )
    work: queue.Queue = queue.Queue()
    for index, (error, task) in enumerate(pending, start=1):
        if (error, task) not in done:
            work.put((index, error, task))
    write_lock = threading.Lock()
    none_lock = threading.Lock()

    def finish_one(index: int, error: str, task: str, gpu: str, core_offset: int) -> None:
        save_path = os.path.join(matrix, error, task)
        with write_lock:
            print(f"[{index}/{len(pending)}] gpu={gpu} {error} {task}", flush=True)
        collect = run_cell(
            task, error, save_path, f"{error}:1.0", args.episodes, gpu, args.timeout, core_offset, per_gpu,
        )
        row = {"error": error, "task": task, "path": save_path, "gpu": gpu}
        row.update(classify(save_path, collect, args.episodes))
        if row["status"] == "insufficient":
            none_path = os.path.join(args.root, "none_check", task)
            with none_lock:
                if len(_episode_ids(none_path)) >= args.episodes:
                    none = {"timed_out": False}
                else:
                    none = run_cell(
                        task, "none", none_path, "none", args.episodes, gpu, args.timeout, core_offset, per_gpu,
                    )
                none_saved = len(_episode_ids(none_path))
            row["none_saved"] = none_saved
            row["task_unstable"] = none_saved < args.episodes and not none["timed_out"]
        with write_lock:
            _append(jsonl, row)
            print(f"  gpu={gpu} {error} {task} {row['status']} saved={row['saved']} {row['elapsed_s']}s", flush=True)

    def gpu_worker(slot: int, gpu: str) -> None:
        while True:
            try:
                index, error, task = work.get_nowait()
            except queue.Empty:
                return
            try:
                finish_one(index, error, task, gpu, slot * per_gpu)
            except Exception as exc:
                with write_lock:
                    _append(jsonl, {"error": error, "task": task, "status": "crash", "errors": [str(exc)], "saved": 0})
                    print(f"  gpu={gpu} {error} {task} crash {exc}", flush=True)
            finally:
                work.task_done()

    threads = [
        threading.Thread(target=gpu_worker, args=(slot, gpu), name=f"gpu-{gpu}")
        for slot, gpu in enumerate(gpu_list)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    cmd_report(args)
    return 0


def _latest_rows(path: str) -> List[Dict[str, Any]]:
    rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[(row.get("error"), row.get("task"))] = row
    return list(rows.values())


def cmd_report(args: argparse.Namespace) -> int:
    rows = _latest_rows(os.path.join(args.root, "matrix.jsonl"))
    by_key = {(row["error"], row["task"]): row for row in rows}
    lines = ["# Full matrix", ""]
    problems = [row for row in rows if row.get("status") not in {"pass"}]
    lines.append(f"Cells {len(rows)}, not passed {len(problems)}.")
    lines.append("")
    header = "| Task | " + " | ".join(ERROR_TYPES) + " |"
    lines.append(header)
    lines.append("| --- | " + " | ".join("---" for _ in ERROR_TYPES) + " |")
    for task in tasks():
        cells = []
        for error in ERROR_TYPES:
            row = by_key.get((error, task))
            if row is None:
                cells.append("not run")
                continue
            counts = row.get("counts") or {}
            cells.append(
                f"{row.get('status')} recovered {counts.get('recovered', 0)}/failed {counts.get('recovery_failed', 0)}/not_injected {counts.get('no_injection', 0)} {row.get('elapsed_s', '')}s"
            )
        lines.append("| " + task + " | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## By error type")
    lines.append("")
    lines.append("| Error type | Cells | Recovered | Recovery failed | Not injected | Recovery rate |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for error in ERROR_TYPES:
        group = [row for row in rows if row.get("error") == error]
        recovered = sum((row.get("counts") or {}).get("recovered", 0) for row in group)
        failed = sum((row.get("counts") or {}).get("recovery_failed", 0) for row in group)
        none = sum((row.get("counts") or {}).get("no_injection", 0) for row in group)
        denom = recovered + failed
        rate = f"{recovered / denom:.2f}" if denom else "-"
        lines.append(f"| {error} | {len(group)} | {recovered} | {failed} | {none} | {rate} |")
    if problems:
        lines.append("")
        lines.append("## Not passed")
        lines.append("")
        for row in problems:
            unstable = " task unstable" if row.get("task_unstable") else ""
            lines.append(
                f"- {row.get('error')} / {row.get('task')}: {row.get('status')}{unstable} saved={row.get('saved')} {row.get('errors') or row.get('crashes')}"
            )
    dest = os.path.join(args.root, "matrix_report.md")
    os.makedirs(args.root, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote {dest}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    errors, counts = check_tree(args.path, args.expect, require_parallel_files=not args.episodes_only)
    print(json.dumps({"counts": counts, "errors": errors}, ensure_ascii=False, indent=2))
    return 1 if errors else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Failure-recovery full matrix")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--root", default=os.path.join(ROOT, "error_recovery", "data", "_fr_full"))
    run.add_argument("--episodes", type=int, default=2)
    run.add_argument("--gpus", default="0,1,2,3")
    run.add_argument("--timeout", type=int, default=CELL_TIMEOUT_S)
    run.set_defaults(func=cmd_run)
    report = sub.add_parser("report")
    report.add_argument("--root", default=os.path.join(ROOT, "error_recovery", "data", "_fr_full"))
    report.set_defaults(func=cmd_report)
    check = sub.add_parser("check")
    check.add_argument("--path", required=True)
    check.add_argument("--expect", type=int, default=None)
    check.add_argument("--episodes-only", action="store_true")
    check.set_defaults(func=cmd_check)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
