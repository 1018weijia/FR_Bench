"""
Error-recovery helpers, with support for multiple tasks.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import shutil
import tempfile
import h5py
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from envs.utils.pkl2hdf5 import process_folder_to_hdf5_video


DEFAULT_RECOVERY_MODE = "inplace"
DEFAULT_RECOVERY_EXECUTION_MODE = "qpos_action"
DEFAULT_RECOVERY_QPOS_ACTION_STRIDE = 16
DEFAULT_RECOVERY_VIDEO_SAMPLE_MODE = "sim"
DEFAULT_RECOVERY_VIDEO_SAMPLE_FREQ = 50
DEFAULT_RECOVERY_PLANNING_HOLD_FRAMES = 8
DEFAULT_RECOVERY_THRESHOLD_TYPE = "p95"
DEFAULT_RECOVERY_FORCE_OPEN_GRIPPERS = False


@dataclass(frozen=True)
class EpisodeSaveLayout:
    """Standard eval run layout: data/video at timestamp root; recovery/ and fail/ on outcome."""

    eval_save_dir: str

    @staticmethod
    def episode_name(episode_index: int) -> str:
        return f"episode{episode_index}"

    def episode_data_path(self, episode_index: int) -> str:
        return os.path.join(self.eval_save_dir, "data", f"{self.episode_name(episode_index)}.hdf5")

    def episode_video_path(self, episode_index: int) -> str:
        return os.path.join(self.eval_save_dir, "video", f"{self.episode_name(episode_index)}.mp4")

    def episode_cache_dir(self, episode_index: int) -> str:
        return os.path.join(self.eval_save_dir, ".cache", self.episode_name(episode_index))

    def recovery_root(self) -> str:
        return os.path.join(self.eval_save_dir, "recovery")

    def recovery_data_path(self, episode_index: int) -> str:
        return os.path.join(self.recovery_root(), "data", f"{self.episode_name(episode_index)}.hdf5")

    def recovery_video_path(self, episode_index: int) -> str:
        return os.path.join(self.recovery_root(), "video", f"{self.episode_name(episode_index)}.mp4")

    def recovery_cache_dir(self, episode_index: int, variant: int) -> str:
        return os.path.join(
            self.recovery_root(),
            ".cache",
            f"{self.episode_name(episode_index)}_variant{variant}",
        )

    def fail_root(self) -> str:
        return os.path.join(self.eval_save_dir, "fail")

    def fail_data_path(self, episode_index: int) -> str:
        return os.path.join(self.fail_root(), "data", f"{self.episode_name(episode_index)}.hdf5")

    def fail_video_path(self, episode_index: int) -> str:
        return os.path.join(self.fail_root(), "video", f"{self.episode_name(episode_index)}.mp4")

    def ensure_rollout_dirs(self) -> None:
        os.makedirs(os.path.join(self.eval_save_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(self.eval_save_dir, "video"), exist_ok=True)

    def ensure_recovery_dirs(self) -> None:
        os.makedirs(os.path.join(self.recovery_root(), "data"), exist_ok=True)
        os.makedirs(os.path.join(self.recovery_root(), "video"), exist_ok=True)

    def ensure_fail_dirs(self) -> None:
        os.makedirs(os.path.join(self.fail_root(), "data"), exist_ok=True)
        os.makedirs(os.path.join(self.fail_root(), "video"), exist_ok=True)


def _sorted_pkl_names(cache_dir: str) -> list[str]:
    if not cache_dir or not os.path.isdir(cache_dir):
        return []
    files: list[tuple[int, str]] = []
    for fname in os.listdir(cache_dir):
        if fname.endswith(".pkl") and fname[:-4].isdigit():
            files.append((int(fname[:-4]), fname))
    files.sort()
    return [fname for _, fname in files]


def count_cache_frames(cache_dir: str) -> int:
    return len(_sorted_pkl_names(cache_dir))


def write_hdf5_attrs(hdf5_path: str, attrs: dict[str, Any]) -> None:
    with h5py.File(hdf5_path, "a") as f:
        for key, value in attrs.items():
            f.attrs[key] = value


def merge_cache_dirs_to_hdf5_video(
    cache_dirs: list[str],
    hdf5_path: str,
    video_path: str,
) -> int:
    """Merge multiple pkl cache dirs into one continuous hdf5 + mp4."""
    merged_root = tempfile.mkdtemp(prefix="merged_episode_")
    frame_count = 0
    try:
        for cache_dir in cache_dirs:
            for fname in _sorted_pkl_names(cache_dir):
                shutil.copy2(
                    os.path.join(cache_dir, fname),
                    os.path.join(merged_root, f"{frame_count}.pkl"),
                )
                frame_count += 1
        if frame_count == 0:
            raise FileNotFoundError("No pkl frames to merge")
        os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        process_folder_to_hdf5_video(merged_root, hdf5_path, video_path)
        return frame_count
    finally:
        shutil.rmtree(merged_root, ignore_errors=True)


def save_merged_recovery_episode(
    layout: EpisodeSaveLayout,
    episode_index: int,
    rollout_cache_dir: str,
    recovery_cache_dir: str,
) -> tuple[str, str, int]:
    """Save rollout + recovery as one continuous trajectory under recovery/."""
    layout.ensure_recovery_dirs()
    hdf5_path = layout.recovery_data_path(episode_index)
    video_path = layout.recovery_video_path(episode_index)
    rollout_frame_count = count_cache_frames(rollout_cache_dir)
    recovery_frame_count = count_cache_frames(recovery_cache_dir)
    frame_count = merge_cache_dirs_to_hdf5_video(
        [rollout_cache_dir, recovery_cache_dir],
        hdf5_path,
        video_path,
    )
    write_hdf5_attrs(
        hdf5_path,
        {
            "contains_recovery": True,
            "recovery_success": True,
            "recovery_start_frame": rollout_frame_count,
            "rollout_frame_count": rollout_frame_count,
            "recovery_frame_count": recovery_frame_count,
            "total_frame_count": frame_count,
        },
    )
    return hdf5_path, video_path, frame_count


def remove_episode_cache_dir(cache_dir: str) -> None:
    if not cache_dir:
        return

    cache_dir = cache_dir.rstrip(os.sep)
    cache_root = os.path.dirname(cache_dir)
    cache_prefix = os.path.basename(cache_dir)

    try:
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)

        # Older recovery runs could leave files like .cache/episode3_variant114.pkl
        # when Base_Task._take_picture concatenated a cache path without a trailing slash.
        if os.path.isdir(cache_root):
            for fname in os.listdir(cache_root):
                if fname.startswith(cache_prefix) and fname.endswith(".pkl"):
                    try:
                        os.remove(os.path.join(cache_root, fname))
                    except Exception:
                        pass

        if os.path.basename(cache_root) == ".cache" and os.path.isdir(cache_root) and not os.listdir(cache_root):
            os.rmdir(cache_root)
    except Exception:
        pass


@dataclass(frozen=True)
class TaskFrameThreshold:
    value: int
    source: str


def _threshold_value(thresholds: dict, threshold_type: str) -> Optional[int]:
    value = thresholds.get(threshold_type)
    if value is None:
        return None
    return int(value)


def load_task_frame_threshold_info(task_name: str, threshold_type: str = "p99") -> Optional[TaskFrameThreshold]:
    """
    Load the frame-count threshold for a task.

    Prefer repo_root/task_lim, then look under eval_result, and finally fall back to the built-in full table.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    task_lim_dir = os.path.join(repo_root, "task_lim")
    threshold_file = os.path.join(task_lim_dir, f"{task_name}.json")

    if os.path.exists(threshold_file):
        try:
            with open(threshold_file, "r") as f:
                data = json.load(f)

            value = _threshold_value(data.get("thresholds", {}), threshold_type)
            if value is not None:
                return TaskFrameThreshold(value=value, source=threshold_file)
        except Exception as e:
            print(f"Warning: Failed to load threshold from {threshold_file}: {e}")

    eval_root = os.environ.get("ROBOTWIN_EVAL_ROOT", os.path.join(repo_root, "eval_result"))
    task_eval_root = os.path.join(eval_root, task_name)
    latest_stats_file = None
    latest_stats_mtime = -1.0
    if os.path.exists(task_eval_root):
        for root, _, files in os.walk(task_eval_root):
            if "success_video_stats.json" not in files:
                continue
            stats_file = os.path.join(root, "success_video_stats.json")
            stats_mtime = os.path.getmtime(stats_file)
            if stats_mtime > latest_stats_mtime:
                latest_stats_file = stats_file
                latest_stats_mtime = stats_mtime

    if latest_stats_file is not None:
        try:
            with open(latest_stats_file, "r") as f:
                stats = json.load(f)

            value = _threshold_value(stats, threshold_type)
            if value is not None:
                return TaskFrameThreshold(value=value, source=latest_stats_file)
        except Exception as e:
            print(f"Warning: Failed to load stats from {latest_stats_file}: {e}")

    all_thresholds_file = os.path.join(
        os.path.dirname(__file__),
        "all_task_thresholds.json",
    )
    if os.path.exists(all_thresholds_file):
        try:
            with open(all_thresholds_file, "r") as f:
                all_thresholds = json.load(f)
            for item in all_thresholds:
                if item.get("task_name") != task_name:
                    continue
                value = _threshold_value(item.get("thresholds", {}), threshold_type)
                if value is not None:
                    return TaskFrameThreshold(value=value, source=all_thresholds_file)
        except Exception as e:
            print(f"Warning: Failed to load thresholds from {all_thresholds_file}: {e}")

    return None


def load_task_frame_threshold(task_name: str, threshold_type: str = "p99") -> Optional[int]:
    threshold = load_task_frame_threshold_info(task_name, threshold_type)
    if threshold is None:
        return None
    return threshold.value


def count_play_once_stages(task_class) -> int:
    """
    Detect the number of stages in play_once by counting self.move() calls.
    """
    if not hasattr(task_class, "play_once"):
        return 1

    try:
        source = inspect.getsource(task_class.play_once)

        move_count = 0
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if (
                        isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "self"
                        and node.func.attr == "move"
                    ):
                        move_count += 1
        except (SyntaxError, IndentationError):
            move_count = source.count("self.move(")
        except Exception:
            move_count = source.count("self.move(")

        return max(1, move_count)

    except Exception as e:
        print(f"Warning: Failed to analyze play_once stages for {task_class.__name__}: {e}")
        return 3


def get_all_actor_poses(env) -> Dict[str, Any]:
    poses: Dict[str, Any] = {}

    common_patterns = [
        "block1",
        "block2",
        "block3",
        "block4",
        "bottle",
        "bottle1",
        "bottle2",
        "bottles",
        "hammer",
        "tray",
        "pot",
        "microwave",
        "deskbin",
        "bigbin",
        "microphone",
        "mic",
        "roller",
        "screwdriver",
        "dumbbell",
        "phone",
        "can",
        "stapler",
        "pad",
        "bowl",
        "basket",
        "container",
        "plate",
        "bread",
        "skillet",
        "mouse",
        "laptop",
        "bell",
        "alarmclock",
        "mug",
        "pillbottle",
        "shoe",
        "shoes",
        "burger",
        "fries",
        "fan",
        "playingcard",
        "qrcode",
        "switch",
        "seal",
        "box",
        "target_box",
    ]

    for pattern in common_patterns:
        if hasattr(env, pattern):
            actor = getattr(env, pattern)
            if hasattr(actor, "get_pose"):
                try:
                    poses[f"{pattern}_pose"] = actor.get_pose()
                except Exception:
                    pass

    if hasattr(env, "scene"):
        try:
            for entity in env.scene.get_all_entities():
                if hasattr(entity, "get_pose"):
                    name = entity.get_name() if hasattr(entity, "get_name") else None
                    if name and name not in ["table", "ground", "robot"]:
                        name_key = f"{name}_pose"
                        if name_key not in poses:
                            try:
                                poses[name_key] = entity.get_pose()
                            except Exception:
                                pass
        except Exception:
            pass

    try:
        for attr_name in dir(env):
            if attr_name.startswith("_") or attr_name in ["scene", "robot", "table", "ground"]:
                continue
            if attr_name in common_patterns:
                continue
            try:
                attr_obj = getattr(env, attr_name)
                if hasattr(attr_obj, "get_pose") and hasattr(attr_obj, "actor"):
                    pose_key = f"{attr_name}_pose"
                    if pose_key not in poses:
                        poses[pose_key] = attr_obj.get_pose()
            except Exception:
                pass
    except Exception:
        pass

    return poses


def restore_all_actor_poses(env, poses: Dict[str, Any]) -> None:
    restored_count = 0
    for key, pose in poses.items():
        if not key.endswith("_pose"):
            continue
        actor_name = key[:-5]

        if hasattr(env, actor_name):
            actor = getattr(env, actor_name)
            restored = False
            if hasattr(actor, "actor") and hasattr(actor.actor, "set_pose"):
                try:
                    actor.actor.set_pose(pose)
                    restored = True
                    restored_count += 1
                except Exception:
                    pass
            elif hasattr(actor, "set_pose"):
                try:
                    actor.set_pose(pose)
                    restored = True
                    restored_count += 1
                except Exception:
                    pass

            if restored:
                print(f"\033[94m[Recovery] Restored {actor_name} pose\033[0m")

    if restored_count == 0 and hasattr(env, "scene"):
        try:
            for key, pose in poses.items():
                if not key.endswith("_pose"):
                    continue
                entity_name = key[:-5]
                for entity in env.scene.get_all_entities():
                    if hasattr(entity, "get_name"):
                        name = entity.get_name()
                        if name == entity_name or name.replace("_", "") == entity_name.replace("_", ""):
                            if hasattr(entity, "set_pose"):
                                try:
                                    entity.set_pose(pose)
                                    print(
                                        f"\033[94m[Recovery] Restored {entity_name} pose via scene\033[0m"
                                    )
                                    restored_count += 1
                                except Exception:
                                    pass
        except Exception:
            pass

    if restored_count > 0:
        print(f"\033[94m[Recovery] Successfully restored {restored_count} actor pose(s)\033[0m")
    else:
        print(f"\033[93m[Recovery] Warning: Failed to restore any actor poses from snapshot\033[0m")

