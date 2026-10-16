"""Inspect and analyze failure-recovery collection results.

The data directory is the output of one collection run, for example ./error_recovery/data/lift_pot.
"""

import os
import sys
import json
from collections import defaultdict
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import h5py
import numpy as np


def get_episode_hdf5_path(data_dir: str, episode_id: int) -> str:
    """Return data/episode{id}.hdf5."""
    return os.path.join(data_dir, "data", f"episode{episode_id}.hdf5")


def open_episode_file(data_dir: str, episode_id: int) -> Tuple[str, Optional[h5py.File]]:
    """
    Open the hdf5 file for the given episode.

    Returns (hdf5_path, file_obj). file_obj is None if the file does not exist.
    The caller must close file_obj when finished.
    """
    hdf5_path = get_episode_hdf5_path(data_dir, episode_id)
    if not os.path.exists(hdf5_path):
        return hdf5_path, None
    f = h5py.File(hdf5_path, "r")
    return hdf5_path, f


def _decode_value(value):
    if isinstance(value, np.ndarray):
        value = value.item() if value.shape == () else value
    if isinstance(value, np.generic) and not isinstance(value, np.bytes_):
        value = value.item()
    if isinstance(value, bytes) and not value.startswith(b"\xff\xd8"):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value
    return value


def _as_bool(value) -> bool:
    decoded = _decode_value(value)
    if isinstance(decoded, str):
        return decoded.strip().lower() in {"1", "true", "yes"}
    return bool(decoded)


def frame_count(h5_file: h5py.File) -> int:
    """The current HDF5 stores frames as datasets. Frame count is the length of error_recovery or the camera sequence."""
    recovery = h5_file.get("error_recovery")
    if isinstance(recovery, h5py.Group) and "phase" in recovery:
        return int(recovery["phase"].shape[0])
    rgb = h5_file.get("observation/head_camera/rgb")
    if isinstance(rgb, h5py.Dataset):
        return int(rgb.shape[0])
    return 0


def get_sorted_frame_ids(h5_file: h5py.File) -> List[int]:
    """Return frame ids in ascending order."""
    return list(range(frame_count(h5_file)))


def _load_frame(h5_file: h5py.File, frame_id: int) -> Dict:
    frame_data: Dict = {}
    recovery = h5_file.get("error_recovery")
    if isinstance(recovery, h5py.Group):
        info = {}
        for key, dataset in recovery.items():
            if not isinstance(dataset, h5py.Dataset) or frame_id >= dataset.shape[0]:
                continue
            value = _decode_value(dataset[frame_id])
            if key in {"enabled", "is_error_frame", "is_error_frame_precise"}:
                value = _as_bool(value)
            info[key] = value
        frame_data["error_recovery"] = info

    rgb = h5_file.get("observation/head_camera/rgb")
    if isinstance(rgb, h5py.Dataset) and frame_id < rgb.shape[0]:
        raw = _decode_value(rgb[frame_id])
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        if isinstance(raw, (bytes, bytearray)):
            encoded = np.frombuffer(bytes(raw).rstrip(b"\0"), dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is not None:
                frame_data["head_rgb"] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return frame_data


def iter_frames(
    h5_file: h5py.File,
) -> Generator[Tuple[int, Dict], None, None]:
    """
    Iterate over every frame in the episode.

    Yields:
        (frame_id, frame_data_dict)
    """
    for frame_id in get_sorted_frame_ids(h5_file):
        yield frame_id, _load_frame(h5_file, frame_id)


def get_error_recovery_info(frame_data: Dict) -> Optional[Dict]:
    """
    Extract error_recovery info from one frame.

    Returns None if it is missing or not enabled.
    """
    if "error_recovery" not in frame_data:
        return None
    error_info = frame_data["error_recovery"]
    if not isinstance(error_info, dict):
        return None
    if not error_info.get("enabled", False):
        return None
    return error_info


def list_episodes(data_dir: str) -> List[int]:
    """
    List episode IDs under the data directory and return them in ascending order.

    Reads episode*.hdf5 under data/.
    """
    data_path = os.path.join(data_dir, "data")

    if not os.path.exists(data_path):
        print(f"❌ Data directory does not exist: {data_path}")
        return []

    hdf5_files = sorted(f for f in os.listdir(data_path) if f.endswith(".hdf5"))

    if len(hdf5_files) == 0:
        print("❌ No data files found")
        return []

    print(f"\n{'=' * 80}")
    print(f"📁 Found {len(hdf5_files)} episodes")
    print(f"{'=' * 80}\n")

    episode_ids: List[int] = []
    for hdf5_file in hdf5_files:
        episode_id = int(hdf5_file.replace("episode", "").replace(".hdf5", ""))
        episode_ids.append(episode_id)

        hdf5_path = os.path.join(data_path, hdf5_file)
        with h5py.File(hdf5_path, "r") as f:
            print(f"Episode {episode_id}: {frame_count(f)} frames")

    return episode_ids


# =========================
# Logic for visualizing a single episode
# =========================

def view_episode(data_dir: str, episode_id: int) -> None:
    """View one trajectory frame by frame."""
    hdf5_path, f = open_episode_file(data_dir, episode_id)

    if f is None:
        print(f"❌ Episode {episode_id} does not exist: {hdf5_path}")
        return

    print(f"\n{'=' * 80}")
    print(f"📹 Episode {episode_id} visualization")
    print(f"{'=' * 80}\n")

    with f:
        frame_ids = get_sorted_frame_ids(f)

        print(f"Total frames: {len(frame_ids)}")
        print(f"Press 'q' to quit, space to pause/resume, 's' to save the current frame\n")

        paused = False
        frame_idx = 0

        while frame_idx < len(frame_ids):
            frame_id = frame_ids[frame_idx]
            frame_data = _load_frame(f, frame_id)

            # Get the image
            if "head_rgb" in frame_data:
                image = frame_data["head_rgb"].copy()
            elif "wrist_rgb" in frame_data:
                image = frame_data["wrist_rgb"].copy()
            else:
                print(f"⚠️  Frame {frame_id} has no image data")
                frame_idx += 1
                continue

            # Convert to BGR for OpenCV display
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image

            # Add an information overlay
            h, w = image_bgr.shape[:2]
            overlay = image_bgr.copy()

            # Semi-transparent background
            cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
            image_bgr = cv2.addWeighted(overlay, 0.6, image_bgr, 0.4, 0)

            # Frame number
            cv2.putText(
                image_bgr,
                f"Frame: {frame_id}/{len(frame_ids) - 1}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            # Error-recovery info
            error_info = get_error_recovery_info(frame_data)
            if error_info is not None:
                phase = error_info["phase"]
                phase_colors = {
                    "normal": (0, 255, 0),
                    "error_attempt": (0, 0, 255),
                    "error_detected": (0, 165, 255),
                    "recovery": (0, 255, 255),
                    "recovery_grasp": (255, 0, 0),
                    "recovered": (255, 255, 0),
                    "recovery_failed": (0, 0, 128),
                }
                color = phase_colors.get(phase, (255, 255, 255))

                cv2.putText(
                    image_bgr,
                    f"Phase: {phase}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

                if error_info.get("error_type"):
                    cv2.putText(
                        image_bgr,
                        f"Error: {error_info['error_type']}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

                if error_info.get("attempt_count", 0) > 0:
                    cv2.putText(
                        image_bgr,
                        f"Attempt: {error_info['attempt_count']}",
                        (w - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                    )

                if error_info.get("is_error_frame"):
                    cv2.putText(
                        image_bgr,
                        "ERROR FRAME",
                        (w - 200, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

            # Display
            cv2.imshow(f"Episode {episode_id}", image_bgr)

            # Keyboard control
            if paused:
                key = cv2.waitKey(0)
            else:
                key = cv2.waitKey(50)  # 20 FPS

            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused
            elif key == ord("s"):
                save_path = f"episode{episode_id}_frame{frame_id}.png"
                cv2.imwrite(save_path, image_bgr)
                print(f"💾 Saved: {save_path}")
            elif key == ord("n"):  # next frame
                frame_idx += 1
            elif key == ord("p"):  # previous frame
                frame_idx = max(0, frame_idx - 1)
            else:
                frame_idx += 1

        cv2.destroyAllWindows()


def run_view_episode_cli(default_data_dir: str = "./error_recovery/data/lift_pot") -> None:
    """
    View ./error_recovery/data/lift_pot.

    - list: list episodes
    - <ID>: view one episode
    - with no arguments, list episodes first and then choose one
    """
    data_dir = default_data_dir

    if len(sys.argv) > 1:
        if sys.argv[1] == "list":
            list_episodes(data_dir)
            return
        else:
            episode_id = int(sys.argv[1])
            view_episode(data_dir, episode_id)
            return

    # List episodes and let the user choose
    episode_ids = list_episodes(data_dir)

    if not episode_ids:
        return

    print(f"\nSelect an episode to view (0-{max(episode_ids)}): ", end="")
    try:
        episode_id = int(input().strip())
        if episode_id in episode_ids:
            view_episode(data_dir, episode_id)
        else:
            print(f"❌ Invalid episode ID: {episode_id}")
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except ValueError:
        print("❌ Please enter a number")


# =========================
# Failure-analysis logic
# =========================

def analyze_episode(data_dir: str, episode_id: int) -> None:
    """Analyze the phase distribution of one trajectory. The conclusion follows metadata end_status."""
    hdf5_path, f = open_episode_file(data_dir, episode_id)

    if f is None:
        print(f"❌ Episode {episode_id} does not exist: {hdf5_path}")
        return

    print(f"\n{'=' * 80}")
    print(f"🔍 Episode {episode_id} failure analysis")
    print(f"{'=' * 80}\n")

    phases: List[str] = []
    error_frames: List[int] = []
    recovery_attempts: List[Dict] = []

    with f:
        frame_ids = get_sorted_frame_ids(f)

        print(f"Total frames: {len(frame_ids)}\n")

        for frame_id in frame_ids:
            frame_data = _load_frame(f, frame_id)

            error_info = get_error_recovery_info(frame_data)
            if error_info is None:
                continue

            phase = error_info["phase"]
            phases.append(phase)

            if error_info.get("is_error_frame"):
                error_frames.append(frame_id)

            if error_info.get("attempt_count", 0) > 0:
                recovery_attempts.append(
                    {
                        "frame": frame_id,
                        "phase": phase,
                        "attempt": error_info["attempt_count"],
                        "error_type": error_info.get("error_type"),
                    }
                )

    if not phases:
        print("⚠️  No error_recovery info detected")
        print(f"\n{'=' * 80}")
        return

    # Phase distribution
    phase_counts: Dict[str, int] = defaultdict(int)
    for phase in phases:
        phase_counts[phase] += 1

    print("📊 Phase distribution:")
    for phase, count in sorted(phase_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {phase:20s}: {count:4d} frames ({count / len(phases) * 100:.1f}%)")

    print(f"\n🔴 Error frames: {len(error_frames)}")
    if error_frames:
        print(f"   First error frame: {error_frames[0]}")
        print(f"   Last error frame: {error_frames[-1]}")

    print(
        f"\n🔄 Recovery attempts: "
        f"{len(set(r['attempt'] for r in recovery_attempts)) if recovery_attempts else 0}"
    )

    if recovery_attempts:
        print("\nRecovery process:")
        for r in recovery_attempts[:10]:  # Show only the first 10
            print(
                f"  Frame {r['frame']}: Attempt {r['attempt']}, "
                f"Phase={r['phase']}, Error={r['error_type']}"
            )

        if len(recovery_attempts) > 10:
            print(f"  ... {len(recovery_attempts) - 10} more recovery frames")

    # Check for recovery_failed
    if "recovery_failed" in phase_counts:
        print(f"\n❌ Recovery failed: {phase_counts['recovery_failed']} frames")
        print("\nPossible failure causes:")
        print("  1. Object position was computed incorrectly")
        print("  2. Grasp pose was wrong")
        print("  3. Collision detection problem")
        print("  4. Gripper parameters were wrong")

    if "recovered" in phase_counts:
        print(f"\n✅ Recovery succeeded: {phase_counts['recovered']} frames")

    print(f"\n{'=' * 80}")


def analyze_all_episodes(data_dir: str) -> None:
    """Read collection_stats.json and summarize this collection run."""
    # Read the stats file
    stats_path = os.path.join(data_dir, "collection_stats.json")

    if not os.path.exists(stats_path):
        print(f"❌ Stats file does not exist: {stats_path}")
        return

    with open(stats_path, "r") as f:
        stats = json.load(f)

    print(f"\n{'=' * 80}")
    print(f"📊 Global statistics")
    print(f"{'=' * 80}\n")

    print(f"Total episodes: {stats['total_episodes']}")
    print(
        f"Succeeded: {stats['successful_episodes']} "
        f"({stats['successful_episodes'] / stats['total_episodes'] * 100:.1f}%)"
    )
    print(
        f"Failed: {stats['total_episodes'] - stats['successful_episodes']} "
        f"({(stats['total_episodes'] - stats['successful_episodes']) / stats['total_episodes'] * 100:.1f}%)"
    )
    print(f"Errors injected: {stats['error_injected_episodes']}")
    print(f"Recoveries succeeded: {stats['error_recovery_episodes']}")

    print(f"\nPer-episode details:")
    print(
        f"{'ID':<6} {'Success':<8} {'Injected':<10} "
        f"{'Attempted':<10} {'Attempts':<10} {'Error type':<15}"
    )
    print("-" * 80)

    for ep in stats["episodes"]:
        error_types = ep.get("error_types")
        if isinstance(error_types, list):
            error_type = ",".join(str(item) for item in error_types) or "None"
        else:
            error_type = str(ep.get("error_type") or "None")
        print(
            f"{ep['episode_id']:<6} "
            f"{'✓' if ep['success'] else '✗':<8} "
            f"{'✓' if ep['error_injected'] else '✗':<10} "
            f"{'✓' if ep.get('error_recovery_attempted', False) else '✗':<10} "
            f"{ep.get('attempt_count', 0):<10} "
            f"{error_type:<15}"
        )

    print(f"\n{'=' * 80}")

    # Failure-cause analysis
    print(f"\n💡 Failure-cause analysis:")

    failed_without_error = sum(
        1
        for ep in stats["episodes"]
        if not ep["success"] and not ep["error_injected"]
    )
    failed_with_error = sum(
        1
        for ep in stats["episodes"]
        if not ep["success"] and ep["error_injected"]
    )

    if failed_without_error > 0:
        print(f"\n  ❌ {failed_without_error} episodes failed without an injected error")
        print("     → The base task itself has a problem (grasp/place failed)")

    if failed_with_error > 0:
        print(f"\n  ❌ {failed_with_error} episodes failed after an injected error")
        print("     → Possible causes:")
        print("        1. Error-recovery logic has a problem")
        print("        2. The grasp pose after recovery was wrong")
        print("        3. The object position update was delayed")

    print(f"\n🔧 Suggestions:")
    print("  1. Use --error_types none to see whether the base task fails on its own")
    print("  2. Read the recovery conclusion from end_status in metadata; per-frame phase is only the stage at collection time")
    print("  3. Adjust grasp and recovery parameters in the corresponding injector")

    print(f"\n{'=' * 80}")


def run_analyze_failure_cli(default_data_dir: str = "./error_recovery/data/lift_pot") -> None:
    """
    Read one failure-recovery collection directory. The default is ./error_recovery/data/lift_pot.

    - no arguments: analyze every episode in that directory
    - all: same as above
    - <ID>: analyze one episode
    """
    data_dir = default_data_dir

    if len(sys.argv) > 1:
        if sys.argv[1] == "all":
            analyze_all_episodes(data_dir)
        else:
            episode_id = int(sys.argv[1])
            analyze_episode(data_dir, episode_id)
    else:
        analyze_all_episodes(data_dir)

        print("\nTips:")
        print("  run_analyze_failure_cli()             # Analyze every episode in the default directory")
        print("  run_view_episode_cli()                # View trajectories in the same directory")
