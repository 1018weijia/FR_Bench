"""Medium-grain support helpers for ErrorRecoveryFramework."""

from __future__ import annotations

import gc
import os
import shutil
import tempfile
import importlib
import time
import logging
from typing import Any, Dict, List, Optional, Type

from .data_collector import ConfigManager, DataCollector
from .error_injectors import create_error_injector
from .lifecycle import backfill_fake_error_detected, close_open_attempt, is_error_phase
from .logging_utils import normalize_log_level, should_log
from .task_wrapper import TaskWrapper


def resolve_task(task_class: Optional[Type], task_name: Optional[str]) -> tuple:
    """Resolve the task class and task name from either input form."""
    if task_class is not None:
        if task_name is None:
            task_name = task_class.__name__
        return task_class, task_name

    if task_name is not None:
        try:
            module = importlib.import_module(f"envs.{task_name}")
            task_class = getattr(module, task_name)
            return task_class, task_name
        except (ImportError, AttributeError) as exc:
            raise ValueError(f"Unable to import task '{task_name}': {exc}")

    raise ValueError("Either task_class or task_name must be provided")


def create_error_injectors_for_framework(
    framework: Any,
    error_types: List[str],
    error_probabilities: Optional[Dict[str, float]],
    error_params: Optional[Dict[str, Dict[str, Any]]],
    logger: Any,
) -> list:
    """Create injectors while preserving the existing framework semantics."""
    error_injectors = []

    for error_type in error_types:
        if error_type not in framework.supported_error_types:
            logger.warning("[ErrorRecoveryFramework] Unsupported error type: %s, skipping", error_type)
            continue

        probability = error_probabilities.get(error_type, 0.5) if error_probabilities else 0.5
        kwargs = {"probability": probability}
        if error_params and error_type in error_params:
            kwargs.update(error_params[error_type])

        injector = create_error_injector(error_type, **kwargs)
        error_injectors.append(injector)

        extra = ", ".join(f"{key}: {value}" for key, value in (error_params or {}).get(error_type, {}).items())
        logger.info(
            "[ErrorRecoveryFramework] Created error injector: %s (probability: %s%s)",
            error_type,
            probability,
            f", {extra}" if extra else "",
        )

    if not error_injectors:
        logger.info("[ErrorRecoveryFramework] Error injection is not enabled; only normal data will be collected")

    return error_injectors


def load_and_setup_config(task_name: str, save_path: str, render: bool, **kwargs) -> Dict[str, Any]:
    """Build the collection config with the same defaults as before."""
    config = ConfigManager.load_task_config(task_name)
    config = ConfigManager.setup_embodiment_config(config)
    config.update(
        {
            "save_path": save_path,
            "save_data": True,
            "save_freq": int(config.get("save_freq", 15)),
            "render_freq": 10 if render else 0,
            "data_type": {
                "rgb": True,
                "depth": False,
                "qpos": True,
                "endpose": True,
            },
        }
    )
    scene = kwargs.pop("scene", "clean")
    config.update(kwargs)
    config["scene_profile"] = scene
    randomization = ConfigManager.scene_randomization(scene)
    if randomization is not None:
        config["domain_randomization"] = randomization
    return config


_WORKER_CONFIG_KEYS = {
    "seed_start",
    "seed_end",
    "episode_offset",
    "worker_id",
    "events_path",
    "fail_fast",
    "max_episodes_per_process",
    "clear_cache_freq",
}


def execute_data_collection(
    task_class: Type,
    task_name: str,
    error_injectors: List[Any],
    config: Dict[str, Any],
    episodes: int,
    save_path: str,
    logger: Any,
) -> Dict[str, Any]:
    """Run the collection loop with the same high-level workflow."""
    log_level = normalize_log_level(config.get("error_recovery_log_level", "quiet"))
    if log_level == "quiet":
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.getLogger().setLevel(logging.INFO)
        logger.info(
            "Starting error-recovery data collection | task=%s | episodes=%s | error_types=%s | seed_start=%s | save_path=%s",
            task_name,
            episodes,
            [injector.name for injector in error_injectors],
            config.get("seed_start", 0),
            save_path,
        )
    os.environ["ROBOTWIN_VERBOSE_VIDEO_SAVE"] = "1" if log_level == "debug" else "0"

    worker_id = config.get("worker_id")
    data_collector = DataCollector(save_path, task_name, worker_id=worker_id)
    if log_level == "quiet":
        logging.getLogger().setLevel(logging.WARNING)
    try:
        from envs.utils.create_actor import UnStableError
    except Exception:
        UnStableError = None

    from .worker_runtime import (
        EXIT_RESTART,
        append_event,
        apply_thread_limits,
        detect_planner_backend,
        is_cuda_fatal,
        is_planner_failure,
        read_rss_mb,
        read_vram_mb,
    )

    apply_thread_limits()
    max_errors_per_episode = 1
    task_wrapper = TaskWrapper(task_class, error_injectors, max_errors_per_episode=max_errors_per_episode)
    task_wrapper.error_recovery_log_level = log_level
    save_data = bool(config.get("save_data"))
    trim_failure_trajectory_on_save = bool(config.get("trim_failure_trajectory_on_save", True))
    render_enabled = bool(config.get("render_freq"))
    max_attempts = episodes * 5
    seed_start = int(config.get("seed_start", 0))
    if seed_start < 0:
        raise ValueError(f"seed_start must be non-negative, got {seed_start}")
    seed_end = config.get("seed_end")
    if seed_end is not None:
        seed_end = int(seed_end)
        if seed_end < seed_start:
            raise ValueError(f"seed_end must be >= seed_start, got {seed_end} < {seed_start}")
    episode_offset = int(config.get("episode_offset") or 0)
    if episode_offset < 0:
        raise ValueError(f"episode_offset must be non-negative, got {episode_offset}")
    max_per_process = int(config.get("max_episodes_per_process") or 0)
    fail_fast = bool(config.get("fail_fast"))
    events_path = config.get("events_path") or None
    raw_clear_cache_freq = config.get("clear_cache_freq", 10)
    clear_cache_freq = 10 if raw_clear_cache_freq is None else int(raw_clear_cache_freq)
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES")
    demo_config = {
        key: value
        for key, value in config.items()
        if key not in _WORKER_CONFIG_KEYS
    }
    actual_ep = 0
    attempted_seed = 0
    process_saved = 0
    collection_status_active = False

    def _print_collection_status(message: str) -> None:
        nonlocal collection_status_active
        print(f"\r\033[K{message}", end="", flush=True)
        collection_status_active = True

    def _clear_collection_status() -> None:
        nonlocal collection_status_active
        if collection_status_active:
            print("\r\033[K", end="", flush=True)
            collection_status_active = False

    while actual_ep < episodes:
        if max_per_process and process_saved >= max_per_process:
            logger.info("Reached this process limit of %s episodes, exiting so it can be restarted", max_per_process)
            break
        ep = episode_offset + actual_ep
        seed = seed_start + attempted_seed
        if seed_end is not None and seed >= seed_end:
            _clear_collection_status()
            logger.warning("Seed range exhausted (%s), finished %s/%s", seed_end, actual_ep, episodes)
            break
        task = None
        _print_collection_status(f"[Collect] Episode {ep} ({actual_ep + 1}/{episodes}) | Seed={seed}")

        success = False
        unstable_skip = False
        plan_fail_skip = False
        discard_failed_episode = False
        abort_process = False
        outcome = "crash"
        attempt_started = time.time()
        append_event(
            events_path,
            {
                "type": "start",
                "worker_id": worker_id,
                "gpu": gpu_id,
                "pid": os.getpid(),
                "episode_id": ep,
                "seed": seed,
            },
        )

        try:
            data_collector.start_episode(ep, task_wrapper)
            task_wrapper.collection_seed = seed
            task = task_wrapper.create_wrapped_task()
            task.setup_demo(now_ep_num=ep, seed=seed, **demo_config)
            wrap_get_obs(task, task_wrapper)
            task.play_once()
            close_open_attempt(task_wrapper)

            plan_success = getattr(task, "plan_success", True)
            if not plan_success:
                recovery_error = getattr(task_wrapper, "last_recovery_error", None)
                status = (
                    f"recovery failed: step={recovery_error.get('step')}/{recovery_error.get('total')} "
                    f"{recovery_error.get('action_type')}, skip"
                    if recovery_error
                    else f"Path planning failed, skipping (error_injection_count={task_wrapper.error_injection_count})"
                )
                _print_collection_status(
                    f"[Collect] Episode {ep}/{episodes} | Seed={seed} {status}"
                )
                plan_fail_skip = True
                outcome = "plan_fail"
                if recovery_error and save_data:
                    trim_info = data_collector.compute_failure_trim_info(
                        task_wrapper.get_error_recovery_data(),
                        success=False,
                    )
                    cache_info = _inspect_failure_cache(task, trim_info)
                    if _can_save_plan_failure(cache_info):
                        discard_failed_episode = True
                        try:
                            save_result = save_episode_outputs(
                                task,
                                task_wrapper,
                                trim_info,
                                trim_failure_trajectory_on_save,
                                False,
                                logger,
                            )
                            if save_result == "saved":
                                discard_failed_episode = False
                                plan_fail_skip = False
                                outcome = "saved"
                                _clear_collection_status()
                                print(f"💾 Saved failed recovery episode {ep}: hdf5+video")
                            else:
                                outcome = "discard"
                                logger.warning(
                                    "⚠️  Episode %s failed to save the failed-recovery trajectory slice; discarding this seed and continuing",
                                    ep,
                                )
                        except Exception as save_exc:
                            outcome = "discard"
                            logger.error("❌ Episode %s failed while saving failed-recovery data: %s", ep, save_exc)
                        if hasattr(task, "remove_data_cache"):
                            task.remove_data_cache()
                    else:
                        _log_failure_cache_diagnostics(logger, cache_info)
            else:
                check_success = getattr(task, "check_success", None)
                success = check_success() if check_success else True
                _clear_collection_status()
                recovery_state = _summarize_recovery_state(task_wrapper)
                print(
                    f"{'✅' if success else '❌'} Episode {ep} {'succeeded' if success else 'failed'} "
                    f"seed={seed} frames={getattr(task_wrapper, 'frame_count', 0)} {recovery_state}"
                )
                if save_data:
                    trim_info = data_collector.compute_failure_trim_info(
                        task_wrapper.get_error_recovery_data(),
                        success,
                    )
                    if should_log(log_level, "normal"):
                        logger.info(f"💾 Saving episode {ep} data...")
                    try:
                        save_result = save_episode_outputs(
                            task,
                            task_wrapper,
                            trim_info,
                            trim_failure_trajectory_on_save,
                            success,
                            logger,
                        )
                        if save_result == "discard":
                            discard_failed_episode = True
                            outcome = "discard"
                            logger.warning(
                                f"⚠️  Episode {ep} is a failed trajectory, but it cannot be trimmed and saved by error_detected_frame; "
                                "discarded this seed and will try the next seed to produce a trajectory with the same index"
                            )
                    except Exception as save_exc:
                        discard_failed_episode = True
                        outcome = "discard"
                        logger.error("❌ Episode %s failed while saving data: %s", ep, save_exc)
                    if hasattr(task, "remove_data_cache"):
                        task.remove_data_cache()
                    if not discard_failed_episode:
                        outcome = "saved"
                        print(f"💾 Saved episode {ep}: hdf5+video")
                else:
                    outcome = "saved"
        except Exception as exc:
            success = False
            if UnStableError and isinstance(exc, UnStableError):
                _print_collection_status(f"[Collect] Episode {ep}/{episodes} | Seed={seed} object unstable, skipping")
                unstable_skip = True
                outcome = "unstable"
            else:
                planner_failure = is_planner_failure(exc)
                cuda_failure = is_cuda_fatal(exc)
                if fail_fast and (planner_failure or cuda_failure):
                    abort_process = True
                    outcome = "crash"
                    discard_failed_episode = True
                    _clear_collection_status()
                    logger.error("Planner or CUDA entered an unrecoverable state, exiting process: %s", exc)
                elif planner_failure:
                    _print_collection_status(
                        f"[Collect] Episode {ep}/{episodes} | Seed={seed} planner initialization error, skipping "
                        f"({type(exc).__name__}: {exc})"
                    )
                    unstable_skip = True
                    outcome = "unstable"
                    discard_failed_episode = True
                else:
                    _clear_collection_status()
                    logger.exception("❌ Episode %s failed", ep)
                    discard_failed_episode = True
                    outcome = "crash"
        finally:
            effective_plan_fail_skip = plan_fail_skip
            planner_backend = detect_planner_backend(task) if task is not None else None
            _finalize_episode(
                task,
                data_collector,
                ep,
                task_wrapper,
                success,
                unstable_skip,
                effective_plan_fail_skip,
                discard_failed_episode,
                render_enabled,
                logger,
            )
            append_event(
                events_path,
                {
                    "type": "attempt",
                    "worker_id": worker_id,
                    "gpu": gpu_id,
                    "pid": os.getpid(),
                    "episode_id": ep,
                    "seed": seed,
                    "outcome": outcome,
                    "planner_backend": planner_backend,
                    "frames": getattr(task_wrapper, "frame_count", None),
                    "elapsed_s": round(time.time() - attempt_started, 3),
                    "rss_mb": read_rss_mb() if events_path else None,
                    "vram_mb": read_vram_mb() if events_path else None,
                    "error_types": _event_error_types(task_wrapper),
                    "recovery": _event_recovery(task_wrapper),
                    "success": bool(success) and outcome == "saved",
                },
            )
            task = None
            gc.collect()
            if clear_cache_freq and (attempted_seed + 1) % clear_cache_freq == 0:
                _clear_sapien_cache(logger)

        if abort_process:
            _clear_collection_status()
            data_collector.save_collection_stats()
            os._exit(EXIT_RESTART)

        if _should_count_episode(unstable_skip, effective_plan_fail_skip, discard_failed_episode):
            actual_ep += 1
            process_saved += 1
        attempted_seed += 1
        if seed_end is None and attempted_seed > max_attempts:
            _clear_collection_status()
            logger.warning("\n⚠️  Warning: tried %s seeds, but only finished %s episodes", attempted_seed, actual_ep)
            logger.warning("Scene setup may be problematic, causing many seeds to be unstable")
            break

    _clear_collection_status()
    data_collector.save_collection_stats()
    return data_collector.get_stats()


def wrap_get_obs(task: Any, task_wrapper: Any) -> None:
    """Attach error recovery metadata to task observations."""
    if not hasattr(task, "get_obs"):
        return

    original_get_obs = task.get_obs
    time_time = time.time
    update_frame_count = task_wrapper.update_frame_count
    get_error_recovery = task_wrapper.get_error_recovery_data
    sanitize_meta_item = getattr(task_wrapper, "sanitize_error_metadata_item", None)

    def wrapped_get_obs(*args, **kwargs):
        base_obs = original_get_obs(*args, **kwargs) or {}
        update_frame_count()
        error_recovery = get_error_recovery()
        meta_list = error_recovery.get("error_metadata") or []
        last_meta = meta_list[-1] if meta_list else None
        if last_meta is not None:
            last_meta = sanitize_meta_item(last_meta) if sanitize_meta_item is not None else task_wrapper.sanitize_error_recovery_data({"error_metadata": [last_meta]}).get("error_metadata", [None])[-1]

        payload = {
            "enabled": True,
            "phase": error_recovery["current_phase"],
            "error_stage_precise": error_recovery.get("error_stage_precise", "normal"),
            "frame_count": error_recovery["frame_count"],
            "timestamp": time_time(),
            "is_error_frame": is_error_phase(error_recovery["current_phase"]),
            "is_error_frame_precise": bool(error_recovery.get("is_error_frame_precise", False)),
            "active_injectors": error_recovery["active_injectors"],
            "error_metadata": last_meta,
        }
        if isinstance(base_obs, dict):
            base_obs["error_recovery"] = payload
        return base_obs

    task.get_obs = wrapped_get_obs


def _recovery_status(task_wrapper: Any) -> str:
    metadata = getattr(task_wrapper, "error_metadata", None) or []
    if not metadata:
        return "none"
    last = metadata[-1]
    status = last.get("end_status") if isinstance(last, dict) else None
    if status == "recovered":
        return "success"
    if status == "recovery_failed":
        return "failed"
    if status:
        return str(status)
    return "attempted"


def _summarize_recovery_state(task_wrapper: Any) -> str:
    metadata = getattr(task_wrapper, "error_metadata", None) or []
    if not metadata:
        return "injected=none recovery=none"

    last = metadata[-1]
    injector = last.get("injector", "unknown") if isinstance(last, dict) else "unknown"
    if hasattr(injector, "name"):
        injector = injector.name
    return f"injected={injector} recovery={_recovery_status(task_wrapper)}"


def _event_error_types(task_wrapper: Any) -> List[str]:
    names = []
    for item in getattr(task_wrapper, "error_metadata", None) or []:
        if not isinstance(item, dict):
            continue
        injector = item.get("injector", "unknown")
        if hasattr(injector, "name"):
            injector = injector.name
        names.append(str(injector))
    return list(dict.fromkeys(names))


def _event_recovery(task_wrapper: Any) -> str:
    return _recovery_status(task_wrapper)


def _clear_sapien_cache(logger: Any) -> None:
    try:
        from sapien.render import clear_cache as sapien_clear_cache

        sapien_clear_cache()
    except Exception as exc:
        logger.warning("Failed to clear SAPIEN cache: %s", exc)


def save_episode_outputs(
    task: Any,
    task_wrapper: Any,
    trim_info: Dict[str, Any],
    trim_failure_trajectory_on_save: bool,
    success: bool,
    logger: Any,
) -> str:
    """Save one episode, or discard untrimmable failed samples when trim mode is strict."""
    if success or not trim_failure_trajectory_on_save:
        task.merge_pkl_to_hdf5_video()
        return "saved"

    trim_info = _resolve_trim_info_for_failed_episode(task_wrapper, trim_info, logger)
    if trim_info.get("trim_detected_frame") is not None:
        return "saved" if save_trimmed_failure_output(task, trim_info, logger) else "discard"

    logger.warning("⚠️  This failed trajectory has no usable error_detected_frame; refusing to write the untrimmed failed trajectory into the dataset")
    return "discard"


def _resolve_trim_info_for_failed_episode(task_wrapper: Any, trim_info: Dict[str, Any], logger: Any) -> Dict[str, Any]:
    if trim_info.get("trim_detected_frame") is not None or not backfill_fake_error_detected(task_wrapper):
        return trim_info

    logger.warning("⚠️  This failed trajectory lacks a real error_detected_frame; backfilled a fake error_detected_frame for trimming")
    return DataCollector.compute_failure_trim_info(task_wrapper.get_error_recovery_data(), success=False)


def _should_count_episode(unstable_skip: bool, plan_fail_skip: bool, discard_failed_episode: bool) -> bool:
    return not (unstable_skip or plan_fail_skip or discard_failed_episode)


def _remove_if_exists(*paths: str) -> None:
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def _resolve_failure_cache_path(task: Any) -> Dict[str, Any]:
    cache_path = getattr(task, "folder_path", {}).get("cache")
    fallback_cache_path = None
    if getattr(task, "save_dir", None) is not None and getattr(task, "ep_num", None) is not None:
        fallback_cache_path = os.path.join(task.save_dir, ".cache", f"episode{task.ep_num}")

    resolved_cache_path = cache_path
    if (not resolved_cache_path or not os.path.exists(resolved_cache_path)) and fallback_cache_path and os.path.exists(fallback_cache_path):
        resolved_cache_path = fallback_cache_path

    pkl_count = 0
    if resolved_cache_path and os.path.exists(resolved_cache_path):
        try:
            pkl_count = sum(1 for name in os.listdir(resolved_cache_path) if name.endswith(".pkl"))
        except Exception:
            pkl_count = 0

    return {
        "cache_path": resolved_cache_path,
        "cache_exists": bool(resolved_cache_path and os.path.exists(resolved_cache_path)),
        "fallback_cache_path": fallback_cache_path,
        "fallback_exists": bool(fallback_cache_path and os.path.exists(fallback_cache_path)),
        "frame_idx": getattr(task, "FRAME_IDX", None),
        "pkl_count": pkl_count,
    }


def _inspect_failure_cache(task: Any, trim_info: Dict[str, Any]) -> Dict[str, Any]:
    cache_info = _resolve_failure_cache_path(task)
    cache_info.update(
        {
            "trim_keep_cache_idx": trim_info.get("trim_keep_cache_idx"),
            "trim_detected_frame": trim_info.get("trim_detected_frame"),
        }
    )
    return cache_info


def _can_save_plan_failure(cache_info: Dict[str, Any]) -> bool:
    frame_idx = cache_info.get("frame_idx")
    try:
        has_frames = int(frame_idx) > 0
    except Exception:
        has_frames = False
    return bool(
        cache_info.get("cache_exists")
        and cache_info.get("pkl_count", 0) > 0
        and has_frames
        and cache_info.get("trim_keep_cache_idx") is not None
    )


def _log_failure_cache_diagnostics(logger: Any, cache_info: Dict[str, Any]) -> None:
    logger.warning(
        "⚠️  Trim-save conditions are incomplete, dropping this failed trajectory | "
        "cache_path=%s cache_exists=%s fallback_cache_path=%s fallback_exists=%s "
        "keep_cache_idx=%s FRAME_IDX=%s trim_detected_frame=%s pkl_count=%s",
        cache_info.get("cache_path"),
        cache_info.get("cache_exists"),
        cache_info.get("fallback_cache_path"),
        cache_info.get("fallback_exists"),
        cache_info.get("trim_keep_cache_idx"),
        cache_info.get("frame_idx"),
        cache_info.get("trim_detected_frame"),
        cache_info.get("pkl_count"),
    )


def save_trimmed_failure_output(task: Any, trim_info: Dict[str, Any], logger: Any) -> bool:
    """Trim a failed trajectory and save it to the standard data/video output location."""
    cache_info = _inspect_failure_cache(task, trim_info)
    cache_path = cache_info.get("cache_path")
    keep_cache_idx = cache_info.get("trim_keep_cache_idx")
    if not cache_path or keep_cache_idx is None or not cache_info.get("cache_exists"):
        _log_failure_cache_diagnostics(logger, cache_info)
        return False

    try:
        from envs.utils.pkl2hdf5 import process_folder_to_hdf5_video
    except Exception as exc:
        logger.warning(f"⚠️  Could not import the trim-save tool, dropping this failed trajectory: {exc}")
        return False

    target_data_dir = os.path.join(task.save_dir, "data")
    target_video_dir = os.path.join(task.save_dir, "video")
    os.makedirs(target_data_dir, exist_ok=True)
    os.makedirs(target_video_dir, exist_ok=True)

    temp_root = os.path.join(task.save_dir, ".trimmed_save_cache", f"pid{os.getpid()}")
    os.makedirs(temp_root, exist_ok=True)
    target_hdf5_path = os.path.join(target_data_dir, f"episode{task.ep_num}.hdf5")
    target_video_path = os.path.join(target_video_dir, f"episode{task.ep_num}.mp4")
    tmp_hdf5_path = os.path.join(target_data_dir, f"episode{task.ep_num}.tmp.hdf5")
    tmp_video_path = os.path.join(target_video_dir, f"episode{task.ep_num}.tmp.mp4")
    preview_cache_dir = None
    copied = 0
    try:
        preview_cache_dir = tempfile.mkdtemp(prefix=f"episode{task.ep_num}_", dir=temp_root)
        for file_name in sorted(os.listdir(cache_path)):
            if not file_name.endswith(".pkl"):
                continue
            try:
                frame_idx = int(os.path.splitext(file_name)[0])
            except Exception:
                continue
            if frame_idx > keep_cache_idx:
                continue
            shutil.copy2(
                os.path.join(cache_path, file_name),
                os.path.join(preview_cache_dir, file_name),
            )
            copied += 1

        if copied == 0:
            logger.warning("⚠️  No cached frames available for trim-save, dropping this failed trajectory")
            return False

        _remove_if_exists(target_hdf5_path, target_video_path, tmp_hdf5_path, tmp_video_path)

        process_folder_to_hdf5_video(preview_cache_dir, tmp_hdf5_path, tmp_video_path)
        os.replace(tmp_hdf5_path, target_hdf5_path)
        if os.path.exists(tmp_video_path):
            os.replace(tmp_video_path, target_video_path)
        logger.info(
            f"✂️  Trimmed official output at the failure-detection frame: episode{task.ep_num} "
            f"(detected_frame={trim_info.get('trim_detected_frame')}, kept_frames={copied})"
        )
        return True
    except Exception as exc:
        logger.warning(f"⚠️  Trim-save failed, dropping this failed trajectory: {exc}")
        try:
            _remove_if_exists(target_hdf5_path, target_video_path, tmp_hdf5_path, tmp_video_path)
        except Exception as cleanup_exc:
            logger.debug("Error cleaning up trim output files", exc_info=cleanup_exc)
        return False
    finally:
        try:
            if preview_cache_dir and os.path.exists(preview_cache_dir):
                shutil.rmtree(preview_cache_dir)
        except Exception as cleanup_exc:
            logger.debug("Error cleaning up the trim temporary directory", exc_info=cleanup_exc)
        try:
            if os.path.exists(temp_root) and not os.listdir(temp_root):
                os.rmdir(temp_root)
        except Exception as cleanup_exc:
            logger.debug("Error cleaning up the trim temporary root directory", exc_info=cleanup_exc)


def _close_episode_task(task: Any, render_enabled: bool, logger: Any, remove_cache: bool = False) -> None:
    if task is None:
        return
    wrapper = getattr(task, "_error_recovery_wrapper", None)
    if wrapper is not None and hasattr(wrapper, "detach"):
        try:
            wrapper.detach(task)
        except Exception as exc:
            logger.warning(f"Error detaching the task wrapper: {exc}")
    if remove_cache and hasattr(task, "remove_data_cache"):
        try:
            task.remove_data_cache()
        except Exception as exc:
            logger.warning(f"Error clearing the episode cache: {exc}")
    try:
        task.close()
    except Exception as exc:
        logger.warning(f"Error closing the task: {exc}")
    if render_enabled and hasattr(task, "viewer") and task.viewer:
        try:
            task.viewer.close()
        except Exception as exc:
            logger.warning(f"Error closing the viewer: {exc}")


def _finalize_episode(
    task: Any,
    data_collector: DataCollector,
    ep: int,
    task_wrapper: Any,
    success: bool,
    unstable_skip: bool,
    plan_fail_skip: bool,
    discard_failed_episode: bool,
    render_enabled: bool,
    logger: Any,
) -> None:
    should_count = _should_count_episode(unstable_skip, plan_fail_skip, discard_failed_episode)
    has_error_metadata = bool(getattr(task_wrapper, "error_metadata", None))
    should_save_metadata = should_count and (getattr(task_wrapper, "frame_count", 0) > 0 or has_error_metadata)

    if should_save_metadata:
        try:
            data_collector.end_episode(ep, task, task_wrapper, success)
        except Exception as exc:
            logger.error(f"Error cleaning up the episode: {exc}")

    _close_episode_task(task, render_enabled, logger, remove_cache=not should_count)
