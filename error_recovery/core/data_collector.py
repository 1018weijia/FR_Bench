"""
Data collector - unified interface for collecting error-recovery data

This module provides a unified data-collection interface that can collect
error-recovery data for any task and automatically add error-recovery labels and metadata.
"""

import os
import json
import numpy as np
import copy
from typing import Dict, List, Any, Optional
from datetime import datetime
import yaml
import logging
import sys

# Configure logging (base config; overridden by DataCollector)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DATA_SUBDIRS = ("data", "video", "metadata", "logs")
_TERMINAL_STATUSES = {"recovered", "recovery_failed", "no_failure"}
_ATTEMPTED_STATUSES = {"recovered", "recovery_failed"}
_ATTEMPT_FRAME_KEYS = (
    "inject_decision_frame",
    "error_onset_frame",
    "error_detected_frame",
    "recovery_start_frame",
    "recovery_end_frame",
    "error_end_frame",
)


def _episode_id(episode: Any) -> Optional[int]:
    if not isinstance(episode, dict) or episode.get("episode_id") is None:
        return None
    try:
        return int(episode["episode_id"])
    except (TypeError, ValueError):
        return None


def _attempt_end_status(error_metadata: List[Dict[str, Any]]) -> Optional[str]:
    if not error_metadata or not isinstance(error_metadata[-1], dict):
        return None
    status = error_metadata[-1].get("end_status")
    return status if isinstance(status, str) else None


def _drop_unsaved_attempt_frames(error_metadata: List[Any], kept_frames: int) -> None:
    """Clear attempt timestamps that point past the frames kept after a failure trim."""
    for item in error_metadata:
        if not isinstance(item, dict):
            continue
        for key in _ATTEMPT_FRAME_KEYS:
            value = item.get(key)
            if value is None:
                continue
            try:
                frame = int(value)
            except (TypeError, ValueError):
                continue
            if frame > kept_frames:
                item[key] = None


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class DataCollector:
    """Error-recovery data collector"""
    
    def __init__(self, save_path: str, task_name: str, worker_id: Optional[str] = None):
        self.save_path = save_path
        self.task_name = task_name
        self.worker_id = worker_id
        self.pid = os.getpid()
        self.gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
        self.episode_count = 0
        self.collection_stats = {
            'task_name': task_name,
            'save_path': save_path,
            'start_time': datetime.now().isoformat(),
            'total_episodes': 0,
            'successful_episodes': 0,
            'error_recovery_episodes': 0,
            'error_injected_episodes': 0,
            'episodes': []
        }
        
        # Create the save directory
        self._setup_directories()
        
        # Configure log file output
        self._setup_logging()
    
    def _setup_directories(self):
        """Set up the save directory structure"""
        for directory in (self.save_path, *(os.path.join(self.save_path, subdir) for subdir in DATA_SUBDIRS)):
            os.makedirs(directory, exist_ok=True)

    @staticmethod
    def _remove_previous_data_collector_handlers(root_logger: logging.Logger) -> None:
        for handler in list(root_logger.handlers):
            if isinstance(handler, logging.FileHandler) and getattr(handler, "_robotwin_data_collector_handler", False):
                try:
                    root_logger.removeHandler(handler)
                finally:
                    try:
                        handler.close()
                    except Exception as exc:
                        logger.debug("Failed to close the previous DataCollector log handler", exc_info=exc)

    @staticmethod
    def _has_console_handler(root_logger: logging.Logger) -> bool:
        return any(
            isinstance(handler, logging.StreamHandler)
            and not isinstance(handler, logging.FileHandler)
            and getattr(handler, "stream", None) in (sys.stdout, sys.stderr)
            for handler in root_logger.handlers
        )

    def _setup_logging(self):
        """Configure log file output
        
        Write logs to both the console and a file under the logs/ directory
        """
        logs_dir = os.path.join(self.save_path, "logs")
        
        # Create the log file name (a timestamp avoids overwriting)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        worker_tag = f"_{self.worker_id}" if self.worker_id else ""
        log_file = os.path.join(logs_dir, f"collection_{timestamp}{worker_tag}_p{self.pid}.log")
        
        # Get the root logger (captures logs from every module)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # Avoid accumulating FileHandlers on the root logger when DataCollector is
        # created repeatedly in the same process, which would write the same logs
        # multiple times (slower) and scatter them across several files.
        # Only remove file handlers added by DataCollector, so user and third-party
        # logging configuration is left alone.
        self._remove_previous_data_collector_handlers(root_logger)

        # Create the file log handler (marked as added by DataCollector so it can be replaced or removed later)
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler._robotwin_data_collector_handler = True  # type: ignore[attr-defined]
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Make sure there is a console log handler (if there is not one yet).
        # Note: logging.basicConfig adds a StreamHandler to sys.stderr by default.
        # Avoid adding a second console handler just because a check only looks at
        # stdout, which would duplicate terminal output.
        if not self._has_console_handler(root_logger):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        # Record the log file path
        logger.debug(f"📝 Log file saved to: {log_file}")
        
        # Save the log file path into collection_stats
        self.collection_stats['log_file'] = log_file
    
    def start_episode(self, episode_id: int, task_wrapper) -> Dict[str, Any]:
        """Start data collection for a new episode"""
        self.episode_count = episode_id
        
        # Reset the task wrapper
        task_wrapper.reset_for_new_episode()
        
        # Create episode metadata
        episode_metadata = {
            'episode_id': episode_id,
            'task_name': self.task_name,
            'start_time': datetime.now().isoformat(),
            'error_injectors': [inj.name for inj in task_wrapper.error_injectors],
            'error_injection_probabilities': {
                inj.name: inj.probability for inj in task_wrapper.error_injectors
            }
        }
        
        return episode_metadata
    
    def end_episode(self, episode_id: int, task, task_wrapper, success: bool) -> Dict[str, Any]:
        """Finish the episode and collect the final data"""
        self._current_seed = getattr(task_wrapper, "collection_seed", None)
        try:
            from .worker_runtime import detect_planner_backend

            self._planner_backend = detect_planner_backend(task)
        except Exception:
            self._planner_backend = None
        raw_error_recovery_data = task_wrapper.get_error_recovery_data()
        trim_info = self.compute_failure_trim_info(raw_error_recovery_data, success)
        error_recovery_data = self._prepare_metadata_error_recovery_data(raw_error_recovery_data, trim_info, success)
        episode_summary = self._build_episode_summary(
            episode_id,
            error_recovery_data,
            trim_info,
            success,
        )
        self._update_stats(episode_summary)
        self._save_episode_metadata(episode_id, episode_summary, error_recovery_data)
        return episode_summary

    @staticmethod
    def compute_failure_trim_info(error_recovery_data: Dict[str, Any], success: bool) -> Dict[str, Any]:
        """Compute a suggested trim point for a failed trajectory, used to build comparison preview videos."""
        info = {
            'trim_detected_frame': None,
            'trim_keep_cache_idx': None,
            'trimmed_total_frames': None,
        }
        if success:
            return info

        detected_frames = DataCollector._collect_detected_frames(error_recovery_data)
        if not detected_frames:
            return info

        first_detected_frame = min(detected_frames)
        keep_cache_idx = max(0, first_detected_frame - 1)
        info.update({
            'trim_detected_frame': first_detected_frame,
            'trim_keep_cache_idx': keep_cache_idx,
            'trimmed_total_frames': keep_cache_idx + 1,
        })
        return info

    @staticmethod
    def _collect_detected_frames(error_recovery_data: Dict[str, Any]) -> List[int]:
        detected_frames: List[int] = []
        for item in error_recovery_data.get('error_metadata') or []:
            if not isinstance(item, dict):
                continue
            detected_frame = item.get('error_detected_frame')
            if detected_frame is None:
                continue
            try:
                frame = int(detected_frame)
            except (TypeError, ValueError):
                continue
            if frame >= 1:
                detected_frames.append(frame)
        return detected_frames
    
    def _recount_stats(self) -> None:
        """Derive totals and rates from the episode list."""
        episodes = [
            episode for episode in self.collection_stats.get("episodes") or []
            if isinstance(episode, dict)
        ]
        stats = self.collection_stats
        stats["total_episodes"] = len(episodes)
        stats["successful_episodes"] = sum(1 for episode in episodes if episode.get("success"))
        stats["error_injected_episodes"] = sum(1 for episode in episodes if episode.get("error_injected"))
        stats["error_recovery_episodes"] = sum(1 for episode in episodes if episode.get("final_phase") == "recovered")
        total_episodes = max(stats["total_episodes"], 1)
        injected_episodes = max(stats["error_injected_episodes"], 1)
        stats["success_rate"] = stats["successful_episodes"] / total_episodes
        stats["error_recovery_rate"] = stats["error_recovery_episodes"] / injected_episodes

    def _update_stats(self, episode_summary: Dict[str, Any]) -> None:
        """Update collection statistics"""
        self.collection_stats["episodes"].append(episode_summary)
        self._recount_stats()

    def _build_episode_summary(
        self,
        episode_id: int,
        error_recovery_data: Dict[str, Any],
        trim_info: Dict[str, Any],
        success: bool,
    ) -> Dict[str, Any]:
        error_metadata = error_recovery_data['error_metadata']
        end_status = _attempt_end_status(error_metadata)
        return {
            'episode_id': episode_id,
            'task_name': self.task_name,
            'success': success,
            'error_injected': bool(error_metadata),
            'error_recovery_attempted': end_status in _ATTEMPTED_STATUSES,
            'final_phase': end_status if end_status in _TERMINAL_STATUSES else error_recovery_data['current_phase'],
            'total_frames': error_recovery_data['frame_count'],
            'trim_detected_frame': trim_info['trim_detected_frame'],
            'trimmed_total_frames': trim_info['trimmed_total_frames'],
            'error_types': self._extract_error_types(error_metadata),
            'attempt_count': len(error_metadata),
            'seed': getattr(self, '_current_seed', None),
            'planner_backend': getattr(self, '_planner_backend', None),
            'worker_id': self.worker_id,
            'gpu': self.gpu,
            'end_time': datetime.now().isoformat()
        }

    @staticmethod
    def _extract_error_types(error_metadata: List[Dict[str, Any]]) -> List[str]:
        return list(set(
            meta['injector'].name if hasattr(meta.get('injector'), 'name') else str(meta.get('injector', 'unknown'))
            for meta in error_metadata
        ))

    @staticmethod
    def _prepare_metadata_error_recovery_data(
        error_recovery_data: Dict[str, Any],
        trim_info: Dict[str, Any],
        success: bool,
    ) -> Dict[str, Any]:
        prepared = copy.deepcopy(error_recovery_data)
        keep_cache_idx = trim_info.get('trim_keep_cache_idx')
        trimmed_total_frames = trim_info.get('trimmed_total_frames')
        if success or keep_cache_idx is None or trimmed_total_frames is None:
            return prepared

        prepared['frame_count'] = trimmed_total_frames
        prepared['action_history'] = [
            item for item in (prepared.get('action_history') or [])
            if not isinstance(item, dict) or item.get('frame_count') is None or int(item['frame_count']) <= keep_cache_idx
        ]
        _drop_unsaved_attempt_frames(prepared.get('error_metadata') or [], int(trimmed_total_frames))
        return prepared

    def _save_episode_metadata(self, episode_id: int, episode_summary: Dict[str, Any], error_recovery_data: Dict[str, Any]):
        """Save episode metadata"""
        metadata_file = os.path.join(
            self.save_path, "metadata", f"episode{episode_id}_metadata.json"
        )
        
        full_metadata = {
            'episode_summary': episode_summary,
            'error_recovery_data': error_recovery_data,
            'collection_timestamp': datetime.now().isoformat()
        }
        
        # Prefer readable metadata so annotations can be inspected by hand.
        # Write a temporary file first, then replace it, so a killed process does not leave a partial JSON.
        tmp_metadata_file = metadata_file + ".tmp"
        with open(tmp_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(
                full_metadata,
                f,
                cls=NumpyEncoder,
                ensure_ascii=False,
                indent=2,
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_metadata_file, metadata_file)
    
    def _absorb_previous_episodes(self, stats_file: str) -> None:
        """Keep episodes already written by an earlier process with the same stats file.

        A restarted worker only has its own episodes in memory. Replacing the file
        would drop the earlier batch from the merged collection stats.
        """
        if not os.path.exists(stats_file):
            return
        try:
            with open(stats_file, "r", encoding="utf-8") as handle:
                previous = json.load(handle)
        except (OSError, ValueError):
            return
        if not isinstance(previous, dict):
            return

        by_id: Dict[int, Dict[str, Any]] = {}
        for episode in previous.get("episodes") or []:
            episode_id = _episode_id(episode)
            if episode_id is not None:
                by_id[episode_id] = episode
        for episode in self.collection_stats.get("episodes") or []:
            episode_id = _episode_id(episode)
            if episode_id is not None:
                by_id[episode_id] = episode
        ordered = [by_id[key] for key in sorted(by_id)]

        self.collection_stats["episodes"] = ordered
        self._recount_stats()
        stats = self.collection_stats
        previous_start = previous.get("start_time")
        if isinstance(previous_start, str) and previous_start < stats.get("start_time", previous_start):
            stats["start_time"] = previous_start

    def save_collection_stats(self):
        """Save collection statistics"""
        stats_name = (
            f"collection_stats_{self.worker_id}.json" if self.worker_id else "collection_stats.json"
        )
        stats_file = os.path.join(self.save_path, stats_name)
        # A restarted worker only holds its new episodes. A serial rerun should
        # replace the previous file instead of keeping episodes it did not collect.
        if self.worker_id:
            self._absorb_previous_episodes(stats_file)
        self._recount_stats()
        self.collection_stats['end_time'] = datetime.now().isoformat()
        # Prefer compact JSON on disk (format it temporarily with `python -m json.tool` when you need to read it).
        tmp_stats_file = stats_file + ".tmp"
        with open(tmp_stats_file, 'w', encoding='utf-8') as f:
            json.dump(
                self.collection_stats,
                f,
                cls=NumpyEncoder,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_stats_file, stats_file)
        
        logger.info(
            "[DataCollector] task=%s total=%d success=%d(%.1f%%) injected=%d recovery=%d recovery_rate=%.1f%% path=%s",
            self.collection_stats['task_name'],
            self.collection_stats['total_episodes'],
            self.collection_stats['successful_episodes'],
            self.collection_stats['success_rate'] * 100,
            self.collection_stats['error_injected_episodes'],
            self.collection_stats['error_recovery_episodes'],
            self.collection_stats['error_recovery_rate'] * 100 if self.collection_stats['error_injected_episodes'] > 0 else 0.0,
            self.save_path,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get the current statistics"""
        return self.collection_stats.copy()


class ConfigManager:
    """Configuration manager"""

    # Cache embodiment type mapping to avoid repeated YAML parsing in a single process.
    _embodiment_types_cache: Optional[Dict[str, Any]] = None
    _embodiment_types_mtime: Optional[float] = None
    
    @staticmethod
    def load_task_config(task_name: str) -> Dict[str, Any]:
        """Load the task config"""
        config_path = f"./task_config/{task_name}.yml"
        
        if not os.path.exists(config_path):
            # Create the default config
            return ConfigManager.create_default_config(task_name)
        
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        return config

    SCENE_CONFIGS = {"clean": None, "random": "./task_config/demo_randomized.yml"}

    @staticmethod
    def scene_randomization(scene: str) -> Optional[Dict[str, Any]]:
        """For a random scene, read domain_randomization from demo_randomized.yml; clean returns None and leaves everything disabled."""
        if scene not in ConfigManager.SCENE_CONFIGS:
            raise ValueError(f"unknown scene {scene!r}, expected one of {sorted(ConfigManager.SCENE_CONFIGS)}")
        path = ConfigManager.SCENE_CONFIGS[scene]
        if path is None:
            return None
        with open(path, "r") as f:
            randomization = dict(yaml.load(f, Loader=yaml.FullLoader)["domain_randomization"])
        randomization["random_embodiment"] = False
        return randomization
    
    @staticmethod
    def create_default_config(task_name: str) -> Dict[str, Any]:
        """Create the default config"""
        config = {
            "task_name": task_name,
            "embodiment": ["aloha-agilex"],
            "camera": {
                "head_camera_type": "D435",
                "wrist_camera_type": "D435",
                "collect_head_camera": True,
                "collect_wrist_camera": True,
            }
        }
        
        # Save the default config
        config_path = f"./task_config/{task_name}.yml"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Created default config: {config_path}")
        return config

    @staticmethod
    def _load_embodiment_types(embodiment_config_path: str) -> Dict[str, Any]:
        """Load embodiment type mapping with an mtime-based cache (logic unchanged)."""
        try:
            mtime = os.path.getmtime(embodiment_config_path)
        except Exception:
            # If we can't stat the file, fall back to always reading it
            # (matches the previous behavior more closely).
            mtime = None

        if mtime is None:
            with open(embodiment_config_path, "r") as f:
                return yaml.load(f, Loader=yaml.FullLoader) or {}

        if ConfigManager._embodiment_types_cache is None or mtime != ConfigManager._embodiment_types_mtime:
            with open(embodiment_config_path, "r") as f:
                ConfigManager._embodiment_types_cache = yaml.load(f, Loader=yaml.FullLoader)
            ConfigManager._embodiment_types_mtime = mtime

        # mypy: cache is populated above
        return ConfigManager._embodiment_types_cache or {}
    
    @staticmethod
    def setup_embodiment_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Set up the robot config"""
        from script.rollout_recovery.config import get_embodiment_config
        
        # Load the robot config mapping
        embodiment_config_path = "./task_config/_embodiment_config.yml"
        _embodiment_types = ConfigManager._load_embodiment_types(embodiment_config_path)
        
        def get_embodiment_file(embodiment_type):
            robot_file = _embodiment_types[embodiment_type]["file_path"]
            if robot_file is None:
                raise Exception("No embodiment files")
            return robot_file if os.path.isabs(robot_file) else os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", robot_file)
            )
        
        embodiment_type = config.get("embodiment", ["aloha-agilex"])
        
        if len(embodiment_type) == 1:
            # A single arm type (may be a single arm or a combined dual-arm robot)
            robot_file = get_embodiment_file(embodiment_type[0])
            # Load the config once and reuse it to avoid parsing twice (output stays the same)
            base_embodiment_config = get_embodiment_config(robot_file)
            dual_arm = base_embodiment_config.get('dual_arm', False)
            
            config.update({
                'left_robot_file': robot_file,
                'right_robot_file': robot_file,
                'dual_arm_embodied': dual_arm,
                'embodiment_name': embodiment_type[0]
            })
        elif len(embodiment_type) == 3:
            # Independent dual-arm robot
            config.update({
                'left_robot_file': get_embodiment_file(embodiment_type[0]),
                'right_robot_file': get_embodiment_file(embodiment_type[1]),
                'embodiment_dis': embodiment_type[2],
                'dual_arm_embodied': False,
                'embodiment_name': f"{embodiment_type[0]}+{embodiment_type[1]}"
            })
        else:
            raise Exception("Embodiment items should be 1 or 3")
        
        # Load the detailed robot config
        if len(embodiment_type) == 1:
            # Keep the same semantics as two separate loads: provide independent dict objects.
            config['left_embodiment_config'] = copy.deepcopy(base_embodiment_config)
            config['right_embodiment_config'] = copy.deepcopy(base_embodiment_config)
        else:
            config['left_embodiment_config'] = get_embodiment_config(config['left_robot_file'])
            config['right_embodiment_config'] = get_embodiment_config(config['right_robot_file'])
        
        # Add other required config
        config.update({
            'need_plan': True,
            'domain_randomization': {
                'random_background': False,
                'cluttered_table': False,
                'clean_background_rate': 0.0,
                'random_head_camera_dis': 0,
                'random_table_height': 0.0,
                'random_light': False,
                'crazy_random_light_rate': 0.0,
                'random_embodiment': False,
            }
        })
        
        return config
