from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any, Dict

import yaml


def get_repo_root() -> Path:
    # script/rollout_recovery/config.py -> script/rollout_recovery -> script -> repo_root
    return Path(__file__).resolve().parents[2]


def parse_args_and_config(argv: list[str] | None = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    def parse_override_pairs(pairs: list[str]) -> Dict[str, Any]:
        if len(pairs) % 2 != 0:
            raise ValueError("overrides must be provided as --key value pairs")
        override_dict: Dict[str, Any] = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value: Any = pairs[i + 1]
            try:
                value = ast.literal_eval(value)
            except Exception:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


def get_camera_config(camera_type: str) -> dict:
    repo_root = get_repo_root()
    camera_config_path = repo_root / "task_config" / "_camera_config.yml"
    assert camera_config_path.is_file(), "task config file is missing"
    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file: str) -> dict:
    robot_config_file = Path(robot_file) / "config.yml"
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f, Loader=yaml.FullLoader)
    return embodiment_args
