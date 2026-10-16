import importlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ACTION_TYPE = os.getenv("YOUR_POLICY_ACTION_TYPE", "qpos")


def encode_obs(observation):
    if isinstance(observation, dict) and "images" in observation and "state" in observation:
        return observation

    try:
        head = observation["observation"]["head_camera"]["rgb"]
        right = observation["observation"]["right_camera"]["rgb"]
        left = observation["observation"]["left_camera"]["rgb"]
        state = observation["joint_action"]["vector"]

        def to_chw(img):
            arr = np.asarray(img)
            if arr.ndim == 3 and arr.shape[-1] >= 3:
                arr = arr[..., :3]
                arr = np.transpose(arr, (2, 0, 1))
            return arr

        return {
            "images": {
                "cam_high": to_chw(head),
                "cam_left_wrist": to_chw(left),
                "cam_right_wrist": to_chw(right),
            },
            "state": np.asarray(state, dtype=np.float32),
        }
    except Exception:
        return observation


class PolicyAdapter:
    def __init__(self, policy: Any, action_type: str = "qpos"):
        self.policy = policy
        self.action_type = action_type
        self.obs_cache = []

    def set_instruction(self, instruction):
        for fn_name in ("set_instruction", "set_language"):
            fn = getattr(self.policy, fn_name, None)
            if callable(fn):
                fn(instruction)
                return

    def update_obs(self, obs):
        self.obs_cache.append(obs)
        for fn_name in ("update_obs", "update_observation_window"):
            fn = getattr(self.policy, fn_name, None)
            if callable(fn):
                fn(obs)
                return

    def get_action(self, obs=None):
        fn = getattr(self.policy, "get_action", None)
        if not callable(fn):
            raise AttributeError("Model must implement get_action().")
        if obs is not None:
            try:
                return fn(obs)
            except TypeError:
                return fn()
        return fn()

    def reset_model(self):
        self.obs_cache.clear()
        for fn_name in ("reset_model", "reset", "reset_obsrvationwindows", "reset_observation_window"):
            fn = getattr(self.policy, fn_name, None)
            if callable(fn):
                fn()
                return


class WebsocketPolicyAdapter:
    def __init__(self, host: str, port: int, action_type: str = "qpos", reward: float = 1.0):
        try:
            from openpi_client import websocket_client_policy
        except ModuleNotFoundError:
            # Fallback to local source tree: policy/pi0/packages/openpi-client/src
            repo_root = Path(__file__).resolve().parents[2]
            local_client_src = repo_root / "policy" / "pi0" / "packages" / "openpi-client" / "src"
            if local_client_src.is_dir():
                sys.path.insert(0, str(local_client_src))
            try:
                from openpi_client import websocket_client_policy
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "Cannot import `openpi_client`. Please install it in current env:\n"
                    "cd policy/pi0/packages/openpi-client && pip install -e ."
                ) from e
        self._ws_client_cls = websocket_client_policy.WebsocketClientPolicy
        self.host = str(host)
        self.port = int(port)
        self.client = self._connect_client()
        self.action_type = action_type
        self.reward = float(reward)
        self.obs_cache = []
        self.instruction = None
        print(f"[Your_Policy] Connected websocket policy server: ws://{self.host}:{self.port}")

    def _connect_client(self):
        try:
            return self._ws_client_cls(host=self.host, port=self.port)
        except TimeoutError as e:
            raise TimeoutError(
                f"WebSocket connect timeout to ws://{self.host}:{self.port}. "
                "Please verify remote_host/remote_port and that a websocket server is listening."
            ) from e

    @staticmethod
    def _is_ws_disconnect_error(err: Exception) -> bool:
        text = str(err).lower()
        return (
            "keepalive ping timeout" in text
            or "connection closed" in text
            or "connectionclosederror" in text
            or "no close frame received" in text
            or isinstance(err, TimeoutError)
        )

    def set_instruction(self, instruction):
        self.instruction = instruction

    def update_obs(self, obs):
        self.obs_cache.append(obs)

    def get_action(self, obs=None):
        if obs is None:
            if not self.obs_cache:
                raise RuntimeError("No observation cached.")
            obs = self.obs_cache[-1]

        if isinstance(obs, dict):
            obs = self._prepare_remote_obs(obs)
            if self.instruction and "prompt" not in obs:
                obs["prompt"] = self.instruction

        try:
            ret = self.client.infer(obs)
        except Exception as e:
            if not self._is_ws_disconnect_error(e):
                raise
            print("[Your_Policy] WebSocket disconnected, reconnecting and retrying once...")
            self.client = self._connect_client()
            ret = self.client.infer(obs)
        return ret["actions"] if isinstance(ret, dict) and "actions" in ret else ret

    def _prepare_remote_obs(self, obs: dict):
        out = dict(obs)
        images = out.get("images")
        if not isinstance(images, dict):
            return out

        def to_chw_image(x):
            arr = np.asarray(x)
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim != 3:
                raise ValueError(f"Expected image ndim=3, got shape={arr.shape}")

            # Ensure CHW for openpi aloha_policy input.
            # HWC -> CHW
            if arr.shape[-1] in (1, 3, 4) and arr.shape[0] not in (1, 3, 4):
                arr = np.transpose(arr, (2, 0, 1))
            return arr

        # Normalize to Aloha camera names expected by openpi.policies.aloha_policy.
        # We accept both local keys (cam_*) and legacy aliases (*_camera).
        source_priority = {
            "cam_high": ("cam_high", "head_camera"),
            "cam_low": ("cam_low",),
            "cam_left_wrist": ("cam_left_wrist", "left_camera"),
            "cam_right_wrist": ("cam_right_wrist", "right_camera"),
        }

        normalized = {}
        for dst_key, src_keys in source_priority.items():
            for src_key in src_keys:
                if src_key not in images:
                    continue
                normalized[dst_key] = to_chw_image(images[src_key])
                break

        # Some RoboTwin tasks do not provide cam_low; use cam_high as fallback.
        if "cam_low" not in normalized and "cam_high" in normalized:
            normalized["cam_low"] = normalized["cam_high"]

        if "cam_high" not in normalized:
            raise ValueError(f"Missing required camera `cam_high` in obs images keys: {tuple(images.keys())}")

        out["images"] = normalized
        reward = out.get("reward", self.reward)
        reward_arr = np.asarray(reward, dtype=np.float32).reshape(-1)
        if reward_arr.size == 0:
            reward_arr = np.asarray([self.reward], dtype=np.float32)
        out["reward"] = reward_arr
        return out

    def reset_model(self):
        self.obs_cache.clear()
        self.instruction = None
        reset_fn = getattr(self.client, "reset", None)
        if callable(reset_fn):
            reset_fn()


def _load_factory(factory_path: str):
    module_name, fn_name = factory_path.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, fn_name, None)
    if not callable(factory):
        raise ValueError(f"Cannot find callable `{fn_name}` in module `{module_name}`.")
    return factory


def _build_policy(usr_args):
    model_factory = usr_args.get("model_factory")
    model_module = usr_args.get("model_module")
    model_class = usr_args.get("model_class")
    model_kwargs = usr_args.get("model_kwargs", {})

    if model_factory:
        return _load_factory(model_factory)(usr_args)

    if model_module and model_class:
        module = importlib.import_module(model_module)
        klass = getattr(module, model_class, None)
        if klass is None:
            raise ValueError(f"Cannot find class `{model_class}` in module `{model_module}`.")
        return klass(**model_kwargs)

    raise ValueError("Set model_factory or model_module+model_class for local/cloud_server mode.")


def get_model(usr_args):
    global _ACTION_TYPE
    _ACTION_TYPE = usr_args.get("action_type", _ACTION_TYPE)
    inference_mode = str(usr_args.get("inference_mode", "local")).lower()

    if inference_mode == "cloud_client":
        return WebsocketPolicyAdapter(
            host=usr_args.get("remote_host", "127.0.0.1"),
            port=usr_args.get("remote_port", 8000),
            action_type=_ACTION_TYPE,
            reward=float(usr_args.get("reward", 1.0)),
        )

    policy = _build_policy(usr_args)
    return PolicyAdapter(policy, action_type=_ACTION_TYPE)


def _normalize_actions(actions):
    if actions is None:
        return []
    if isinstance(actions, np.ndarray):
        return [actions] if actions.ndim == 1 else list(actions)
    if isinstance(actions, (list, tuple)):
        return list(actions)
    return [actions]


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()

    if hasattr(model, "set_instruction"):
        model.set_instruction(instruction)

    model_obs = dict(obs) if isinstance(obs, dict) else obs
    if isinstance(model_obs, dict) and instruction and "prompt" not in model_obs:
        model_obs["prompt"] = instruction

    model.update_obs(model_obs)
    actions = _normalize_actions(model.get_action(model_obs))

    for action in actions:
        TASK_ENV.take_action(action, action_type=_ACTION_TYPE)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model_obs = dict(obs) if isinstance(obs, dict) else obs
        if isinstance(model_obs, dict) and instruction and "prompt" not in model_obs:
            model_obs["prompt"] = instruction
        model.update_obs(model_obs)


def reset_model(model):
    if hasattr(model, "reset_model"):
        model.reset_model()
    elif hasattr(model, "reset"):
        model.reset()
