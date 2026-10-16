from __future__ import annotations

import importlib
import inspect
import os
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from script.rollout_recovery.recovery_utils import (
    DEFAULT_RECOVERY_EXECUTION_MODE,
    DEFAULT_RECOVERY_FORCE_OPEN_GRIPPERS,
    DEFAULT_RECOVERY_PLANNING_HOLD_FRAMES,
    DEFAULT_RECOVERY_QPOS_ACTION_STRIDE,
    DEFAULT_RECOVERY_THRESHOLD_TYPE,
    DEFAULT_RECOVERY_VIDEO_SAMPLE_FREQ,
    DEFAULT_RECOVERY_VIDEO_SAMPLE_MODE,
    EpisodeSaveLayout,
    count_play_once_stages,
    get_all_actor_poses,
    load_task_frame_threshold_info,
    remove_episode_cache_dir,
    restore_all_actor_poses,
    save_merged_recovery_episode,
)


def run_recovery_play_once(env, skip_stages: int, variant: int) -> bool:
    if not hasattr(env, "play_once"):
        return env.check_success() if hasattr(env, "check_success") else False

    sig = inspect.signature(env.play_once)
    if "skip_stages" in sig.parameters:
        env.play_once(skip_stages=skip_stages)
    elif skip_stages > 0:
        original_move = env.move
        move_call_count = [0]

        def move_wrapper(*args, **kwargs):
            move_call_count[0] += 1
            if move_call_count[0] <= skip_stages:
                print(
                    f"\033[93m[Recovery] [Variant {variant}] Skipping stage {move_call_count[0]}/{skip_stages} (move call {move_call_count[0]})\033[0m"
                )
                return True
            executed_stage = move_call_count[0] - skip_stages
            print(
                f"\033[92m[Recovery] [Variant {variant}] Executing stage {executed_stage} (move call {move_call_count[0]})\033[0m"
            )
            return original_move(*args, **kwargs)

        env.move = move_wrapper
        try:
            print(f"\033[94m[Recovery] Executing play_once with {skip_stages} stages skipped\033[0m")
            env.play_once()
        finally:
            env.move = original_move
            print(
                f"\033[94m[Recovery] Completed play_once, executed {move_call_count[0] - skip_stages} move calls\033[0m"
            )
    else:
        print("\033[94m[Recovery] Executing play_once with all stages (variant 1)\033[0m")
        env.play_once()

    return env.check_success() if hasattr(env, "check_success") else False


def save_env_hdf5_video(env, custom_name: str) -> None:
    try:
        env.merge_pkl_to_hdf5_video(custom_video_name=custom_name)
    except TypeError as e:
        if "custom_video_name" not in str(e):
            raise
        original_ep_num = getattr(env, "ep_num", None)
        env.ep_num = custom_name.replace("episode", "", 1) if custom_name.startswith("episode") else custom_name
        try:
            env.merge_pkl_to_hdf5_video()
        finally:
            if original_ep_num is not None:
                env.ep_num = original_ep_num


def write_env_frame_to_cache(env) -> None:
    if not getattr(env, "save_data", False):
        return
    cache_path = env.folder_path["cache"]
    from envs.utils.save_file import save_pkl

    save_pkl(os.path.join(cache_path, f"{env.FRAME_IDX}.pkl"), env.get_obs())
    env.FRAME_IDX += 1


def recovery_succeeded(env, candidate_success: Any) -> bool:
    if not bool(candidate_success):
        return False
    if getattr(env, "plan_success", True) is False:
        return False
    if not hasattr(env, "check_success"):
        return bool(candidate_success)
    try:
        return bool(env.check_success())
    except Exception:
        return False


def install_recovery_move_frame_capture(env):
    original_move = env.move

    def move_with_frame_save(*args, **kwargs):
        result = original_move(*args, **kwargs)
        if getattr(env, "save_data", False):
            write_env_frame_to_cache(env)
        return result

    env.move = move_with_frame_save
    return original_move


def _control_seq_to_qpos_actions(env, control_seq, stride: int = 1) -> list[np.ndarray]:
    left_state = np.array(env.robot.get_left_arm_jointState(), dtype=np.float64)
    right_state = np.array(env.robot.get_right_arm_jointState(), dtype=np.float64)

    left_arm = control_seq.get("left_arm")
    left_gripper = control_seq.get("left_gripper")
    right_arm = control_seq.get("right_arm")
    right_gripper = control_seq.get("right_gripper")

    max_control_len = max(
        left_arm["position"].shape[0] if left_arm is not None else 0,
        left_gripper["num_step"] if left_gripper is not None else 0,
        right_arm["position"].shape[0] if right_arm is not None else 0,
        right_gripper["num_step"] if right_gripper is not None else 0,
    )

    actions: list[np.ndarray] = []
    stride = max(1, int(stride))
    for control_idx in range(max_control_len):
        if left_arm is not None and control_idx < left_arm["position"].shape[0]:
            left_state[:-1] = left_arm["position"][control_idx]
        if left_gripper is not None and control_idx < left_gripper["num_step"]:
            left_state[-1] = left_gripper["result"][control_idx]

        if right_arm is not None and control_idx < right_arm["position"].shape[0]:
            right_state[:-1] = right_arm["position"][control_idx]
        if right_gripper is not None and control_idx < right_gripper["num_step"]:
            right_state[-1] = right_gripper["result"][control_idx]

        if control_idx % stride == 0 or control_idx == max_control_len - 1:
            actions.append(np.concatenate([left_state[:-1], [left_state[-1]], right_state[:-1], [right_state[-1]]]))

    return actions


def _build_parallel_move_control_seq(env, left_action, right_action):
    if env.need_plan:
        left_result = env._plan_path_with_retry(
            arm_tag="left",
            target_pose=left_action.target_pose,
            constraint_pose=left_action.args.get("constraint_pose"),
        )
        right_result = env._plan_path_with_retry(
            arm_tag="right",
            target_pose=right_action.target_pose,
            constraint_pose=right_action.args.get("constraint_pose"),
        )
        env.left_joint_path.append(deepcopy(left_result))
        env.right_joint_path.append(deepcopy(right_result))
    else:
        left_result = deepcopy(env.left_joint_path[env.left_cnt])
        right_result = deepcopy(env.right_joint_path[env.right_cnt])
        env.left_cnt += 1
        env.right_cnt += 1

    if not env._plan_result_success(left_result) or not env._plan_result_success(right_result):
        env.plan_success = False
        return None

    control_seq = env._build_empty_control_seq()
    control_seq["left_arm"] = left_result
    control_seq["right_arm"] = right_result
    return control_seq


def install_recovery_qpos_action_move(
    env,
    action_type: str = "qpos",
    action_stride: int = DEFAULT_RECOVERY_QPOS_ACTION_STRIDE,
):
    original_move = env.move
    original_eval_video_path = getattr(env, "eval_video_path", None)
    env.eval_video_path = None

    def move_with_qpos_actions(actions_by_arm1, actions_by_arm2=None, save_freq=-1):
        if env.plan_success is False:
            return False

        left_actions = env._get_arm_plan_tuple(actions_by_arm1, actions_by_arm2, "left")
        right_actions = env._get_arm_plan_tuple(actions_by_arm1, actions_by_arm2, "right")

        max_len = max(len(left_actions), len(right_actions))
        left_actions += [None] * (max_len - len(left_actions))
        right_actions += [None] * (max_len - len(right_actions))

        original_step_lim = getattr(env, "step_lim", None)
        if original_step_lim is not None:
            env.step_lim = max(int(original_step_lim), int(env.take_action_cnt) + 100000)

        try:
            for left, right in zip(left_actions, right_actions):
                env._validate_parallel_actions(left, right)
                env._invoke_before_move_hook(left, right)

                if (
                    left is not None
                    and left.action == "move"
                    and right is not None
                    and right.action == "move"
                ):
                    control_seq = _build_parallel_move_control_seq(env, left, right)
                else:
                    control_seq = env._build_single_arm_actions(left, right)

                if control_seq is None or env.plan_success is False:
                    return False

                for qpos_action in _control_seq_to_qpos_actions(env, control_seq, stride=action_stride):
                    env.take_action(qpos_action, action_type=action_type)
                    if getattr(env, "eval_success", False):
                        return True
                    if env.plan_success is False:
                        return False
        finally:
            if original_step_lim is not None:
                env.step_lim = original_step_lim

        return True

    env.move = move_with_qpos_actions
    env._recovery_original_eval_video_path = original_eval_video_path
    return original_move


def restore_recovery_qpos_action_move(env, original_move) -> None:
    env.move = original_move
    if hasattr(env, "_recovery_original_eval_video_path"):
        env.eval_video_path = env._recovery_original_eval_video_path
        del env._recovery_original_eval_video_path


def recovery_video_sample_config(args: dict, default_freq: int) -> tuple[str, int]:
    mode = str(args.get("recovery_video_sample_mode", DEFAULT_RECOVERY_VIDEO_SAMPLE_MODE)).lower()
    if mode not in {"action", "sim"}:
        raise ValueError(f"Invalid recovery_video_sample_mode: {mode}")
    # Use a coarser sim-step sample for evaluation videos so rollout/recovery playback
    # stays close to the original policy-eval visual pace.
    freq = int(
        args.get(
            "recovery_video_sample_freq",
            DEFAULT_RECOVERY_VIDEO_SAMPLE_FREQ if mode == "sim" else default_freq,
        )
    )
    return mode, freq


def write_hold_frames(env, count: int) -> None:
    for _ in range(max(0, int(count))):
        write_env_frame_to_cache(env)


def cleanup_recovery_attempt_outputs(layout: EpisodeSaveLayout, episode_index: int, variant: int) -> None:
    remove_episode_cache_dir(layout.recovery_cache_dir(episode_index, variant))
    remove_episode_cache_dir(os.path.join(layout.recovery_root(), ".cache", layout.episode_name(episode_index)))
    for path in (
        layout.recovery_data_path(episode_index),
        layout.recovery_video_path(episode_index),
    ):
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    for subdir in ("data", "video", ".cache", "index"):
        dir_path = os.path.join(layout.recovery_root(), subdir)
        try:
            if os.path.isdir(dir_path) and not os.listdir(dir_path):
                os.rmdir(dir_path)
        except Exception:
            pass
    try:
        if os.path.isdir(layout.recovery_root()) and not os.listdir(layout.recovery_root()):
            os.rmdir(layout.recovery_root())
    except Exception:
        pass


@dataclass(frozen=True)
class RecoveryVariantResult:
    variant: int
    success: bool
    variant_video_path: Optional[str]
    variant_hdf5_path: Optional[str]
    recovery_save_dir: Optional[str]
    custom_name: str


@dataclass
class RecoveryContext:
    task_name: str
    eval_video_freq: int = 4
    frame_threshold: Optional[int] = None
    force_open_grippers: bool = True
    num_stages: int = 1
    variant_success_counts: dict[int, int] | None = None
    variant_total_counts: dict[int, int] | None = None

    def snapshot_env(self, env) -> dict[str, Any]:
        snap: dict[str, Any] = {
            "take_action_cnt": env.take_action_cnt,
            "left_jointstate": env.robot.get_left_arm_jointState(),
            "right_jointstate": env.robot.get_right_arm_jointState(),
        }
        actor_poses = get_all_actor_poses(env)
        snap.update(actor_poses)
        snap["_actor_poses"] = actor_poses
        return snap

    def restore_env(self, env, snapshot: dict[str, Any]) -> None:
        left = snapshot.get("left_jointstate")
        right = snapshot.get("right_jointstate")
        if left is not None and right is not None:
            env.robot.set_arm_joints(left[:-1], np.zeros_like(left[:-1]), "left")
            env.robot.set_gripper(left[-1], "left")
            env.robot.set_arm_joints(right[:-1], np.zeros_like(right[:-1]), "right")
            env.robot.set_gripper(right[-1], "right")

        actor_poses = snapshot.get("_actor_poses")
        if actor_poses is None:
            actor_poses = {
                k: v
                for k, v in snapshot.items()
                if k.endswith("_pose") and k not in ["left_jointstate", "right_jointstate"]
            }
        restore_all_actor_poses(env, actor_poses)

        env.plan_success = True
        env.eval_success = False

        env.scene.step()
        env._update_render()

    def should_trigger_recovery(
        self,
        env,
        *,
        min_step: int = 50,
        check_window: int = 20,
    ) -> bool:
        if env.take_action_cnt < min_step:
            return False

        if self.frame_threshold is not None:
            # Threshold stats are per saved video frame; rollout saves one frame per take_action.
            current_frames = int(env.take_action_cnt)
            effective_threshold = int(self.frame_threshold) * int(self.eval_video_freq)
            if current_frames > effective_threshold:
                print(
                    f"\n\033[93m[Recovery Trigger] Frame count {current_frames} > "
                    f"threshold {effective_threshold} ({self.frame_threshold} x eval_video_freq "
                    f"{self.eval_video_freq}), triggering recovery\033[0m"
                )
                return True

        if not hasattr(env, "_recent_ee_positions"):
            env._recent_ee_positions = deque(maxlen=check_window)

        robot = env.robot
        left_ee_pos = robot.get_left_tcp_pose()[:3]
        right_ee_pos = robot.get_right_tcp_pose()[:3]
        env._recent_ee_positions.append((left_ee_pos, right_ee_pos))

        if len(env._recent_ee_positions) >= check_window:
            recent_left = np.array([pos[0] for pos in env._recent_ee_positions])
            recent_right = np.array([pos[1] for pos in env._recent_ee_positions])
            left_motion = float(np.max(np.linalg.norm(recent_left - recent_left[0], axis=1)))
            right_motion = float(np.max(np.linalg.norm(recent_right - recent_right[0], axis=1)))
            if left_motion < 0.003 and right_motion < 0.003:
                print(
                    f"\n\033[93m[Recovery Trigger] End-effector stagnant for {check_window} steps "
                    f"(left_motion={left_motion:.6f}, right_motion={right_motion:.6f}), "
                    f"triggering recovery\033[0m"
                )
                return True

        if self.frame_threshold is None and env.take_action_cnt >= env.step_lim * 0.95 and not env.eval_success:
            print(
                f"\n\033[93m[Recovery Trigger] Step count {env.take_action_cnt} >= "
                f"{env.step_lim * 0.95:.0f} (95% of max), triggering recovery (fallback)\033[0m"
            )
            return True

        return False


    def try_in_place_recovery(
        self,
        env,
        *,
        episode_index: int,
        eval_save_dir: str,
        args: dict,
        recovery_seed: int,
        rollout_cache_dir: Optional[str] = None,
    ) -> tuple[bool, list[RecoveryVariantResult]]:
        layout = EpisodeSaveLayout(eval_save_dir)
        variant = 1
        skip_stages = max(0, int(args.get("recovery_inplace_skip_stages", 0)))
        custom_name = layout.episode_name(episode_index)
        recovery_cache_dir = layout.recovery_cache_dir(episode_index, variant)

        print(
            f"\n\033[93m[Recovery] Starting in-place recovery "
            f"(skip_stages={skip_stages})\033[0m"
        )

        if self.variant_total_counts is not None:
            self.variant_total_counts[variant] = self.variant_total_counts.get(variant, 0) + 1

        original_save_data = getattr(env, "save_data", False)
        original_save_dir = getattr(env, "save_dir", None)
        original_save_freq = getattr(env, "save_freq", None)
        original_folder_path = getattr(env, "folder_path", None)
        original_frame_idx = getattr(env, "FRAME_IDX", 0)
        original_record_policy_dense_frames = getattr(env, "record_policy_dense_frames", False)
        original_grasp_actor = env.grasp_actor
        original_move = env.move

        success = False
        variant_video_path = None
        variant_hdf5_path = None

        try:
            recovery_sample_mode, recovery_sample_freq = recovery_video_sample_config(args, self.eval_video_freq)
            env.save_data = True
            env.save_dir = layout.recovery_root()
            env.ep_num = episode_index
            env.folder_path = {"cache": recovery_cache_dir + os.sep}
            env.FRAME_IDX = 0
            env.save_freq = recovery_sample_freq if recovery_sample_mode == "sim" else None
            env.record_policy_dense_frames = recovery_sample_mode == "sim"
            print(
                f"\033[94m[Recovery] [In-place] Recording mode={recovery_sample_mode}, "
                f"save_freq={recovery_sample_freq}\033[0m"
            )
            # Save the exact handoff state before any recovery-side mutation.
            write_env_frame_to_cache(env)

            should_force_open_grippers = self.force_open_grippers and skip_stages == 0
            if self.force_open_grippers and not should_force_open_grippers:
                print(
                    "\033[94m[Recovery] [In-place] Preserving gripper state because "
                    "the grasp stage is skipped\033[0m"
                )

            if should_force_open_grippers:
                try:
                    left_before = env.robot.get_left_gripper_val()
                    right_before = env.robot.get_right_gripper_val()
                    env.robot.set_gripper(1.0, "left")
                    env.robot.set_gripper(1.0, "right")
                    for _ in range(20):
                        env.scene.step()
                        env._sim_step_count += 1
                        if env.render_freq:
                            env._update_render()
                        if recovery_sample_mode == "sim" and env._sim_step_count % recovery_sample_freq == 0:
                            env._update_render()
                            env._take_picture()
                    if recovery_sample_mode == "action":
                        write_env_frame_to_cache(env)
                    print(
                        f"\033[94m[Recovery] [In-place] Opened both grippers before recovery: "
                        f"left {left_before:.3f}->{env.robot.get_left_gripper_val():.3f}, "
                        f"right {right_before:.3f}->{env.robot.get_right_gripper_val():.3f}\033[0m"
                    )
                except Exception as e:
                    print(f"\033[93m[Recovery] [In-place] Warning: Failed to open grippers: {e}\033[0m")

            if hasattr(env.robot, "set_origin_endpose"):
                env.robot.set_origin_endpose()
                print("\033[94m[Recovery] [In-place] Set origin_pose to current state\033[0m")

            write_hold_frames(
                env,
                int(args.get("recovery_planning_hold_frames", DEFAULT_RECOVERY_PLANNING_HOLD_FRAMES)),
            )
            recovery_execution_mode = str(
                args.get("recovery_execution_mode", DEFAULT_RECOVERY_EXECUTION_MODE)
            ).lower()
            if recovery_execution_mode == "qpos_action":
                action_stride = int(args.get("recovery_qpos_action_stride", DEFAULT_RECOVERY_QPOS_ACTION_STRIDE))
                original_move = install_recovery_qpos_action_move(
                    env,
                    str(args.get("action_type", "qpos")),
                    action_stride,
                )
                print(
                    f"\033[94m[Recovery] [In-place] Executing expert plan as qpos actions "
                    f"(stride={action_stride})\033[0m"
                )
            elif recovery_execution_mode == "native":
                if recovery_sample_mode == "action":
                    original_move = install_recovery_move_frame_capture(env)
            else:
                raise ValueError(f"Invalid recovery_execution_mode: {recovery_execution_mode}")

            def grasp_actor_wrapper(actor, arm_tag, **kwargs):
                arm_tag_str = str(arm_tag) if hasattr(arm_tag, "__str__") else arm_tag
                current_gripper_val = (
                    env.robot.get_left_gripper_val()
                    if arm_tag_str == "left"
                    else env.robot.get_right_gripper_val()
                )
                result = original_grasp_actor(actor, arm_tag, **kwargs)
                if result is None or result[0] is None:
                    return result

                returned_arm_tag, action_list = result
                from envs.utils.action import Action

                new_action_list = []
                injected_open_before_first_move = False
                for i, action in enumerate(action_list):
                    if action.action == "move":
                        if i == 0 or (i > 0 and action_list[i - 1].action != "gripper"):
                            if current_gripper_val < 0.8:
                                new_action_list.append(Action(returned_arm_tag, "open", target_gripper_pos=1.0))
                                injected_open_before_first_move = True
                                print(
                                    f"\033[94m[Recovery] [In-place] Inserted open gripper action before "
                                    f"move action {i+1} (gripper was closed: {current_gripper_val:.3f})\033[0m"
                                )
                            elif should_force_open_grippers and not injected_open_before_first_move:
                                new_action_list.append(Action(returned_arm_tag, "open", target_gripper_pos=1.0))
                                injected_open_before_first_move = True
                                print(
                                    f"\033[94m[Recovery] [In-place] Inserted maintain-open gripper action "
                                    f"before move action {i+1}\033[0m"
                                )
                        new_action_list.append(action)
                    else:
                        new_action_list.append(action)
                        if action.action == "gripper" and action.target_gripper_pos is not None:
                            current_gripper_val = action.target_gripper_pos
                return returned_arm_tag, new_action_list

            env.grasp_actor = grasp_actor_wrapper

            if hasattr(env, "expert_recovery_sequence"):
                try:
                    success = env.expert_recovery_sequence(variant)
                except NotImplementedError:
                    success = run_recovery_play_once(env, skip_stages, variant)
            else:
                success = run_recovery_play_once(env, skip_stages, variant)

            success = recovery_succeeded(env, success)
            if success:
                if rollout_cache_dir:
                    hdf5_path, video_path, frame_count = save_merged_recovery_episode(
                        layout,
                        episode_index,
                        rollout_cache_dir,
                        recovery_cache_dir,
                    )
                    remove_episode_cache_dir(rollout_cache_dir)
                    remove_episode_cache_dir(recovery_cache_dir)
                    variant_hdf5_path = hdf5_path
                    variant_video_path = video_path
                    print(
                        f"\033[92m[Recovery] In-place recovery succeeded, saved merged trajectory "
                        f"({frame_count} frames) to {video_path}, {hdf5_path}\033[0m"
                    )
                else:
                    save_env_hdf5_video(env, custom_name)
                    env.remove_data_cache()
                    variant_hdf5_path = layout.recovery_data_path(episode_index)
                    variant_video_path = layout.recovery_video_path(episode_index)
                    print("\033[92m[Recovery] In-place recovery succeeded\033[0m")
            else:
                cleanup_recovery_attempt_outputs(layout, episode_index, variant)
                print("\033[93m[Recovery] In-place recovery failed, recovery outputs discarded\033[0m")

        except KeyboardInterrupt:
            success = False
            cleanup_recovery_attempt_outputs(layout, episode_index, variant)
            print("\033[93m[Recovery] In-place recovery interrupted, recovery cache discarded\033[0m")
            raise
        except Exception as e:
            print(f"\033[91m[Recovery] In-place recovery failed with error: {e}\033[0m")
            import traceback

            traceback.print_exc()
            success = False
            cleanup_recovery_attempt_outputs(layout, episode_index, variant)
        finally:
            env.grasp_actor = original_grasp_actor
            restore_recovery_qpos_action_move(env, original_move)
            env.save_data = original_save_data
            env.save_dir = original_save_dir
            env.save_freq = original_save_freq
            env.record_policy_dense_frames = original_record_policy_dense_frames
            env.folder_path = original_folder_path
            env.FRAME_IDX = original_frame_idx

        result = RecoveryVariantResult(
            variant=variant,
            success=bool(success),
            variant_video_path=variant_video_path,
            variant_hdf5_path=variant_hdf5_path,
            recovery_save_dir=layout.recovery_root(),
            custom_name=custom_name,
        )
        if success and self.variant_success_counts is not None:
            self.variant_success_counts[variant] = self.variant_success_counts.get(variant, 0) + 1
        if not success:
            print("\033[91m[Recovery] In-place recovery failed\033[0m")
        return bool(success), [result]

    def try_multi_env_recovery(
        self,
        snapshot: dict[str, Any],
        *,
        episode_index: int,
        eval_save_dir: str,
        args: dict,
        recovery_seed: int,
        rollout_cache_dir: Optional[str] = None,
    ) -> tuple[bool, list[RecoveryVariantResult]]:
        print(
            f"\n\033[93m[Recovery] Starting multi-environment recovery with {self.num_stages} stages\033[0m"
        )

        recovery_results: list[RecoveryVariantResult] = []
        env_class = None
        layout = EpisodeSaveLayout(eval_save_dir)
        episode_name = layout.episode_name(episode_index)

        excluded_keys = frozenset(
            {
                "table_xy_bias",
                "table_height_bias",
                "seed",
                "task_name",
                "task_config",
                "save_path",
                "now_ep_num",
                "render_freq",
                "data_type",
                "save_data",
                "dual_arm",
                "eval_mode",
                "domain_randomization",
                "need_plan",
                "left_joint_path",
                "right_joint_path",
                "eval_video_save_dir",
                "save_freq",
            }
        )
        extra_kwargs = {k: v for k, v in args.items() if k not in excluded_keys}
        common_init_kwargs = dict(
            table_xy_bias=args.get("table_xy_bias", [0, 0]),
            table_height_bias=args.get("table_height_bias", 0),
            skip_robot_init=True,
            seed=recovery_seed,
            task_name=args.get("task_name"),
            task_config=args.get("task_config"),
            save_path=args.get("save_path", "data"),
            now_ep_num=episode_index,
            render_freq=args.get("render_freq", 10),
            data_type=args.get("data_type", None),
            save_data=False,
            dual_arm=args.get("dual_arm", True),
            eval_mode=True,
            domain_randomization=args.get("domain_randomization", {}),
            need_plan=args.get("need_plan", True),
            left_joint_path=args.get("left_joint_path", []),
            right_joint_path=args.get("right_joint_path", []),
            eval_video_save_dir=args.get("eval_video_save_dir", None),
            save_freq=args.get("save_freq"),
            **extra_kwargs,
        )

        for variant in range(1, int(self.num_stages) + 1):
            skip_stages = variant - 1
            print(f"\n\033[93m[Recovery] Variant {variant}: Skipping first {skip_stages} stage(s)\033[0m")

            env = None
            try:
                print(f"[Recovery] Creating recovery environment {variant}/{self.num_stages}")
                if env_class is None:
                    envs_module = importlib.import_module(f"envs.{self.task_name}")
                    env_class = getattr(envs_module, self.task_name)
                env = env_class()

                env.save_data = False
                env.save_freq = None

                env._init_task_env_(**common_init_kwargs)

                print(
                    f"\033[94m[Recovery] [Variant {variant}] Restoring robot and object states from snapshot\033[0m"
                )
                self.restore_env(env, snapshot)

                actor_poses = snapshot.get("_actor_poses")
                if actor_poses is None:
                    actor_poses = {
                        k: v
                        for k, v in snapshot.items()
                        if k.endswith("_pose") and k not in ["left_jointstate", "right_jointstate"]
                    }
                if actor_poses:
                    print(
                        f"\033[94m[Recovery] [Variant {variant}] Re-applying {len(actor_poses)} actor pose(s) after restore\033[0m"
                    )
                    restore_all_actor_poses(env, actor_poses)

                print(
                    f"\033[94m[Recovery] [Variant {variant}] Stabilizing robot position (save_data=False, no frames saved)\033[0m"
                )
                for _ in range(100):
                    env.scene.step()
                    if env.render_freq:
                        env._update_render()

            except Exception as e:
                print(f"\033[91m[Recovery] Failed to create recovery environment {variant}: {e}\033[0m")
                recovery_results.append(
                    RecoveryVariantResult(
                        variant=variant,
                        success=False,
                        variant_video_path=None,
                        variant_hdf5_path=None,
                        recovery_save_dir=None,
                        custom_name=episode_name,
                    )
                )
                continue

            if self.variant_total_counts is not None:
                self.variant_total_counts[variant] = self.variant_total_counts.get(variant, 0) + 1

            print(
                f"\033[94m[Recovery] [Variant {variant}] Will skip first {skip_stages} stage(s), execute remaining {self.num_stages - skip_stages} stage(s)\033[0m"
            )

            should_force_open_grippers = self.force_open_grippers and skip_stages == 0
            if self.force_open_grippers and not should_force_open_grippers:
                print(
                    f"\033[94m[Recovery] [Variant {variant}] Preserving gripper state because "
                    f"the grasp stage is skipped\033[0m"
                )

            if should_force_open_grippers:
                try:
                    left_before = env.robot.get_left_gripper_val()
                    right_before = env.robot.get_right_gripper_val()
                    env.robot.set_gripper(1.0, "left")
                    env.robot.set_gripper(1.0, "right")
                    for _ in range(20):
                        env.scene.step()
                        if env.render_freq:
                            env._update_render()
                    left_after = env.robot.get_left_gripper_val()
                    right_after = env.robot.get_right_gripper_val()
                    print(
                        f"\033[94m[Recovery] [Variant {variant}] Opened both grippers before play_once:"
                    )
                    print(
                        f"  Left:  {left_before:.3f} -> {left_after:.3f} {'✓' if left_after > 0.8 else '✗ (may not be fully open)'}"
                    )
                    print(
                        f"  Right: {right_before:.3f} -> {right_after:.3f} {'✓' if right_after > 0.8 else '✗ (may not be fully open)'}\033[0m"
                    )
                    if left_after < 0.8 or right_after < 0.8:
                        print(
                            f"\033[93m[Recovery] [Variant {variant}] Warning: Grippers not fully open, retrying...\033[0m"
                        )
                        env.robot.set_gripper(1.0, "left")
                        env.robot.set_gripper(1.0, "right")
                        for _ in range(20):
                            env.scene.step()
                            if env.render_freq:
                                env._update_render()
                        left_final = env.robot.get_left_gripper_val()
                        right_final = env.robot.get_right_gripper_val()
                        print(
                            f"\033[94m[Recovery] [Variant {variant}] After retry - Left: {left_final:.3f}, Right: {right_final:.3f}\033[0m"
                        )
                except Exception as e:
                    print(f"\033[93m[Recovery] [Variant {variant}] Warning: Failed to open grippers: {e}\033[0m")

            original_save_data = getattr(env, "save_data", False)
            original_save_dir = getattr(env, "save_dir", None)
            original_save_freq = getattr(env, "save_freq", None)
            original_record_policy_dense_frames = getattr(env, "record_policy_dense_frames", False)
            original_move = env.move
            if hasattr(env, "folder_path"):
                env.folder_path = None

            env.robot.set_origin_endpose()
            print(
                f"\033[94m[Recovery] [Variant {variant}] Set origin_pose to current recovery state for collision avoidance\033[0m"
            )

            recovery_sample_mode, recovery_sample_freq = recovery_video_sample_config(args, self.eval_video_freq)
            env.save_data = True
            env.save_dir = layout.recovery_root()
            env.ep_num = episode_index
            env.folder_path = {"cache": layout.recovery_cache_dir(episode_index, variant) + os.sep}
            env.FRAME_IDX = 0
            env.save_freq = recovery_sample_freq if recovery_sample_mode == "sim" else None
            env.record_policy_dense_frames = recovery_sample_mode == "sim"
            print(
                f"\033[94m[Recovery] [Variant {variant}] Recording mode={recovery_sample_mode}, "
                f"save_freq={recovery_sample_freq}\033[0m"
            )

            try:
                write_env_frame_to_cache(env)
                write_hold_frames(
                    env,
                    int(args.get("recovery_planning_hold_frames", DEFAULT_RECOVERY_PLANNING_HOLD_FRAMES)),
                )
                print(f"\033[94m[Recovery] [Variant {variant}] Saved recovery start keyframe\033[0m")
            except Exception as e:
                print(
                    f"\033[93m[Recovery] [Variant {variant}] Warning: Failed to save recovery start keyframe: {e}\033[0m"
                )

            recovery_execution_mode = str(
                args.get("recovery_execution_mode", DEFAULT_RECOVERY_EXECUTION_MODE)
            ).lower()
            if recovery_execution_mode == "qpos_action":
                action_stride = int(args.get("recovery_qpos_action_stride", DEFAULT_RECOVERY_QPOS_ACTION_STRIDE))
                original_move = install_recovery_qpos_action_move(
                    env,
                    str(args.get("action_type", "qpos")),
                    action_stride,
                )
                print(
                    f"\033[94m[Recovery] [Variant {variant}] Executing expert plan as qpos actions "
                    f"(stride={action_stride})\033[0m"
                )
            elif recovery_execution_mode == "native":
                if recovery_sample_mode == "action":
                    original_move = install_recovery_move_frame_capture(env)
            else:
                raise ValueError(f"Invalid recovery_execution_mode: {recovery_execution_mode}")

            success = False
            variant_video_path = None
            variant_hdf5_path = None
            custom_name = episode_name

            original_grasp_actor = env.grasp_actor

            def grasp_actor_wrapper(actor, arm_tag, **kwargs):
                arm_tag_str = str(arm_tag) if hasattr(arm_tag, "__str__") else arm_tag
                if arm_tag_str == "left":
                    current_gripper_val = env.robot.get_left_gripper_val()
                else:
                    current_gripper_val = env.robot.get_right_gripper_val()

                result = original_grasp_actor(actor, arm_tag, **kwargs)
                if result is None or result[0] is None:
                    return result

                returned_arm_tag, action_list = result

                from envs.utils.action import Action

                action_arm_tag = returned_arm_tag
                tasks_need_closed_gripper = ["open_laptop", "press_stapler", "turn_switch"]
                is_task_need_closed_gripper = self.task_name in tasks_need_closed_gripper

                new_action_list = []
                injected_open_before_first_move = False
                for i, action in enumerate(action_list):
                    if action.action == "move":
                        if i == 0 or (i > 0 and action_list[i - 1].action != "gripper"):
                            if current_gripper_val < 0.8:
                                if is_task_need_closed_gripper:
                                    task_reason = {
                                        "open_laptop": "push screen",
                                        "press_stapler": "press stapler",
                                        "turn_switch": "turn switch",
                                    }.get(self.task_name, "perform action")
                                    print(
                                        f"\033[94m[Recovery] [Variant {variant}] {self.task_name} task: gripper is closed ({current_gripper_val:.3f}), allowing direct move without opening ({task_reason})\033[0m"
                                    )
                                else:
                                    open_action = Action(action_arm_tag, "open", target_gripper_pos=1.0)
                                    new_action_list.append(open_action)
                                    injected_open_before_first_move = True
                                    print(
                                        f"\033[94m[Recovery] [Variant {variant}] Inserted open gripper action before move action {i+1} (gripper was closed: {current_gripper_val:.3f})\033[0m"
                                    )
                            elif should_force_open_grippers:
                                if not injected_open_before_first_move:
                                    maintain_open_action = Action(action_arm_tag, "open", target_gripper_pos=1.0)
                                    new_action_list.append(maintain_open_action)
                                    injected_open_before_first_move = True
                                    print(
                                    f"\033[94m[Recovery] [Variant {variant}] Inserted maintain-open gripper action before move action {i+1} (prevent passive closing, recovery_force_open_grippers=True)\033[0m"
                                )
                        new_action_list.append(action)
                    elif action.action == "gripper":
                        new_action_list.append(action)
                        if action.target_gripper_pos is not None:
                            current_gripper_val = action.target_gripper_pos
                    else:
                        new_action_list.append(action)

                if not new_action_list:
                    return result

                print(
                    f"\033[94m[Recovery] [Variant {variant}] Modified action sequence: {len(action_list)} -> {len(new_action_list)} actions\033[0m"
                )
                return returned_arm_tag, new_action_list

            env.grasp_actor = grasp_actor_wrapper

            try:
                if hasattr(env, "expert_recovery_sequence"):
                    try:
                        success = env.expert_recovery_sequence(variant)
                    except NotImplementedError:
                        print(
                            f"\033[93m[Recovery] expert_recovery_sequence not implemented, using play_once with skip_stages={skip_stages}\033[0m"
                        )
                        success = run_recovery_play_once(env, skip_stages, variant)
                else:
                    success = run_recovery_play_once(env, skip_stages, variant)

                success = recovery_succeeded(env, success)

                if success:
                    try:
                        recovery_cache_dir = layout.recovery_cache_dir(episode_index, variant)
                        if rollout_cache_dir:
                            hdf5_path, video_path, frame_count = save_merged_recovery_episode(
                                layout,
                                episode_index,
                                rollout_cache_dir,
                                recovery_cache_dir,
                            )
                            remove_episode_cache_dir(rollout_cache_dir)
                            remove_episode_cache_dir(recovery_cache_dir)
                            variant_video_path = video_path
                            variant_hdf5_path = hdf5_path
                            print(
                                f"\033[92m[Recovery] Variant {variant} succeeded, saved merged trajectory "
                                f"({frame_count} frames) to {video_path}, {hdf5_path}\033[0m"
                            )
                        else:
                            save_env_hdf5_video(env, custom_name)
                            env.remove_data_cache()
                            variant_video_path = layout.recovery_video_path(episode_index)
                            variant_hdf5_path = layout.recovery_data_path(episode_index)
                            print(
                                f"\033[92m[Recovery] Variant {variant} succeeded, saved to "
                                f"{variant_video_path}, {variant_hdf5_path}\033[0m"
                            )
                    except Exception as e:
                        print(
                            f"\033[91m[Recovery] Warning: failed to save recovery outputs for variant {variant}: {e}\033[0m"
                        )
                        success = False
                        cleanup_recovery_attempt_outputs(layout, episode_index, variant)
                else:
                    cleanup_recovery_attempt_outputs(layout, episode_index, variant)
                    print(f"\033[93m[Recovery] Variant {variant} failed, recovery outputs discarded\033[0m")

            except KeyboardInterrupt:
                success = False
                cleanup_recovery_attempt_outputs(layout, episode_index, variant)
                print(f"\033[93m[Recovery] Variant {variant} interrupted, recovery cache discarded\033[0m")
                raise
            except Exception as e:
                print(f"\033[91m[Recovery] Variant {variant} failed with error: {e}\033[0m")
                import traceback

                traceback.print_exc()
                success = False
                cleanup_recovery_attempt_outputs(layout, episode_index, variant)
            finally:
                env.grasp_actor = original_grasp_actor
            restore_recovery_qpos_action_move(env, original_move)

            variant_save_dir = getattr(env, "save_dir", None)
            env.save_data = original_save_data
            env.save_dir = original_save_dir
            env.save_freq = original_save_freq
            env.record_policy_dense_frames = original_record_policy_dense_frames

            recovery_results.append(
                RecoveryVariantResult(
                    variant=variant,
                    success=bool(success),
                    variant_video_path=variant_video_path,
                    variant_hdf5_path=variant_hdf5_path,
                    recovery_save_dir=variant_save_dir,
                    custom_name=custom_name,
                )
            )

            try:
                env.close_env()
            except Exception:
                pass

            if success:
                if self.variant_success_counts is not None:
                    self.variant_success_counts[variant] = self.variant_success_counts.get(variant, 0) + 1
                print(f"\033[92m[Recovery] Success with variant {variant}\033[0m")
                break
            else:
                print(f"\033[93m[Recovery] Variant {variant} failed\033[0m")

        any_success = any(r.success for r in recovery_results)
        if not any_success:
            print("\033[91m[Recovery] All recovery variants failed\033[0m")
        return any_success, recovery_results


def build_recovery_context(task_name: str, task_env, usr_args: dict, *, eval_video_freq: int = 4) -> RecoveryContext:
    threshold_type = usr_args.get("recovery_threshold_type", DEFAULT_RECOVERY_THRESHOLD_TYPE)
    threshold_info = load_task_frame_threshold_info(task_name, threshold_type)
    frame_threshold = threshold_info.value if threshold_info is not None else None
    if threshold_info is not None:
        print(
            f"\033[94m[Recovery] Loaded frame threshold for {task_name}: "
            f"{frame_threshold} ({threshold_type}, source={threshold_info.source})\033[0m"
        )
    else:
        print(
            f"\033[93m[Recovery] Warning: No frame threshold found for {task_name}, using step-based trigger\033[0m"
        )

    recovery_force_open_grippers = usr_args.get(
        "recovery_force_open_grippers",
        DEFAULT_RECOVERY_FORCE_OPEN_GRIPPERS,
    )
    if recovery_force_open_grippers:
        print("\033[94m[Recovery] Will force open both grippers at recovery start\033[0m")

    num_stages = count_play_once_stages(task_env.__class__)
    if hasattr(task_env, "get_recovery_steps"):
        try:
            num_stages = task_env.get_recovery_steps()
            print(f"\033[94m[Recovery] Task implements get_recovery_steps(): {num_stages} stages\033[0m")
        except Exception:
            pass
    print(f"\033[94m[Recovery] Detected {num_stages} stages for {task_name}\033[0m")

    return RecoveryContext(
        task_name=task_name,
        eval_video_freq=eval_video_freq,
        frame_threshold=frame_threshold,
        force_open_grippers=recovery_force_open_grippers,
        num_stages=num_stages,
        variant_success_counts={},
        variant_total_counts={},
    )
