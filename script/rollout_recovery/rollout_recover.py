import sys
import os
import shutil

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError
import numpy as np
from pathlib import Path

import yaml
from datetime import datetime
import importlib

from description.utils.generate_episode_instructions import generate_episode_descriptions
from script.rollout_recovery.config import (
    get_camera_config,
    get_embodiment_config,
    parse_args_and_config,
)
from script.rollout_recovery.recovery import build_recovery_context, save_env_hdf5_video
from script.rollout_recovery.recovery_utils import (
    DEFAULT_RECOVERY_MODE,
    DEFAULT_RECOVERY_VIDEO_SAMPLE_FREQ,
    DEFAULT_RECOVERY_VIDEO_SAMPLE_MODE,
    EpisodeSaveLayout,
    count_cache_frames,
    remove_episode_cache_dir,
    write_hdf5_attrs,
)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e


def finalize_rollout_episode(
    env,
    episode_index: int,
    reason: str,
) -> None:
    episode_name = EpisodeSaveLayout.episode_name(episode_index)
    try:
        save_env_hdf5_video(env, episode_name)
        env.remove_data_cache()
        print(f"\033[94m[Rollout] Saved rollout data/video {reason}\033[0m")
    except Exception as e:
        print(f"\033[93m[Rollout] Warning: Failed to save rollout data: {e}\033[0m")


def discard_rollout_episode(env, reason: str) -> None:
    try:
        env.remove_data_cache()
        print(f"\033[93m[Rollout] Discarded rollout cache {reason}\033[0m")
    except Exception as e:
        print(f"\033[93m[Rollout] Warning: Failed to discard rollout cache: {e}\033[0m")


def freeze_rollout_cache(cache_dir: str | None) -> str | None:
    if not cache_dir or not os.path.isdir(cache_dir):
        return None
    frozen_dir = f"{cache_dir}_pre_recovery"
    if os.path.exists(frozen_dir):
        shutil.rmtree(frozen_dir)
    shutil.copytree(cache_dir, frozen_dir)
    frame_count = count_cache_frames(frozen_dir)
    print(f"\033[94m[Rollout] Frozen clean pre-recovery cache: {frozen_dir} ({frame_count} frames)\033[0m")
    return frozen_dir


def cleanup_cache_dir(cache_dir: str | None) -> None:
    remove_episode_cache_dir(cache_dir)


def save_cache_episode(
    cache_dir: str,
    hdf5_path: str,
    video_path: str,
    *,
    attrs: dict | None = None,
) -> None:
    from envs.utils.pkl2hdf5 import process_folder_to_hdf5_video

    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    process_folder_to_hdf5_video(cache_dir, hdf5_path, video_path)
    if attrs:
        write_hdf5_attrs(hdf5_path, attrs)


def install_rollout_frame_capture(env, sample_mode: str):
    original_take_action = env.take_action
    if sample_mode == "sim":
        env.record_policy_dense_frames = True
        return original_take_action

    env.record_policy_dense_frames = False

    def take_action_with_frame_save(action, action_type="qpos"):
        result = original_take_action(action, action_type=action_type)
        if getattr(env, "save_data", False):
            env._update_render()
            env._take_picture()
        return result

    env.take_action = take_action_with_frame_save
    return original_take_action


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f, Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f, Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir
        EpisodeSaveLayout(str(video_save_dir)).ensure_rollout_dirs()

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = 500

    model = get_model(usr_args)
    st_seed, suc_num = eval_policy(
        task_name,
        TASK_ENV,
        args,
        model,
        st_seed,
        usr_args,
        test_num=test_num,
        video_size=video_size,
        instruction_type=instruction_type,
    )
    suc_nums.append(suc_num)

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")
    # return task_reward


def eval_policy(
    task_name,
    TASK_ENV,
    args,
    model,
    st_seed,
    usr_args,
    test_num=100,
    video_size=None,
    instruction_type=None,
):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True
    EVAL_VIDEO_FREQ = 4  # Must match eval_video_freq in eval_policy.py

    # ================ Universal Recovery System (modularized) ================
    recovery_ctx = build_recovery_context(task_name, TASK_ENV, usr_args, eval_video_freq=EVAL_VIDEO_FREQ)
    variant_success_counts = recovery_ctx.variant_success_counts
    variant_total_counts = recovery_ctx.variant_total_counts
    # ===========================================================

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print("error occurs !")
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        episode_index = TASK_ENV.test_num
        succ = False
        reset_func(model)
        recovery_attempted = False
        rollout_failed = False
        rollout_cache_dir = None
        pre_recovery_cache_dir = None

        original_save_data_rollout = getattr(TASK_ENV, "save_data", False)
        original_save_dir_rollout = getattr(TASK_ENV, "save_dir", None)
        original_save_freq_rollout = getattr(TASK_ENV, "save_freq", None)
        original_ep_num_rollout = getattr(TASK_ENV, "ep_num", None)
        original_eval_video_path = getattr(TASK_ENV, "eval_video_path", None)
        original_take_action_rollout = None

        eval_save_dir = args.get("eval_video_save_dir")
        eval_save_dir = str(eval_save_dir) if eval_save_dir is not None else None
        eval_video_log_enabled = eval_save_dir is not None
        if eval_video_log_enabled:
            # Rollout uses pkl->hdf5/video save; disable ffmpeg streaming in take_action.
            TASK_ENV.eval_video_path = None
            print(f"\033[94m[Rollout] eval_video_log is enabled, will save data/video during rollout\033[0m")
            layout = EpisodeSaveLayout(eval_save_dir)
            layout.ensure_rollout_dirs()
            video_sample_mode = str(
                args.get("recovery_video_sample_mode", DEFAULT_RECOVERY_VIDEO_SAMPLE_MODE)
            ).lower()
            if video_sample_mode not in {"action", "sim"}:
                raise ValueError(f"Invalid recovery_video_sample_mode: {video_sample_mode}")
            # Use a coarser sim-step sample for evaluation videos so rollout/recovery playback
            # stays close to the original policy-eval visual pace.
            video_sample_freq = int(
                args.get(
                    "recovery_video_sample_freq",
                    DEFAULT_RECOVERY_VIDEO_SAMPLE_FREQ if video_sample_mode == "sim" else EVAL_VIDEO_FREQ,
                )
            )

            TASK_ENV.save_data = True
            TASK_ENV.save_dir = eval_save_dir
            TASK_ENV.ep_num = episode_index
            TASK_ENV.save_freq = video_sample_freq
            TASK_ENV.folder_path = {"cache": layout.episode_cache_dir(episode_index)}
            TASK_ENV.FRAME_IDX = 0
            TASK_ENV.record_policy_dense_frames = video_sample_mode == "sim"
            print(
                f"\033[94m[Rollout] Recording mode={video_sample_mode}, "
                f"save_freq={video_sample_freq}\033[0m"
            )
            original_take_action_rollout = install_rollout_frame_capture(TASK_ENV, video_sample_mode)
            rollout_cache_dir = layout.episode_cache_dir(episode_index)
        else:
            if args.get("eval_video_log", False):
                print(
                    "\033[93m[Rollout] eval_video_log requested but eval_video_path is None; "
                    "data/video will NOT be saved during rollout\033[0m"
                )
            else:
                print(
                    "\033[93m[Rollout] eval_video_log is NOT enabled, data/video will NOT be saved during rollout\033[0m"
                )

        env = TASK_ENV
        get_obs = env.get_obs
        policy_step = eval_func
        should_trigger = recovery_ctx.should_trigger_recovery
        recovery_variant_results = None

        while env.take_action_cnt < env.step_lim:
            observation = get_obs()
            policy_step(env, model, observation)

            if (not recovery_attempted) and should_trigger(env) and (not env.eval_success):
                recovery_attempted = True
                rollout_failed = True
                if eval_video_log_enabled:
                    env._update_render()
                    env._take_picture()
                pre_recovery_cache_dir = freeze_rollout_cache(rollout_cache_dir)

                recovery_mode = str(args.get("recovery_mode", DEFAULT_RECOVERY_MODE)).lower()
                print(f"\033[94m[Recovery] Using recovery_mode={recovery_mode}\033[0m")
                if recovery_mode == "inplace":
                    recovered, recovery_variant_results = recovery_ctx.try_in_place_recovery(
                        env,
                        episode_index=episode_index,
                        eval_save_dir=eval_save_dir,
                        args=args,
                        recovery_seed=now_seed,
                        rollout_cache_dir=pre_recovery_cache_dir,
                    )
                else:
                    snapshot = recovery_ctx.snapshot_env(env)
                    recovered, recovery_variant_results = recovery_ctx.try_multi_env_recovery(
                        snapshot,
                        episode_index=episode_index,
                        eval_save_dir=eval_save_dir,
                        args=args,
                        recovery_seed=now_seed,
                        rollout_cache_dir=pre_recovery_cache_dir,
                    )

                if recovered:
                    succ = True
                break

            if env.eval_success:
                succ = True
                break

        if eval_video_log_enabled:
            if succ and not rollout_failed:
                finalize_rollout_episode(env, episode_index, "(direct success)")
            elif succ and rollout_failed:
                cleanup_cache_dir(pre_recovery_cache_dir)
                cleanup_cache_dir(rollout_cache_dir)
                TASK_ENV.folder_path = None
            elif rollout_failed and not succ:
                layout.ensure_fail_dirs()
                try:
                    cache_to_save = pre_recovery_cache_dir
                    if cache_to_save is None:
                        raise FileNotFoundError("No frozen pre-recovery rollout cache available; refusing to save possibly recovery-polluted fail trajectory")
                    fail_frame_count = count_cache_frames(cache_to_save)
                    save_cache_episode(
                        cache_to_save,
                        layout.fail_data_path(episode_index),
                        layout.fail_video_path(episode_index),
                        attrs={
                            "contains_recovery": False,
                            "recovery_success": False,
                            "recovery_start_frame": fail_frame_count,
                            "rollout_frame_count": fail_frame_count,
                            "recovery_frame_count": 0,
                            "total_frame_count": fail_frame_count,
                        },
                    )
                    print("\033[94m[Rollout] Saved clean pre-recovery rollout to fail/\033[0m")
                except Exception as e:
                    print(f"\033[93m[Rollout] Warning: Failed to save clean failed rollout: {e}\033[0m")
                finally:
                    cleanup_cache_dir(pre_recovery_cache_dir)
                    cleanup_cache_dir(rollout_cache_dir)
                    TASK_ENV.folder_path = None
            elif not succ:
                discard_rollout_episode(env, "(episode failed; rollout not saved)")

            TASK_ENV.save_data = original_save_data_rollout
            TASK_ENV.save_dir = original_save_dir_rollout
            TASK_ENV.save_freq = original_save_freq_rollout
            if original_ep_num_rollout is not None:
                TASK_ENV.ep_num = original_ep_num_rollout
            TASK_ENV.eval_video_path = original_eval_video_path
            TASK_ENV.record_policy_dense_frames = False
            if original_take_action_rollout is not None:
                TASK_ENV.take_action = original_take_action_rollout

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += 1

    # Print per-variant recovery statistics
    if variant_total_counts:
        print("\n\033[94m[Recovery Summary per variant]\033[0m")
        for v in sorted(variant_total_counts.keys()):
            total = variant_total_counts.get(v, 0)
            succ_v = variant_success_counts.get(v, 0)
            if total == 0:
                rate = 0.0
            else:
                rate = round(succ_v / total * 100.0, 1)
            print(f"  Variant {v}: {succ_v}/{total} => {rate}%")

    return now_seed, TASK_ENV.suc
if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
