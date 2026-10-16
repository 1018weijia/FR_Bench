import sys
import os
import subprocess
import ast

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
sys.path.append("./script")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path

import yaml
from datetime import datetime
import importlib
import argparse

# Try to use imageio-ffmpeg's ffmpeg if available
try:
    import imageio_ffmpeg
    FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG_BIN = "ffmpeg"

from generate_episode_instructions import *

current_file_path = Path(__file__).resolve()
repo_root = current_file_path.parents[2]

DEFAULT_TEST_NUM = 50
DEFAULT_OPEN_GRIPPER_PROB = 0.5
DEFAULT_OPEN_GRIPPER_CLOSED_THRESHOLD = 0.5
DEFAULT_OPEN_GRIPPER_MAX_INJECTIONS = 3
DEFAULT_OPEN_GRIPPER_DURATION = 20
DEFAULT_OPEN_GRIPPER_START_STEP = 50


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name, None)
    if env_class is None:
        raise SystemExit("No Task")
    return env_class()


def eval_function_decorator(policy_name, model_name):
    policy_model = importlib.import_module(policy_name)
    return getattr(policy_model, model_name)


def get_camera_config(camera_type):
    camera_config_path = repo_root / "task_config" / "_camera_config.yml"

    assert camera_config_path.is_file(), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def get_gripper_action_indices(action_type, left_arm_dim, right_arm_dim):
    action_type = action_type or "qpos"
    action_left_dim = int(left_arm_dim) if action_type == "qpos" else 7
    action_right_dim = int(right_arm_dim) if action_type == "qpos" else 7
    return action_left_dim, action_left_dim + 1 + action_right_dim


def action_gripper_is_closing(action, action_type, left_arm_dim, right_arm_dim, closed_threshold):
    left_gripper_idx, right_gripper_idx = get_gripper_action_indices(
        action_type,
        left_arm_dim,
        right_arm_dim,
    )
    arr = np.asarray(action)
    if arr.ndim == 1:
        if arr.shape[0] <= right_gripper_idx:
            return False, False
        return arr[left_gripper_idx] <= closed_threshold, arr[right_gripper_idx] <= closed_threshold
    if arr.ndim == 2:
        if arr.shape[1] <= right_gripper_idx:
            return False, False
        return (
            bool(np.any(arr[:, left_gripper_idx] <= closed_threshold)),
            bool(np.any(arr[:, right_gripper_idx] <= closed_threshold)),
        )
    return False, False


def force_open_gripper_action(action, action_type, left_arm_dim, right_arm_dim, *, open_left, open_right):
    if not open_left and not open_right:
        return action, 0, 0

    left_gripper_idx, right_gripper_idx = get_gripper_action_indices(
        action_type,
        left_arm_dim,
        right_arm_dim,
    )
    arr = np.array(action, copy=True)
    if arr.ndim == 1:
        if arr.shape[0] <= right_gripper_idx:
            return action, 0, 0
        if open_left:
            arr[left_gripper_idx] = 1.0
        if open_right:
            arr[right_gripper_idx] = 1.0
        return arr, int(open_left), int(open_right)

    if arr.ndim == 2:
        if arr.shape[1] <= right_gripper_idx:
            return action, 0, 0
        if open_left:
            arr[:, left_gripper_idx] = 1.0
        if open_right:
            arr[:, right_gripper_idx] = 1.0
        return arr, int(open_left), int(open_right)

    return action, 0, 0


def install_open_gripper_perturbation(
    env,
    left_arm_dim,
    right_arm_dim,
    *,
    probability,
    closed_threshold,
    max_injections,
    duration,
    start_step,
    rng_seed,
):
    original_take_action = env.take_action
    rng = np.random.default_rng(rng_seed)
    stats = {
        "actions_checked": 0,
        "actions_perturbed": 0,
        "left_forced_actions": 0,
        "right_forced_actions": 0,
        "events_started": 0,
        "left_events": 0,
        "right_events": 0,
        "max_injections": max_injections,
        "duration": duration,
        "start_step": start_step,
    }
    active_steps = {"left": 0, "right": 0}

    def take_action_with_open_gripper(action, action_type="qpos"):
        stats["actions_checked"] += 1
        if stats["actions_checked"] <= start_step:
            return original_take_action(action, action_type=action_type)

        try:
            left_is_closed = float(env.robot.get_left_gripper_val()) <= closed_threshold
            right_is_closed = float(env.robot.get_right_gripper_val()) <= closed_threshold
        except Exception:
            left_is_closed = False
            right_is_closed = False

        left_action_closing, right_action_closing = action_gripper_is_closing(
            action,
            action_type,
            left_arm_dim,
            right_arm_dim,
            closed_threshold,
        )

        remaining_events = max_injections - stats["events_started"]
        if remaining_events > 0:
            candidates = []
            if active_steps["left"] <= 0 and (left_is_closed or left_action_closing):
                if rng.random() < probability:
                    candidates.append("left")
            if active_steps["right"] <= 0 and (right_is_closed or right_action_closing):
                if rng.random() < probability:
                    candidates.append("right")

            if len(candidates) > remaining_events:
                candidates = rng.choice(candidates, size=remaining_events, replace=False).tolist()

            for arm in candidates:
                active_steps[arm] = duration
                stats["events_started"] += 1
                stats[f"{arm}_events"] += 1

        open_left = active_steps["left"] > 0
        open_right = active_steps["right"] > 0
        perturbed_action, left_count, right_count = force_open_gripper_action(
            action,
            action_type,
            left_arm_dim,
            right_arm_dim,
            open_left=open_left,
            open_right=open_right,
        )
        if open_left:
            active_steps["left"] -= 1
        if open_right:
            active_steps["right"] -= 1

        if left_count or right_count:
            stats["actions_perturbed"] += 1
            stats["left_forced_actions"] += left_count
            stats["right_forced_actions"] += right_count
        return original_take_action(perturbed_action, action_type=action_type)

    env.take_action = take_action_with_open_gripper
    return original_take_action, stats


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

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
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = save_dir

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
    test_num = usr_args.get("test_num", DEFAULT_TEST_NUM)

    model = get_model(usr_args)
    open_gripper = usr_args.get("open_gripper", False)
    open_gripper_prob = float(usr_args.get("open_gripper_prob", DEFAULT_OPEN_GRIPPER_PROB))
    open_gripper_closed_threshold = float(
        usr_args.get("open_gripper_closed_threshold", DEFAULT_OPEN_GRIPPER_CLOSED_THRESHOLD)
    )
    open_gripper_max_injections = int(
        usr_args.get("open_gripper_max_injections", DEFAULT_OPEN_GRIPPER_MAX_INJECTIONS)
    )
    open_gripper_duration = int(usr_args.get("open_gripper_duration", DEFAULT_OPEN_GRIPPER_DURATION))
    open_gripper_start_step = int(usr_args.get("open_gripper_start_step", DEFAULT_OPEN_GRIPPER_START_STEP))
    if not 0.0 <= open_gripper_prob <= 1.0:
        raise ValueError(f"open_gripper_prob must be in [0, 1], got {open_gripper_prob}")
    if open_gripper_max_injections < 0:
        raise ValueError(f"open_gripper_max_injections must be >= 0, got {open_gripper_max_injections}")
    if open_gripper_duration < 1:
        raise ValueError(f"open_gripper_duration must be >= 1, got {open_gripper_duration}")
    if open_gripper_start_step < 0:
        raise ValueError(f"open_gripper_start_step must be >= 0, got {open_gripper_start_step}")
    left_arm_dim = usr_args.get("left_arm_dim", 6)
    right_arm_dim = usr_args.get("right_arm_dim", 6)

    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type,
                                   open_gripper=open_gripper,
                                   open_gripper_prob=open_gripper_prob,
                                   open_gripper_closed_threshold=open_gripper_closed_threshold,
                                   open_gripper_max_injections=open_gripper_max_injections,
                                   open_gripper_duration=open_gripper_duration,
                                   open_gripper_start_step=open_gripper_start_step,
                                   left_arm_dim=left_arm_dim,
                                   right_arm_dim=right_arm_dim)
    suc_nums.append(suc_num)

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                open_gripper=False,
                open_gripper_prob=DEFAULT_OPEN_GRIPPER_PROB,
                open_gripper_closed_threshold=DEFAULT_OPEN_GRIPPER_CLOSED_THRESHOLD,
                open_gripper_max_injections=DEFAULT_OPEN_GRIPPER_MAX_INJECTIONS,
                open_gripper_duration=DEFAULT_OPEN_GRIPPER_DURATION,
                open_gripper_start_step=DEFAULT_OPEN_GRIPPER_START_STEP,
                left_arm_dim=6,
                right_arm_dim=6):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")
    if open_gripper:
        print(
            f"\033[93m[Inject Error] Enabled: open closed/closing gripper actions "
            f"with probability={open_gripper_prob}, threshold={open_gripper_closed_threshold}, "
            f"duration={open_gripper_duration}, max_events={open_gripper_max_injections}, "
            f"start_after={open_gripper_start_step} actions\033[0m"
        )

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

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError:
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception:
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

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    FFMPEG_BIN,
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        reset_func(model)
        original_take_action = None
        perturb_stats = None
        if open_gripper:
            original_take_action, perturb_stats = install_open_gripper_perturbation(
                TASK_ENV,
                left_arm_dim,
                right_arm_dim,
                probability=open_gripper_prob,
                closed_threshold=open_gripper_closed_threshold,
                max_injections=open_gripper_max_injections,
                duration=open_gripper_duration,
                start_step=open_gripper_start_step,
                rng_seed=now_seed,
            )
        try:
            while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
                observation = TASK_ENV.get_obs()
                eval_func(TASK_ENV, model, observation)
                if TASK_ENV.eval_success:
                    succ = True
                    break
        finally:
            if original_take_action is not None:
                TASK_ENV.take_action = original_take_action
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if perturb_stats is not None:
            print(
                f"\033[93m[Inject Error] Open-gripper perturbations applied: "
                f"{perturb_stats['events_started']}/{perturb_stats['max_injections']} events, "
                f"duration={perturb_stats['duration']}, start_after={perturb_stats['start_step']} actions, "
                f"{perturb_stats['actions_perturbed']}/{perturb_stats['actions_checked']} actions "
                f"(left_events={perturb_stats['left_events']}, right_events={perturb_stats['right_events']}, "
                f"left_forced={perturb_stats['left_forced_actions']}, "
                f"right_forced={perturb_stats['right_forced_actions']})\033[0m"
            )

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
        now_seed += 1

    return now_seed, TASK_ENV.suc


def parse_override_pairs(pairs):
    override_dict = {}
    if len(pairs) % 2 != 0:
        raise ValueError("overrides must be provided as --key value pairs")
    for i in range(0, len(pairs), 2):
        key = pairs[i].lstrip("--")
        value = pairs[i + 1]
        try:
            value = ast.literal_eval(value)
        except Exception:
            pass
        override_dict[key] = value
    return override_dict


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--open_gripper", action="store_true", help="Enable open-gripper perturbation during policy eval")
    parser.add_argument("--open_gripper_prob", type=float, default=None, help="Probability of opening a closed/closing gripper action")
    parser.add_argument("--open_gripper_closed_threshold", type=float, default=None, help="Gripper value <= this threshold is treated as closed/closing")
    parser.add_argument("--open_gripper_max_injections", type=int, default=None, help="Maximum number of gripper-open perturbations per episode")
    parser.add_argument("--open_gripper_duration", type=int, default=None, help="Number of consecutive policy actions to force open after each perturbation event")
    parser.add_argument("--open_gripper_start_step", type=int, default=None, help="Do not inject during the first N policy actions")
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Add open_gripper to config if specified
    if args.open_gripper:
        config["open_gripper"] = True
    if args.open_gripper_prob is not None:
        config["open_gripper_prob"] = args.open_gripper_prob
    if args.open_gripper_closed_threshold is not None:
        config["open_gripper_closed_threshold"] = args.open_gripper_closed_threshold
    if args.open_gripper_max_injections is not None:
        config["open_gripper_max_injections"] = args.open_gripper_max_injections
    if args.open_gripper_duration is not None:
        config["open_gripper_duration"] = args.open_gripper_duration
    if args.open_gripper_start_step is not None:
        config["open_gripper_start_step"] = args.open_gripper_start_step

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
