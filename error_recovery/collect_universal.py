#!/usr/bin/env python3
"""
Universal error-recovery data collection script

Usage:
    # Collect premature-close error data for the pick_diverse_bottles task
    python error_recovery/collect_universal.py --task pick_diverse_bottles --error_types premature_close --episodes 100
    
    # Collect multiple error types
    python error_recovery/collect_universal.py --task pick_diverse_bottles --error_types premature_close grasp_slip --episodes 200
    
    # Custom error probabilities
    python error_recovery/collect_universal.py --task pick_diverse_bottles --error_types premature_close:0.3 grasp_position_offset:0.4 --episodes 150
    
    # Collect only normal data (no error injection)
    python error_recovery/collect_universal.py --task stack_blocks_two --error_types none --episodes 50
"""

import sys
import os
import argparse
import logging
import warnings
from typing import Dict, List

warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"The symbol `warp\.torch\.device_from_torch`.*",
    category=DeprecationWarning,
)

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _apply_thread_limits_early() -> None:
    """Cap torch/OpenCV pools before the simulator import creates them.

    Libraries size their pools from the machine's CPU count at first use.
    The supervisor sets FRBENCH_NUM_THREADS before exec; applying it here
    keeps a worker from starting one thread per core.
    """
    raw = os.environ.get("FRBENCH_NUM_THREADS", "")
    if not raw:
        return
    try:
        limit = int(raw)
    except ValueError:
        return
    if limit <= 0:
        return
    try:
        import torch

        torch.set_num_threads(limit)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except Exception:
        pass
    try:
        import cv2

        cv2.setNumThreads(limit)
    except Exception:
        pass


_apply_thread_limits_early()

from error_recovery import ErrorRecoveryFramework


def parse_error_types(error_types_str: List[str]) -> tuple:
    """Parse error types, probabilities, and parameters
    
    Args:
        error_types_str: List of error-type strings, for example:
            - 'premature_close:0.3'  # Set the probability only
            - 'premature_close:0.3:0.7'  # Set the probability and close_ratio
            - 'premature_close'  # Use the defaults
    
    Returns:
        (error_types, error_probabilities, error_params)
    """
    if not error_types_str or error_types_str == ['none']:
        return [], {}, {}
    
    error_types = []
    error_probabilities = {}
    error_params = {}
    
    for error_str in error_types_str:
        parts = error_str.split(':')
        error_type = parts[0]
        
        # Parse the probability
        if len(parts) >= 2:
            try:
                probability = float(parts[1])
                if not 0.0 <= probability <= 1.0:
                    raise ValueError(f"probability must be between 0.0 and 1.0: {probability}")
            except ValueError as e:
                raise ValueError(f"Invalid probability value '{parts[1]}': {e}")
        else:
            probability = 0.5  # Default probability
        
        # Parse parameters for specific error types
        params = {}
        if error_type == 'premature_close' and len(parts) >= 3:
            # The third premature_close parameter is close_ratio
            try:
                close_ratio = float(parts[2])
                if not 0.0 <= close_ratio <= 1.0:
                    raise ValueError(f"close_ratio must be between 0.0 and 1.0: {close_ratio}")
                params['close_ratio'] = close_ratio
            except ValueError as e:
                raise ValueError(f"Invalid close_ratio value '{parts[2]}': {e}")
        elif error_type == 'grasp_position_offset' and len(parts) >= 3:
            # The third grasp_position_offset parameter is offset_range_min
            try:
                offset_min = float(parts[2])
                if offset_min < 0:
                    raise ValueError(f"offset_min must be >= 0: {offset_min}")
                params['offset_range'] = (offset_min, offset_min * 2.0)  # Default maximum is twice the minimum
                # If a fourth parameter is provided, use it as the maximum
                if len(parts) >= 4:
                    offset_max = float(parts[3])
                    if offset_max < offset_min:
                        raise ValueError(f"offset_max must be >= offset_min: {offset_max} < {offset_min}")
                    params['offset_range'] = (offset_min, offset_max)
            except ValueError as e:
                raise ValueError(f"Invalid offset_range value: {e}")
        elif error_type == 'grasp_slip' and len(parts) >= 3:
            # The third grasp_slip parameter is force_reduction (grip-force reduction ratio)
            try:
                force_reduction = float(parts[2])
                if not 0.0 <= force_reduction <= 1.0:
                    raise ValueError(f"force_reduction must be between 0.0 and 1.0: {force_reduction}")
                params['force_reduction'] = force_reduction
            except ValueError as e:
                raise ValueError(f"Invalid force_reduction value '{parts[2]}': {e}")
        elif error_type == 'grasp_orientation_mismatch' and len(parts) >= 3:
            # The third and fourth grasp_orientation_mismatch parameters are rotation_range (min/max; units follow the implementation)
            try:
                rot_min = float(parts[2])
                rot_max = rot_min * 2.0
                if len(parts) >= 4:
                    rot_max = float(parts[3])
                if rot_max < rot_min:
                    raise ValueError(f"rot_max must be >= rot_min: {rot_max} < {rot_min}")
                params['rotation_range'] = (rot_min, rot_max)
            except ValueError as e:
                raise ValueError(f"Invalid rotation_range value: {e}")
        
        error_types.append(error_type)
        error_probabilities[error_type] = probability
        if params:
            error_params[error_type] = params
    
    return error_types, error_probabilities, error_params


def main():
    parser = argparse.ArgumentParser(
        description="Universal error-recovery data collection script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Error types:
    - To see which error types are supported, run: python error_recovery/collect_universal.py --list_error_types
    - none means do not inject errors and collect only normal data

Parameter format:
    Format: error_type:probability[:param1:param2...]
    Probability range: 0.0-1.0 (default: 0.5)

Examples:
    # Basic use (premature_close by default, probability 50%)
    python error_recovery/collect_universal.py --task pick_diverse_bottles --episodes 100
    
    # Select a GPU (GPU 0, or GPUs 0 and 1)
    python error_recovery/collect_universal.py --task pick_diverse_bottles --episodes 100 --gpu 0 # or --gpu 0, 1
    
    # Specify multiple error types
    python error_recovery/collect_universal.py --task place_bread_basket \
        --error_types premature_close grasp_slip
    
    # Custom probabilities
    python error_recovery/collect_universal.py --task handover_block \
        --error_types premature_close:0.3 grasp_slip:0.4
    
    # Collect only normal data (no error injection)
    python error_recovery/collect_universal.py --task stack_blocks_two --error_types none
    
    # Collect rich error-recovery data (recommended)
    python error_recovery/collect_universal.py --task lift_pot \
        --error_types premature_close:0.4 grasp_position_offset:0.3 grasp_slip:0.2 --episodes 200
    
    # List all available tasks and error types
    python error_recovery/collect_universal.py --list_tasks # or --list_error_types
        """
    )
    
    parser.add_argument(
        "--task",
        type=str,
        required=False,  # Optional, because the user may only want to list tasks
        help="Task name (a matching .py file must exist under envs/)"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to collect (default: 100)"
    )
    
    parser.add_argument(
        "--error_types",
        nargs='+',
        default=['premature_close'],
        help="Error type list; a probability may be specified, e.g. 'premature_close:0.3' (default: premature_close)"
    )
    
    parser.add_argument(
        "--save_path",
        type=str,
        help="Data save path (default: ./error_recovery/data/{task_name})"
    )
    
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering (for debugging)"
    )
    parser.add_argument(
        "--scene",
        choices=["clean", "random"],
        default="clean",
        help="clean=white tabletop with no randomization; random=load cluttered tabletop, background, and lighting randomization from task_config/demo_randomized.yml (default: clean)",
    )
    parser.add_argument(
        "--log_level",
        choices=["quiet", "normal", "debug"],
        default="quiet",
        help="Collection log verbosity: quiet=concise by default, normal=recovery-flow summary, debug=detailed diagnostics (default: quiet)",
    )

    parser.add_argument(
        "--data_types",
        nargs="+",
        default=None,
        help=(
            "Observation data types to save (overrides the default config). For example: "
            "--data_types rgb qpos endpose  (do not save depth or point cloud)\n"
            "Options (depend on the task implementation): rgb depth pointcloud qpos endpose third_view mesh_segmentation actor_segmentation conbine"
        ),
    )
    parser.add_argument(
        "--no_depth",
        action="store_true",
        help="Do not save depth images",
    )
    parser.add_argument(
        "--pointcloud",
        action="store_true",
        help="Save point clouds",
    )
    parser.add_argument(
        "--trim_failure_trajectory_on_save",
        action="store_true",
        default=True,
        help="Trim the official failed-trajectory output to before the first failure-detection frame (enabled by default)",
    )
    parser.add_argument(
        "--no_trim_failure_trajectory_on_save",
        dest="trim_failure_trajectory_on_save",
        action="store_false",
        help="Disable official failed-trajectory trimming and keep the full failed trajectory (for debugging)",
    )
    
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU IDs (e.g. '0' or '0,1'); sets the CUDA_VISIBLE_DEVICES environment variable (default: system default GPU)"
    )
    parser.add_argument("--seed_start", type=int, default=0, help="First random seed to try (default: 0)")
    parser.add_argument(
        "--seed_end",
        type=int,
        default=None,
        help="Exclusive seed upper bound. When set, the episodes*5 attempt cap is not used",
    )
    parser.add_argument("--episode_offset", type=int, default=0, help="Starting episode index written by this process")
    parser.add_argument("--worker_id", type=str, default=None, help="Parallel worker id, written into logs and metadata")
    parser.add_argument("--events_path", type=str, default=None, help="JSONL path to append each attempt result")
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Exit immediately (exit code 75) when CUDA or cuRobo enters an unrecoverable state, instead of failing silently",
    )
    parser.add_argument(
        "--max_episodes_per_process",
        type=int,
        default=0,
        help="Maximum number of successfully saved episodes before this process exits normally; 0 means unlimited",
    )
    parser.add_argument(
        "--wait_stable_before_grasp",
        action="store_true",
        help="Wait for the target object to settle before grasping (disabled by default)",
    )
    parser.add_argument(
        "--wait_stable_only_in_recovery",
        type=int,
        choices=[0, 1],
        default=1,
        help="Whether to wait for stability only during recovery (1=yes, 0=no, default: 1)",
    )
    parser.add_argument(
        "--wait_stable_require_success_in_recovery",
        type=int,
        choices=[0, 1],
        default=0,
        help="Whether to forbid the grasp when recovery stability detection times out (1=yes, 0=no, default: 0)",
    )
    parser.add_argument(
        "--wait_stable_max_steps",
        type=int,
        default=120,
        help="Maximum steps to wait for stability before grasping (default: 120)",
    )
    parser.add_argument(
        "--wait_stable_window",
        type=int,
        default=20,
        help="Stability detection window length before grasping (default: 20)",
    )
    parser.add_argument(
        "--wait_stable_pos_eps",
        type=float,
        default=0.002,
        help="Position stability threshold before grasping (meters, default: 0.002)",
    )
    parser.add_argument(
        "--wait_stable_post_extra_steps",
        type=int,
        default=10,
        help="Extra wait steps after stability is confirmed (default: 10)",
    )
    parser.add_argument(
        "--wait_stable_consecutive_windows",
        type=int,
        default=3,
        help="Number of consecutive windows that must meet the stability threshold (default: 3)",
    )
    parser.add_argument(
        "--wait_stable_lin_vel_eps",
        type=float,
        default=0.03,
        help="Linear-velocity threshold before grasping (m/s, default: 0.03)",
    )
    parser.add_argument(
        "--wait_stable_ang_vel_eps_deg",
        type=float,
        default=20.0,
        help="Angular-velocity threshold before grasping (deg/s, default: 20.0)",
    )
    parser.add_argument(
        "--wait_stable_record_frames",
        action="store_true",
        help="Record trajectory frames while waiting for stability (so the wait is visible in the video)",
    )
    
    parser.add_argument(
        "--list_tasks",
        action="store_true",
        help="List all available tasks"
    )
    
    parser.add_argument(
        "--list_error_types",
        action="store_true",
        help="List all supported error types"
    )
    
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO if args.log_level != "quiet" else logging.WARNING)
    
    # Set the GPU (if specified)
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        print(f"🔧 Set CUDA_VISIBLE_DEVICES={args.gpu}")
    
    # List available tasks
    if args.list_tasks:
        print("Available tasks:")
        envs_dir = "./envs"
        if os.path.exists(envs_dir):
            for file in sorted(os.listdir(envs_dir)):
                if file.endswith('.py') and not file.startswith('_'):
                    task_name = file[:-3]  # Remove the .py suffix
                    print(f"  - {task_name}")
        return
    
    # List supported error types
    if args.list_error_types:
        framework = ErrorRecoveryFramework()
        print("Supported error types:")
        for error_type in framework.list_supported_error_types():
            print(f"  - {error_type}")
        return
    
    # Check that a task name was provided (unless only listing tasks)
    if not args.task:
        parser.error("The --task argument is required (or use --list_tasks to see available tasks)")
    
    # Parse error types
    try:
        error_types, error_probabilities, error_params = parse_error_types(args.error_types)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Create the framework
    framework = ErrorRecoveryFramework()

    # data_type selection (if omitted, use the framework default config)
    data_type = None
    if args.data_types is not None:
        data_type = {k: True for k in args.data_types}
    elif args.no_depth or args.pointcloud:
        data_type = {
            "rgb": True,
            "depth": not args.no_depth,
            "qpos": True,
            "endpose": True,
        }
        if args.pointcloud:
            data_type["pointcloud"] = True
    
    # Collect data
    try:
        stats = framework.collect_data(
            task_name=args.task,
            episodes=args.episodes,
            error_types=error_types,
            error_probabilities=error_probabilities,
            error_params=error_params,
            save_path=args.save_path,
            render=args.render,
            scene=args.scene,
            data_type=data_type,
            wait_stable_before_grasp_enable=args.wait_stable_before_grasp,
            wait_stable_only_in_recovery=bool(args.wait_stable_only_in_recovery),
            wait_stable_require_success_in_recovery=bool(args.wait_stable_require_success_in_recovery),
            wait_stable_max_steps=args.wait_stable_max_steps,
            wait_stable_window=args.wait_stable_window,
            wait_stable_pos_eps=args.wait_stable_pos_eps,
            wait_stable_post_extra_steps=args.wait_stable_post_extra_steps,
            wait_stable_consecutive_windows=args.wait_stable_consecutive_windows,
            wait_stable_lin_vel_eps=args.wait_stable_lin_vel_eps,
            wait_stable_ang_vel_eps_deg=args.wait_stable_ang_vel_eps_deg,
            wait_stable_record_frames=args.wait_stable_record_frames,
            trim_failure_trajectory_on_save=args.trim_failure_trajectory_on_save,
            error_recovery_log_level=args.log_level,
            seed_start=args.seed_start,
            episode_offset=args.episode_offset,
            fail_fast=args.fail_fast,
            max_episodes_per_process=args.max_episodes_per_process,
            **(
                {"seed_end": args.seed_end}
                if args.seed_end is not None
                else {}
            ),
            **(
                {"worker_id": args.worker_id, "events_path": args.events_path}
                if args.worker_id or args.events_path
                else {}
            ),
        )
        
        print(f"\n🎉 Data collection complete!")
        resolved_save_path = stats.get('save_path', args.save_path or f'./error_recovery/data/{args.task}')
        print(f"Statistics saved to: {resolved_save_path}/collection_stats.json")
        
    except Exception as e:
        print(f"❌ Data collection failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
