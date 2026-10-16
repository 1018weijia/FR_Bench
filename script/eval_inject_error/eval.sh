#!/usr/bin/env bash
# Example eval with open_gripper perturbation.
# Export variables before calling this script to override defaults.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
OPENPI_SRC="${OPENPI_SRC:-${ROBOTWIN_ROOT}/policy/openpi/packages/openpi-client/src}"
VENV_PYTHON="${VENV_PYTHON:-python}"

TASK_NAME="${TASK_NAME:-lift_pot}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
GPU_ID="${GPU_ID:-0}"
TEST_NUM="${TEST_NUM:-1}"
REWARD_VALUE="${REWARD_VALUE:-1.0}"
CKPT_SETTING="${CKPT_SETTING:-open_gripper_perturb}"
CONFIG_PATH="${CONFIG_PATH:-policy/Your_Policy/deploy_policy.yml}"
POLICY_NAME="${POLICY_NAME:-Your_Policy}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-seen}"
OPEN_GRIPPER_PROB="${OPEN_GRIPPER_PROB:-0.5}"
OPEN_GRIPPER_CLOSED_THRESHOLD="${OPEN_GRIPPER_CLOSED_THRESHOLD:-0.5}"
OPEN_GRIPPER_MAX_INJECTIONS="${OPEN_GRIPPER_MAX_INJECTIONS:-3}"
OPEN_GRIPPER_DURATION="${OPEN_GRIPPER_DURATION:-20}"
OPEN_GRIPPER_START_STEP="${OPEN_GRIPPER_START_STEP:-50}"

cd "${ROBOTWIN_ROOT}"
if [[ -d "${OPENPI_SRC}" ]]; then
  export PYTHONPATH="${OPENPI_SRC}:${PYTHONPATH:-}"
fi

EXTRA_OVERRIDES=()
if [[ -n "${RVS_WEIGHT_DIR:-}" ]]; then
  EXTRA_OVERRIDES+=(--torch_checkpoint_dir "${RVS_WEIGHT_DIR}")
fi
if [[ -n "${NORM_STATS_JSON:-}" ]]; then
  EXTRA_OVERRIDES+=(--norm_stats_path "${NORM_STATS_JSON}")
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${VENV_PYTHON}" script/eval_inject_error/eval.py \
  --config "${CONFIG_PATH}" \
  --open_gripper \
  --open_gripper_prob "${OPEN_GRIPPER_PROB}" \
  --open_gripper_closed_threshold "${OPEN_GRIPPER_CLOSED_THRESHOLD}" \
  --open_gripper_max_injections "${OPEN_GRIPPER_MAX_INJECTIONS}" \
  --open_gripper_duration "${OPEN_GRIPPER_DURATION}" \
  --open_gripper_start_step "${OPEN_GRIPPER_START_STEP}" \
  --overrides \
  --task_name "${TASK_NAME}" \
  --task_config "${TASK_CONFIG}" \
  --ckpt_setting "${CKPT_SETTING}" \
  --seed 0 \
  --policy_name "${POLICY_NAME}" \
  --instruction_type "${INSTRUCTION_TYPE}" \
  --reward_value "${REWARD_VALUE}" \
  --test_num "${TEST_NUM}" \
  "${EXTRA_OVERRIDES[@]}"
