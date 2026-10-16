#!/usr/bin/env bash
set -euo pipefail

# FRdata_collect_test.sh — Batch-collect error-recovery data (run collect_universal.py once for each of four error types)
#
# Run from the repository root. By default, collect EPISODES_PER_TYPE episodes for each of 4 error types and save them to
# error_recovery/data/<task>_<RUN_TAG>_<timestamp>/<error_type>/
#
# Usage:
#   bash script/FRdata_collect_test.sh
#   bash script/FRdata_collect_test.sh pick_diverse_bottles
#   bash script/FRdata_collect_test.sh place_bread_basket 10
#   TASK_NAME=place_bread_basket bash script/FRdata_collect_test.sh 10
#   ERROR_SPECS='premature_close:1.0:0.5;grasp_slip:1.0:0.2' bash script/FRdata_collect_test.sh pick_diverse_bottles 20
#
# Args:
#   $1  task name (default: pick_diverse_bottles)
#   $2  episodes per error type (default: 10)
#
# Env (optional):
#   TASK_NAME, EPISODES_PER_TYPE, RUN_TAG (default: error_tests)
#   ERROR_SPECS — semicolon-separated; each item has the form: <save_subdir>|<collect_universal --error_types argument>
#   WAIT_STABLE_* / TRIM_FAILURE_TRAJECTORY_ON_SAVE — see the defaults below

TASK_NAME="${1:-${TASK_NAME:-pick_diverse_bottles}}"
EPISODES_PER_TYPE="${2:-${EPISODES_PER_TYPE:-10}}"
RUN_TAG="${RUN_TAG:-error_tests}"
ROOT_SAVE_DIR="error_recovery/data/${TASK_NAME}_${RUN_TAG}_$(date +%Y%m%d_%H%M%S)"

# Stability wait switches (enabled by default for this batch script)
WAIT_STABLE_BEFORE_GRASP="${WAIT_STABLE_BEFORE_GRASP:-1}"
WAIT_STABLE_ONLY_IN_RECOVERY="${WAIT_STABLE_ONLY_IN_RECOVERY:-1}"
WAIT_STABLE_REQUIRE_SUCCESS_IN_RECOVERY="${WAIT_STABLE_REQUIRE_SUCCESS_IN_RECOVERY:-0}"
WAIT_STABLE_MAX_STEPS="${WAIT_STABLE_MAX_STEPS:-2000}"
WAIT_STABLE_POST_EXTRA_STEPS="${WAIT_STABLE_POST_EXTRA_STEPS:-10}"
WAIT_STABLE_LIN_VEL_EPS="${WAIT_STABLE_LIN_VEL_EPS:-0.03}"
WAIT_STABLE_ANG_VEL_EPS_DEG="${WAIT_STABLE_ANG_VEL_EPS_DEG:-20.0}"
WAIT_STABLE_RECORD_FRAMES="${WAIT_STABLE_RECORD_FRAMES:-1}"
TRIM_FAILURE_TRAJECTORY_ON_SAVE="${TRIM_FAILURE_TRAJECTORY_ON_SAVE:-1}"

echo "Task: ${TASK_NAME}"
echo "Episodes per type: ${EPISODES_PER_TYPE}"
echo "Save root: ${ROOT_SAVE_DIR}"
echo

DEFAULT_ERROR_SPECS=(
  "premature_close|premature_close:1.0:0.5"
  "grasp_position_offset|grasp_position_offset:1.0:0.05:0.08"
  "grasp_slip|grasp_slip:1.0:0.2"
  "grasp_orientation_mismatch|grasp_orientation_mismatch:1.0"
)

IFS=';' read -r -a ERROR_SPECS <<< "${ERROR_SPECS:-}"
if [ "${#ERROR_SPECS[@]}" -eq 0 ] || [ -z "${ERROR_SPECS[0]}" ]; then
  ERROR_SPECS=("${DEFAULT_ERROR_SPECS[@]}")
fi

run_one() {
  local type_name="$1"
  local error_spec="$2"
  local save_dir="${ROOT_SAVE_DIR}/${type_name}"
  local stable_args=()

  echo "============================================================"
  echo "Running: ${type_name}"
  echo "Spec:    ${error_spec}"
  echo "Save to: ${save_dir}"
  echo "============================================================"

  if [ "${WAIT_STABLE_BEFORE_GRASP}" = "1" ]; then
    stable_args+=(--wait_stable_before_grasp)
  fi
  if [ "${WAIT_STABLE_RECORD_FRAMES}" = "1" ]; then
    stable_args+=(--wait_stable_record_frames)
  fi
  if [ "${TRIM_FAILURE_TRAJECTORY_ON_SAVE}" = "1" ]; then
    stable_args+=(--trim_failure_trajectory_on_save)
  fi

  python error_recovery/collect_universal.py \
    --task "${TASK_NAME}" \
    --error_types "${error_spec}" \
    --episodes "${EPISODES_PER_TYPE}" \
    --save_path "${save_dir}" \
    --wait_stable_only_in_recovery "${WAIT_STABLE_ONLY_IN_RECOVERY}" \
    --wait_stable_require_success_in_recovery "${WAIT_STABLE_REQUIRE_SUCCESS_IN_RECOVERY}" \
    --wait_stable_max_steps "${WAIT_STABLE_MAX_STEPS}" \
    --wait_stable_post_extra_steps "${WAIT_STABLE_POST_EXTRA_STEPS}" \
    --wait_stable_lin_vel_eps "${WAIT_STABLE_LIN_VEL_EPS}" \
    --wait_stable_ang_vel_eps_deg "${WAIT_STABLE_ANG_VEL_EPS_DEG}" \
    "${stable_args[@]}"
}

for spec in "${ERROR_SPECS[@]}"; do
  type_name="${spec%%|*}"
  error_spec="${spec#*|}"
  run_one "${type_name}" "${error_spec}"
done

echo
echo "Done. Data generated under:"
echo "  ${ROOT_SAVE_DIR}"
