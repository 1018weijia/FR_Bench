#!/bin/bash
set -euo pipefail

# Error-type test script (smoke/regression)
#
# Goals:
# - Cover error types the code actually supports (the registry is the source of truth)
# - Write each error to its own directory so collection_stats.json is not overwritten
#
# Configurable environment variables:
# - EPISODES_PER_ERROR: episodes to collect per error (default 3, suitable for smoke tests)
# - P: injection probability (default 0.7)
# - SAVE_ROOT: save root (default ./error_recovery/data/_smoke_tests)

EPISODES_PER_ERROR="${EPISODES_PER_ERROR:-3}"
P="${P:-0.7}"
SAVE_ROOT="${SAVE_ROOT:-./error_recovery/data/_smoke_tests}"

echo "Starting error_recovery tests (episodes_per_error=${EPISODES_PER_ERROR}, p=${P})"
echo "Save directory: ${SAVE_ROOT}"

run_one () {
  local task="$1"
  local error_spec="$2"
  local name="$3"

  echo ""
  echo "==> ${name}  task=${task}  error=${error_spec}"
  python error_recovery/collect_universal.py \
    --task "${task}" \
    --error_types "${error_spec}" \
    --episodes "${EPISODES_PER_ERROR}" \
    --save_path "${SAVE_ROOT}/${task}__${name}"
}

# 1) premature_close
run_one "pick_diverse_bottles" "premature_close:${P}:0.5" "premature_close"

# 2) grasp_slip
run_one "pick_diverse_bottles" "grasp_slip:${P}:0.8" "grasp_slip"

# 3) grasp_position_offset
run_one "pick_diverse_bottles" "grasp_position_offset:${P}:0.02:0.05" "grasp_position_offset"

# 4) grasp_orientation_mismatch
run_one "place_shoe" "grasp_orientation_mismatch:${P}:45:90" "grasp_orientation_mismatch"

echo ""
echo "All error-type tests finished!"







