#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/sys02/FRBench

TASK_NAME="${TASK_NAME:-pick_diverse_bottles}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
CKPT_SETTING="${CKPT_SETTING:-pi05_local_2000}"
SEED="${SEED:-0}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-8000}"

PI05_CKPT_DIR="${PI05_CKPT_DIR:-$HOME/ckpt/RoboTwin_org_ckpt_pi05/20000}"
PI05_CONFIG="${PI05_CONFIG:-pi0_base_aloha_robotwin_lora}"

cd "${ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/policy/pi0/src:${ROOT}/policy/pi0/packages/openpi-client/src:${PYTHONPATH:-}"
export YOUR_POLICY_ACTION_TYPE=qpos

echo "[1/2] Starting pi0.5 websocket server"
echo "  config: ${PI05_CONFIG}"
echo "  ckpt:   ${PI05_CKPT_DIR}"
echo "  port:   ${PORT}"

python policy/pi0/scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config="${PI05_CONFIG}" \
  --policy.dir="${PI05_CKPT_DIR}" \
  --port="${PORT}" &
SERVER_PID=$!

cleanup() {
  echo "[cleanup] killing server PID=${SERVER_PID}"
  kill "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "[wait] waiting for websocket server..."
python - <<PY
import socket, time, sys
host = "127.0.0.1"
port = int("${PORT}")
for _ in range(120):
    try:
        with socket.create_connection((host, port), timeout=1):
            print("server is ready")
            sys.exit(0)
    except OSError:
        time.sleep(1)
raise SystemExit("server did not become ready")
PY

echo "[2/2] Running rollout + recovery"

PYTHONWARNINGS=ignore::UserWarning \
python -m script.rollout_recovery.cli recover \
  --config policy/Your_Policy/deploy_policy.yml \
  --overrides \
  --task_name "${TASK_NAME}" \
  --task_config "${TASK_CONFIG}" \
  --ckpt_setting "${CKPT_SETTING}" \
  --seed "${SEED}" \
  --policy_name Your_Policy \
  --instruction_type seen \
  --inference_mode cloud_client \
  --remote_host 127.0.0.1 \
  --remote_port "${PORT}" \
  --rpc_timeout_s 30 \
  --action_type qpos \
  --recovery_threshold_type p95 \
  --recovery_force_open_grippers True