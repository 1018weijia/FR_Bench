#!/bin/bash
set -euo pipefail

# Compatible with the old positional args, and extended for direct cloud deployment.
# Old style:
#   bash policy/Your_Policy/eval_double_env.sh TASK TASK_CONFIG CKPT EXPERT_NUM SEED GPU CONDA_ENV
# New style:
#   bash policy/Your_Policy/eval_double_env.sh TASK TASK_CONFIG CKPT EXPERT_NUM SEED GPU CONDA_ENV MODE HOST PORT ACTION_TYPE RPC_TIMEOUT
# MODE: both | client | server

policy_name=Your_Policy
task_name=${1:-}
task_config=${2:-}
ckpt_setting=${3:-}
expert_data_num=${4:-0} # kept for compatibility, currently unused
seed=${5:-0}
gpu_id=${6:-0}
policy_conda_env=${7:-}
mode=${8:-both}
remote_host=${9:-127.0.0.1}
remote_port=${10:-}
action_type=${11:-qpos}
rpc_timeout_s=${12:-30}

if [[ -z "${task_name}" || -z "${task_config}" || -z "${ckpt_setting}" ]]; then
    echo "Usage: bash policy/Your_Policy/eval_double_env.sh <task_name> <task_config> <ckpt_setting> [expert_data_num] [seed] [gpu_id] [policy_conda_env] [mode] [remote_host] [remote_port] [action_type] [rpc_timeout_s]"
    exit 1
fi

if [[ "${mode}" != "both" && "${mode}" != "client" && "${mode}" != "server" ]]; then
    echo "Invalid mode: ${mode}. Use one of: both | client | server"
    exit 1
fi

pick_free_port() {
    python3 - << 'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("", 0))
    print(s.getsockname()[1])
PY
}

if [[ -z "${remote_port}" ]]; then
    if [[ "${mode}" == "both" ]]; then
        remote_port=$(pick_free_port)
    else
        remote_port=9999
    fi
fi

activate_conda_if_requested() {
    if [[ -n "${policy_conda_env}" && "${policy_conda_env}" != "none" ]]; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${policy_conda_env}"
    fi
}

run_server() {
    local bind_host=$1
    PYTHONWARNINGS=ignore::UserWarning \
    python script/policy_model_server.py \
        --host "${bind_host}" \
        --port "${remote_port}" \
        --config "policy/${policy_name}/deploy_policy.yml" \
        --overrides \
        --task_name "${task_name}" \
        --task_config "${task_config}" \
        --ckpt_setting "${ckpt_setting}" \
        --seed "${seed}" \
        --policy_name "${policy_name}" \
        --inference_mode cloud_server \
        --action_type "${action_type}" \
        --remote_host "${remote_host}" \
        --remote_port "${remote_port}" \
        --rpc_timeout_s "${rpc_timeout_s}"
}

run_client() {
    local host=$1
    PYTHONWARNINGS=ignore::UserWarning \
    python script/eval_policy_client.py \
        --host "${host}" \
        --port "${remote_port}" \
        --rpc_timeout_s "${rpc_timeout_s}" \
        --config "policy/${policy_name}/deploy_policy.yml" \
        --overrides \
        --task_name "${task_name}" \
        --task_config "${task_config}" \
        --ckpt_setting "${ckpt_setting}" \
        --seed "${seed}" \
        --policy_name "${policy_name}" \
        --inference_mode cloud_client \
        --action_type "${action_type}" \
        --remote_host "${host}" \
        --remote_port "${remote_port}" \
        --rpc_timeout_s "${rpc_timeout_s}"
}

export CUDA_VISIBLE_DEVICES=${gpu_id}
export YOUR_POLICY_ACTION_TYPE=${action_type}

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mmode: ${mode}, host: ${remote_host}, port: ${remote_port}, action_type: ${action_type}\033[0m"

cd ../..
activate_conda_if_requested

if [[ "${mode}" == "server" ]]; then
    echo -e "\033[32m[server] Starting cloud policy server on 0.0.0.0:${remote_port}\033[0m"
    run_server "0.0.0.0"
    exit 0
fi

if [[ "${mode}" == "client" ]]; then
    echo -e "\033[34m[client] Connecting to remote policy server at ${remote_host}:${remote_port}\033[0m"
    run_client "${remote_host}"
    exit 0
fi

# mode == both
echo -e "\033[32m[both] Starting local server at 127.0.0.1:${remote_port}\033[0m"
run_server "127.0.0.1" &
SERVER_PID=$!

trap "echo -e '\033[31m[cleanup] Killing server (PID=${SERVER_PID})\033[0m'; kill ${SERVER_PID} 2>/dev/null || true" EXIT
sleep 2

echo -e "\033[34m[both] Running local client against 127.0.0.1:${remote_port}\033[0m"
run_client "127.0.0.1"

echo -e "\033[33m[both] Evaluation finished.\033[0m"
