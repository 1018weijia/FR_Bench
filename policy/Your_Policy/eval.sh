#!/bin/bash

policy_name=Your_Policy # [TODO] 
task_name=${1}
task_config=${2}
ckpt_setting=${3}
seed=${4}
gpu_id=${5}
action_type=${6:-qpos}
# [TODO] add parameters here

export CUDA_VISIBLE_DEVICES=${gpu_id}
export YOUR_POLICY_ACTION_TYPE=${action_type}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../.. # move to root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --inference_mode local \
    --action_type ${action_type}
    # [TODO] add parameters here
