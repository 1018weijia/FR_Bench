# eval_inject_error

`script/eval_inject_error` injects gripper-open perturbations during policy evaluation and measures how the success rate changes when a gripper is forced open at random.

The perturbation applies only during the policy rollout. It does not apply during the expert check that runs before the rollout.

## Perturbation

With `--open_gripper`, the default behavior is:

- The first 50 policy actions are left alone.
- From policy action 51 onward, the grippers are checked.
- If one gripper is currently closed, or the policy command for that gripper is a close, a perturbation event fires with probability 50%.
- One event forces that gripper open for 20 consecutive policy actions.
- An episode fires at most 3 events.

Defaults:

| Parameter | Default | Meaning |
|------|--------|------|
| `open_gripper_prob` | `0.5` | Probability of firing when the closed condition is met |
| `open_gripper_closed_threshold` | `0.5` | A gripper value at or below this threshold counts as closed, or as a close command |
| `open_gripper_max_injections` | `3` | Maximum events per episode |
| `open_gripper_duration` | `20` | Consecutive policy actions forced open by one event |
| `open_gripper_start_step` | `50` | Number of leading policy actions that are never perturbed |

Example log line:

```text
[Inject Error] Open-gripper perturbations applied: 3/3 events, duration=20, start_after=50 actions, 60/400 actions
```

This episode fired 3 events. Each lasted 20 steps, so 60 of 400 policy actions had a gripper forced open.

## Before running

From the repository root, activate the environment:

```bash
cd /path/to/FRBench
conda activate RoboTwin
```

If `Your_Policy` uses the websocket client, the policy server must already be listening. The default config is:

```text
policy/Your_Policy/deploy_policy.yml
```

The default connection is:

```yaml
inference_mode: cloud_client
action_type: qpos
remote_host: 127.0.0.1
remote_port: 8000
rpc_timeout_s: 30
```

For the openpi client:

```bash
export PYTHONPATH=/path/to/FRBench/policy/openpi/packages/openpi-client/src:${PYTHONPATH:-}
```

## One task

50 episodes in the clean scene:

```bash
PYTHONWARNINGS=ignore::UserWarning python script/eval_inject_error/eval.py \
  --config policy/Your_Policy/deploy_policy.yml \
  --open_gripper \
  --overrides \
  --task_name lift_pot \
  --task_config demo_clean \
  --ckpt_setting open_gripper_clean \
  --seed 0 \
  --policy_name Your_Policy \
  --instruction_type seen \
  --test_num 50
```

50 episodes in the randomized scene:

```bash
PYTHONWARNINGS=ignore::UserWarning python script/eval_inject_error/eval.py \
  --config policy/Your_Policy/deploy_policy.yml \
  --open_gripper \
  --overrides \
  --task_name lift_pot \
  --task_config demo_randomized \
  --ckpt_setting open_gripper_randomized \
  --seed 0 \
  --policy_name Your_Policy \
  --instruction_type seen \
  --test_num 50
```

To override the perturbation:

```bash
--open_gripper_prob 0.5 \
--open_gripper_max_injections 3 \
--open_gripper_duration 20 \
--open_gripper_start_step 50
```

## Wrapper script

```bash
TEST_NUM=50 TASK_NAME=lift_pot TASK_CONFIG=demo_clean bash script/eval_inject_error/eval.sh
```

Common environment variables:

```bash
TASK_NAME=lift_pot
TASK_CONFIG=demo_clean
TEST_NUM=50
GPU_ID=0
CONFIG_PATH=policy/Your_Policy/deploy_policy.yml
POLICY_NAME=Your_Policy
OPEN_GRIPPER_PROB=0.5
OPEN_GRIPPER_MAX_INJECTIONS=3
OPEN_GRIPPER_DURATION=20
OPEN_GRIPPER_START_STEP=50
```

## Several tasks and scenes

`eval.py` runs one `task_name` and one `task_config` per process. Loop in the shell for more tasks or for both clean and randomized scenes:

```bash
for task in lift_pot pick_diverse_bottles; do
  for cfg in demo_clean demo_randomized; do
    PYTHONWARNINGS=ignore::UserWarning python script/eval_inject_error/eval.py \
      --config policy/Your_Policy/deploy_policy.yml \
      --open_gripper \
      --overrides \
      --task_name "$task" \
      --task_config "$cfg" \
      --ckpt_setting open_gripper_${cfg} \
      --seed 0 \
      --policy_name Your_Policy \
      --instruction_type seen \
      --test_num 50
  done
done
```

## Output

Results are written to:

```text
eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{timestamp}/
├── _result.txt
└── episode*.mp4
```

- `_result.txt` stores the success rate for this run.
- `episode*.mp4` is the evaluation video.
- This script does not save HDF5. Use `rollout_recovery`, or add a pickle-to-HDF5 save path, if you need HDF5.
