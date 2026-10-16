# Bimanual Robotic Manipulation For Failure Recovery

A bimanual failure-recovery experiment repository based on RoboTwin. New data collection uses failure recovery: `error_recovery/collect_parallel.py`. Two existing flows remain: normal expert trajectories via `script/collect_data.py`, and automatic recovery after a failed policy rollout via `script.rollout_recovery.cli recover`.

---

## Where to start

Pick the path that matches the job:

### A) Collect normal expert data

1. “Collect normal data” in this README
2. Task config: `task_config/<config_name>.yml`
3. Task implementation: `envs/<task_name>.py`

### B) Collect failure-recovery data

1. [error_recovery/README.md](error_recovery/README.md)
2. [error_recovery/docs/COMMANDS_REFERENCE.md](error_recovery/docs/COMMANDS_REFERENCE.md)
3. [error_recovery/docs/DATA_FORMAT.md](error_recovery/docs/DATA_FORMAT.md)

### C) Evaluate a policy and recover failed rollouts

1. “Automatic policy-rollout recovery” in this README
2. [script/rollout_recovery/README.md](script/rollout_recovery/README.md)
3. Policy config: `policy/Your_Policy/deploy_policy.yml`

### D) Add a task

Copy a similar task from `envs/*.py`. Add an error type under `error_recovery/core/injectors/`. See [error_recovery/docs/DEVELOPER_GUIDE.md](error_recovery/docs/DEVELOPER_GUIDE.md).

---

## Quick start

### 1) Activate the environment

Run commands from the repository root. The data-generation environment is described in [ENVIRONMENT.md](ENVIRONMENT.md).

```bash
conda activate RoboTwin
```

### 2) Collect normal data

Entry point: `script/collect_data.py`

```bash
python script/collect_data.py pick_diverse_bottles demo_clean
```

- `pick_diverse_bottles` is `envs/pick_diverse_bottles.py`
- `demo_clean` is `task_config/demo_clean.yml`
- `task_config/demo_clean.yml` currently defaults to `save_freq: 15`

### 3) Collect failure-recovery data

```bash
python error_recovery/collect_parallel.py \
  --task move_can_pot \
  --episodes 20 \
  --gpus 0,1 \
  --workers_per_gpu 2 \
  --scene clean \
  --error_types premature_close:1.0
```

Run this from the repository root in the `RoboTwin` environment. Each episode receives one injection. `--scene clean` is a white tabletop. `--scene random` uses the clutter, background, and lighting in `task_config/demo_randomized.yml` and does not change the robot. Collection reads `task_config/{task}.yml` and defaults `save_freq` to 15 when that key is absent. Output goes to `error_recovery/data/{task}/`. Single-process runs, multiple error types, and checks are in [error_recovery/README.md](error_recovery/README.md).

### 4) Automatic policy-rollout recovery

Entry point: `script.rollout_recovery.cli recover`

The policy rolls out normally. When the frame count passes the task threshold and the episode has not succeeded, expert recovery takes over in the same environment. A successful recovery saves the combined rollout and recovery trajectory. A failed recovery saves only the pre-recovery failure trajectory, without the failed recovery actions.

```bash
PYTHONWARNINGS=ignore::UserWarning python -m script.rollout_recovery.cli recover \
  --config policy/Your_Policy/deploy_policy.yml \
  --overrides \
  --task_name lift_pot \
  --task_config demo_clean \
  --ckpt_setting pi05_local_2000 \
  --seed 0 \
  --policy_name Your_Policy \
  --instruction_type seen
```

`policy/Your_Policy/deploy_policy.yml` already contains these inference settings:

```yaml
inference_mode: cloud_client
action_type: qpos
remote_host: 127.0.0.1
remote_port: 8000
rpc_timeout_s: 30
```

Default parameters:

| Parameter | Default | Meaning |
|------|--------|------|
| `recovery_mode` | `inplace` | Recover inside the original rollout environment, so the scene is not rebuilt |
| `recovery_execution_mode` | `qpos_action` | Turn the expert plan into qpos actions, closer to the policy action that gets saved |
| `recovery_qpos_action_stride` | `16` | Stride used when sampling the expert plan. It changes how the recovery segment executes and how fast the saved video looks |
| `recovery_video_sample_mode` | `sim` | Sample both rollout and recovery on simulation steps |
| `recovery_video_sample_freq` | `50` | Default sample rate, chosen so the policy rollout and recovery videos look closer to the original evaluation speed |
| `recovery_threshold_type` | `p95` | Trigger recovery at the p95 task-frame threshold |
| `recovery_force_open_grippers` | `False` | Do not force the grippers open |

Expert collection uses `save_freq` 15. To match that rate:

```bash
--recovery_video_sample_mode sim --recovery_video_sample_freq 15
```

For the current policy-rollout videos, `50` is usually closer to the original evaluation speed than `15`.

Default output:

```text
eval_result/{task}/{policy}/{task_config}/{ckpt_setting}/{timestamp}/
├── data/                 # policy succeeded without recovery
├── video/
├── recovery/             # recovery succeeded: rollout + recovery
│   ├── data/episode*.hdf5
│   └── video/episode*.mp4
└── fail/                 # recovery failed: clean trajectory from before recovery
    ├── data/episode*.hdf5
    └── video/episode*.mp4
```

What is saved:

- `data/`: the policy succeeded before recovery was triggered
- `recovery/`: recovery succeeded. The file is the rollout segment plus the recovery segment
- `fail/`: recovery failed. Only the pre-recovery rollout is kept. The failed recovery attempt is dropped
- `.cache` is a temporary pickle cache and should not remain after a successful save
- `recovery/` does not write an `index/` directory. The recovery start frame is stored in the HDF5 attributes

HDF5 attributes written by `rollout_recovery`:

| Field | Meaning |
|------|------|
| `contains_recovery` | Whether the file contains a recovery segment |
| `recovery_success` | Whether recovery succeeded |
| `recovery_start_frame` | Frame index where recovery starts |
| `rollout_frame_count` | Frames in the rollout segment |
| `recovery_frame_count` | Frames in the recovery segment |
| `total_frame_count` | Total frames |

For a successful file under `recovery/`, `frames[:recovery_start_frame]` is the policy rollout and `frames[recovery_start_frame:]` is expert recovery. A file under `fail/` has `contains_recovery=False` and does not contain the failed recovery actions.

---

## Layout

```text
.
├── envs/                         # task environments
│   ├── _base_task.py             # task base class
│   ├── _base_task_error_recovery.py
│   └── ...
├── error_recovery/               # error injection and expert-recovery collection
│   ├── collect_parallel.py       # multi-GPU collection
│   ├── collect_universal.py      # single-process collection
│   ├── examples/batch_collect.py # one collect_parallel.py run per task
│   ├── core/
│   └── README.md
├── script/
│   ├── collect_data.py           # normal expert collection
│   └── rollout_recovery/         # policy rollout plus automatic recovery
├── task_config/                  # task YAML configs
├── policy/                       # policy deployment and training code
└── assets/                       # model assets
```

---

## Features

### Normal data collection

- RoboTwin task configs
- Bimanual, multi-camera observations
- HDF5 and video output

### Failure-recovery data collection

- Error injection, failure detection, expert recovery, and metadata labels
- Single-process and multi-GPU collection
- A failed episode is trimmed at the first detected failure. A successful episode keeps the full trajectory

### Automatic policy-rollout recovery

- Recovery starts from a task frame-count threshold
- Expert recovery runs in the same environment by default, so the state does not jump
- Successful recovery is written only to `recovery/`
- Failed recovery is written only as a clean pre-recovery trajectory under `fail/`
- HDF5 attributes record the recovery start and frame counts for later training or filtering

---

## Documentation

| Topic | Document |
|------|------|
| Environment and paths | [ENVIRONMENT.md](ENVIRONMENT.md) |
| Failure recovery | [error_recovery/README.md](error_recovery/README.md) |
| Failure-recovery commands | [error_recovery/docs/COMMANDS_REFERENCE.md](error_recovery/docs/COMMANDS_REFERENCE.md) |
| Failure-recovery data format | [error_recovery/docs/DATA_FORMAT.md](error_recovery/docs/DATA_FORMAT.md) |
| Failure-recovery development | [error_recovery/docs/DEVELOPER_GUIDE.md](error_recovery/docs/DEVELOPER_GUIDE.md) |
| Automatic rollout recovery | [script/rollout_recovery/README.md](script/rollout_recovery/README.md) |

---

## Configuration

Task configs live in `task_config/`. For example:

```yaml
embodiment:
  - aloha-agilex

camera:
  head_camera_type: D435
  wrist_camera_type: D435
  collect_head_camera: true
  collect_wrist_camera: true
```

The full field list is in `task_config/_config_template.yml`.
