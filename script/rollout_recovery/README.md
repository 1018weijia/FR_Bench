# rollout_recovery

`script/rollout_recovery` attaches expert recovery during policy evaluation and saves the result as trainable data, split by outcome.

- Policy succeeds on its own: save under `data/` and `video/` at the eval root
- Recovery succeeds: save the combined rollout and recovery trajectory under `recovery/`
- Recovery fails: drop the failed recovery attempt and save only the clean rollout from before recovery under `fail/`
- An ordinary failure that never triggers recovery is not written to disk

Entry point:

```bash
python -m script.rollout_recovery.cli recover \
  --config policy/Your_Policy/deploy_policy.yml \
  --overrides \
  --task_name lift_pot \
  --task_config demo_clean \
  --ckpt_setting pi05_local_2000 \
  --seed 0 \
  --policy_name Your_Policy \
  --instruction_type seen
```

## Layout

- `cli.py`: CLI for `recover`, `video-stats`, and `batch-video-stats`
- `config.py`: reads `--config` and applies `--overrides`
- `recovery_utils.py`: defaults, threshold loading, cache merge, HDF5 attributes, and cache cleanup
- `recovery.py`: recovery trigger context, in-place recovery, and multi-env recovery
- `rollout_recover.py`: policy rollout plus the recovery loop
- `video_stats.py`: counts frames in successful videos and writes a threshold file
- `batch_video_stats.py`: computes thresholds for many tasks

## Defaults

The defaults live in `recovery_utils.py`:

| Parameter | Default | Meaning |
|------|--------|------|
| `recovery_mode` | `inplace` | Run recovery in the original rollout environment |
| `recovery_execution_mode` | `qpos_action` | Turn the expert plan into qpos actions |
| `recovery_qpos_action_stride` | `16` | Sample stride for qpos actions from the expert plan |
| `recovery_video_sample_mode` | `sim` | Sample rollout and recovery on simulation steps |
| `recovery_video_sample_freq` | `50` | Default save rate, chosen to look closer to the current policy evaluation |
| `recovery_planning_hold_frames` | `8` | Extra held frames saved before recovery planning, used as the handoff |
| `recovery_threshold_type` | `p95` | Trigger recovery at the p95 frame threshold |
| `recovery_force_open_grippers` | `False` | Do not force the grippers open |

Expert collection uses `save_freq` 15. To match that rate:

```bash
--recovery_video_sample_mode sim --recovery_video_sample_freq 15
```

The default `50` is the better match. Policy evaluation and expert collection do not step at the same rate, and `15` usually makes the video look slow.

## Evaluation output

One eval run:

```text
eval_result/{task}/{policy}/{task_config}/{ckpt_setting}/{timestamp}/
├── data/episode{N}.hdf5          # policy succeeded on its own
├── video/episode{N}.mp4
├── recovery/                     # recovery succeeded
│   ├── data/episode{N}.hdf5
│   └── video/episode{N}.mp4
└── fail/                         # recovery failed
    ├── data/episode{N}.hdf5
    └── video/episode{N}.mp4
```

- `recovery/` stores only a successful combined trajectory.
- `fail/` stores the pre-recovery rollout frozen at the trigger. It does not contain the failed recovery segment.
- `.cache` files written during a recovery attempt are temporary pickles. They should not remain after a successful save or a failed cleanup.
- `recovery/` no longer writes `index/`. The recovery start and frame counts are HDF5 attributes.

## HDF5 attributes

Files under `recovery/` and `fail/` get these attributes:

| Field | Recovery succeeded | Recovery failed |
|------|---------------|---------------|
| `contains_recovery` | `True` | `False` |
| `recovery_success` | `True` | `False` |
| `recovery_start_frame` | Frame where recovery starts | Equal to the fail trajectory length |
| `rollout_frame_count` | Frames in the rollout segment | Fail trajectory length |
| `recovery_frame_count` | Frames in the recovery segment | `0` |
| `total_frame_count` | Combined length | Fail trajectory length |

For a successful recovery:

```text
frames[:recovery_start_frame]   -> policy rollout
frames[recovery_start_frame:]   -> expert recovery
```

A fail file has `contains_recovery=False` and does not contain the failed recovery actions.

## Commands

### Evaluate and recover

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

`policy/Your_Policy/deploy_policy.yml` already contains the usual inference settings:

```yaml
inference_mode: cloud_client
action_type: qpos
remote_host: 127.0.0.1
remote_port: 8000
rpc_timeout_s: 30
```

Extra overrides go after `--overrides`, for example:

```bash
--recovery_qpos_action_stride 16 --recovery_video_sample_freq 50
```

### Frame threshold for one eval directory

```bash
python -m script.rollout_recovery.cli video-stats \
  --eval_dir <eval_dir> \
  --task_name <task_name> \
  --task_lim_dir task_lim
```

### Thresholds for many tasks

This scans `./eval_result` and, for each task, uses the eval directory that holds the newest `_result.txt`:

```bash
python -m script.rollout_recovery.cli batch-video-stats \
  --eval_root ./eval_result \
  --task_lim_dir task_lim
```

## Where the threshold comes from

The recovery trigger reads these, in order:

1. `task_lim/<task>.json`
2. The newest `success_video_stats.json` under `eval_result`
3. `script/rollout_recovery/all_task_thresholds.json`

The default statistic is `p95`. The trigger converts that video-frame threshold into the policy evaluation's internal action-step clock before comparing.
