# Failure-recovery data format

## 1. Error types and parameter format

```bash
python error_recovery/collect_universal.py --list_error_types
```

### 1.1 The four grasp errors used here

| Error type | Meaning | Trigger action / stage | Parameter format | Parameters | Onset label |
| --- | --- | --- | --- | --- | --- |
| `grasp_position_offset` | The grasp target is shifted, so the grasp misses, closes on air, or is unstable | `grasp_actor` | `grasp_position_offset:prob[:offset_min[:offset_max]]` | `offset_min` / `offset_max` are in meters. If `offset_max` is omitted it defaults to `2 * offset_min` | The label sits between the move to the offset grasp pose and the following gripper close |
| `grasp_orientation_mismatch` | The grasp orientation does not match the object | `grasp_actor` | `grasp_orientation_mismatch:prob[:rot_min[:rot_max]]` | `rot_min` / `rot_max` are the rotation range. If `rot_max` is omitted it defaults to `2 * rot_min` | The label sits between the move to the rotated grasp pose and the following gripper close |
| `grasp_slip` | The object slips while being lifted after a grasp | Lift stage | `grasp_slip:prob[:force_reduction]` | `force_reduction` is the fraction of force removed | The label sits after the lift and before the gripper opens and the object slips |
| `premature_close` | The gripper closes early along the grasp path | `grasp_actor` | `premature_close:prob[:close_ratio]` | `close_ratio` is how far along the grasp path the early close happens | The label is on the grasp action that closes early |

### 1.2 Fields shared by every error

An attempt usually carries these fields:

| Field | Meaning |
| --- | --- |
| `injector` | Error type |
| `error_onset_frame` | Frame where the error actually begins |
| `error_detected_frame` | Frame where the system first decides the attempt failed |
| `recovery_start_frame` | Frame where recovery starts |
| `recovery_end_frame` | Frame where recovery ends |
| `end_status` | `recovered`, `recovery_failed`, or `no_failure` |

- `error_onset_frame` and `error_detected_frame` are not required to be the same frame
- If a real `error_detected_frame` is missing, the system may fill in a fake detected label

## 2. Output directory

Default output:

```text
error_recovery/data/{task}/
├── data/
├── video/
├── metadata/
├── logs/                      # collection_{time}[_wK]_p{pid}.log; parallel runs also have supervisor wK.out files
├── events/                    # one jsonl per worker in a parallel run
├── collection_stats.json      # stats for the run; merged by the supervisor when parallel
├── collection_stats_wK.json   # one stats file per worker when parallel
└── status.json                # parallel progress
```

The task config comes from `task_config/{task}.yml`. `save_freq` defaults to 15 when that key is absent. `--scene clean` (the default) turns domain randomization off. `--scene random` uses the `domain_randomization` section of `task_config/demo_randomized.yml`. Changing the robot embodiment stays off.

## 3. How to read the labels

| Field | Where it is written | Meaning |
| --- | --- | --- |
| `phase` / `is_error_frame` | Every HDF5 frame | State at collection time. `recovery` and sub-phases such as `recovery_perceive` are error frames. The recovery-end frame and every frame after it are `normal`, whether `end_status` is `recovered` or `recovery_failed` |
| `error_stage_precise` | Every HDF5 frame | A look back over onset, detection, and recovery bounds: `pre_error`, `error_active`, `error_detected`, `recovery`, `normal`. The recovery-end frame is `recovered` or `recovery_failed` |
| `final_phase` | `episode_summary` | `end_status` of the last attempt: `recovered`, `recovery_failed`, or `no_failure`. With no injection this falls back to `current_phase` at the time, usually `normal` |

## 4. HDF5 output

The HDF5 layout of one episode follows the task saver. It usually contains:

```text
episode*.hdf5
├── observation/        # cameras, state, and other observations
├── endpose/            # end-effector poses
├── joint_action/       # actions
└── error_recovery/     # per-frame failure-recovery labels
```

Common fields under `error_recovery/`:

| Field | Meaning |
| --- | --- |
| `enabled` | Whether failure-recovery labels are enabled on this frame |
| `phase` | Phase at collection time, including `recovery_*` sub-phases |
| `error_stage_precise` | Phase reconstructed from the timeline |
| `frame_count` | Frame counter, starting at 1 |
| `timestamp` | Frame timestamp |
| `is_error_frame` | True when `phase` is `error_attempt`, `error_detected`, or starts with `recovery` |
| `is_error_frame_precise` | Whether `error_stage_precise` falls in the error or recovery span |
| `active_injectors` | Injectors that are still active |
| `error_metadata` | Summary of the latest attempt |

The framework writes these fields on every frame so the labels stay aligned with the trajectory.

## 5. Metadata file

Each episode writes one file:

```text
metadata/episode{N}_metadata.json
```

Top-level structure:

```json
{
  "episode_summary": {...},
  "error_recovery_data": {...},
  "collection_timestamp": "..."
}
```

### 5.1 `episode_summary`

Episode-level summary. Common fields:

| Field | Meaning |
| --- | --- |
| `episode_id` | Episode id |
| `task_name` | Task name |
| `success` | Whether the task finally succeeded |
| `error_injected` | Whether an error was injected |
| `error_recovery_attempted` | `end_status` is `recovered` or `recovery_failed` |
| `final_phase` | `end_status` of the last attempt. With no injection this is `current_phase` |
| `total_frames` | Frame count recorded in the final metadata. A trimmed failure uses the kept length |
| `trim_detected_frame` | Detection frame used as the trim point for a failure |
| `trimmed_total_frames` | Number of frames kept |
| `error_types` | Error types present in this trajectory |
| `attempt_count` | Number of error attempts |
| `end_time` | End time |

### 5.2 `error_recovery_data`

The fuller episode-level recovery record. Common fields:

| Field | Meaning |
| --- | --- |
| `current_phase` | Live phase when collection ends. After recovery closes this is `normal`. The outcome is `end_status` inside `error_metadata` |
| `error_stage_precise` | Timeline phase at the end |
| `is_error_frame_precise` | Finer error-frame flag |
| `error_metadata` | The attempt-level labels |
| `action_history` | Action history |
| `recovery_actions` | Recovery action sequence |
| `frame_count` | Frame count |
| `active_injectors` | Injectors that are still active |

### 5.3 `error_metadata`

The attempt-level label list. Each element is one error attempt:

| Field | Meaning |
| --- | --- |
| `attempt_id` | Attempt id |
| `injector` | Error type |
| `phase` | How far this attempt got. After it closes, this is set to `end_status` |
| `inject_decision_frame` | Frame where injection was decided |
| `error_onset_frame` | Frame labeled as the real start of the error |
| `error_detected_frame` | Frame where the system first decides the attempt failed |
| `error_detected_is_fake` | Whether the detected frame was filled in as a fallback |
| `recovery_start_frame` | Frame where recovery starts |
| `recovery_end_frame` | Frame where recovery ends |
| `error_end_frame` | Frame where the error stage ends |
| `end_status` | Recovery outcome: `recovered`, `recovery_failed`, or `no_failure` |
| `onset_reason` | Why the onset frame was chosen |
| `metadata` | Extra parameters for this error type |

## 6. Failure trimming

Trimming is on by default. When the task finally fails, the saved output is cut at the first valid `error_detected_frame`. A detection frame below 1 is ignored. A successful task keeps the full trajectory. A trajectory that both fails recovery and fails the task is trimmed too. Turn this off for debugging:

```bash
--no_trim_failure_trajectory_on_save
```

| Item | What happens |
| --- | --- |
| `data/` | The HDF5 is trimmed |
| `video/` | The video is trimmed |
| `metadata/` | `action_history` and `frame_count` are updated to the kept length |
| `error_metadata` | Kept. Timestamps past the kept length, such as `recovery_end_frame`, are cleared |

If a real `error_detected_frame` is missing, the system tries to fill a fake detected frame. If it still cannot trim, that failed trajectory is dropped and does not enter the dataset.

## 7. collection_stats.json

A single process writes `collection_stats.json` directly. In a parallel run each worker writes `collection_stats_wK.json`, and the supervisor merges them into `collection_stats.json`. When the same worker restarts and writes again, existing records are merged by `episode_id`. Earlier episodes are not replaced.

| Field | Meaning |
| --- | --- |
| `total_episodes` | Episodes that were saved |
| `successful_episodes` | Episodes with `success` true |
| `error_injected_episodes` | Episodes that received an injection |
| `error_recovery_episodes` | Episodes whose `final_phase` is `recovered` |
| `success_rate` | `successful_episodes / total_episodes` |
| `error_recovery_rate` | `error_recovery_episodes / error_injected_episodes` |
| `episodes` | `episode_summary` for each trajectory |
