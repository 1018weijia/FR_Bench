# Development overview

This document is for anyone extending, debugging, or changing the `error_recovery` implementation.
Commands are in [COMMANDS_REFERENCE.md](COMMANDS_REFERENCE.md). Field definitions are in [DATA_FORMAT.md](DATA_FORMAT.md).

## 1. Layers

```text
core/
├── framework.py          # public entry
├── framework_support.py  # collection loop, saving, status line
├── task_wrapper.py       # task wrapper and shared scheduling
├── data_collector.py     # metadata, stats, and log files
├── logging_utils.py      # quiet / normal / debug
├── injectors/            # error injectors
├── execution/            # error-action execution and action rewriting
├── runtime/              # runtime state and deferred detection
├── recovery/             # recovery actions and the success check
├── lifecycle/            # attempt lifecycle labels
└── serialization/        # cleanup and serialization before export
```

Error types are registered in `core/injectors/registry.py`. `core/error_injectors.py` is only a compatibility entry.

The parallel entry point is [collect_parallel.py](../collect_parallel.py). It does not import the simulator. It launches `collect_universal.py` on each GPU. Several tasks use [examples/batch_collect.py](../examples/batch_collect.py), which starts one parallel collection per task.

## 2. Call chain

1. [collect_universal.py](../collect_universal.py)
2. [core/framework.py](../core/framework.py)
3. [core/framework_support.py](../core/framework_support.py)
4. [core/task_wrapper.py](../core/task_wrapper.py)

## 3. Injection, detection, and recovery

The framework does not rewrite the whole task. It inserts logic around the important action calls. The result is still the original task plus a local recovery step.

| When | What is inserted |
| --- | --- |
| Before the action | Build the context and decide whether to inject |
| During the action | Run the modified action |
| After the action | Detect immediately, or schedule a deferred detection |

| Component | Responsibility |
| --- | --- |
| `TaskWrapper` | Decide when to detect and when to enter recovery |
| `ErrorInjector.detect_failure()` | Failure check owned by each error type |
| `plan_recovery()` | Build the recovery actions |

After a failure is detected:

| Step | What happens |
| --- | --- |
| 1 | The phase switches to `error_detected` |
| 2 | `error_detected_frame` is recorded |
| 3 | `plan_recovery()` is called |
| 4 | Recovery actions run in order |
| 5 | The attempt `end_status` becomes `recovered` or `recovery_failed`. After either ending, the live `phase` returns to `normal` |

## 4. Labels and saving

Per-frame labels are written in `wrap_get_obs` in `framework_support.py`. When recovery closes, the cached frame whose index equals `recovery_end_frame` is rewritten to `phase=normal`, and `error_stage_precise` becomes `recovered` or `recovery_failed`. Earlier frames keep the phase they had at collection time.

After `play_once` returns, an attempt that still has no `end_status` is closed as `recovery_failed` if it has a detection frame of at least 1, and as `no_failure` otherwise. The live phase returns to `normal`, and `phase` inside `error_metadata` is set to that `end_status`.

The episode result is the last attempt's `end_status`, stored as `final_phase`. `error_recovery_rate` is the number of episodes with `final_phase == recovered` divided by the number of injected episodes. Each episode is injected once, however many error types are configured.

`_recount_stats()` in `data_collector.py` recomputes the totals from the `episodes` list. When a `worker_id` is set, the existing `collection_stats_{worker_id}.json` is merged by `episode_id` before saving, so a worker restart does not drop earlier episodes. A single-process `collection_stats.json` keeps only the episodes from that process.

`_RECOVERY_PHASES` and `is_error_phase` in `execution/flow.py` are different lists. Do not merge them. Injection count has a separate cap of 1. When a worker restarts and continues, the next seed is one past the highest seed that worker already used.

The field tables are in [DATA_FORMAT.md](DATA_FORMAT.md).

## 5. Design rules

| Rule | Why | What the code does |
| --- | --- | --- |
| The target is a failure-recovery pair, not an ordinary success | The context before the error, the moment the failure is noticed, the recovery actions, and the outcome all need to stay | Per-frame labels and attempt-level metadata are both kept |
| Onset and detection stay separate | The error taking effect is not the same as the system noticing it | Both `error_onset_frame` and `error_detected_frame` are recorded |
| Detection may be deferred | A shifted grasp or a slip is often visible only on a later action | Both immediate and deferred detection are supported |
| Deciding to inject is not the start of the error | Otherwise the error span starts too early | Onset is marked only where the error starts to change behavior |
| Failed recoveries are kept | They are useful for evaluating recovery and for boundary cases | Failed trajectories can be saved, and the labels have to explain them |
| Saved failures are trimmed by default | Actions after the failure detection are often a redundant tail | When task `success` is false, trim at the first `error_detected_frame`. A successful task keeps the whole trajectory |
| A fake detection is only a fallback | A real failure can occasionally be missing its detected frame | Prefer a real detection, then a fake one, and drop the episode if neither exists |
| Attempt-level labels stay | An episode summary alone is not enough to analyze | Keep `injector`, onset, detected, recovery, and `end_status` |

## 6. Common changes

| Change | Main files | Notes |
| --- | --- | --- |
| Add an error type | [core/injectors/base.py](../core/injectors/base.py) / [core/injectors/registry.py](../core/injectors/registry.py) / `core/injectors/grasp/` | Built-in injectors live under `grasp/`. Implement `should_inject`, `inject_error`, `detect_failure`, and `plan_recovery`, then register the type |
| Change when failure is detected or recovery starts | [core/task_wrapper.py](../core/task_wrapper.py) / `runtime/` / `recovery/` | Check immediate detection, deferred detection, and the recovery entry |
| Change metadata, stats, or the save format | [core/data_collector.py](../core/data_collector.py) / [core/framework_support.py](../core/framework_support.py) / [core/serialization/sanitize.py](../core/serialization/sanitize.py) | Start from the save path |
| Change trimming | [core/framework_support.py](../core/framework_support.py) / [core/data_collector.py](../core/data_collector.py) | Keep the HDF5, video, and metadata consistent |
| Change collection logs | [core/logging_utils.py](../core/logging_utils.py) / [core/framework_support.py](../core/framework_support.py) / `envs/_base_task*.py` | The default is `quiet`. Use `--log_level debug` while debugging |
| Debug one error type | That injector plus [core/task_wrapper.py](../core/task_wrapper.py) | Check that onset, detected, and recovery line up. Turn on `--log_level debug` if needed |

Put a new error type under `error_recovery/core/injectors/` and enable it with `--error_types`. Do not add another entry such as `grasp_actor_with_error_recovery` on the task class.

## 7. What to run after a change

Log levels and inspection commands are in [COMMANDS_REFERENCE.md](COMMANDS_REFERENCE.md). After a label or stats change, run:

```bash
python error_recovery/tests/test_labels.py
python error_recovery/tests/test_parallel.py
```
