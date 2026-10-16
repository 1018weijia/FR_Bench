# Collection commands

Run these from the repository root in the `RoboTwin` environment. Error types and fields are in [DATA_FORMAT.md](DATA_FORMAT.md).

## 1. Parallel

```bash
python error_recovery/collect_parallel.py \
  --task move_can_pot \
  --episodes 20 \
  --gpus 0,1 \
  --workers_per_gpu 2 \
  --scene clean \
  --error_types premature_close:1.0
```

`collect_parallel.py` does not load the simulator. It splits episodes across GPUs, and each worker is a new `collect_universal.py` process. Arguments it does not recognize, such as `--scene`, `--error_types`, and `--log_level`, are forwarded to the worker. `--gpu`, `--seed_end`, `--episode_offset`, `--worker_id`, and `--events_path` are filled in by the supervisor. Passing them again has no effect.

| Argument | Default | Meaning |
| --- | --- | --- |
| `--gpus` | `0` | Comma-separated GPU ids |
| `--workers_per_gpu` | `2` | Processes per GPU |
| `--seed_start` | `0` | First seed |
| `--max_episodes_per_process` | `50` | Exit after saving this many episodes. The next process continues writing |
| `--max_restarts` | `5` | Restarts after a worker exits with an error |
| `--episode_timeout` | `1200` | Seconds before a stuck attempt is killed |

## 2. Single process

```bash
python error_recovery/collect_universal.py --list_tasks
python error_recovery/collect_universal.py --list_error_types

python error_recovery/collect_universal.py \
  --task move_can_pot \
  --episodes 10 \
  --error_types premature_close:1.0 \
  --log_level quiet
```

| Argument | Meaning |
| --- | --- |
| `--episodes` | Number of saved episodes. Default `100` |
| `--error_types` | `TYPE[:probability[:parameters]]`. `none` injects nothing |
| `--save_path` | Default `error_recovery/data/{task}` |
| `--scene` | `clean` (default, white tabletop) or `random` (clutter, background, and lighting from `task_config/demo_randomized.yml`) |
| `--log_level` | `quiet` / `normal` / `debug` |
| `--data_types` | For example `rgb qpos endpose` |
| `--no_depth` / `--pointcloud` | Depth is off by default. Point clouds are enabled separately |
| `--no_trim_failure_trajectory_on_save` | A failed episode is trimmed at the first detection frame by default. This flag keeps the full trajectory |

`--error_types` accepts several entries, for example `premature_close:0.4 grasp_slip:0.2`. Whatever the list contains, each episode is injected once, using the first type that hits. A bare type name uses probability `0.5`. Extra parameters for each type are in [DATA_FORMAT.md](DATA_FORMAT.md).

| Level | Output |
| --- | --- |
| `quiet` | Status line, episode result, save summary, and important exceptions |
| `normal` | Also injection, detection, and recovery summaries |
| `debug` | Also planning retries, grasp checks, per-frame logs, and video saving |

## 3. Several tasks

```bash
python error_recovery/examples/batch_collect.py \
  --tasks move_can_pot lift_pot \
  --episodes 20 \
  --gpus 0,1 \
  --error_types premature_close:1.0
```

Each task starts its own `collect_parallel.py` and writes to `--save_root/{task}`. This process does not load the simulator.

## 4. Inspect results

```bash
python error_recovery/utils/parse_hdf5.py error_recovery/data/move_can_pot/data/episode0.hdf5 --summary
python error_recovery/utils/data_analyzer.py error_recovery/data/move_can_pot
python error_recovery/tools/verify_dataset.py --path error_recovery/data/move_can_pot --expect 10
```

Count frames by phase for one episode:

```bash
python -c "from utils.utils4episode import analyze_episode; analyze_episode('error_recovery/data/move_can_pot', 0)"
```

Checks that do not start the simulator:

```bash
python error_recovery/tests/test_labels.py
python error_recovery/tests/test_parallel.py
python error_recovery/utils/test_basic.py
```

Run `test_labels.py` and `test_parallel.py` as scripts. `python -m unittest` imports `error_recovery/__init__.py` first and initializes CUDA. `test_basic.py` does not start the simulator, but it does import the collection framework.

A short run of the four error types:

```bash
bash error_recovery/test_all_error_types.sh
```

`EPISODES_PER_ERROR` defaults to 3, `P` defaults to 0.7, and `SAVE_ROOT` defaults to `./error_recovery/data/_smoke_tests`.

To check that collection survives a worker crash:

```bash
bash script/er_smoke_test.sh
```

It runs the unit tests, one serial regression, and one run with no injection. It then collects 8 episodes on two GPUs, sends `kill -9` to one worker, and checks that all 8 episodes are still present. `GPUS` defaults to `0,1`.
