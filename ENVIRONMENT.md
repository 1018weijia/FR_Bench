# Data-generation environment

This covers expert data collection only: normal trajectories and failure-recovery trajectories. It does not cover policy inference or evaluation.

Run every command from the repository root. Existing docs use the environment name `RoboTwin`. This repository does not include the command that creates that environment. The dependency list is `script/requirements.txt`.

## Install

```bash
conda activate RoboTwin
pip install -r script/requirements.txt
```

Versions that the simulator depends on directly:

| Package | Version |
| --- | --- |
| torch | `2.4.1` |
| sapien | `3.0.0b1` |
| mplib | `0.2.1` |
| scipy | `1.10.1` |
| gymnasium | `0.29.1` |
| trimesh | `4.4.3` |
| open3d | `0.18.0` |
| transforms3d | `0.4.2` |
| imageio | `2.34.2` |

`openai`, `azure`, and `azure-ai-inference` are used only to generate task language descriptions. The simulator does not need them.

Curobo is not in this list. If it is not installed, `envs/robot/robot.py` falls back to Mplib. To force Mplib:

```bash
export FRBENCH_FORCE_MPLIB=1
```

## Paths

Scripts find configs and tasks through relative paths such as `task_config/` and `envs/`. The current directory must be the repository root.

| Use | Relative path |
| --- | --- |
| Dependency list | `script/requirements.txt` |
| Normal collection | `script/collect_data.py` |
| Failure-recovery collection | `error_recovery/collect_universal.py` |
| Parallel failure-recovery collection | `error_recovery/collect_parallel.py` |
| Task implementation | `envs/<task_name>.py` |
| Collection config | `task_config/<config_name>.yml` |
| Config field template | `task_config/_config_template.yml` |
| Robot model map | `task_config/_embodiment_config.yml` |
| Robot assets | `assets/embodiments/<embodiment>/` |

`task_config/demo_clean.yml` is a full collection config. Its current `save_freq` is 15. `task_config/<task_name>.yml` only sets the cameras, robot model, and task name. It does not replace `demo_clean.yml`.

The default robot is `aloha-agilex`. Its assets are in `assets/embodiments/aloha-agilex/`.

## Run

Normal trajectories:

```bash
conda activate RoboTwin
python script/collect_data.py pick_diverse_bottles demo_clean
```

Here `pick_diverse_bottles` is `envs/pick_diverse_bottles.py` and `demo_clean` is `task_config/demo_clean.yml`.

Failure-recovery trajectories:

```bash
python error_recovery/collect_parallel.py \
  --task pick_diverse_bottles --episodes 10 --gpus 0 \
  --error_types premature_close:1.0
```

Normal collection writes under the `save_path` in the config. `demo_clean.yml` sets that to `./data`. Failure recovery reads `task_config/{task}.yml`, not `demo_clean.yml`, and writes to `error_recovery/data/{task_name}/` by default. Parameters are in `error_recovery/docs/COMMANDS_REFERENCE.md`.
