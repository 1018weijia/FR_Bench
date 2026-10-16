# Failure-recovery data collection

Collect trajectories that inject an error, detect the failure, run expert recovery, and save the result. The implementation is in [core/](core).

```bash
# Single process
python error_recovery/collect_universal.py \
  --task move_can_pot --episodes 10 --error_types premature_close:1.0

# Multi-GPU. --scene, --error_types, and similar arguments are forwarded to collect_universal.py
python error_recovery/collect_parallel.py \
  --task move_can_pot --episodes 20 --gpus 0,1 --workers_per_gpu 2 \
  --scene random --error_types premature_close:1.0

# Several tasks. Each task starts its own collect_parallel.py
python error_recovery/examples/batch_collect.py \
  --tasks move_can_pot lift_pot --episodes 20 --gpus 0,1 \
  --error_types premature_close:1.0
```

Run these from the repository root in the `RoboTwin` environment. Each episode receives one injection. Set the probability to `1.0` to keep a single error type. `--scene clean` (the default) is a white tabletop. `--scene random` randomizes clutter, background, and lighting.

The default output is `error_recovery/data/{task}/`, with `data/`, `video/`, `metadata/`, and `collection_stats.json`. In a parallel run each worker also writes `collection_stats_wK.json`, and the supervisor merges those into the file above. A failed episode is trimmed at the first detected failure. A successful episode keeps the full trajectory.

Error types, parameters, and label fields are in [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md). Commands are in [docs/COMMANDS_REFERENCE.md](docs/COMMANDS_REFERENCE.md). Code changes are in [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md). Collection output under `error_recovery/data/` is listed in the repository `.gitignore`.
