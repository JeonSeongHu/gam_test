# Pretraining Data

GAM pretraining combines public Open X-Embodiment, MimicGen, and RoboCasa365
data. The full multi-dataset training loader is not part of this release, but
the dataset list and scripts for depth, index, and normalization statistics are
provided here.

## Setup

```bash
export GAM_ROOT="$(git rev-parse --show-toplevel)"
export DATA_ROOT="${DATA_ROOT:-$HOME/gam-data}"
mkdir -p "$DATA_ROOT"
```

Check each upstream repository's current size before downloading. The complete
mixture requires several terabytes after extraction and preprocessing.

## Sources

| Source | Paper ratio | Data | Depth |
|---|---:|---|---|
| Open X-Embodiment | 72% | 23 LeRobot datasets | DA3 teacher pseudo-depth |
| MimicGen | 18% | Official `core` demonstrations | Simulator depth |
| RoboCasa365 | 10% | Human pretraining subset | Simulator depth |

The base setting uses 224x224 images, two views, and an 8-step action chunk.
External and wrist cameras are the defaults; camera lists remain configurable.

## Download

### Open X

The downloader contains the 23 Hugging Face repository IDs used by GAM:

```bash
python "$GAM_ROOT/scripts/pretraining/download_openx.py" \
  --output-root "$DATA_ROOT/openx_lerobot"
```

Use `--dry-run` to print the repository list without downloading.

### MimicGen

```bash
hf download amandlek/mimicgen_datasets \
  --repo-type dataset \
  --include 'core/*.hdf5' \
  --local-dir "$DATA_ROOT/mimicgen"
```

### RoboCasa365

Install RoboCasa, then use its official downloader:

```bash
python -m robocasa.scripts.download_datasets \
  --split pretrain --source human
```

Use the downloader's resulting `v1.0` directory as `--robocasa-root` below.

## Depth

Run the exporters in environments containing the matching upstream simulator
packages. Both scripts accept a comma-separated `--cameras` list.

### MimicGen

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python "$GAM_ROOT/scripts/pretraining/export_mimicgen_depth.py" \
  --mimicgen-root "$DATA_ROOT/mimicgen" \
  --output-root "$DATA_ROOT/mimicgen_depth" \
  --splits core \
  --cameras agentview,robot0_eye_in_hand
```

The exporter restores each HDF5 simulator state, renders the selected cameras,
and converts the MuJoCo depth buffer to metric depth with
`get_real_depth_map`.

### RoboCasa365

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python "$GAM_ROOT/scripts/pretraining/export_robocasa_depth.py" \
  --robocasa-root /path/to/robocasa/v1.0 \
  --output-root "$DATA_ROOT/robocasa365_depth" \
  --sources atomic,composite \
  --target-demos-per-task 100 \
  --cameras robot0_agentview_left,robot0_eye_in_hand
```

This selects successful episodes without mobile-base motion, removes no-op
frames, replays the saved state and XML, and writes metric depth plus
`index.json`.

## Actions

All sources are converted to:

```text
[delta_position(3), delta_rotation_axis_angle(3), gripper_close(1)]
```

| Source | Conversion |
|---|---|
| MimicGen | OSC scale: position `0.05`, rotation `0.5`; signed gripper to close polarity |
| RoboCasa365 | Drop base/control slots; keep 6D arm delta; signed gripper to close polarity |
| Open X | Apply source-specific frame, rotation, gripper, and valid-dimension conversion |

Statistics are computed separately after conversion. For MimicGen raw HDF5:

```bash
python "$GAM_ROOT/scripts/pretraining/compute_action_stats.py" \
  "$DATA_ROOT/mimicgen/core" \
  --key actions \
  --layout mimicgen-osc \
  --output "$DATA_ROOT/stats/mimicgen.json"
```

For canonical 7D NumPy, HDF5, or Parquet actions, use the default
`--layout canonical7`. RoboCasa365 raw 12D arrays can use
`--layout robocasa365-12d`.

The normalization is:

```text
a_norm = 2 * (a - q01) / (q99 - q01 + eps) - 1
```

## License

Follow the licenses and citation requirements of every upstream dataset,
MimicGen, RoboCasa365, robosuite, and MuJoCo. Generated depth does not replace
the upstream terms.

## Links

- [GAM paper](https://arxiv.org/abs/2606.17046)
- [Open X-Embodiment](https://robotics-transformer-x.github.io/)
- [MimicGen](https://huggingface.co/datasets/amandlek/mimicgen_datasets)
- [RoboCasa](https://github.com/robocasa/robocasa)
