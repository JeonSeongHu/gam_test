# GAM: Geometric Action Model for LIBERO / LIBERO-Plus

This repository is the public implementation of **GAM (Geometric Action
Model)** for LIBERO robot manipulation. GAM is a single-stage **world-action
model** built on **DA3 (Depth Anything 3)**: blocks 0-12 of DA3-Giant act as a
frozen per-view geometric encoder; a **`GAMFuturePredictor`** (dense
block-autoregressive transformer) sits at block 12 and predicts next-step
visual / proprio / action tokens; DA3 blocks 13-39 then consume the predicted
sequence and emit action tokens that an MLP action head turns into robot
actions.

This repository contains **only** the DA3-Giant `gam` code. Training is on
**LIBERO** (HDF5 demos); evaluation is closed-loop in simulation on **LIBERO** and
**LIBERO-Plus** (the LIBERO-Plus perturbed-task benchmark). LIBERO-Plus is an
eval-only benchmark.

## Architecture (one stage)

```
Input: (B, H, V, 3, 224, 224)      H observed timesteps, V camera views
  → DA3 blocks 0-12 (frozen):       per-view shallow visual tokens
  → GAMFuturePredictor:     dense block-AR over observed history,
                                    emits next-step visual/proprio/action tokens
                                    (language conditioning + 4D axial RoPE)
  → DA3 blocks 13-39 (trainable):   consume the predicted sequence [o_1,a_0 ... o_H,a_{H-1}]
  → action tokens → ActionHeadV2 (MLP) → (B, T, chunk, 7) delta actions

Losses (compute_gam_forward_loss): action L1 + future-feature distillation
        + proprio L1 + optional GT-depth (DA3 DPT decode) + optional SIGReg.
```

## Layout

| Path | Purpose |
|------|---------|
| `src/train_robot.py` | Training entry (`run_da3_finetune_training`) |
| `src/eval_libero_unified.py` | LIBERO / LIBERO-Plus closed-loop + open-loop eval |
| `scripts/run_hf_gam_libero_plus_eval.sh` | Standalone local-GPU LIBERO-Plus eval for the public GAM checkpoints |
| `src/robot/future_predictor.py` | `GAMFuturePredictor` |
| `src/robot/unified_loss.py` | `compute_gam_forward_loss` |
| `src/robot/da3_giant_encoder.py` | DA3-Giant backbone + action-token injection + depth/camera decode |
| `src/robot/action_head_v2.py` | MLP action head (`action_head_oft.py` = OFT ablation) |
| `src/robot/dataset.py` | LIBERO HDF5 training dataset + action/proprio normalizers |
| `src/robot/rollout_env.py`, `closed_loop_libero_eval.py` | LIBERO simulator rollout |
| `configs/training/libero_unified/` | Training configs (smoke / baseline / gam / data_variants / resume_variants / deepspeed) |
| `Depth-Anything-3/` | DA3 backbone source. **Clone separately** (see Setup) |

## Setup

Reproducible from scratch on a normal CUDA GPU server. Pick **one** of the three. All install the identical
pinned package set (see `requirements.txt`). The validated reference environment is
Python 3.12 + a torch 2.8 GH200 build; the pins are public and work on torch >= 2.5.

### Option A: Docker (recommended; fully self-contained)
Installs the pinned Python stack, the DA3 backbone, LIBERO, LIBERO-Plus, the
LIBERO-Plus perturbation libraries, and the headless-MuJoCo GL/EGL system
libraries.

```bash
docker build -t da3-libero .
docker run --gpus all -it --rm \
  -e DA3_ROOT=/data -v /host/data_root:/data \    # mount data + weights (below)
  -e WANDB_API_KEY=$WANDB_API_KEY da3-libero
```

### Option B: conda
```bash
conda env create -f environment.yml && conda activate da3-libero
bash scripts/setup_sources.sh          # clone + install DA3 backbone, LIBERO, LIBERO-Plus
```

### Option C: venv + pip
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
bash scripts/setup_sources.sh
# system libs for headless rendering and LIBERO-Plus motion blur (Debian/Ubuntu):
#   sudo apt-get install libgl1 libglvnd0 libegl1 libgles2 libosmesa6 libglfw3 ffmpeg imagemagick libmagickwand-dev
```

`scripts/setup_sources.sh` clones the **DA3 backbone** (`ByteDance-Seed/Depth-Anything-3`,
present on disk for `da3_giant_encoder`, which adds `Depth-Anything-3/src` to `sys.path`),
the **LIBERO** benchmark (installed with `--no-deps` to keep the pinned package set stable),
and **LIBERO-Plus** (eval only).
Commits are pinned in the script / Dockerfile.

Then per shell:
```bash
export DA3_ROOT=/path/to/your/data_root  # configs resolve all data/ckpt paths under ${DA3_ROOT}
export DA3_LIBERO_SOURCE_DIR=$DA3_ROOT/LIBERO
export PYTHONPATH=$DA3_ROOT/src:$DA3_LIBERO_SOURCE_DIR:$PYTHONPATH   # LIBERO repo root
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl    # osmesa for software rendering
export WANDB_API_KEY=...                       # optional; only with --wandb
```

`requirements.txt` is the curated, public, pinned list (the install target).
`requirements-cscs.lock` is a reference freeze of the exact validated cluster env.
The public launchers assume these Docker/conda/venv dependencies are installed
in the active environment and leave runtime library paths untouched.

### Data & weights (download separately, lay out under `$DA3_ROOT`)
All machine-specific paths in the configs are `${oc.env:DA3_ROOT,.}/...` (OmegaConf):

```
$DA3_ROOT/
  checkpoints/track4world_da3.pth        # DA3-Giant base weights (stage_1.ckpt_path)
  data/libero_noop/<suite>/*.hdf5        # LIBERO HDF5 demos (libero_spatial/object/goal/10)
  data/libero_noop/_stats/               # action/proprio normalizer stats (auto-computed if absent)
```

- **DA3-Giant base weights** `track4world_da3.pth`: DA3-Giant initialization the encoder fine-tunes from; from the Depth-Anything-3 release.
- **LIBERO HDF5 demos**: replayed no-op LIBERO HDF5s with embedded GT depth (`obs/agentview_depth`, `obs/eye_in_hand_depth`). Configs set `gt_depth_root: null`, and the loader reads depth directly from the HDF5.
- **LIBERO-Plus (eval only)**: perturbed-task benchmark from source (`sylvestf/LIBERO-plus`), used by `eval_libero_unified.py --plus` (simulator rollout).

## Public HF checkpoints

The standalone LIBERO-Plus rollout checkpoint set is hosted at:

```bash
hf download SeonghuJeon/3da-libero-gam \
  --local-dir checkpoints_hf/3da-libero-gam
```

Expected layout:

| Suite key | LIBERO suite | Checkpoint | Config |
|-----------|--------------|------------|--------|
| `spatial` | `libero_spatial` | `spatial/0084000.pt` | `spatial/config.yaml` |
| `object` | `libero_object` | `object/0022000.pt` | `object/config.yaml` |
| `goal` | `libero_goal` | `goal/0068500.pt` | `goal/config.yaml` |
| `long` | `libero_10` | `long/0091500.pt` | `long/config.yaml` |

These are GAM checkpoints with `predictor.enabled: true` and
`predictor.type: gam`.

## Train

```bash
# single-GPU smoke (1 step). Point DA3_ROOT at a tiny LIBERO subset first.
PYTHONPATH=src:$PYTHONPATH python src/train_robot.py \
  --config configs/training/libero_unified/smoke/gam_chunk2.yaml \
  --single-gpu --set training.max_steps=1

# multi-GPU (DeepSpeed ZeRO-2)
PYTHONPATH=src:$PYTHONPATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
deepspeed --include localhost:0,1,2,3 src/train_robot.py \
  --config configs/training/libero_unified/gam/chunk8_150k_2node.yaml \
  --deepspeed_config configs/training/libero_unified/deepspeed/micro2.json \
  --wandb --wandb-name libero_da3giant
```

In-training closed-loop LIBERO eval runs when `training.closed_loop_evals[]` is set in the config.

## Evaluate (closed-loop)

```bash
PYTHONPATH=src:$PYTHONPATH python src/eval_libero_unified.py \
  --ckpt /path/to/checkpoint.pt \
  --config configs/training/libero_unified/gam/chunk8_150k_2node.yaml \
  --suites libero_spatial,libero_object,libero_goal,libero_10 \
  --num-trials-per-task 5

# LIBERO-Plus (per official perturbation category)
PYTHONPATH=src:$PYTHONPATH python src/eval_libero_unified.py \
  --ckpt /path/to/checkpoint.pt --plus --plus-perturbation all ...
```

`scripts/run_libero_eval.sh` and `scripts/run_libero_batched_eval.py` are convenience launchers.

## Standalone LIBERO-Plus eval on local GPUs

The public standalone path uses local CUDA GPU processes. Run one process per
GPU, shard with explicit `--shard-index` / `--shard-count`, and keep LIBERO-Plus
robot initialization in the official original-qpos mode.

The wrapper below does that and aggregates shard outputs into one suite-level
`summary.json` and `per_task.csv`:

```bash
# One suite on four local GPUs.
GAM_EVAL_GPUS=0,1,2,3 \
scripts/run_hf_gam_libero_plus_eval.sh spatial

# Other suites.
GAM_EVAL_GPUS=0,1,2,3 scripts/run_hf_gam_libero_plus_eval.sh object
GAM_EVAL_GPUS=0,1,2,3 scripts/run_hf_gam_libero_plus_eval.sh goal
GAM_EVAL_GPUS=0,1,2,3 scripts/run_hf_gam_libero_plus_eval.sh long

# Sequentially run all four suites.
GAM_EVAL_GPUS=0,1,2,3 scripts/run_hf_gam_libero_plus_eval.sh all
```

Default protocol:

- `--plus`
- `--plus-perturbation all`
- `--plus-official-category all`
- `--num-trials-per-task 1`
- `--libero-plus-robot-init-qpos-mode original`
- `--history-horizon 1`
- `--rollout-decode-horizon 1`
- `--action-horizon 1`
- `--action-repeat 1`
- `--action-repeat-mode split_delta`
- `--camera-size 256`
- `--parallel-envs 16`
- `--max-batch-size 16`
- `--env-process-isolation`

`--libero-plus-robot-init-qpos-mode original` is intentional: LIBERO-Plus
rollout uses the original benchmark robot init qpos.

The full LIBERO-Plus suite contains 10,030 one-trial episodes:

| Suite | Episodes |
|-------|---------:|
| `libero_spatial` | 2,402 |
| `libero_object` | 2,518 |
| `libero_goal` | 2,591 |
| `libero_10` | 2,519 |

Cluster schedulers can wrap the same local CUDA process and explicit-shard
workflow.

## Notes
- Checkpoints saved under `torch.compile` carry an `_orig_mod.` prefix; the loader strips it.
- The DPT depth head runs in float32 even under bf16 autocast.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is recommended.

## License
See `LICENSE`.
