# DA3-Giant GAM-AR — LIBERO / LIBERO-Plus

A single-stage **world-action model** for robot manipulation built on
**DA3 (Depth Anything 3)**. Blocks 0–12 of DA3-Giant act as a frozen per-view
encoder; a **`GAMFuturePredictor`** (a dense block-autoregressive
transformer) sits at block 12 and predicts next-step visual / proprio / action
tokens; DA3 blocks 13–39 then consume the predicted sequence and emit action
tokens that an MLP action head turns into robot actions.

This repository contains **only** the DA3-Giant `gam` code. Training is on
**LIBERO** (HDF5 demos); evaluation is closed-loop in simulation on **LIBERO** and
**LIBERO-Plus** (the LIBERO-Plus perturbed-task benchmark). LIBERO-Plus is an
eval-only benchmark — there is no offline LIBERO-Plus training path here.

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
| `src/robot/future_predictor.py` | `GAMFuturePredictor` |
| `src/robot/unified_loss.py` | `compute_gam_forward_loss` |
| `src/robot/da3_giant_encoder.py` | DA3-Giant backbone + action-token injection + depth/camera decode |
| `src/robot/action_head_v2.py` | MLP action head (`action_head_oft.py` = OFT ablation) |
| `src/robot/dataset.py` | LIBERO HDF5 training dataset + action/proprio normalizers |
| `src/robot/rollout_env.py`, `closed_loop_libero_eval.py` | LIBERO simulator rollout |
| `configs/training/libero_unified/` | Training configs (smoke / baseline / gam / data_variants / resume_variants / deepspeed) |
| `Depth-Anything-3/` | DA3 backbone source — **clone separately** (see Setup) |

## Setup

```bash
pip install -r requirements.txt          # see requirements-cscs.lock for the exact pinned env
# DA3 backbone source: clone the Depth-Anything-3 repo into ./Depth-Anything-3 so that
# ./Depth-Anything-3/src/depth_anything_3 is importable (da3_giant_encoder adds it to sys.path),
# then install its requirements.
export PYTHONPATH=src:$PYTHONPATH
export DA3_ROOT=/path/to/your/data_root  # all data/checkpoint paths in configs resolve under ${DA3_ROOT}
export WANDB_API_KEY=...                  # optional; only if --wandb is used
```

All machine-specific paths in the configs are written as `${oc.env:DA3_ROOT,.}/...`
(OmegaConf env resolution). Lay out your data/checkpoints under `$DA3_ROOT`:

```
$DA3_ROOT/
  checkpoints/track4world_da3.pth                  # DA3-Giant base weights (stage_1.ckpt_path)
  data/libero_hdf5/yifengzhu-hf/LIBERO-datasets/   # LIBERO HDF5 demos (training)
  gt_depth/...                                     # optional aligned GT-depth sidecars (gt_depth_root)
```

### Environment (conda or docker)
Reproducible environments are provided; pick one.

```bash
# Option A — conda
conda env create -f environment.yml && conda activate da3-libero

# Option B — docker (CUDA + headless MuJoCo/EGL preinstalled)
docker build -t da3-libero .
docker run --gpus all -it --rm -e DA3_ROOT=/data -v /host/data_root:/data \
  -e WANDB_API_KEY=$WANDB_API_KEY da3-libero
```

`requirements.txt` is the portable dependency list; `requirements-cscs.lock` is the
exact pin set validated on the original cluster (GH200/aarch64, torch 2.8). Closed-loop
eval renders MuJoCo headlessly — set `MUJOCO_GL=egl` (NVIDIA driver) or `osmesa` (software);
the Docker image installs the needed GL/EGL libraries.

### Data & weights (download separately)
- **DA3-Giant base weights** `track4world_da3.pth` — the DA3-Giant initialization the encoder fine-tunes from. Obtain from the Depth-Anything-3 release.
- **LIBERO (training + eval)** — HDF5 demos from `yifengzhu-hf/LIBERO-datasets` (HuggingFace) for training; plus the `libero` simulator/assets (`robosuite` + `LIBERO`) for closed-loop eval.
- **LIBERO-Plus (eval only)** — the perturbed-task benchmark, installed from source (`sylvestf/LIBERO-plus`). Used for closed-loop eval via `eval_libero_unified.py --plus` (simulator rollout). It is **not** a training dataset.

## Train

```bash
# single-GPU smoke (1 step) — point DA3_ROOT at a tiny LIBERO subset first
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

## Notes
- Checkpoints saved under `torch.compile` carry an `_orig_mod.` prefix; the loader strips it.
- The DPT depth head runs in float32 even under bf16 autocast.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is recommended.

## License
See `LICENSE`.
