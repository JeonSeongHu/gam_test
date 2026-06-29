# GAM: Geometric Action Model for Robot Policy Learning

<p align="center">
  <a href="https://cvlab-kaist.github.io/Geometric-Action-Model/">Project Page</a> |
  <a href="https://arxiv.org/abs/2606.17046">Paper</a> |
  <a href="https://huggingface.co/SeonghuJeon/3da-libero-gam">Checkpoints</a> |
  <a href="#citation">BibTeX</a>
</p>

<p align="center">
  Jisang Han<sup>*1</sup>, Seonghu Jeon<sup>*1</sup>, Jaewoo Jung<sup>1,2</sup>,
  Rene Zurbrugg<sup>2,3</sup>, Honggyu An<sup>1</sup>, Tifanny Portela<sup>2,3</sup>,
  Marco Hutter<sup>2</sup>, Marc Pollefeys<sup>2</sup>,
  Seungryong Kim<sup>1</sup>, Sunghwan Hong<sup>2,3</sup>
</p>

<p align="center">
  <sup>1</sup>KAIST AI &nbsp;&nbsp;
  <sup>2</sup>ETH Zurich &nbsp;&nbsp;
  <sup>3</sup>ETH AI Center &nbsp;&nbsp;
  <sup>*</sup>Equal contribution
</p>

<p align="center">
  <a href="https://cvlab-kaist.github.io/Geometric-Action-Model/">
    <img src="https://cvlab-kaist.github.io/Geometric-Action-Model/static/images/teaser3.webp?v=20260616" alt="GAM paper teaser: overall pipeline and quantitative results" width="100%">
  </a>
</p>

GAM is a language-conditioned robot manipulation policy that repurposes a
pretrained geometric foundation model as one shared backbone for perception,
future prediction, and action decoding. This repository is the public
implementation for LIBERO and LIBERO-Plus evaluation and fine-tuning.

## Highlights

| Model | Params | LIBERO | LIBERO-Plus | Camera split | Latency |
|-------|-------:|-------:|------------:|-------------:|--------:|
| GAM | 1.4B | 97.6 | 85.5 | 83.1 | 6.9 ms |

Numbers follow the paper/project page. LIBERO and LIBERO-Plus entries are
success rates in percent. Latency is single-pass policy inference with cached
history.

## Method

GAM turns a geometric foundation model into a world-action policy with a single
split-backbone design:

```text
Multi-view RGB, proprioception, language, action history
  -> DA3-Giant blocks 0-12
       shallow geometric observation tokens
  -> GAMFuturePredictor
       causal future latent prediction conditioned on language,
       proprioception, and action history
  -> DA3-Giant blocks 13-39
       feature propagation and geometric decoding
  -> ActionHeadV2
       action chunks in the LIBERO 7D delta-action space
```

Training uses action regression, future-feature distillation, proprioception
prediction, optional DA3 depth decoding, and optional SIGReg.

## Repository

| Path | Role |
|------|------|
| `src/train_robot.py` | GAM fine-tuning entrypoint |
| `src/gam/training/` | Training helpers for metrics, data loading, checkpoints, distributed setup, and debug guards |
| `src/eval_libero_unified.py` | LIBERO and LIBERO-Plus rollout evaluation entrypoint |
| `src/gam/evaluation/` | Evaluation helpers for LIBERO-Plus metadata, registry records, and rollout videos |
| `scripts/run_hf_gam_libero_plus_eval.sh` | Standalone local-GPU LIBERO-Plus checkpoint eval |
| `src/robot/future_predictor.py` | `GAMFuturePredictor` |
| `src/robot/unified_loss.py` | GAM training losses |
| `src/robot/da3_giant_encoder.py` | DA3-Giant backbone wrapper and action-token path |
| `src/robot/action_head_v2.py` | MLP action head |
| `src/robot/dataset.py` | LIBERO HDF5 dataset and normalizers |
| `src/robot/rollout_env.py` | LIBERO and LIBERO-Plus simulator integration |
| `configs/training/libero_unified/` | GAM training configs |
| `Dockerfile`, `environment.yml`, `requirements.txt` | Public runtime definitions |

## Installation

Use Docker for a fully specified environment, or create the same Python stack
with conda or venv.

### Docker

```bash
docker build -t gam-libero .
docker run --gpus all -it --rm \
  -v /host/gam_workspace/checkpoints:/workspace/da3-libero/checkpoints \
  -v /host/gam_workspace/data:/workspace/da3-libero/data \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  gam-libero
```

### Conda

```bash
conda env create -f environment.yml
conda activate da3-libero
bash scripts/setup_sources.sh
bash scripts/setup_libero_plus.sh --download-assets
```

### venv

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
bash scripts/setup_sources.sh
bash scripts/setup_libero_plus.sh --download-assets
```

For Debian or Ubuntu venv installs, install the system packages used by Docker:

```bash
sudo apt-get install \
  libgl1 libglvnd0 libegl1 libgles2 libosmesa6 libglfw3 \
  ffmpeg imagemagick libmagickwand-dev
```

Set the runtime paths for local shells:

```bash
export DA3_ROOT=/path/to/this_repo
export DA3_BASE_CKPT=$DA3_ROOT/checkpoints/track4world_da3.pth
export DA3_LIBERO_SOURCE_DIR=$DA3_ROOT/LIBERO
export DA3_LIBERO_PLUS_DIR=$DA3_ROOT/LIBERO-plus
export PYTHONPATH=$DA3_ROOT/src:$DA3_LIBERO_PLUS_DIR:$DA3_LIBERO_SOURCE_DIR:$PYTHONPATH
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

The launchers assume the active Docker, conda, or venv already provides the
runtime libraries. They leave `LD_LIBRARY_PATH`, ImageMagick paths, and external
Python package fallbacks untouched.

`scripts/setup_sources.sh` installs source checkouts. Run
`scripts/setup_libero_plus.sh --download-assets` to download
`Sylvest/LIBERO-plus` `assets.zip`, strip the nested archive prefix, and
install assets under `$DA3_LIBERO_PLUS_DIR/libero/libero/assets`. Override the location with
`DA3_LIBERO_PLUS_ASSETS_DIR` when assets live outside the checkout.

## Data And Weights

Place data and base weights under `$DA3_ROOT`. By default, `$DA3_ROOT` is the
repository root:

```text
$DA3_ROOT/
  checkpoints/track4world_da3.pth
  data/libero_noop/<suite>/*.hdf5
  data/libero_noop/_stats/
```

Files:

| Item | Description |
|------|-------------|
| `checkpoints/track4world_da3.pth` | DA3-Giant base checkpoint |
| `data/libero_noop/<suite>/*.hdf5` | LIBERO demonstrations with embedded RGB, proprioception, actions, and depth |
| `data/libero_noop/_stats/` | Action and proprioception normalization stats |

Configs resolve data and checkpoint paths through `${oc.env:DA3_ROOT,.}`.

Download the released training assets:

```bash
hf download SeonghuJeon/3da-libero-training-assets \
  --repo-type dataset \
  --local-dir .
```

## Public Checkpoints

Download the released GAM checkpoints:

```bash
hf download SeonghuJeon/3da-libero-gam \
  --local-dir checkpoints_hf/3da-libero-gam
```

For a single-suite smoke rollout, download only that suite:

```bash
hf download SeonghuJeon/3da-libero-gam \
  spatial/gam.pt spatial/config.yaml \
  --local-dir checkpoints_hf/3da-libero-gam
```

Expected layout:

| Suite key | LIBERO suite | Checkpoint | Config |
|-----------|--------------|------------|--------|
| `spatial` | `libero_spatial` | `spatial/gam.pt` | `spatial/config.yaml` |
| `object` | `libero_object` | `object/gam.pt` | `object/config.yaml` |
| `goal` | `libero_goal` | `goal/gam.pt` | `goal/config.yaml` |
| `long` | `libero_10` | `long/gam.pt` | `long/config.yaml` |

Each config uses `predictor.enabled: true` and `predictor.type: gam`.
The released configs resolve the DA3 base checkpoint and LIBERO data under
`${DA3_ROOT}` by default.

## LIBERO-Plus Evaluation

The standalone script runs one process per GPU, shards the task list, and writes
suite-level `summary.json` and `per_task.csv` files.

Install LIBERO-Plus assets before the first rollout:

```bash
bash scripts/setup_libero_plus.sh --download-assets
```

```bash
GAM_EVAL_GPUS=0,1,2,3 \
scripts/run_hf_gam_libero_plus_eval.sh spatial

GAM_EVAL_GPUS=0,1,2,3 scripts/run_hf_gam_libero_plus_eval.sh object
GAM_EVAL_GPUS=0,1,2,3 scripts/run_hf_gam_libero_plus_eval.sh goal
GAM_EVAL_GPUS=0,1,2,3 scripts/run_hf_gam_libero_plus_eval.sh long

GAM_EVAL_GPUS=0,1,2,3 scripts/run_hf_gam_libero_plus_eval.sh all
```

Default protocol:

| Argument | Value |
|----------|-------|
| `--plus` | enabled |
| `--plus-perturbation` | `all` |
| `--plus-official-category` | `all` |
| `--num-trials-per-task` | `1` |
| `--libero-plus-robot-init-qpos-mode` | `original` |
| `--history-horizon` | `1` |
| `--rollout-decode-horizon` | `1` |
| `--action-horizon` | `1` |
| `--action-repeat` | `1` |
| `--action-repeat-mode` | `split_delta` |
| `--camera-size` | `256` |
| `--parallel-envs` | `16` |
| `--max-batch-size` | `16` |
| `--env-process-isolation` | enabled |

`original` robot qpos follows the official LIBERO-Plus robot initialization.

Parallelism is controlled by environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `GAM_EVAL_GPUS` | `CUDA_VISIBLE_DEVICES` or `0` | One eval process and one task shard per listed GPU |
| `PARALLEL_ENVS_PER_GPU` | `16` | Simulator workers per GPU process |
| `MAX_BATCH_SIZE` | `16` | Maximum observations per policy forward |
| `MAX_WAIT_TIME` | `0.5` | Batch wait time in seconds |
| `ENV_CACHE_SIZE` | `1` | Cached simulator instances per worker |
| `GAM_PLUS_PERTURBATION` | `all` | LIBERO-Plus perturbation filter |
| `GAM_PLUS_OFFICIAL_CATEGORY` | `all` | Official category filter |

The launcher sets `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, and
`EGL_PLATFORM=device`. On a bare GPU machine, install the GL, EGL, OSMesa, GLFW,
ffmpeg, and ImageMagick packages listed in the installation section.

Full LIBERO-Plus contains 10,030 one-trial episodes:

| Suite | Episodes |
|-------|---------:|
| `libero_spatial` | 2,402 |
| `libero_object` | 2,518 |
| `libero_goal` | 2,591 |
| `libero_10` | 2,519 |

## LIBERO Evaluation

```bash
PYTHONPATH=src:$PYTHONPATH python src/eval_libero_unified.py \
  --ckpt /path/to/checkpoint.pt \
  --config configs/training/libero_unified/gam/chunk8_150k_2node.yaml \
  --suites libero_spatial,libero_object,libero_goal,libero_10 \
  --num-trials-per-task 5
```

## Training

Single-GPU smoke run:

```bash
PYTHONPATH=src:$PYTHONPATH python src/train_robot.py \
  --config configs/training/libero_unified/smoke/gam_chunk2.yaml \
  --single-gpu \
  --set training.max_steps=1
```

Multi-GPU GAM fine-tuning with DeepSpeed ZeRO-2:

```bash
PYTHONPATH=src:$PYTHONPATH \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
deepspeed --include localhost:0,1,2,3 src/train_robot.py \
  --config configs/training/libero_unified/gam/chunk8_150k_2node.yaml \
  --deepspeed_config configs/training/libero_unified/deepspeed/micro2.json \
  --wandb \
  --wandb-name gam_libero
```

In-training closed-loop eval is configured through `training.closed_loop_evals`.

### Training Config Reference

`src/train_robot.py` reads an OmegaConf YAML and accepts `--set key=value`
overrides. The main GAM config is
`configs/training/libero_unified/gam/chunk8_150k_2node.yaml`.

| YAML key | Meaning |
|----------|---------|
| `stage_1.ckpt_path` | DA3-Giant base checkpoint loaded before robot fine-tuning |
| `da3_finetune.enabled` | Enables the DA3-Giant GAM fine-tuning path |
| `da3_finetune.freeze_blocks_before` | Freezes DA3 blocks before this index, with GAM using blocks 0-12 as the geometric encoder |
| `da3_finetune.n_action_steps` | Number of low-level actions represented by one GAM action token sequence |
| `da3_finetune.n_views` | Number of camera views per timestep |
| `action_head.chunk_size` | Low-level actions predicted per action-head token |
| `action_head.n_dims` | Action dimensionality, 7 for LIBERO delta actions |
| `predictor.enabled` | Enables `GAMFuturePredictor` |
| `predictor.type` | Must be `gam` for this release |
| `predictor.H_choices` | Observed history lengths sampled during training |
| `predictor.H_weights` | Sampling weights for `H_choices` |
| `predictor.lambda_feat_future` | Future latent feature distillation weight |
| `predictor.lambda_sigreg` | SIGReg regularization weight |
| `regularization.lambda_depth` | DA3 depth decode loss weight |
| `training.global_batch_size` | Target global batch size across data-parallel ranks |
| `training.micro_batch_size` | Per-GPU batch before gradient accumulation |
| `training.grad_accum_steps` | Gradient accumulation factor |
| `training.base_lr` | DA3 backbone base learning rate |
| `training.head_lr_mult` | Multiplier for the action head learning rate |
| `training.predictor_lr_mult` | Multiplier for the GAM predictor learning rate |
| `training.max_steps` | Total optimizer steps |
| `training.ckpt_every` | Checkpoint save interval in steps |
| `training.vis_every` | Visualization interval in steps |
| `training.bf16` | Uses bf16 autocast for training |
| `training.compile` | Enables `torch.compile` for the training model |
| `dataset.hdf5_root` | LIBERO HDF5 root |
| `dataset.stats_dir` | Action and proprioception stats root |
| `dataset.future_steps` | Future action/observation horizon in dataset samples |
| `dataset.chunk_size` | Low-level action chunk length from the dataset |
| `dataset.camera_keys` | Camera keys read from HDF5 |
| `dataset.da3_input_rotate180` | Applies the train-time DA3 image rotation convention |
| `dataset.gt_depth_root` | `null` means depth is read from embedded HDF5 keys |

Training CLI flags:

| Flag | Meaning |
|------|---------|
| `--config` | YAML config path |
| `--results-dir` | Output root for checkpoints, logs, and visualizations |
| `--ckpt` | Resume checkpoint |
| `--single-gpu` | Run one local GPU process |
| `--wandb` | Enable W&B logging |
| `--wandb-name` | W&B run display name |
| `--wandb-project` | W&B project override |
| `--wandb-new-run` | Start a fresh W&B run during resume |
| `--wandb-resume-from` | Rewind W&B resume point, for example `<run_id>?_step=72000` |
| `--reset-schedule` | Reset optimizer and LR scheduler on resume |
| `--reset-optimizer-state` | Load model weights while starting optimizer, scheduler, and scaler fresh |
| `--refresh-action-stats` | Reload normalizers from `dataset.stats_dir` during resume |
| `--eval-only` | Load checkpoint, run configured eval split, then exit |
| `--eval-max-batches` | Cap eval batches per rank for `--eval-only` |
| `--ddp-timeout-minutes` | Distributed process group timeout |
| `--set key=value` | Override YAML keys with OmegaConf dotlist syntax |
| `--deepspeed_config` | DeepSpeed config path, added by DeepSpeed |

Training compile controls:

| Setting | Meaning |
|---------|---------|
| `training.compile=true` | Calls `torch.compile` around the training model |
| `DA3_TRAIN_COMPILE_MODE=default` | Standard training compile mode |
| `training.compile=false` | Public configs use eager training by default |

DeepSpeed ZeRO-2 is selected by the JSON passed to `--deepspeed_config`.
`configs/training/libero_unified/deepspeed/micro2.json` sets:

| JSON key | Meaning |
|----------|---------|
| `train_micro_batch_size_per_gpu` | Per-GPU micro batch seen by DeepSpeed |
| `gradient_accumulation_steps` | DeepSpeed accumulation factor |
| `zero_optimization.stage` | ZeRO stage, `2` for optimizer-state sharding |
| `bf16.enabled` | bf16 training |
| `optimizer.type` | AdamW |
| `optimizer.params.lr` | Base optimizer LR, overridden by train param groups |
| `gradient_clipping` | Global grad clipping value |

Example resume with a fresh optimizer:

```bash
PYTHONPATH=src:$PYTHONPATH \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
deepspeed --include localhost:0,1,2,3 src/train_robot.py \
  --config configs/training/libero_unified/gam/chunk8_150k_2node.yaml \
  --deepspeed_config configs/training/libero_unified/deepspeed/micro2.json \
  --ckpt /path/to/checkpoint.pt \
  --reset-optimizer-state \
  --wandb \
  --wandb-name gam_resume_fresh_optimizer
```

### In-Training Closed-Loop Eval

Closed-loop eval during training is configured in YAML with
`training.closed_loop_evals`. Each entry is a rollout profile consumed by
`src/robot/closed_loop_libero_eval.py`.

```yaml
training:
  closed_loop_evals:
    - name: plus_spatial_smoke
      benchmark: libero_plus
      suites: [libero_spatial]
      num_trials_per_task: 1
      max_tasks_per_suite: 4
      every_steps: 2000
      plus_official_category: camera
      libero_plus_robot_init_qpos_mode: original
      action_horizon: 1
      rollout_decode_horizon: 1
      action_repeat: 1
      action_repeat_mode: split_delta
      camera_size: 256
      env_process_isolation: true
```

| Profile key | Meaning |
|-------------|---------|
| `name` | Label used in logs and W&B metrics |
| `benchmark` | `libero` or `libero_plus` |
| `suites` | Suite list such as `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` |
| `num_trials_per_task` | Rollout trials per task |
| `max_tasks_per_suite` | Optional task cap for smoke profiles |
| `every_steps` | Training step interval |
| `plus_official_category` | LIBERO-Plus category filter |
| `libero_plus_robot_init_qpos_mode` | Use `original` for official LIBERO-Plus qpos |
| `action_horizon` | GAM model-step chunks executed per policy call |
| `rollout_decode_horizon` | GAM AR model steps decoded before action selection |
| `action_repeat` | Env steps per predicted action |
| `action_repeat_mode` | `split_delta` divides motion deltas across repeats |
| `camera_size` | Simulator render resolution |
| `env_process_isolation` | Runs each env inside a child process |

## Evaluation Config Reference

Standalone rollout eval uses `src/eval_libero_unified.py`.

| Flag | Meaning |
|------|---------|
| `--ckpt` | Stage 1 GAM checkpoint |
| `--config` | Training YAML used to rebuild model architecture |
| `--use-ema` | Load EMA weights saved in the checkpoint |
| `--suites` | Comma-separated suite list |
| `--task-ids` | Comma-separated task ids within each suite |
| `--num-trials-per-task` | Trials per task |
| `--preset` | Preset horizon and wait-step bundle |
| `--max-steps` | Rollout horizon override |
| `--num-steps-wait` | Initial dummy wait steps |
| `--history-horizon` | Observed history length given to GAM |
| `--rollout-decode-horizon` | GAM AR decode length before action selection |
| `--action-horizon` | Model-step chunks executed per policy call |
| `--action-repeat` | Env steps per predicted action |
| `--action-repeat-mode` | `hold` or `split_delta` |
| `--policy-hz` | Policy frequency used by `action-repeat=auto` |
| `--env-control-hz` | LIBERO robosuite control frequency |
| `--camera-size` | Simulator camera resolution |
| `--render-gpu-device-id` | robosuite EGL render GPU override |
| `--env-process-isolation` | Spawn child env workers |
| `--output-dir` | Eval output root |
| `--run-name` | Eval run folder name |
| `--shard-index` | Shard id for distributed eval |
| `--shard-count` | Total shard count |
| `--video-every` | Save one diagnostic video every N global episodes |
| `--detailed-video` | Save RGB/depth/action diagnostic video |
| `--trace-actions` | Write per-step action/proprio diagnostics |
| `--decode-visuals` | Decode depth/RGB for diagnostics |
| `--temporal-ensemble` | ACT-style low-level action ensemble |
| `--execution-strategy` | Diagnostic execution strategy |
| `--execute-chunk-prefix` | Execute a prefix of each chunk before re-observing |
| `--partial-chunk-history` | Previous-action history policy for prefix execution |
| `--rotate-policy-input` | Rotate live RGB by 180 degrees |
| `--proprio-orientation` | `auto`, `rpy`, or `axis_angle` live proprio convention |
| `--text-prompt-normalization` | Text normalization before encoding |
| `--action-frame` | Model action frame override |
| `--wandb` | Enable W&B logging |
| `--action-stats-key` | Normalizer stats key override |
| `--plus` | Enable LIBERO-Plus |
| `--plus-root` | LIBERO-Plus source checkout |
| `--plus-perturbation` | LIBERO-Plus perturbation filter |
| `--plus-official-category` | Official category filter such as `camera` or `noise` |
| `--libero-plus-robot-init-qpos-mode` | Use `original` for official Plus qpos |
| `--plus-sample-group-by` | Deterministic Plus task sampling group |
| `--plus-samples-per-group` | Tasks per sampled group |

## CUDA Graph Latency Mode

The paper latency path is an eval-time CUDA graph path inside
`src/eval_libero_unified.py`. It fuses GAM h=1 inference into one compiled
callable:

```text
DA3 shallow encode -> GAMFuturePredictor -> DA3 deep propagation -> ActionHeadV2
```

Use this mode for model-forward latency measurement:

```bash
DA3_MAX_OPTIMIZE=1 \
DA3_COMPILE_INFERENCE_MODE=reduce-overhead \
DA3_FUSE_SHALLOW=1 \
DA3_SKIP_FULL_ENCODE=1 \
DA3_PROFILE_INFERENCE=1 \
PYTHONPATH=src:$PYTHONPATH python src/eval_libero_unified.py \
  --ckpt /path/to/checkpoint.pt \
  --config configs/training/libero_unified/gam/chunk8_150k_2node.yaml \
  --suites libero_spatial \
  --task-ids 0 \
  --num-trials-per-task 1 \
  --history-horizon 1 \
  --rollout-decode-horizon 1 \
  --action-horizon 1 \
  --action-repeat 1 \
  --action-repeat-mode split_delta \
  --camera-size 256 \
  --env-process-isolation
```

The log line has this form:

```text
[INFER PROFILE] H_eff=1 dec_vis=0 full_enc=skip total=...ms ar(no_cache)=...ms ...
```

For the paper latency number, read the `ar(...)` model-forward field after
warmup. `total` includes preprocessing, CPU copies, normalization, and logging
guards.

CUDA graph environment variables:

| Variable | Value | Meaning |
|----------|-------|---------|
| `DA3_MAX_OPTIMIZE` | `1` | Enables the fused h=1 path |
| `DA3_COMPILE_INFERENCE_MODE` | `reduce-overhead` | Uses PyTorch CUDA graph replay mode |
| `DA3_FUSE_SHALLOW` | `1` | Folds DA3 blocks 0-12 into the fused graph |
| `DA3_SKIP_FULL_ENCODE` | `1` | Skips the separate full DA3 encode in GAM action selection |
| `DA3_PROFILE_INFERENCE` | `1` | Prints `[INFER PROFILE]` timing lines |
| `DA3_CUDAGRAPH_CLONE` | `0` | Keeps fused graph output clone-free for the single graph path |
| `DA3_MAX_OPTIMIZE_NO_BF16` | `1` | Leaves inference weights in fp32 for ablation |

General per-submodule compile is also available:

| Variable | Values | Meaning |
|----------|--------|---------|
| `DA3_COMPILE_INFERENCE` | `all`, `predictor`, `shallow`, `propagate`, `action_head` | Compiles selected eval modules |
| `DA3_COMPILE_INFERENCE_MODE` | `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`, `default` | PyTorch compile mode |
| `DA3_CUDAGRAPH_CLONE` | `1` or `0` | Clones outputs from CUDA graph buffers for separate compiled modules |

## Notes

- Checkpoints saved under `torch.compile` may contain `_orig_mod.` prefixes;
  the loader strips them.
- The DPT depth head runs in float32 under bf16 autocast.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is recommended for
  multi-GPU fine-tuning.

## Acknowledgements

We thank the teams behind
[Track4World](https://github.com/TencentARC/Track4World),
[OpenPI](https://github.com/Physical-Intelligence/openpi),
[Pi0.5](https://huggingface.co/docs/lerobot/en/pi05),
[Cosmos Policy](https://github.com/nvlabs/cosmos-policy), and
[OpenVLA-OFT](https://github.com/moojink/openvla-oft) for releasing their
research, code, and models to the robotics community.

## Citation

```bibtex
@misc{han2026geometricactionmodelrobot,
      title={Geometric Action Model for Robot Policy Learning},
      author={Jisang Han and Seonghu Jeon and Jaewoo Jung and Ren{\'e} Zurbr{\"u}gg and Honggyu An and Tifanny Portela and Marco Hutter and Marc Pollefeys and Seungryong Kim and Sunghwan Hong},
      year={2026},
      eprint={2606.17046},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2606.17046}
}
```

## License

See `LICENSE`.
