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
| `src/eval_libero_unified.py` | LIBERO and LIBERO-Plus rollout evaluation |
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
```

### venv

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
bash scripts/setup_sources.sh
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
export DA3_LIBERO_SOURCE_DIR=$DA3_ROOT/LIBERO
export DA3_LIBERO_PLUS_DIR=$DA3_ROOT/LIBERO-plus
export PYTHONPATH=$DA3_ROOT/src:$DA3_LIBERO_SOURCE_DIR:$PYTHONPATH
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

The launchers assume the active Docker, conda, or venv already provides the
runtime libraries. They leave `LD_LIBRARY_PATH`, ImageMagick paths, and external
Python package fallbacks untouched.

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

## Public Checkpoints

Download the released GAM checkpoints:

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

Each config uses `predictor.enabled: true` and `predictor.type: gam`.

## LIBERO-Plus Evaluation

The standalone script runs one process per GPU, shards the task list, and writes
suite-level `summary.json` and `per_task.csv` files.

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

## Notes

- Checkpoints saved under `torch.compile` may contain `_orig_mod.` prefixes;
  the loader strips them.
- The DPT depth head runs in float32 under bf16 autocast.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is recommended for
  multi-GPU fine-tuning.

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
