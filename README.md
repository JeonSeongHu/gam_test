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

**GAM (Geometric Action Model)** is a language-conditioned robot manipulation
policy that adapts a pretrained geometric foundation model into one shared
backbone for perception, future prediction, and action decoding. This public
release contains the LIBERO and LIBERO-Plus implementation, training configs,
standalone rollout scripts, and released checkpoints. The released 1.4B GAM
model reports **97.6% LIBERO**, **85.5% LIBERO-Plus**, **83.1% camera split**,
and **6.9 ms** model-forward latency with the CUDA graph inference path.

## Repository Layout

| Area | Paths |
|------|-------|
| Core model and training | `src/robot/`, `src/gam/training/`, `src/train_robot.py` |
| Rollout evaluation | `src/eval_libero_unified.py`, `src/gam/evaluation/`, `scripts/run_hf_gam_libero_plus_eval.sh` |
| Configs and runtime | `configs/training/libero_unified/`, `Dockerfile`, `environment.yml`, `requirements.txt` |
| Setup utilities | `scripts/setup_sources.sh`, `scripts/setup_libero_plus.sh` |

## Installation

Use Docker for the most reproducible setup:

```bash
docker build -t gam-libero .
docker run --gpus all -it --rm \
  -v /host/gam_workspace/checkpoints:/workspace/gam-libero/checkpoints \
  -v /host/gam_workspace/data:/workspace/gam-libero/data \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  gam-libero
```

Conda and venv are also supported:

```bash
conda env create -f environment.yml
conda activate gam-libero
bash scripts/setup_sources.sh
bash scripts/setup_libero_plus.sh --download-assets
```

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

## Data And Weights

Place data and base weights under `$DA3_ROOT`. By default, `$DA3_ROOT` is the
repository root:

```text
$DA3_ROOT/
  checkpoints/track4world_da3.pth
  data/libero_noop/<suite>/*.hdf5
  data/libero_noop/_stats/
```

Download the released training assets:

```bash
hf download SeonghuJeon/3da-libero-training-assets \
  --repo-type dataset \
  --local-dir .
```

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

Expected checkpoint layout:

| Suite key | LIBERO suite | Checkpoint | Config |
|-----------|--------------|------------|--------|
| `spatial` | `libero_spatial` | `spatial/gam.pt` | `spatial/config.yaml` |
| `object` | `libero_object` | `object/gam.pt` | `object/config.yaml` |
| `goal` | `libero_goal` | `goal/gam.pt` | `goal/config.yaml` |
| `long` | `libero_10` | `long/gam.pt` | `long/config.yaml` |

## Evaluation

Install LIBERO-Plus assets before the first rollout:

```bash
bash scripts/setup_libero_plus.sh --download-assets
```

Run standalone LIBERO-Plus evaluation from the released HF checkpoint:

```bash
GAM_EVAL_GPUS=0,1,2,3 scripts/run_hf_gam_libero_plus_eval.sh spatial
GAM_EVAL_GPUS=0,1,2,3 scripts/run_hf_gam_libero_plus_eval.sh all
```

The standalone launcher uses `qpos=original` for LIBERO-Plus robot
initialization and writes `summary.json`, `per_task.csv`, and logs under the
result directory. See [docs/evaluation.md](docs/evaluation.md) for rollout
protocols, parallelism, config flags, and CUDA graph latency measurement.

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

See [docs/training.md](docs/training.md) for config keys, CLI flags, W&B resume,
optimizer-state resume, DeepSpeed ZeRO-2, compile settings, and in-training
closed-loop eval.

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
