# =============================================================================
# DA3-Giant `gam` — LIBERO / LIBERO-Plus (training + closed-loop inference)
#
# Self-contained, reproducible build: installs the full pinned Python stack,
# the LIBERO benchmark, and the DA3 backbone from source. The only things NOT
# baked in (license / size) are the datasets + base weights — mount them at run
# time under $DA3_ROOT (see README).
#
#   docker build -t da3-libero .
#   docker run --gpus all -it --rm \
#     -e DA3_ROOT=/data -v /host/data_root:/data \
#     -e WANDB_API_KEY=$WANDB_API_KEY da3-libero
#
# The base image fixes torch==2.5.1 (+cu124) and Python 3.11. The validated
# reference environment used Python 3.12 + a torch 2.8 GH200 build; the pinned
# package set in requirements.txt is identical and works on both. torch >= 2.5
# is required for the predictor's flex_attention / BlockMask path.
# =============================================================================
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# System libs for headless MuJoCo / OpenGL (LIBERO closed-loop rollout) + build tools.
# libegl1/libglvnd give EGL for MUJOCO_GL=egl; libosmesa6 is the software fallback.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential ca-certificates \
      libgl1 libglib2.0-0 libglvnd0 libegl1 libgles2 libosmesa6 libglfw3 \
      libx11-6 libxext6 libxrender1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/da3-libero

# ---- Python deps (pinned). torch (2.5.1) already in the base image satisfies
#      `torch>=2.5`, so pip does not reinstall it. -------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- DA3 backbone --------------------------------------------------------------
# da3_giant_encoder.py adds  $DA3_ROOT/Depth-Anything-3/src  to sys.path, so the
# backbone only needs to be PRESENT (not pip-installed — that would pull heavy
# NVS-only deps such as open3d/pycolmap/xformers that the gam encoder never uses).
ARG DA3_BACKBONE_COMMIT=2c21ea849ceec7b469a3e62ea0c0e270afc3281a
RUN git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git Depth-Anything-3 \
    && git -C Depth-Anything-3 checkout ${DA3_BACKBONE_COMMIT}

# ---- LIBERO benchmark ----------------------------------------------------------
# Installed with --no-deps: LIBERO's own requirements.txt pins old, conflicting
# versions (numpy 1.22, transformers 4.21, gym 0.25, robosuite 1.4.0). The
# runtime deps its env classes actually need (bddl, easydict, future) are pinned
# in requirements.txt at the validated versions.
RUN git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO \
    && pip install --no-cache-dir --no-deps -e /opt/LIBERO

# ---- LIBERO-Plus (optional: needed only for `--plus` perturbed-task eval) ------
ARG LIBERO_PLUS_COMMIT=4976dc30028e805ff8094b55501d532c48fec182
RUN git clone https://github.com/sylvestf/LIBERO-plus.git /opt/LIBERO-plus \
    && git -C /opt/LIBERO-plus checkout ${LIBERO_PLUS_COMMIT}

# ---- Project source ------------------------------------------------------------
COPY . .

# Headless rendering + runtime defaults. PYTHONPATH includes the project src, the
# LIBERO source, and the local python deps; DA3 backbone is found via DA3_ROOT.
ENV MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    DA3_ROOT=/workspace/da3-libero \
    DA3_LIBERO_SOURCE_DIR=/opt/LIBERO \
    DA3_LIBERO_PLUS_DIR=/opt/LIBERO-plus \
    PYTHONPATH=/workspace/da3-libero/src:/opt/LIBERO:/opt/LIBERO/libero \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Datasets + base weights are NOT bundled — mount/download under $DA3_ROOT:
#   $DA3_ROOT/checkpoints/track4world_da3.pth       (DA3-Giant base weights)
#   $DA3_ROOT/data/libero_noop/<suite>/*.hdf5       (LIBERO HDF5 demos, embedded depth)
#   $DA3_ROOT/data/libero_noop/_stats               (action/proprio normalizer stats)
# LIBERO benchmark assets are downloaded by the LIBERO package on first use.
CMD ["bash"]
