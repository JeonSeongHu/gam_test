# DA3-Giant gam — LIBERO / LIBERO-Plus (train + inference)
#
# Build:  docker build -t da3-libero .
# Run  :  docker run --gpus all -it --rm \
#            -e DA3_ROOT=/data -v /host/data_root:/data \
#            -e WANDB_API_KEY=$WANDB_API_KEY da3-libero
#
# torch 2.5 base gives flex_attention/BlockMask used by the predictor (a dense-mask
# fallback exists for older torch). Use a -devel base so DeepSpeed can JIT its ops.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
# System libs for headless MuJoCo / OpenGL (LIBERO closed-loop rollout) + build tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential ca-certificates \
      libgl1 libglib2.0-0 libegl1 libgles2 libosmesa6 libglfw3 \
      libx11-6 libxext6 libxrender1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/da3-libero

# Install python deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project source.
COPY . .

# Headless rendering + runtime defaults.
ENV MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    PYTHONPATH=/workspace/da3-libero/src \
    DA3_ROOT=/workspace/da3-libero \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NOT bundled (license/size) — provide at build- or run-time per README:
#   * DA3 backbone source  -> ./Depth-Anything-3  (so depth_anything_3 is importable)
#   * LIBERO benchmark + assets (pip install the LIBERO package from source)
#   * data + base weights under $DA3_ROOT (checkpoints/, data/libero_hdf5, data/libero_plus)
CMD ["bash"]
