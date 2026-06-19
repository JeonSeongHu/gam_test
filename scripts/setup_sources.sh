#!/usr/bin/env bash
# Clone + install the source-only dependencies that are not on PyPI:
#   - DA3 backbone  (Depth-Anything-3)  -> ./Depth-Anything-3   (sys.path import)
#   - LIBERO benchmark                  -> ./LIBERO             (pip install --no-deps)
#   - LIBERO-Plus  (optional, --plus)   -> ./LIBERO-plus
#
# Run AFTER creating the Python env (Docker does this automatically):
#   conda activate da3-libero            # or: source .venv/bin/activate
#   bash scripts/setup_sources.sh
#
# Pinned commits match the validated reference environment. Override the install
# root with DA3_ROOT (defaults to the repo root = this script's parent dir).
set -euo pipefail

ROOT="${DA3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
echo "[setup_sources] installing source deps under: $ROOT"

DA3_BACKBONE_COMMIT="${DA3_BACKBONE_COMMIT:-2c21ea849ceec7b469a3e62ea0c0e270afc3281a}"
LIBERO_PLUS_COMMIT="${LIBERO_PLUS_COMMIT:-4976dc30028e805ff8094b55501d532c48fec182}"

# ---- DA3 backbone: present-on-disk only (da3_giant_encoder adds its src to sys.path) ----
if [ ! -d Depth-Anything-3/src/depth_anything_3 ]; then
  git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git Depth-Anything-3
  git -C Depth-Anything-3 checkout "$DA3_BACKBONE_COMMIT"
else
  echo "[setup_sources] Depth-Anything-3 already present, skipping."
fi

# ---- LIBERO benchmark: used via PYTHONPATH only (NOT pip-installed) ----
# Put the LIBERO repo ROOT on PYTHONPATH so `import libero.libero.*` resolves.
# Do NOT add LIBERO/libero (it is a namespace package; adding it makes `import
# libero` bind to the inner LIBERO/libero/libero and breaks `import libero.libero`).
# LIBERO's requirements.txt is ignored (pins old, conflicting versions); its env
# classes' runtime deps (bddl, easydict, future) come from requirements.txt.
if [ ! -d LIBERO/libero ]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git LIBERO
fi

# robosuite pulls non-headless opencv-python (needs libGL). Force a clean
# headless-only cv2 so eval rendering works on a headless server.
pip uninstall -y opencv-python opencv-python-headless >/dev/null 2>&1 || true
pip install --no-deps "opencv-python-headless==4.11.0.86" >/dev/null 2>&1 || true

# ---- LIBERO-Plus: optional, only for `--plus` perturbed-task eval ----
if [ ! -d LIBERO-plus/libero ]; then
  git clone https://github.com/sylvestf/LIBERO-plus.git LIBERO-plus
  git -C LIBERO-plus checkout "$LIBERO_PLUS_COMMIT"
else
  echo "[setup_sources] LIBERO-plus already present, skipping."
fi

cat <<EOF

[setup_sources] done. Now set per shell:
  export DA3_ROOT="$ROOT"
  export DA3_LIBERO_SOURCE_DIR="$ROOT/LIBERO"
  export DA3_LIBERO_PLUS_DIR="$ROOT/LIBERO-plus"
  export PYTHONPATH="$ROOT/src:$ROOT/LIBERO:\${PYTHONPATH:-}"   # NOT \$ROOT/LIBERO/libero
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
EOF
