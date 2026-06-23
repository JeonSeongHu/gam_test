#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ "$(hostname)" == nid* && -f "$SCRIPT_DIR/../.claude/env/clariden-ln001.env" ]]; then
    source "$SCRIPT_DIR/../.claude/env/clariden-ln001.env"
else
    source "$SCRIPT_DIR/../.claude/env/resolve.sh"
fi
DA3_RUNTIME_ROOT="${DA3_CODE_ROOT:-$SCRIPT_ROOT}"
export DA3_CODE_ROOT="$SCRIPT_ROOT"

export DA3_LIBERO_DIR="${DA3_LIBERO_DIR:-$DA3_DATA_ROOT/libero}"
export DA3_EVAL_LAUNCHER="${DA3_EVAL_LAUNCHER:-local}"
export DA3_LIBERO_SOURCE_DIR="${DA3_LIBERO_SOURCE_DIR:-$DA3_RUNTIME_ROOT/LIBERO}"
export DA3_LOCAL_GLVND="${DA3_LOCAL_GLVND:-$DA3_RUNTIME_ROOT/.local/glvnd_runtime}"
export DA3_LOCAL_PYTHON_DEPS="${DA3_LOCAL_PYTHON_DEPS:-$DA3_RUNTIME_ROOT/.local/python_deps/site}"
export DA3_LOCAL_IMAGEMAGICK="${DA3_LOCAL_IMAGEMAGICK:-$DA3_RUNTIME_ROOT/.local/imagemagick}"
if [[ "$(hostname)" == nid* && "${DA3_MUJOCO_GL:-auto}" == "auto" ]]; then
    export DA3_MUJOCO_GL=egl
fi

if [[ -d "$DA3_LIBERO_SOURCE_DIR/libero" ]]; then
    export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$DA3_LIBERO_SOURCE_DIR/.libero_config_da3}"
fi
if [[ -d "$DA3_LOCAL_GLVND/lib" ]]; then
    export LD_LIBRARY_PATH="$DA3_LOCAL_GLVND/lib:${LD_LIBRARY_PATH:-}"
fi
if [[ -d "$DA3_LOCAL_GLVND/glvnd/egl_vendor.d" ]]; then
    export __EGL_VENDOR_LIBRARY_DIRS="$DA3_LOCAL_GLVND/glvnd/egl_vendor.d"
fi
if [[ -d "$DA3_LOCAL_IMAGEMAGICK/lib" ]]; then
    export LD_LIBRARY_PATH="$DA3_LOCAL_IMAGEMAGICK/lib:${LD_LIBRARY_PATH:-}"
    export PATH="$DA3_LOCAL_IMAGEMAGICK/bin:${PATH:-}"
    export MAGICK_HOME="$DA3_LOCAL_IMAGEMAGICK"
    export MAGICK_CONFIGURE_PATH="$DA3_LOCAL_IMAGEMAGICK/etc/ImageMagick-7:$DA3_LOCAL_IMAGEMAGICK/lib/ImageMagick-7.1.1/config-7_Q16HDRI10"
    export MAGICK_CODER_MODULE_PATH="$DA3_LOCAL_IMAGEMAGICK/lib/ImageMagick-7.1.1/modules-7_Q16HDRI10/coders"
    export MAGICK_FILTER_MODULE_PATH="$DA3_LOCAL_IMAGEMAGICK/lib/ImageMagick-7.1.1/modules-7_Q16HDRI10/filters"
fi

if [[ ! -d "$DA3_LIBERO_DIR" ]]; then
    echo "ERROR: LIBERO data missing at DA3_LIBERO_DIR=$DA3_LIBERO_DIR" >&2
    echo "Set DA3_LIBERO_DIR to the LIBERO data root before running." >&2
    exit 1
fi

validate_cuda_visible_devices() {
    local visible="${CUDA_VISIBLE_DEVICES:-${DA3_SMOKE_GPU:-0}}"
    local available=",${DA3_AVAILABLE_GPUS:-},"
    local gpu
    IFS=',' read -ra gpus <<< "$visible"
    for gpu in "${gpus[@]}"; do
        gpu="${gpu//[[:space:]]/}"
        [[ -z "$gpu" ]] && continue
        if [[ "$available" != *",$gpu,"* ]]; then
            echo "ERROR: CUDA_VISIBLE_DEVICES=$visible uses GPU $gpu outside DA3_AVAILABLE_GPUS=${DA3_AVAILABLE_GPUS:-unset}" >&2
            exit 1
        fi
    done
    export CUDA_VISIBLE_DEVICES="$visible"
}

detect_mujoco_gl() {
    if [[ -n "${DA3_MUJOCO_GL:-}" && "$DA3_MUJOCO_GL" != "auto" ]]; then
        return
    fi
    if [[ -n "${PYOPENGL_PLATFORM:-}" ]]; then
        DA3_MUJOCO_GL="$PYOPENGL_PLATFORM"
        return
    fi

    local display_candidates=()
    if [[ -n "${DISPLAY:-}" ]]; then
        display_candidates+=("$DISPLAY")
    fi
    if [[ -S /tmp/.X11-unix/X1 ]]; then
        display_candidates+=(":1")
    fi
    local candidate
    for candidate in "${display_candidates[@]}"; do
        if DISPLAY="$candidate" MUJOCO_GL=glx "$DA3_PYTHON" -c 'import mujoco; ctx = mujoco.GLContext(64, 64); ctx.free()' >/dev/null 2>&1; then
            export DISPLAY="$candidate"
            DA3_MUJOCO_GL=glx
            return
        fi
    done

    if MUJOCO_GL=egl "$DA3_PYTHON" -c 'import mujoco; ctx = mujoco.GLContext(64, 64); ctx.free()' >/dev/null 2>&1; then
        DA3_MUJOCO_GL=egl
    else
        DA3_MUJOCO_GL=osmesa
    fi
}

detect_mujoco_gl
case "$DA3_MUJOCO_GL" in
    egl|osmesa)
        if [[ -n "${PYOPENGL_PLATFORM:-}" && "$PYOPENGL_PLATFORM" != "$DA3_MUJOCO_GL" ]]; then
            echo "ERROR: PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM conflicts with DA3_MUJOCO_GL=$DA3_MUJOCO_GL" >&2
            exit 1
        fi
        export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$DA3_MUJOCO_GL}"
        ;;
    glx)
        if [[ -n "${PYOPENGL_PLATFORM:-}" ]]; then
            echo "ERROR: PYOPENGL_PLATFORM must be unset for DA3_MUJOCO_GL=glx (got $PYOPENGL_PLATFORM)" >&2
            exit 1
        fi
        unset PYOPENGL_PLATFORM
        ;;
    *)
        echo "ERROR: unsupported DA3_MUJOCO_GL=$DA3_MUJOCO_GL" >&2
        exit 1
        ;;
esac
if [[ "$DA3_MUJOCO_GL" == "egl" && -n "${CUDA_VISIBLE_DEVICES:-${DA3_SMOKE_GPU:-}}" ]]; then
    _visible_for_egl="${CUDA_VISIBLE_DEVICES:-${DA3_SMOKE_GPU:-}}"
    export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${_visible_for_egl%%,*}}"
    unset _visible_for_egl
fi
export MUJOCO_GL="$DA3_MUJOCO_GL"
export PYTHONPATH="$DA3_CODE_ROOT/src:$DA3_LIBERO_SOURCE_DIR:$DA3_LOCAL_PYTHON_DEPS:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

case "$DA3_EVAL_LAUNCHER" in
    local)
        validate_cuda_visible_devices
        echo "[libero-eval] launcher=local CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES MUJOCO_GL=$MUJOCO_GL"
        exec "$DA3_PYTHON" -u "$DA3_CODE_ROOT/src/eval_libero_unified.py" "$@"
        ;;
    sbatch)
        mkdir -p "$DA3_CODE_ROOT/logs"
        echo "[libero-eval] launcher=sbatch account=${DA3_ACCOUNT:-unset} partition=${DA3_PARTITION_TRAIN:-normal} MUJOCO_GL=$MUJOCO_GL"
        exec sbatch \
            --account="${DA3_ACCOUNT:-a144}" \
            --partition="${DA3_PARTITION_TRAIN:-normal}" \
            "$DA3_CODE_ROOT/sbatch/eval_libero_cscs.sbatch" "$@"
        ;;
    *)
        echo "ERROR: unknown DA3_EVAL_LAUNCHER=$DA3_EVAL_LAUNCHER (expected local or sbatch)" >&2
        exit 1
        ;;
esac
