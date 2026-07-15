#!/usr/bin/env python3
"""Replay MimicGen states and export aligned metric depth sidecars."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimicgen-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", default="core")
    parser.add_argument("--tasks", default=None, help="Optional comma-separated HDF5 stems.")
    parser.add_argument("--cameras", default="agentview,robot0_eye_in_hand")
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-demos-per-task", type=int, default=None)
    parser.add_argument("--render-gpu-device-id", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def csv_values(value: str | None) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value or "").split(",") if item.strip())


def load_env_metadata(path: Path) -> dict[str, Any]:
    import h5py

    with h5py.File(path, "r") as handle:
        raw = handle["data"].attrs["env_args"]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def iter_hdf5_files(
    root: Path, splits: tuple[str, ...], tasks: set[str] | None
) -> Iterator[tuple[str, Path]]:
    for split in splits:
        split_root = root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Missing MimicGen split: {split_root}")
        for path in sorted(split_root.glob("*.hdf5")):
            if tasks is None or path.stem in tasks:
                yield split, path


def make_env(
    hdf5_path: Path,
    cameras: tuple[str, ...],
    camera_size: int,
    render_gpu_device_id: int | None,
) -> tuple[Any, dict[str, Any]]:
    try:
        import mimicgen  # noqa: F401
        import robosuite
    except ImportError as exc:
        raise RuntimeError(
            "Install MimicGen and its matching robosuite environment before rendering depth."
        ) from exc

    metadata = load_env_metadata(hdf5_path)
    kwargs = dict(metadata.get("env_kwargs") or {})
    env_name = str(metadata.get("env_name") or kwargs.pop("env_name", ""))
    if not env_name:
        raise ValueError(f"Missing env_name in {hdf5_path}")
    kwargs.pop("env_name", None)
    kwargs.update(
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_names=list(cameras),
        camera_heights=int(camera_size),
        camera_widths=int(camera_size),
        camera_depths=True,
    )
    if render_gpu_device_id is not None:
        kwargs["render_gpu_device_id"] = int(render_gpu_device_id)
    return robosuite.make(env_name, **kwargs), metadata


def restore_demo_model(env: Any, model_xml: str | bytes | None) -> None:
    env.reset()
    if model_xml is None:
        return
    if isinstance(model_xml, bytes):
        model_xml = model_xml.decode("utf-8")
    edit_model_xml = getattr(env, "edit_model_xml", None)
    if callable(edit_model_xml):
        model_xml = edit_model_xml(model_xml)
    env.reset_from_xml_string(model_xml)
    env.sim.reset()


def camera_geometry(sim: Any, cameras: tuple[str, ...], size: int) -> tuple[np.ndarray, np.ndarray]:
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
    )

    intrinsics = [
        get_camera_intrinsic_matrix(
            sim=sim,
            camera_name=camera,
            camera_height=size,
            camera_width=size,
        ).astype(np.float32)
        for camera in cameras
    ]
    extrinsics = [
        get_camera_extrinsic_matrix(sim, camera).astype(np.float32)
        for camera in cameras
    ]
    return np.stack(intrinsics), np.stack(extrinsics)


def render_metric_depth(sim: Any, camera: str, size: int) -> np.ndarray:
    from robosuite.utils.camera_utils import get_real_depth_map

    rendered = sim.render(height=size, width=size, camera_name=camera, depth=True)
    depth = rendered[1] if isinstance(rendered, tuple) else rendered
    depth = np.ascontiguousarray(np.asarray(depth, dtype=np.float32)[::-1])
    return np.asarray(get_real_depth_map(sim, depth), dtype=np.float32)


def render_demo(
    env: Any,
    states: np.ndarray,
    frame_indices: np.ndarray,
    cameras: tuple[str, ...],
    camera_size: int,
) -> dict[str, np.ndarray]:
    depth = np.empty(
        (len(frame_indices), len(cameras), camera_size, camera_size), dtype=np.float32
    )
    intrinsics = np.empty((len(frame_indices), len(cameras), 3, 3), dtype=np.float32)
    extrinsics = np.empty((len(frame_indices), len(cameras), 4, 4), dtype=np.float32)

    for output_index, frame_index in enumerate(frame_indices):
        env.sim.set_state_from_flattened(np.asarray(states[int(frame_index)]))
        env.sim.forward()
        intrinsics[output_index], extrinsics[output_index] = camera_geometry(
            env.sim, cameras, camera_size
        )
        for camera_index, camera in enumerate(cameras):
            depth[output_index, camera_index] = render_metric_depth(
                env.sim, camera, camera_size
            )

    return {
        "depth_meters": depth,
        "frame_indices": frame_indices.astype(np.int64),
        "camera_names": np.asarray(cameras, dtype="U"),
        "camera_intrinsics": intrinsics,
        "camera_extrinsics_c2w": extrinsics,
    }


def save_sidecar(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    cameras = csv_values(args.cameras)
    splits = csv_values(args.splits)
    tasks = set(csv_values(args.tasks)) or None
    if not cameras:
        raise ValueError("At least one camera is required.")

    import h5py

    total = 0
    rendered = 0
    skipped = 0
    for split, hdf5_path in iter_hdf5_files(args.mimicgen_root.expanduser(), splits, tasks):
        with h5py.File(hdf5_path, "r") as handle:
            demo_names = sorted(handle["data"].keys())
        if args.max_demos_per_task is not None:
            demo_names = demo_names[: max(0, int(args.max_demos_per_task))]
        total += len(demo_names)
        if args.dry_run:
            print(f"{split}/{hdf5_path.name}: {len(demo_names)} demos")
            continue

        env, _ = make_env(
            hdf5_path, cameras, int(args.camera_size), args.render_gpu_device_id
        )
        try:
            for demo_name in demo_names:
                output_path = (
                    args.output_root.expanduser()
                    / split
                    / f"{hdf5_path.stem}__{demo_name}.npz"
                )
                if output_path.exists() and not args.overwrite:
                    skipped += 1
                    continue
                with h5py.File(hdf5_path, "r") as handle:
                    demo = handle["data"][demo_name]
                    states = np.asarray(demo["states"])
                    model_xml = demo.attrs.get("model_file")
                restore_demo_model(env, model_xml)
                frame_indices = np.arange(
                    0, len(states), max(1, int(args.frame_stride)), dtype=np.int64
                )
                save_sidecar(
                    output_path,
                    render_demo(env, states, frame_indices, cameras, int(args.camera_size)),
                )
                rendered += 1
                print(f"wrote {output_path}")
        finally:
            env.close()

    print(f"demos={total} rendered={rendered} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
