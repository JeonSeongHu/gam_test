#!/usr/bin/env python3
"""Select base-static RoboCasa365 episodes and export metric depth plus an index."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


DEFAULT_CAMERAS = ("robot0_agentview_left", "robot0_eye_in_hand")


@dataclass(frozen=True)
class Episode:
    source: str
    task: str
    dataset_root: Path
    episode_index: int
    parquet_path: Path
    frame_indices: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robocasa-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sources", default="atomic,composite")
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--target-demos-per-task", type=int, default=100)
    parser.add_argument("--base-static-threshold", type=float, default=1.0e-6)
    parser.add_argument("--noop-threshold", type=float, default=1.0e-4)
    parser.add_argument("--cameras", default=",".join(DEFAULT_CAMERAS))
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--render-gpu-device-id", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def csv_set(value: str | None) -> set[str] | None:
    values = {item.strip() for item in str(value or "").split(",") if item.strip()}
    return values or None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_episode_rows(root: Path) -> list[dict[str, Any]]:
    jsonl = root / "meta" / "episodes.jsonl"
    if jsonl.exists():
        return [
            json.loads(line)
            for line in jsonl.read_text(encoding="utf-8").splitlines()
            if line
        ]
    parquet = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if parquet.exists():
        import pandas as pd

        return pd.read_parquet(parquet).to_dict("records")
    raise FileNotFoundError(f"Missing episode metadata under {root}")


def episode_parquet_path(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunk_size = max(1, int(info.get("chunks_size") or 1000))
    chunk = int(episode_index) // chunk_size
    template = str(info.get("data_path") or "")
    if not template:
        raise KeyError(f"Missing data_path in {root / 'meta/info.json'}")
    relative = template.format(
        episode_chunk=chunk,
        chunk_index=chunk,
        episode_index=int(episode_index),
        file_index=0,
    )
    path = root / relative
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def stack_actions(values: Any) -> np.ndarray:
    array = values.to_numpy()
    if not len(array):
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(array).astype(np.float32, copy=False)


def canonical_arm_action(actions: np.ndarray) -> np.ndarray:
    if actions.ndim != 2:
        raise ValueError(f"Expected a 2D action table, got {actions.shape}")
    if actions.shape[1] >= 12:
        return np.concatenate([actions[:, 5:11], actions[:, 11:12]], axis=1)
    if actions.shape[1] >= 7:
        return actions[:, :7].copy()
    raise ValueError(f"Expected at least 7 action dimensions, got {actions.shape}")


def keep_indices(actions: np.ndarray, threshold: float) -> np.ndarray:
    if not len(actions):
        return np.empty((0,), dtype=np.int64)
    low_motion = np.linalg.norm(actions[:, :6], axis=1) < float(threshold)
    same_gripper = np.zeros(len(actions), dtype=bool)
    same_gripper[1:] = actions[1:, 6] == actions[:-1, 6]
    no_op = low_motion.copy()
    no_op[1:] &= same_gripper[1:]
    return np.flatnonzero(~no_op).astype(np.int64)


def select_episode(
    root: Path,
    source: str,
    task: str,
    episode_index: int,
    info: dict[str, Any],
    base_static_threshold: float,
    noop_threshold: float,
) -> Episode | None:
    import pandas as pd

    parquet_path = episode_parquet_path(root, info, episode_index)
    frame = pd.read_parquet(parquet_path)
    success = np.zeros(len(frame), dtype=bool)
    if "next.reward" in frame:
        success |= np.asarray(frame["next.reward"], dtype=np.float32) > 0
    if "next.done" in frame:
        success |= np.asarray(frame["next.done"], dtype=bool)
    hits = np.flatnonzero(success)
    if not len(hits):
        return None
    frame = frame.iloc[: int(hits[0]) + 1]
    actions = stack_actions(frame["action"])
    if actions.shape[1] >= 12:
        base_motion = actions[:, :4]
        if len(base_motion) and np.max(np.abs(base_motion)) > float(base_static_threshold):
            return None
    retained = keep_indices(canonical_arm_action(actions), noop_threshold)
    if not len(retained):
        return None
    return Episode(source, task, root, int(episode_index), parquet_path, retained)


def iter_replay_roots(
    robocasa_root: Path,
    sources: set[str] | None,
    tasks: set[str] | None,
) -> Iterator[tuple[str, str, Path]]:
    pretrain_root = robocasa_root / "pretrain"
    if not pretrain_root.is_dir():
        raise FileNotFoundError(f"Missing RoboCasa pretrain directory: {pretrain_root}")
    for metadata_path in sorted(pretrain_root.rglob("extras/dataset_meta.json")):
        root = metadata_path.parent.parent
        relative = root.relative_to(pretrain_root)
        if len(relative.parts) < 3:
            continue
        source, task = relative.parts[:2]
        if sources is not None and source not in sources:
            continue
        if tasks is not None and task not in tasks:
            continue
        is_mimicgen = any(
            relative.parts[index : index + 2] == ("mg", "demo")
            for index in range(len(relative.parts) - 1)
        )
        if is_mimicgen:
            continue
        yield source, task, root


def collect_episodes(args: argparse.Namespace, robocasa_root: Path) -> list[Episode]:
    sources = csv_set(args.sources)
    tasks = csv_set(args.tasks)
    selected: list[Episode] = []
    counts: dict[str, int] = {}
    target = max(0, int(args.target_demos_per_task))
    for source, task, root in iter_replay_roots(robocasa_root, sources, tasks):
        if counts.get(task, 0) >= target:
            continue
        info = read_json(root / "meta" / "info.json")
        for row in read_episode_rows(root):
            if counts.get(task, 0) >= target:
                break
            episode_index = int(row["episode_index"])
            extras = root / "extras" / f"episode_{episode_index:06d}"
            required = (
                extras / "states.npz",
                extras / "model.xml.gz",
                extras / "ep_meta.json",
            )
            if not all(path.exists() for path in required):
                continue
            episode = select_episode(
                root,
                source,
                task,
                episode_index,
                info,
                args.base_static_threshold,
                args.noop_threshold,
            )
            if episode is not None:
                selected.append(episode)
                counts[task] = counts.get(task, 0) + 1
    return selected


def load_controller_config() -> dict[str, Any]:
    try:
        from robosuite import load_controller_config

        return load_controller_config(default_controller="OSC_POSE")
    except (ImportError, AttributeError):
        from robosuite.controllers import load_part_controller_config

        return load_part_controller_config(default_controller="OSC_POSE")


def make_env(
    task: str,
    cameras: tuple[str, ...],
    camera_size: int,
    render_gpu_device_id: int | None,
) -> Any:
    try:
        import robocasa  # noqa: F401
        import robosuite
    except ImportError as exc:
        raise RuntimeError(
            "Install RoboCasa and its simulator dependencies before rendering depth."
        ) from exc

    kwargs: dict[str, Any] = {
        "env_name": task,
        "robots": "PandaOmron",
        "controller_configs": load_controller_config(),
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "ignore_done": True,
        "use_object_obs": True,
        "use_camera_obs": True,
        "camera_names": list(cameras),
        "camera_widths": int(camera_size),
        "camera_heights": int(camera_size),
        "camera_depths": True,
        "split": "pretrain",
        "obj_instance_split": "pretrain",
    }
    if render_gpu_device_id is not None:
        kwargs["render_gpu_device_id"] = int(render_gpu_device_id)
    optional = ("obj_instance_split", "split", "render_gpu_device_id")
    while True:
        try:
            return robosuite.make(**kwargs)
        except TypeError as exc:
            bad_key = next(
                (
                    key
                    for key in optional
                    if key in kwargs and key in str(exc) and "unexpected" in str(exc)
                ),
                None,
            )
            if bad_key is None:
                raise
            kwargs.pop(bad_key)


def env_candidates(env: Any) -> Iterator[Any]:
    seen: set[int] = set()
    pending = [env]
    while pending:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        yield candidate
        pending.extend(getattr(candidate, name, None) for name in ("env", "base_env"))


def restore_episode(env: Any, model_xml: str, ep_meta: dict[str, Any], state: np.ndarray) -> None:
    for candidate in env_candidates(env):
        for name in ("set_attrs_from_ep_meta", "set_ep_meta"):
            setter = getattr(candidate, name, None)
            if callable(setter):
                setter(ep_meta)
                break
    env.reset()
    editor = next(
        (
            candidate
            for candidate in env_candidates(env)
            if hasattr(candidate, "edit_model_xml")
        ),
        env,
    )
    model_xml = editor.edit_model_xml(model_xml) if hasattr(editor, "edit_model_xml") else model_xml
    editor.reset_from_xml_string(model_xml)
    editor.sim.reset()
    editor.sim.set_state_from_flattened(np.asarray(state))
    editor.sim.forward()


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
        )
        for camera in cameras
    ]
    extrinsics = [get_camera_extrinsic_matrix(sim, camera) for camera in cameras]
    return np.asarray(intrinsics, dtype=np.float32), np.asarray(extrinsics, dtype=np.float32)


def render_metric_depth(sim: Any, camera: str, size: int) -> np.ndarray:
    from robosuite.utils.camera_utils import get_real_depth_map

    rendered = sim.render(height=size, width=size, camera_name=camera, depth=True)
    depth = rendered[1] if isinstance(rendered, tuple) else rendered
    depth = np.ascontiguousarray(np.asarray(depth, dtype=np.float32)[::-1])
    return np.asarray(get_real_depth_map(sim, depth), dtype=np.float32)


def render_episode(
    env: Any,
    episode: Episode,
    cameras: tuple[str, ...],
    camera_size: int,
) -> dict[str, np.ndarray]:
    extras = episode.dataset_root / "extras" / f"episode_{episode.episode_index:06d}"
    with np.load(extras / "states.npz", allow_pickle=False) as data:
        states = np.asarray(data["states"])
    frame_indices = episode.frame_indices[episode.frame_indices < len(states)]
    if not len(frame_indices):
        raise ValueError(f"No retained states for episode {episode.episode_index}")
    with gzip.open(extras / "model.xml.gz", "rt", encoding="utf-8") as handle:
        model_xml = handle.read()
    ep_meta = read_json(extras / "ep_meta.json")
    restore_episode(env, model_xml, ep_meta, states[int(frame_indices[0])])

    depth = np.empty(
        (len(frame_indices), len(cameras), camera_size, camera_size), dtype=np.float32
    )
    intrinsics = np.empty((len(frame_indices), len(cameras), 3, 3), dtype=np.float32)
    extrinsics = np.empty((len(frame_indices), len(cameras), 4, 4), dtype=np.float32)
    sim = next(candidate.sim for candidate in env_candidates(env) if hasattr(candidate, "sim"))
    for output_index, frame_index in enumerate(frame_indices):
        sim.set_state_from_flattened(np.asarray(states[int(frame_index)]))
        sim.forward()
        intrinsics[output_index], extrinsics[output_index] = camera_geometry(
            sim, cameras, camera_size
        )
        for camera_index, camera in enumerate(cameras):
            depth[output_index, camera_index] = render_metric_depth(sim, camera, camera_size)
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
    robocasa_root = args.robocasa_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    cameras = tuple(
        item.strip() for item in str(args.cameras).split(",") if item.strip()
    )
    if not cameras:
        raise ValueError("At least one camera is required.")
    episodes = collect_episodes(args, robocasa_root)
    print(f"selected {len(episodes)} episodes")
    if args.dry_run:
        for episode in episodes:
            print(f"{episode.source}/{episode.task} episode={episode.episode_index}")
        return 0

    entries: list[dict[str, Any]] = []
    current_task = None
    env = None
    try:
        for episode in episodes:
            relative_root = episode.dataset_root.relative_to(robocasa_root)
            output_path = (
                output_root
                / "depth"
                / episode.source
                / episode.task
                / f"episode_{episode.episode_index:06d}.npz"
            )
            if not output_path.exists() or args.overwrite:
                if episode.task != current_task:
                    if env is not None:
                        env.close()
                    env = make_env(
                        episode.task, cameras, int(args.camera_size), args.render_gpu_device_id
                    )
                    current_task = episode.task
                save_sidecar(
                    output_path,
                    render_episode(env, episode, cameras, int(args.camera_size)),
                )
                print(f"wrote {output_path}")
            entries.append(
                {
                    "source": episode.source,
                    "task": episode.task,
                    "dataset_root": relative_root.as_posix(),
                    "episode_index": episode.episode_index,
                    "depth_path": output_path.relative_to(output_root).as_posix(),
                    "frames": int(len(episode.frame_indices)),
                }
            )
    finally:
        if env is not None:
            env.close()

    index = {
        "schema_version": 1,
        "camera_names": list(cameras),
        "camera_size": int(args.camera_size),
        "episodes": entries,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
