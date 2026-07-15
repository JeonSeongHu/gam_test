#!/usr/bin/env python3
"""Compute q01/q99 statistics from canonical 7D action arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


SUPPORTED_SUFFIXES = {".npy", ".npz", ".h5", ".hdf5", ".parquet"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Files or directories to scan.")
    parser.add_argument("--key", default="actions", help="Array or column name.")
    parser.add_argument(
        "--layout",
        choices=("canonical7", "mimicgen-osc", "robocasa365-12d"),
        default="canonical7",
        help="Convert a known raw action layout before computing statistics.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reservoir-size", type=int, default=2_000_000)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def discover_files(inputs: Iterable[Path]) -> list[Path]:
    files: set[Path] = set()
    for raw_path in inputs:
        path = raw_path.expanduser()
        if path.is_dir():
            files.update(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES
            )
        elif path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.add(path)
        else:
            raise FileNotFoundError(f"Unsupported or missing input: {path}")
    if not files:
        raise FileNotFoundError("No supported action files were found.")
    return sorted(files)


def canonicalize_actions(array: np.ndarray, layout: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim < 2:
        raise ValueError(f"Expected an action array with ndim >= 2, got {array.shape}")
    array = array.reshape(-1, array.shape[-1])
    if layout == "canonical7":
        if array.shape[-1] != 7:
            raise ValueError(f"Expected canonical 7D actions, got {array.shape}")
        return np.asarray(array, dtype=np.float32)
    if layout == "mimicgen-osc":
        if array.shape[-1] < 7:
            raise ValueError(f"Expected MimicGen 7D OSC actions, got {array.shape}")
        result = np.asarray(array[:, :7], dtype=np.float32).copy()
        result[:, :6] *= np.asarray([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])
        result[:, 6] = (result[:, 6] + 1.0) * 0.5
        return result
    if layout == "robocasa365-12d":
        if array.shape[-1] < 12:
            raise ValueError(f"Expected RoboCasa365 12D actions, got {array.shape}")
        result = np.concatenate([array[:, 5:11], array[:, 11:12]], axis=1).astype(np.float32)
        result[:, 6] = np.clip((result[:, 6] + 1.0) * 0.5, 0.0, 1.0)
        return result
    raise AssertionError(f"Unhandled action layout: {layout}")


def iter_batches(array: np.ndarray, batch_size: int, layout: str) -> Iterator[np.ndarray]:
    array = canonicalize_actions(array, layout)
    if array.shape[-1] != 7:
        raise ValueError(f"Expected canonical 7D actions, got {array.shape}")
    for start in range(0, len(array), batch_size):
        batch = np.asarray(array[start : start + batch_size], dtype=np.float32)
        finite = np.isfinite(batch).all(axis=1)
        if finite.any():
            yield batch[finite]


def iter_hdf5_arrays(path: Path, key: str) -> Iterator[np.ndarray]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to read HDF5 inputs.") from exc

    with h5py.File(path, "r") as handle:
        matches = []

        def collect(name: str, value: object) -> None:
            if isinstance(value, h5py.Dataset) and (name == key or name.rsplit("/", 1)[-1] == key):
                matches.append(value)

        handle.visititems(collect)
        if not matches:
            raise KeyError(f"No HDF5 dataset named {key!r} in {path}")
        for dataset in matches:
            yield np.asarray(dataset)


def iter_file_arrays(path: Path, key: str) -> Iterator[np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        yield np.load(path, mmap_mode="r", allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if key not in data.files:
                raise KeyError(f"Missing array {key!r} in {path}; available={data.files}")
            yield np.asarray(data[key])
    elif suffix in {".h5", ".hdf5"}:
        yield from iter_hdf5_arrays(path, key)
    elif suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "pandas and a parquet engine are required for parquet inputs."
            ) from exc
        frame = pd.read_parquet(path, columns=[key])
        values = frame[key].to_numpy()
        yield np.stack(values) if values.dtype == object else values
    else:
        raise ValueError(f"Unsupported input: {path}")


def update_reservoir(
    reservoir: np.ndarray,
    priorities: np.ndarray,
    batch: np.ndarray,
    rng: np.random.Generator,
    capacity: int,
) -> tuple[np.ndarray, np.ndarray]:
    batch_priorities = rng.random(len(batch), dtype=np.float64)
    rows = np.concatenate([reservoir, batch], axis=0)
    keys = np.concatenate([priorities, batch_priorities], axis=0)
    if len(rows) <= capacity:
        return rows, keys
    selected = np.argpartition(keys, -capacity)[-capacity:]
    return rows[selected], keys[selected]


def main() -> int:
    args = parse_args()
    capacity = int(args.reservoir_size)
    if capacity <= 0:
        raise ValueError("--reservoir-size must be positive.")

    rng = np.random.default_rng(args.seed)
    reservoir = np.empty((0, 7), dtype=np.float32)
    priorities = np.empty((0,), dtype=np.float64)
    total_rows = 0
    files = discover_files(args.inputs)

    for path in files:
        for array in iter_file_arrays(path, args.key):
            for batch in iter_batches(array, max(1, int(args.batch_size)), args.layout):
                total_rows += len(batch)
                reservoir, priorities = update_reservoir(
                    reservoir, priorities, batch, rng, capacity
                )

    if not total_rows:
        raise ValueError("No finite canonical action rows were found.")

    payload = {
        "schema_version": 1,
        "action_space": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper_close"],
        "normalization": "q01_q99",
        "input_layout": args.layout,
        "total_rows": total_rows,
        "sampled_rows": len(reservoir),
        "seed": int(args.seed),
        "q01": np.percentile(reservoir, 1, axis=0).tolist(),
        "q99": np.percentile(reservoir, 99, axis=0).tolist(),
        "mean": reservoir.mean(axis=0).tolist(),
        "std": reservoir.std(axis=0).tolist(),
        "min": reservoir.min(axis=0).tolist(),
        "max": reservoir.max(axis=0).tolist(),
    }
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output} from {total_rows:,} rows ({len(reservoir):,} sampled)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
