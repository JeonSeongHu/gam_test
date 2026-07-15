#!/usr/bin/env python3
"""Download the Open X-Embodiment LeRobot repositories used by GAM."""

from __future__ import annotations

import argparse
from pathlib import Path


OPENX_REPOSITORIES = (
    "BrunoM42/bridge_orig_lerobot",
    "lerobot/droid_1.0.1",
    "lerobot/taco_play",
    "lerobot/utaustin_mutex",
    "lerobot/stanford_hydra_dataset",
    "lerobot/berkeley_autolab_ur5",
    "lerobot/austin_sailor_dataset",
    "lerobot/austin_sirius_dataset",
    "lerobot/berkeley_fanuc_manipulation",
    "lerobot/jaco_play",
    "lerobot/fmb",
    "lerobot/stanford_kuka_multimodal_dataset",
    "BrunoM42/fractal20220817_data_lerobot",
    "lerobot/berkeley_cable_routing",
    "lerobot/roboturk",
    "lerobot/dlr_edan_shared_control",
    "lerobot/austin_buds_dataset",
    "lerobot/nyu_franka_play_dataset",
    "lerobot/nyu_door_opening_surprising_effectiveness",
    "lerobot/cmu_stretch",
    "tailong-wu/furniture_bench_dataset_lerobot_v30",
    "tailong-wu/bc_z_lerobot_v30",
    "tailong-wu/language_table_lerobot_v30",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--repo",
        action="append",
        dest="repositories",
        help="Download only this repository. May be repeated.",
    )
    parser.add_argument("--revision", default=None, help="Optional common HF revision.")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repositories = tuple(args.repositories or OPENX_REPOSITORIES)
    unknown = sorted(set(repositories) - set(OPENX_REPOSITORIES))
    if unknown:
        raise ValueError(f"Unknown GAM Open X repositories: {', '.join(unknown)}")

    output_root = args.output_root.expanduser().resolve()
    for repo_id in repositories:
        local_dir = output_root / repo_id
        print(f"{repo_id} -> {local_dir}")
        if args.dry_run:
            continue
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=args.revision,
            local_dir=local_dir,
            max_workers=max(1, int(args.max_workers)),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
