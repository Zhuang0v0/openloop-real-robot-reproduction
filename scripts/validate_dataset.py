"""Validate the per-episode ModelScope export before GPU preprocessing.

This check is intentionally independent of either policy implementation.  It
checks the episode parquet files, the three camera streams, the 16-D state
and action columns, and finite numeric values.  It does not load video frames
or start robot hardware.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


CAMERAS = (
    "camera_3",
    "left_wrist",
    "right_wrist",
)


def episode_id(path: Path) -> int:
    match = re.search(r"episode_(\d+)\.parquet$", path.name)
    if not match:
        raise ValueError(f"cannot parse episode id from {path}")
    return int(match.group(1))


def find_parquets(root: Path) -> list[Path]:
    files = sorted((root / "data").rglob("episode_*.parquet"))
    return files or sorted(root.rglob("episode_*.parquet"))


def find_video(root: Path, camera: str, episode: int) -> Path | None:
    candidates = (
        root / "videos" / f"observation.images.{camera}" / "chunk-000" / f"episode_{episode:06d}.mp4",
        root / "videos" / f"observation.images.{camera}" / f"episode_{episode:06d}.mp4",
        root / "videos" / camera / "chunk-000" / f"episode_{episode:06d}.mp4",
        root / "videos" / camera / f"episode_{episode:06d}.mp4",
    )
    found = next((path for path in candidates if path.is_file()), None)
    if found is not None:
        return found
    for camera_root in (
        root / "videos" / f"observation.images.{camera}",
        root / "videos" / camera,
        root / "per_episode" / "videos" / f"observation.images.{camera}",
        root / "per_episode" / "videos" / camera,
    ):
        matches = sorted(camera_root.rglob(f"episode_{episode:06d}.mp4"))
        if matches:
            return matches[0]
    return None


def validate(root: Path, expected_episodes: int) -> dict[str, object]:
    root = root.expanduser().resolve()
    parquets = find_parquets(root)
    ids = sorted(episode_id(path) for path in parquets)
    if ids != list(range(expected_episodes)):
        raise ValueError(f"expected parquet episode ids 0..{expected_episodes - 1}, found {ids}")

    rows_per_episode: dict[int, int] = {}
    for path in sorted(parquets, key=episode_id):
        table = pq.read_table(path, columns=["action", "observation.state"])
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        if action.ndim != 2 or action.shape[1] != 16:
            raise ValueError(f"{path}: action shape {action.shape}, expected [T, 16]")
        if state.ndim != 2 or state.shape != action.shape:
            raise ValueError(f"{path}: state shape {state.shape}, expected {action.shape}")
        if not np.isfinite(action).all() or not np.isfinite(state).all():
            raise ValueError(f"{path}: non-finite state/action values")
        rows_per_episode[episode_id(path)] = int(action.shape[0])

    missing = {
        camera: [episode for episode in ids if find_video(root, camera, episode) is None]
        for camera in CAMERAS
    }
    missing = {camera: episodes for camera, episodes in missing.items() if episodes}
    if missing:
        raise FileNotFoundError(f"missing camera videos: {missing}")

    return {
        "root": str(root),
        "episodes": expected_episodes,
        "training_episode_ids": list(range(expected_episodes)),
        "held_out_evaluation_split": False,
        "cameras": list(CAMERAS),
        "state_dim": 16,
        "action_dim": 16,
        "rows_per_episode": rows_per_episode,
        "total_rows": sum(rows_per_episode.values()),
        "status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--episodes",
        type=int,
        default=int(os.environ.get("EXPECTED_EPISODES", "106")),
        help="expected episode count (default: 106)",
    )
    args = parser.parse_args()
    report = validate(args.dataset_root, args.episodes)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
