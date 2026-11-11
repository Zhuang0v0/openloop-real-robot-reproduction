#!/usr/bin/env python3
"""Convert the building-block LeRobot export to Efficient-WAM's clean layout.

The released Efficient-WAM RoboTwin packer consumes one task directory with
per-episode action tensors, videos, and text metadata.  The LeRobot recording
also contains a separate observation.state stream, so this converter writes it
to ``state/`` instead of incorrectly reusing actions as the initial state.

Video layout is a 320x384 mosaic (Efficient-WAM config uses [height, width] =
[384, 320]): camera_3 across the top (320x192), and the left/right wrist
views side by side below (160x192 each).  ffmpeg is used for
the transcode because it is available in the target training environment and
avoids loading an entire episode into Python memory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pyarrow.parquet as pq


CAMERAS = {
    "head": "observation.images.camera_3",
    "left_wrist": "observation.images.left_wrist",
    "right_wrist": "observation.images.right_wrist",
}
INSTRUCTION = "Stack the blocks into a tower."


def _episode_id(path: Path) -> int:
    stem = path.stem
    if stem.startswith("episode_"):
        return int(stem.split("_")[-1])
    raise ValueError(f"Cannot infer episode id from {path}")


def _as_matrix(values: Iterable[Any], name: str, episode: int) -> np.ndarray:
    rows = [np.asarray(row, dtype=np.float32) for row in values]
    if not rows:
        raise ValueError(f"Episode {episode} has no {name} rows")
    result = np.stack(rows, axis=0)
    if result.ndim != 2 or result.shape[1] != 16:
        raise ValueError(f"Episode {episode} {name} shape is {result.shape}, expected [T,16]")
    if not np.isfinite(result).all():
        raise ValueError(f"Episode {episode} {name} contains non-finite values")
    return result


def _find_episode_files(dataset_root: Path) -> List[Path]:
    candidates = sorted((dataset_root / "data").rglob("episode_*.parquet"))
    if not candidates:
        candidates = sorted(dataset_root.rglob("episode_*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No per-episode parquet files below {dataset_root}")
    return candidates


def _video_path(video_root: Path, feature: str, episode: int) -> Path:
    candidates = [
        video_root / "videos" / feature / "chunk-000" / f"episode_{episode:06d}.mp4",
        video_root / "per_episode" / "videos" / feature / "chunk-000" / f"episode_{episode:06d}.mp4",
        video_root / "videos" / feature / f"episode_{episode:06d}.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # ModelScope/LeRobot exports may use a different chunk number.  Search
    # only within the expected camera directory so a duplicate episode from
    # another camera cannot be selected accidentally.
    for camera_root in (
        video_root / "videos" / feature,
        video_root / "per_episode" / "videos" / feature,
    ):
        matches = sorted(camera_root.rglob(f"episode_{episode:06d}.mp4"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Missing {feature} video for episode {episode}; tried {candidates}")


def _run_mosaic(head: Path, left: Path, right: Path, output: Path, *, overwrite: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        return
    # Keep all views temporally aligned.  The source recordings are 30 FPS;
    # -shortest prevents a trailing repeated view if one camera is shorter.
    filters = (
        "[0:v]scale=320:192:force_original_aspect_ratio=decrease,pad=320:192:(ow-iw)/2:(oh-ih)/2:color=black[top];"
        "[1:v]scale=160:192:force_original_aspect_ratio=decrease,pad=160:192:(ow-iw)/2:(oh-ih)/2:color=black[left];"
        "[2:v]scale=160:192:force_original_aspect_ratio=decrease,pad=160:192:(ow-iw)/2:(oh-ih)/2:color=black[right];"
        "[left][right]hstack=inputs=2[bottom];[top][bottom]vstack=inputs=2[out]"
    )
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(head), "-i", str(left), "-i", str(right),
        "-filter_complex", filters, "-map", "[out]", "-r", "30",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    subprocess.run(command, check=True)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def convert(
    dataset_root: Path,
    output_root: Path,
    *,
    video_root: Path | None = None,
    instruction: str = INSTRUCTION,
    start_episode: int = 0,
    end_episode: int | None = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve()
    video_root = (video_root or dataset_root).expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    task_root = output_root / "clean" / "building_block_towers"
    for name in ("qpos", "state", "videos", "metas", "umt5_wan"):
        (task_root / name).mkdir(parents=True, exist_ok=True)

    episodes: List[Dict[str, Any]] = []
    for parquet_path in _find_episode_files(dataset_root):
        episode = _episode_id(parquet_path)
        if episode < start_episode or (end_episode is not None and episode > end_episode):
            continue
        table = pq.read_table(parquet_path)
        action = _as_matrix(table["action"].to_pylist(), "action", episode)
        state = _as_matrix(table["observation.state"].to_pylist(), "state", episode)
        if action.shape[0] != state.shape[0]:
            raise ValueError(f"Episode {episode} action/state lengths differ: {action.shape} vs {state.shape}")
        action_path = task_root / "qpos" / f"episode_{episode:06d}.pt"
        state_path = task_root / "state" / f"episode_{episode:06d}.pt"
        if overwrite or not action_path.exists():
            import torch
            torch.save(torch.from_numpy(action), action_path)
        if overwrite or not state_path.exists():
            import torch
            torch.save(torch.from_numpy(state), state_path)

        source_videos = {role: _video_path(video_root, feature, episode) for role, feature in CAMERAS.items()}
        mosaic_path = task_root / "videos" / f"episode_{episode:06d}.mp4"
        _run_mosaic(source_videos["head"], source_videos["left_wrist"], source_videos["right_wrist"], mosaic_path, overwrite=overwrite)
        meta_path = task_root / "metas" / f"episode_{episode:06d}.txt"
        if overwrite or not meta_path.exists():
            meta_path.write_text(instruction.strip() + "\n", encoding="utf-8")
        episodes.append({
            "episode_index": episode,
            "frames": int(action.shape[0]),
            "state_dim": int(state.shape[1]),
            "action_dim": int(action.shape[1]),
            "qpos": str(action_path),
            "state": str(state_path),
            "video": str(mosaic_path),
            "meta": str(meta_path),
            # This experiment intentionally uses all 100 episodes for training.
            # Real-robot evaluation is performed after fine-tuning, not by
            # marking a subset of episodes as an offline validation split.
            "split": "train",
        })

    if not episodes:
        raise ValueError("No episodes selected")
    manifest = {
        "format": "efficient_wam_robotwin_clean_from_lerobot",
        "task": "building_block_towers",
        "instruction": instruction,
        "state_action_separate": True,
        "state_dim": 16,
        "action_dim": 16,
        "video_layout": {"width": 320, "height": 384, "config_size_hw": [384, 320], "top": "camera_3", "bottom_left": "left_wrist", "bottom_right": "right_wrist"},
        "episodes": episodes,
    }
    _write_json(output_root / "clean_manifest.json", manifest)
    _write_json(task_root / "manifest.json", manifest)
    (output_root / "task_instruction.txt").write_text(instruction.strip() + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--instruction", default=INSTRUCTION)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--end-episode", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = convert(**vars(args))
    print(json.dumps({"episodes": len(result["episodes"]), "output_root": str(args.output_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
