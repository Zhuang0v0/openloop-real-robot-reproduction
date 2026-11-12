#!/usr/bin/env python3
"""Convert a 16D LeRobot export to Efficient-WAM's clean layout.

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
import csv
import json
import re
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
INSTRUCTION = "Put all fruits into the large basket first, then put all blocks into the small basket."


def _episode_id(path: Path) -> int:
    stem = path.stem
    match = re.search(r"episode[_-](\d+)", stem, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
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
    """Find frame parquet files in both LeRobot layouts.

    ModelScope exports are commonly either one ``episode_*.parquet`` per
    episode or LeRobot-v3 shards such as ``data/chunk-000/file-000.parquet``.
    The latter stores the episode id in an ``episode_index`` column, so the
    caller must group rows instead of inferring the id from the filename.
    """

    data_root = dataset_root / "data"
    search_root = data_root if data_root.is_dir() else dataset_root
    candidates: list[Path] = []
    for path in sorted(search_root.rglob("*.parquet")):
        if path.name.lower() in {"tasks.parquet", "episodes.parquet"}:
            continue
        # Avoid treating unrelated metadata tables as frame streams.  The
        # schema check is intentionally cheap and does not read the payload.
        try:
            names = set(pq.read_schema(path).names)
        except Exception:
            names = set()
        if names and not ({"action", "observation.state"} <= names):
            continue
        candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No frame parquet files below {search_root}")
    return candidates


def _episode_records(dataset_root: Path) -> Dict[int, Dict[str, Any]]:
    """Read frame parquet files and return one validated record per episode.

    A shard may contain several episodes and an episode may span several
    shards.  Rows are grouped by ``episode_index`` when present; per-episode
    files continue to use their filename as the fallback id.  If every part
    has ``frame_index``, rows are restored to chronological order across
    shards.  No video or output file is touched by this helper.
    """

    parts: Dict[int, list[dict[str, Any]]] = {}
    for file_order, parquet_path in enumerate(_find_episode_files(dataset_root)):
        table = pq.read_table(parquet_path)
        names = set(table.column_names)
        if "action" not in names or "observation.state" not in names:
            raise ValueError(f"Frame parquet is missing action/state columns: {parquet_path}")
        action_values = table["action"].to_pylist()
        state_values = table["observation.state"].to_pylist()
        if len(action_values) != len(state_values):
            raise ValueError(
                f"Action/state row counts differ in {parquet_path}: "
                f"{len(action_values)} vs {len(state_values)}"
            )
        row_count = len(action_values)
        if "episode_index" in names:
            episode_values = table["episode_index"].to_pylist()
            if len(episode_values) != row_count:
                raise ValueError(f"episode_index length differs in {parquet_path}")
            episode_ids = sorted({int(value) for value in episode_values})
        else:
            episode_ids = [_episode_id(parquet_path)]
            episode_values = [episode_ids[0]] * row_count

        frame_values = table["frame_index"].to_pylist() if "frame_index" in names else None
        if frame_values is not None and len(frame_values) != row_count:
            raise ValueError(f"frame_index length differs in {parquet_path}")
        for episode_id in episode_ids:
            row_indices = [index for index, value in enumerate(episode_values) if int(value) == episode_id]
            if not row_indices:
                continue
            action = _as_matrix((action_values[index] for index in row_indices), "action", episode_id)
            state = _as_matrix((state_values[index] for index in row_indices), "state", episode_id)
            if action.shape[0] != state.shape[0]:
                raise ValueError(f"Episode {episode_id} action/state lengths differ in {parquet_path}")
            frames = None if frame_values is None else np.asarray(
                [frame_values[index] for index in row_indices], dtype=np.int64
            )
            parts.setdefault(episode_id, []).append(
                {
                    "action": action,
                    "state": state,
                    "frame_index": frames,
                    "file_order": file_order,
                    "source": str(parquet_path),
                }
            )

    records: Dict[int, Dict[str, Any]] = {}
    for episode_id, episode_parts in sorted(parts.items()):
        with_frames = all(part["frame_index"] is not None for part in episode_parts)
        if with_frames:
            # Stable sort preserves source order for duplicate frame indices,
            # while still restoring episodes split across shard boundaries.
            order: list[tuple[int, int, int]] = []
            for part_index, part in enumerate(episode_parts):
                order.extend(
                    (int(frame), part_index, row_index)
                    for row_index, frame in enumerate(part["frame_index"])
                )
            order.sort(key=lambda item: item[0])
            action = np.stack(
                [episode_parts[part_index]["action"][row_index] for _, part_index, row_index in order],
                axis=0,
            )
            state = np.stack(
                [episode_parts[part_index]["state"][row_index] for _, part_index, row_index in order],
                axis=0,
            )
            frame_index = np.asarray([frame for frame, _, _ in order], dtype=np.int64)
        else:
            episode_parts.sort(key=lambda part: int(part["file_order"]))
            action = np.concatenate([part["action"] for part in episode_parts], axis=0)
            state = np.concatenate([part["state"] for part in episode_parts], axis=0)
            frame_index = None
        records[int(episode_id)] = {
            "action": action,
            "state": state,
            "frame_index": frame_index,
            "sources": [part["source"] for part in episode_parts],
        }
    if not records:
        raise ValueError(f"No episodes found in frame parquet files below {dataset_root}")
    return records


def _read_episode_ids(path: Path) -> set[int]:
    """Read an explicit episode whitelist from a CSV/JSON manifest.

    The Classify Objects workbook matcher is the authority for labels.  This
    helper deliberately accepts only an explicit ``episode_id`` (or
    ``episode_index``) column and never infers IDs from row order.
    """

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"episode manifest does not exist: {path}")
    suffix = path.suffix.lower()
    rows: list[dict[str, Any]]
    if suffix in {".json", ".jsonl"}:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("episodes", "rows", "manifest"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise ValueError(f"episode manifest must contain a list: {path}")
        rows = [dict(item) for item in payload if isinstance(item, dict)]
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [dict(item) for item in csv.DictReader(handle)]
    ids: set[int] = set()
    for row_number, row in enumerate(rows, 2):
        raw = row.get("episode_id", row.get("episode_index"))
        if raw in (None, ""):
            raise ValueError(f"episode manifest row {row_number} has no episode_id: {path}")
        try:
            value = float(str(raw).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid episode_id {raw!r} on row {row_number}: {path}") from exc
        if not value.is_integer():
            raise ValueError(f"episode_id must be integral, got {raw!r} on row {row_number}: {path}")
        status = str(row.get("match_status", "matched") or "matched").strip().lower()
        if status not in {"matched", "match", "ok", "valid", "true", "1"}:
            continue
        ids.add(int(value))
    if not ids:
        raise ValueError(f"episode manifest contains no matched episode IDs: {path}")
    return ids


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
    task_name: str = "classify_objects",
    start_episode: int = 0,
    end_episode: int | None = None,
    episode_ids_file: Path | None = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve()
    video_root = (video_root or dataset_root).expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not task_name or any(char in task_name for char in "/\\"):
        raise ValueError(f"task_name must be a simple directory name: {task_name!r}")
    task_root = output_root / "clean" / task_name
    episode_ids = _read_episode_ids(episode_ids_file) if episode_ids_file else None
    for name in ("qpos", "state", "videos", "metas", "umt5_wan"):
        (task_root / name).mkdir(parents=True, exist_ok=True)

    episodes: List[Dict[str, Any]] = []
    records = _episode_records(dataset_root)
    for episode, record in records.items():
        if episode < start_episode or (end_episode is not None and episode > end_episode):
            continue
        if episode_ids is not None and episode not in episode_ids:
            continue
        action = np.asarray(record["action"], dtype=np.float32)
        state = np.asarray(record["state"], dtype=np.float32)
        if action.shape[0] != state.shape[0]:
            raise ValueError(
                f"Episode {episode} action/state lengths differ: {action.shape} vs {state.shape}"
            )
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
            # The caller controls the episode whitelist.  Validation episodes
            # must be excluded before conversion rather than mixed into this
            # packed training root.
            "split": "train",
        })

    if not episodes:
        raise ValueError("No episodes selected")
    manifest = {
        "format": "efficient_wam_robotwin_clean_from_lerobot",
        "task": task_name,
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
    parser.add_argument(
        "--task-name",
        default="classify_objects",
        help="task directory/manifest name (default: classify_objects)",
    )
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--end-episode", type=int)
    parser.add_argument(
        "--episode-ids-file",
        type=Path,
        help="CSV/JSON manifest with an explicit episode_id whitelist",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = convert(**vars(args))
    print(json.dumps({"episodes": len(result["episodes"]), "output_root": str(args.output_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
