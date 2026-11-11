#!/usr/bin/env python3
"""Audit and stage a LeRobot v3 dataset for an Efficient-WAM adapter.

This utility deliberately stops before the official Efficient-WAM packed
dataset format.  It preserves LeRobot state and action separately, records the
camera/frame mapping, and writes an intermediate manifest that a model-aware
packer can consume after WAN VAE and UMT5/T5 assets are available.

The script only needs numpy and pyarrow for ``audit``.  ``stage`` additionally
copies (or links) videos and writes numpy arrays; PyTorch is optional and is
only used when ``--write-torch`` is requested.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - dependency error path
    raise SystemExit("numpy is required: pip install numpy") from exc

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - dependency error path
    raise SystemExit("pyarrow is required: pip install pyarrow") from exc


EXPECTED_DIM = 16
EPISODE_RE = re.compile(r"episode[_-](\d+)", re.IGNORECASE)
CAMERA_PREFIX = "observation.images."


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _find_info(dataset_root: Path) -> Optional[Path]:
    candidates = [dataset_root / "meta" / "info.json", dataset_root / "info.json"]
    return next((path for path in candidates if path.exists()), None)


def _find_parquet_files(dataset_root: Path) -> List[Path]:
    data_root = dataset_root / "data"
    search_root = data_root if data_root.exists() else dataset_root
    paths = [path for path in search_root.rglob("*.parquet") if path.is_file()]
    # tasks.parquet and episode metadata are not frame streams.
    paths = [path for path in paths if path.name not in {"tasks.parquet", "episodes.parquet"}]
    if not paths:
        raise FileNotFoundError(f"No frame parquet files found below {search_root}")
    return sorted(paths)


def _episode_id_from_path(path: Path) -> Optional[int]:
    match = EPISODE_RE.search(path.stem)
    return int(match.group(1)) if match else None


def _column_to_array(table: Any, name: str, *, dtype: Any) -> np.ndarray:
    if name not in table.column_names:
        raise KeyError(f"Parquet file is missing required column {name!r}")
    # to_pylist handles Arrow fixed-size-list columns across pyarrow versions.
    values = table[name].to_pylist()
    array = np.asarray(values, dtype=dtype)
    return array


def _numeric_stats(array: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[0] == 0:
        raise ValueError("Cannot compute statistics for an empty array")
    return {
        "shape": [int(value) for value in values.shape],
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "q01": np.quantile(values, 0.01, axis=0),
        "q50": np.quantile(values, 0.50, axis=0),
        "q99": np.quantile(values, 0.99, axis=0),
    }


def _frame_summary(frame_index: np.ndarray, timestamps: np.ndarray) -> Dict[str, Any]:
    frames = np.asarray(frame_index, dtype=np.int64).reshape(-1)
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if len(frames) != len(ts):
        raise ValueError("frame_index and timestamp lengths differ")
    diffs = np.diff(frames)
    time_diffs = np.diff(ts)
    positive_dt = time_diffs[time_diffs > 0]
    fps = float(1.0 / np.median(positive_dt)) if len(positive_dt) else None
    return {
        "count": int(len(frames)),
        "first": int(frames[0]) if len(frames) else None,
        "last": int(frames[-1]) if len(frames) else None,
        "contiguous": bool(len(diffs) == 0 or np.all(diffs == 1)),
        "monotonic": bool(len(diffs) == 0 or np.all(diffs >= 0)),
        "timestamp_first": float(ts[0]) if len(ts) else None,
        "timestamp_last": float(ts[-1]) if len(ts) else None,
        "estimated_fps": fps,
    }


def _camera_name_from_feature(feature: str) -> str:
    name = feature[len(CAMERA_PREFIX) :] if feature.startswith(CAMERA_PREFIX) else feature
    # Efficient-WAM's single-video RoboTwin loader calls the fixed camera view
    # "head".  Keep the source name in the manifest and expose this alias.
    if name in {"camera_3", "camera3", "third", "head"}:
        return "head"
    return name


def _camera_features(info: Mapping[str, Any]) -> List[str]:
    features = info.get("features", {})
    if not isinstance(features, Mapping):
        return []
    return sorted(
        key[len(CAMERA_PREFIX) :]
        for key, value in features.items()
        if isinstance(key, str)
        and key.startswith(CAMERA_PREFIX)
        and isinstance(value, Mapping)
        and value.get("dtype") == "video"
    )


def _video_candidates(dataset_root: Path, camera: str, episode_id: int) -> List[Path]:
    camera_roots = [
        dataset_root / "videos" / f"{CAMERA_PREFIX}{camera}",
        dataset_root / "videos" / camera,
    ]
    names = [f"episode_{episode_id:06d}.mp4", f"episode_{episode_id:06d}.MP4"]
    # The standard LeRobot layout uses file-XXX; per-episode exports use the
    # episode name.  Search both without assuming a particular chunk number.
    candidates: List[Path] = []
    for root in camera_roots:
        if not root.exists():
            continue
        for name in names:
            candidates.extend(root.rglob(name))
        candidates.extend(root.rglob(f"file-{episode_id:03d}.mp4"))
    return sorted(set(path for path in candidates if path.is_file()))


def _probe_video(path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"path": str(path), "bytes": int(path.stat().st_size)}
    try:
        import av  # type: ignore

        container = av.open(str(path))
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is not None:
            result.update(
                {
                    "frames": int(stream.frames) if stream.frames else None,
                    "width": int(stream.width),
                    "height": int(stream.height),
                    "average_rate": str(stream.average_rate) if stream.average_rate else None,
                }
            )
        container.close()
        return result
    except Exception:
        pass

    # ffprobe is optional.  Do not fail an audit when it is unavailable.
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        command = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames,width,height,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ]
        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            streams = json.loads(completed.stdout).get("streams", [])
            if streams:
                stream = streams[0]
                result.update(
                    {
                        "frames": int(stream["nb_frames"]) if stream.get("nb_frames") not in (None, "N/A") else None,
                        "width": int(stream["width"]) if stream.get("width") else None,
                        "height": int(stream["height"]) if stream.get("height") else None,
                        "average_rate": stream.get("avg_frame_rate"),
                    }
                )
        except Exception:
            pass
    return result


def _sampling_summary(frame_count: int, *, downsample: int, video_ratio: int, video_frames: int) -> Dict[str, Any]:
    action_chunk = int(video_ratio) * int(video_frames)
    physical_horizon = action_chunk * int(downsample)
    valid_conditions = max(0, int(frame_count) - physical_horizon)
    action_offsets = [int(downsample) * (index + 1) for index in range(action_chunk)]
    video_offsets = [action_offsets[(index + 1) * int(video_ratio) - 1] for index in range(video_frames)]
    return {
        "global_downsample_rate": int(downsample),
        "video_action_freq_ratio": int(video_ratio),
        "num_video_frames": int(video_frames),
        "action_chunk_size": int(action_chunk),
        "physical_horizon_frames": int(physical_horizon),
        "valid_condition_frames": int(valid_conditions),
        "action_offsets": action_offsets,
        "video_offsets": video_offsets,
        "first_condition_frame": 0 if valid_conditions else None,
        "last_condition_frame": valid_conditions - 1 if valid_conditions else None,
    }


def _read_parquet_episodes(path: Path, *, expected_episode: Optional[int]) -> List[Dict[str, Any]]:
    table = pq.read_table(path)
    state = _column_to_array(table, "observation.state", dtype=np.float32)
    action = _column_to_array(table, "action", dtype=np.float32)
    frame_index = _column_to_array(table, "frame_index", dtype=np.int64).reshape(-1)
    timestamps = _column_to_array(table, "timestamp", dtype=np.float64).reshape(-1)
    if state.ndim != 2 or action.ndim != 2:
        raise ValueError(f"Expected 2D state/action arrays in {path}; got {state.shape}/{action.shape}")
    if state.shape != action.shape:
        raise ValueError(f"State/action shapes differ in {path}: {state.shape} vs {action.shape}")
    if state.shape[1] != EXPECTED_DIM:
        raise ValueError(f"Expected 16D state/action, got {state.shape[1]}D in {path}")
    episode_values = _column_to_array(table, "episode_index", dtype=np.int64).reshape(-1)
    unique_episode_ids = sorted(set(int(value) for value in episode_values.tolist()))
    if expected_episode is not None:
        if expected_episode not in unique_episode_ids:
            raise ValueError(
                f"Path episode {expected_episode} is not present in {path}; found {unique_episode_ids}"
            )
        unique_episode_ids = [expected_episode]

    episodes: List[Dict[str, Any]] = []
    for episode_id in unique_episode_ids:
        mask = episode_values == int(episode_id)
        episode_state = state[mask]
        episode_action = action[mask]
        episode_frames = frame_index[mask]
        episode_timestamps = timestamps[mask]
        if len(episode_state) == 0:
            continue
        episodes.append(
            {
                "episode_id": int(episode_id),
                "parquet": str(path),
                "rows": int(episode_state.shape[0]),
                "state_dim": int(episode_state.shape[1]),
                "action_dim": int(episode_action.shape[1]),
                "state_stats": _numeric_stats(episode_state),
                "action_stats": _numeric_stats(episode_action),
                "frame_index": _frame_summary(episode_frames, episode_timestamps),
                "frame_index_values": episode_frames,
                "timestamp_values": episode_timestamps,
                "state_values": episode_state,
                "action_values": episode_action,
            }
        )
    return episodes


def audit_dataset(
    dataset_root: Path,
    *,
    video_root: Optional[Path],
    probe_videos: bool,
    downsample: int,
    video_ratio: int,
    video_frames: int,
) -> Dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve()
    video_root = (video_root or dataset_root).expanduser().resolve()
    info_path = _find_info(dataset_root)
    info = _read_json(info_path) if info_path else {}
    camera_names = _camera_features(info)
    if not camera_names:
        # Keep the audit useful for datasets with no info.json.
        camera_names = ["camera_3", "left_wrist", "right_wrist"]

    episodes: List[Dict[str, Any]] = []
    for parquet_path in _find_parquet_files(dataset_root):
        expected_episode = _episode_id_from_path(parquet_path)
        for episode in _read_parquet_episodes(parquet_path, expected_episode=expected_episode):
            episode_id = int(episode["episode_id"])
            cameras: Dict[str, Any] = {}
            for source_camera in camera_names:
                matches = _video_candidates(video_root, source_camera, episode_id)
                selected = matches[0] if matches else None
                camera_entry: Dict[str, Any] = {
                    "source_name": source_camera,
                    "efficient_wam_name": _camera_name_from_feature(f"{CAMERA_PREFIX}{source_camera}"),
                    "path": str(selected) if selected else None,
                    "exists": bool(selected),
                }
                if selected and probe_videos:
                    camera_entry["probe"] = _probe_video(selected)
                cameras[source_camera] = camera_entry
            episodes.append(
                {
                    "episode_id": episode_id,
                    "parquet": episode["parquet"],
                    "rows": episode["rows"],
                    "state_dim": episode["state_dim"],
                    "action_dim": episode["action_dim"],
                    "state_stats": episode["state_stats"],
                    "action_stats": episode["action_stats"],
                    "frame_index": episode["frame_index"],
                    "sampling": _sampling_summary(
                        episode["rows"],
                        downsample=downsample,
                        video_ratio=video_ratio,
                        video_frames=video_frames,
                    ),
                    "cameras": cameras,
                }
            )

    episodes.sort(key=lambda item: int(item["episode_id"]))
    all_camera_names = sorted({name for episode in episodes for name in episode["cameras"]})
    camera_mapping = {
        source: {
            "efficient_wam_name": _camera_name_from_feature(f"{CAMERA_PREFIX}{source}"),
            "role": "head" if _camera_name_from_feature(f"{CAMERA_PREFIX}{source}") == "head" else "auxiliary",
            "used_by_native_single_video_loader": source in {"camera_3", "camera3", "third", "head"},
        }
        for source in all_camera_names
    }
    total_rows = sum(int(episode["rows"]) for episode in episodes)
    dims = sorted({(int(episode["state_dim"]), int(episode["action_dim"])) for episode in episodes})
    return {
        "format": "lerobot_efficient_wam_intermediate",
        "version": 1,
        "dataset_root": str(dataset_root),
        "video_root": str(video_root),
        "info_path": str(info_path) if info_path else None,
        "source_codebase_version": info.get("codebase_version"),
        "fps": info.get("fps"),
        "episode_count": int(len(episodes)),
        "total_rows": int(total_rows),
        "state_action_dimensions": [list(pair) for pair in dims],
        "camera_mapping": camera_mapping,
        "episodes": episodes,
        "sampling_defaults": {
            "global_downsample_rate": int(downsample),
            "video_action_freq_ratio": int(video_ratio),
            "num_video_frames": int(video_frames),
        },
        "model_compatibility": {
            "state_dim": EXPECTED_DIM,
            "action_dim": EXPECTED_DIM,
            "efficient_wam_default_state_dim": 14,
            "efficient_wam_default_action_dim": 14,
            "requires_custom_16d_action_expert": True,
            "do_not_crop_dimensions": True,
        },
        "official_packed_safetensors": {
            "status": "not_built_by_this_script",
            "required_assets": [
                "Wan2.2_VAE.pth (WAN-2.2 TI2V-5B VAE) to encode video latents",
                "UMT5-XXL tokenizer and encoder to create per-instruction language embeddings",
                "Efficient-WAM packer and safetensors to write dataset.json/shards/*.safetensors",
            ],
            "reason": "This audit/stage utility preserves raw arrays and videos; it intentionally does not fabricate latents or language embeddings.",
        },
    }


def _strip_runtime_arrays(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_runtime_arrays(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strip_runtime_arrays(item) for item in value]
    return value


def _write_audit_outputs(manifest: Dict[str, Any], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "audit.json", _strip_runtime_arrays(manifest))
    with (output_root / "episodes.jsonl").open("w", encoding="utf-8", newline="") as handle:
        for episode in manifest["episodes"]:
            handle.write(json.dumps(_strip_runtime_arrays(episode), sort_keys=True, default=_json_default) + "\n")
    with (output_root / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["episode_id", "rows", "state_dim", "action_dim", "first_frame", "last_frame", "estimated_fps", "head_video", "left_wrist_video", "right_wrist_video"],
        )
        writer.writeheader()
        for episode in manifest["episodes"]:
            cameras = episode.get("cameras", {})
            row = {
                "episode_id": episode["episode_id"],
                "rows": episode["rows"],
                "state_dim": episode["state_dim"],
                "action_dim": episode["action_dim"],
                "first_frame": episode["frame_index"]["first"],
                "last_frame": episode["frame_index"]["last"],
                "estimated_fps": episode["frame_index"]["estimated_fps"],
                "head_video": cameras.get("camera_3", {}).get("path") or cameras.get("head", {}).get("path"),
                "left_wrist_video": cameras.get("left_wrist", {}).get("path"),
                "right_wrist_video": cameras.get("right_wrist", {}).get("path"),
            }
            writer.writerow(row)


def _copy_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            # Cross-device links are common on cluster scratch volumes.
            mode = "copy"
    if mode == "symlink":
        try:
            destination.symlink_to(source)
            return
        except OSError:
            mode = "copy"
    shutil.copy2(source, destination)


def stage_dataset(
    dataset_root: Path,
    output_root: Path,
    *,
    video_root: Optional[Path],
    copy_mode: str,
    instruction: Optional[str],
    write_torch: bool,
    probe_videos: bool,
    downsample: int,
    video_ratio: int,
    video_frames: int,
) -> Dict[str, Any]:
    manifest = audit_dataset(
        dataset_root,
        video_root=video_root,
        probe_videos=probe_videos,
        downsample=downsample,
        video_ratio=video_ratio,
        video_frames=video_frames,
    )
    output_root = output_root.expanduser().resolve()
    _write_audit_outputs(manifest, output_root)

    if instruction is None:
        instruction_candidates = [
            Path(str(dataset_root)).expanduser() / "task_instruction.txt",
            Path(str(dataset_root)).expanduser() / "meta" / "task_instruction.txt",
        ]
        instruction = next(
            (candidate.read_text(encoding="utf-8").strip() for candidate in instruction_candidates if candidate.exists()),
            "Build a tower with the building blocks.",
        )

    try:
        torch = __import__("torch") if write_torch else None
    except ImportError as exc:
        raise SystemExit("--write-torch requires PyTorch; omit it to stage portable .npy arrays") from exc

    for episode in manifest["episodes"]:
        episode_id = int(episode["episode_id"])
        source_parquet = Path(str(episode["parquet"]))
        # Re-read the row stream so stage mode remains independent of any
        # in-memory audit implementation details.
        table = pq.read_table(source_parquet)
        all_state = _column_to_array(table, "observation.state", dtype=np.float32)
        all_action = _column_to_array(table, "action", dtype=np.float32)
        all_frame_index = _column_to_array(table, "frame_index", dtype=np.int64).reshape(-1)
        all_timestamps = _column_to_array(table, "timestamp", dtype=np.float64).reshape(-1)
        all_episode_values = _column_to_array(table, "episode_index", dtype=np.int64).reshape(-1)
        row_mask = all_episode_values == episode_id
        state = all_state[row_mask]
        action = all_action[row_mask]
        frame_index = all_frame_index[row_mask]
        timestamps = all_timestamps[row_mask]
        episode_dir = output_root / "episodes" / f"episode_{episode_id:06d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        np.save(episode_dir / "state.npy", state)
        np.save(episode_dir / "action.npy", action)
        np.save(episode_dir / "frame_index.npy", frame_index)
        np.save(episode_dir / "timestamp.npy", timestamps)
        if torch is not None:
            torch.save(torch.from_numpy(state), episode_dir / "state.pt")
            torch.save(torch.from_numpy(action), episode_dir / "action.pt")

        (episode_dir / "instruction.txt").write_text(instruction.strip() + "\n", encoding="utf-8")
        staged_cameras: Dict[str, str] = {}
        for source_camera, camera_entry in episode["cameras"].items():
            source = camera_entry.get("path")
            if not source:
                continue
            source_path = Path(source)
            destination_name = f"{_camera_name_from_feature(f'{CAMERA_PREFIX}{source_camera}')}.mp4"
            destination = episode_dir / "videos" / destination_name
            _copy_file(source_path, destination, copy_mode)
            staged_cameras[source_camera] = str(destination)
        episode["staged"] = {
            "root": str(episode_dir),
            "state_npy": str(episode_dir / "state.npy"),
            "action_npy": str(episode_dir / "action.npy"),
            "state_pt": str(episode_dir / "state.pt") if torch is not None else None,
            "action_pt": str(episode_dir / "action.pt") if torch is not None else None,
            "videos": staged_cameras,
            "language_embedding": None,
        }

    manifest["staging"] = {
        "output_root": str(output_root),
        "copy_mode": copy_mode,
        "state_action_are_separate": True,
        "language_embedding_status": "missing; generate with WAN UMT5-XXL",
        "official_packed_safetensors_status": "blocked_until_vae_and_t5_are_available",
    }
    _write_audit_outputs(manifest, output_root)
    _write_json(output_root / "conversion_status.json", {
        "status": "intermediate_only",
        "state_dim": EXPECTED_DIM,
        "action_dim": EXPECTED_DIM,
        "official_efficient_wam_packer_ready": False,
        "blockers": manifest["official_packed_safetensors"]["required_assets"],
        "next_step": "Generate one UMT5 embedding list per episode/instruction, then implement a LeRobot-aware packer that encodes clips with Wan2.2_VAE.pth and writes Efficient-WAM dataset.json plus safetensors shards.",
    })
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "stage"):
        sub = subparsers.add_parser(command, help=f"{command} the LeRobot dataset")
        sub.add_argument("--dataset-root", required=True, type=Path, help="LeRobot v3 root (with data/, videos/, meta/)")
        sub.add_argument("--video-root", type=Path, default=None, help="Optional root containing per-episode videos (defaults to --dataset-root)")
        sub.add_argument("--output", required=True, type=Path, help="Directory for audit/staging outputs")
        sub.add_argument("--probe-videos", action="store_true", help="Probe dimensions/frame counts when PyAV or ffprobe is available")
        sub.add_argument("--downsample", type=int, default=3, help="Recommended 30Hz->10Hz frame stride")
        sub.add_argument("--video-ratio", type=int, default=2, help="Action steps per predicted video frame")
        sub.add_argument("--video-frames", type=int, default=8, help="Predicted future video frames")
        if command == "stage":
            sub.add_argument("--copy-mode", choices=("copy", "hardlink", "symlink"), default="hardlink")
            sub.add_argument("--instruction", default=None, help="Instruction text; defaults to task_instruction.txt when present")
            sub.add_argument("--write-torch", action="store_true", help="Also write state.pt/action.pt (requires PyTorch)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "audit":
        manifest = audit_dataset(
            args.dataset_root,
            video_root=args.video_root,
            probe_videos=args.probe_videos,
            downsample=args.downsample,
            video_ratio=args.video_ratio,
            video_frames=args.video_frames,
        )
        _write_audit_outputs(manifest, args.output.expanduser().resolve())
    else:
        manifest = stage_dataset(
            args.dataset_root,
            args.output,
            video_root=args.video_root,
            copy_mode=args.copy_mode,
            instruction=args.instruction,
            write_torch=args.write_torch,
            probe_videos=args.probe_videos,
            downsample=args.downsample,
            video_ratio=args.video_ratio,
            video_frames=args.video_frames,
        )
    print(json.dumps({
        "command": args.command,
        "episodes": manifest["episode_count"],
        "total_rows": manifest["total_rows"],
        "dimensions": manifest["state_action_dimensions"],
        "output": str(args.output.expanduser().resolve()),
        "official_packed_safetensors": manifest["official_packed_safetensors"],
    }, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
