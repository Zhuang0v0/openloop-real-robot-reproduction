#!/usr/bin/env python3
"""Audit the Classify Objects Excel manifest against a real dataset export.

The audit is independent of either policy implementation. It matches episode
IDs first, then checks parquet schema, state/action dimensions, finite values,
timestamps and the three camera streams. Both flattened LeRobot exports and
per-episode exports are supported. The source dataset is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pq = None  # type: ignore[assignment]

try:
    from tools.classify_common import (
        DEFAULT_EPISODE_COUNT,
        DEFAULT_START_EPISODE,
        FIXED_PROMPT,
        MANIFEST_FIELDS,
        SHEET_NAME,
        UNMATCHED_FIELDS,
        build_manifest,
        discover_episode_paths,
        load_classify_rows,
        read_manifest,
        write_audit_markdown,
        write_csv,
    )
except ModuleNotFoundError:  # pragma: no cover
    from classify_common import (  # type: ignore
        DEFAULT_EPISODE_COUNT,
        DEFAULT_START_EPISODE,
        FIXED_PROMPT,
        MANIFEST_FIELDS,
        SHEET_NAME,
        UNMATCHED_FIELDS,
        build_manifest,
        discover_episode_paths,
        load_classify_rows,
        read_manifest,
        write_audit_markdown,
        write_csv,
    )


CAMERAS = (
    "observation.images.camera_3",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)
_EPISODE_RE = re.compile(r"episode[_-]?(\d+)", re.IGNORECASE)


def _as_int(value: Any) -> int | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        number = float(text)
        return int(number) if number.is_integer() else None
    except (TypeError, ValueError):
        return None


def _path_episode_id(path: Path) -> int | None:
    for part in reversed(path.parts):
        match = _EPISODE_RE.search(part)
        if match:
            return int(match.group(1))
    return None


def _episode_parquets(path: Path, episode_id: int) -> list[Path]:
    """Find frame parquet files without trusting file-name order."""

    if path.is_file():
        return [path] if path.suffix.lower() == ".parquet" else []
    roots = [path / "data", path / "per_episode" / "data", path]
    found: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for candidate in sorted(root.rglob("*.parquet")):
            if candidate.name.lower() in {"tasks.parquet", "episodes.parquet"}:
                continue
            if _path_episode_id(candidate) == episode_id:
                found.append(candidate)
                continue
            if pq is not None:
                try:
                    schema = pq.read_schema(candidate)
                    if "episode_index" in schema.names:
                        values = pq.read_table(candidate, columns=["episode_index"])["episode_index"].to_pylist()
                        if any(_as_int(value) == episode_id for value in values):
                            found.append(candidate)
                except Exception:
                    continue
        if found:
            break
    return sorted(set(found))


def _video_candidates(path: Path, camera: str, episode_id: int) -> list[Path]:
    names = [camera, camera.replace("observation.images.", "")]
    result: list[Path] = []
    for root in (path, path / "per_episode"):
        for name in names:
            camera_root = root / "videos" / name
            if not camera_root.is_dir():
                continue
            result.extend(sorted(camera_root.rglob(f"episode_{episode_id:06d}.mp4")))
            result.extend(sorted(camera_root.rglob(f"file-{episode_id:03d}.mp4")))
    return sorted(set(result))


def _merge_tables(paths: Sequence[Path], episode_id: int) -> tuple[dict[str, list[Any]], list[str]]:
    if pq is None:
        return {}, ["pyarrow_missing"]
    errors: list[str] = []
    columns: dict[str, list[Any]] = {}
    for path in paths:
        try:
            table = pq.read_table(path)
        except Exception as exc:
            errors.append(f"parquet_read:{path.name}:{type(exc).__name__}")
            continue
        episode_values = table["episode_index"].to_pylist() if "episode_index" in table.column_names else None
        selected_indices = (
            [index for index, value in enumerate(episode_values) if _as_int(value) == episode_id]
            if episode_values is not None
            else None
        )
        for name in table.column_names:
            values = table[name].to_pylist()
            if selected_indices is not None and len(values) == len(episode_values):
                values = [values[index] for index in selected_indices]
            columns.setdefault(name, []).extend(values)
    return columns, errors


def _finite_matrix(values: Any, dimension: int) -> tuple[bool, list[int], str]:
    if values is None:
        return False, [], "missing"
    if np is None:
        return False, [], "numpy_missing"
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False, [], "not_numeric"
    shape = list(array.shape)
    if array.ndim != 2 or array.shape[1] != dimension:
        return False, shape, f"expected_[T,{dimension}]"
    if not np.isfinite(array).all():
        return False, shape, "non_finite"
    return True, shape, "ok"


def _video_probe(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "bytes": path.stat().st_size}
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        result.update({"fps": None, "frames": None, "probe": "ffprobe_missing"})
        return result
    command = [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate,nb_frames", "-of", "json", str(path)]
    try:
        raw = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=30)
        stream = json.loads(raw).get("streams", [{}])[0]
        rate = str(stream.get("r_frame_rate", ""))
        if "/" in rate:
            numerator, denominator = rate.split("/", 1)
            fps = float(numerator) / float(denominator) if float(denominator) else math.nan
        else:
            fps = float(rate) if rate else None
        result.update({"fps": fps, "frames": _as_int(stream.get("nb_frames")), "probe": "ok"})
    except Exception as exc:
        result.update({"fps": None, "frames": None, "probe": type(exc).__name__})
    return result


def inspect_episode(row: Mapping[str, Any], dataset_root: str | Path | None, *, probe_videos: bool = False) -> dict[str, Any]:
    episode_id = _as_int(row.get("episode_id"))
    path_value = str(row.get("episode_path", "") or "").strip()
    root = Path(dataset_root).expanduser().resolve() if dataset_root else None
    path = Path(path_value).expanduser() if path_value else None
    if path is not None and root is not None and not path.is_absolute():
        path = root / path
    if path is None or not path.exists():
        if root is not None:
            try:
                selected, _ = discover_episode_paths(root)
                path = selected.get(episode_id) if episode_id is not None else None
            except Exception:
                path = None
    result: dict[str, Any] = {"episode_id": episode_id, "path": str(path) if path else "", "matched": bool(path and path.exists()), "parquet": {}, "videos": {}, "issues": []}
    if episode_id is None:
        result["issues"].append("invalid_episode_id")
        return result
    if path is None or not path.exists():
        result["issues"].append("episode_path_missing")
        return result
    parquet_paths = _episode_parquets(path, episode_id)
    if not parquet_paths:
        result["issues"].append("frame_parquet_missing")
    else:
        columns, read_errors = _merge_tables(parquet_paths, episode_id)
        result["parquet"].update({"files": [str(item) for item in parquet_paths], "rows": len(columns.get("action", [])), "columns": sorted(columns)})
        result["issues"].extend(read_errors)
        action_ok, action_shape, action_reason = _finite_matrix(columns.get("action"), 16)
        state_ok, state_shape, state_reason = _finite_matrix(columns.get("observation.state"), 16)
        result["parquet"].update({"action_shape": action_shape, "state_shape": state_shape, "action_ok": action_ok, "state_ok": state_ok, "action_reason": action_reason, "state_reason": state_reason})
        if not action_ok:
            result["issues"].append(f"action:{action_reason}")
        if not state_ok:
            result["issues"].append(f"state:{state_reason}")
        if action_ok and state_ok and action_shape[0] != state_shape[0]:
            result["issues"].append("state_action_length_mismatch")
        timestamps = columns.get("timestamp")
        if timestamps and np is not None:
            try:
                values = np.asarray(timestamps, dtype=np.float64)
                timestamp_info = {"count": int(values.size), "monotonic": bool(np.all(np.diff(values) >= 0)), "duration": float(values[-1] - values[0]) if values.size > 1 else 0.0}
                result["parquet"]["timestamp"] = timestamp_info
                if not timestamp_info["monotonic"]:
                    result["issues"].append("timestamp_not_monotonic")
            except (TypeError, ValueError):
                result["issues"].append("timestamp_not_numeric")
    for camera in CAMERAS:
        candidates = _video_candidates(path, camera, episode_id)
        if not candidates:
            result["videos"][camera] = {"present": False}
            result["issues"].append(f"video_missing:{camera}")
        else:
            result["videos"][camera] = {"present": True, "path": str(candidates[0])}
            if probe_videos:
                result["videos"][camera].update(_video_probe(candidates[0]))
    result["ok"] = not result["issues"]
    return result


def audit_manifest(manifest: Sequence[Mapping[str, Any]], *, dataset_root: str | Path | None = None, start_episode: int = DEFAULT_START_EPISODE, episode_count: int = DEFAULT_EPISODE_COUNT, probe_videos: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    expected = list(range(start_episode, start_episode + episode_count))
    expected_set = set(expected)
    normalized: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    inspections: list[dict[str, Any]] = []
    seen: Counter[int] = Counter()
    for source_row in manifest:
        row = {field: source_row.get(field, "") for field in MANIFEST_FIELDS}
        episode_id = _as_int(source_row.get("episode_id"))
        if episode_id is None:
            issues.append({"episode_id": "", "reason": "invalid_manifest_id", "excel_row": source_row.get("excel_row", ""), "episode_path": source_row.get("episode_path", ""), "details": "episode_id is not an integer"})
            row["match_status"] = "invalid_id"
        else:
            row["episode_id"] = episode_id
            seen[episode_id] += 1
            if episode_id not in expected_set:
                issues.append({"episode_id": episode_id, "reason": "manifest_id_out_of_range", "excel_row": source_row.get("excel_row", ""), "episode_path": source_row.get("episode_path", ""), "details": f"Expected ids {start_episode}..{start_episode + episode_count - 1}"})
            if str(source_row.get("prompt", "")) != FIXED_PROMPT:
                issues.append({"episode_id": episode_id, "reason": "prompt_mismatch", "excel_row": source_row.get("excel_row", ""), "episode_path": source_row.get("episode_path", ""), "details": "Prompt differs from fixed prompt"})
            inspection = inspect_episode(row, dataset_root, probe_videos=probe_videos)
            inspections.append(inspection)
            if inspection.get("path"):
                row["episode_path"] = inspection["path"]
            for reason in inspection.get("issues", []):
                issues.append({"episode_id": episode_id, "reason": reason.split(":", 1)[0], "excel_row": source_row.get("excel_row", ""), "episode_path": inspection.get("path", ""), "details": reason})
        normalized.append(row)
    for episode_id in expected:
        if seen[episode_id] == 0:
            issues.append({"episode_id": episode_id, "reason": "missing_manifest_episode", "excel_row": "", "episode_path": "", "details": "No manifest row for expected episode"})
        elif seen[episode_id] > 1:
            issues.append({"episode_id": episode_id, "reason": "duplicate_manifest_episode", "excel_row": "", "episode_path": "", "details": f"Found {seen[episode_id]} rows"})
    if dataset_root is not None:
        try:
            _, candidates = discover_episode_paths(Path(dataset_root).expanduser().resolve())
            for episode_id, paths in candidates.items():
                if episode_id not in expected_set:
                    issues.append({"episode_id": episode_id, "reason": "dataset_id_out_of_range", "excel_row": "", "episode_path": str(paths[0]) if paths else "", "details": f"Expected ids {start_episode}..{start_episode + episode_count - 1}"})
        except Exception as exc:
            issues.append({"episode_id": "", "reason": "dataset_discovery_error", "excel_row": "", "episode_path": str(dataset_root), "details": f"{type(exc).__name__}: {exc}"})
    status_counts = Counter(str(row.get("match_status", "")) for row in normalized)
    group_counts = Counter(str(row.get("group", "")) for row in normalized if row.get("group"))
    summary = {"expected_episode_count": episode_count, "expected_first_episode": start_episode, "expected_last_episode": start_episode + episode_count - 1, "manifest_rows": len(normalized), "matched_episode_count": sum(bool(item.get("matched")) for item in inspections), "valid_episode_count": sum(bool(item.get("ok")) for item in inspections), "unique_episode_ids": len(seen), "issue_count": len(issues), "match_status": dict(sorted(status_counts.items())), "group_counts": dict(sorted(group_counts.items())), "ok": not issues}
    return normalized, issues, summary, inspections


def _append_inspection_markdown(path: Path, inspections: Sequence[Mapping[str, Any]]) -> None:
    """Add compact schema/video evidence to the human-readable audit."""

    lines = [
        "",
        "## Dataset inspection",
        "",
        "| Episode | Matched | Parquet rows | State shape | Action shape | Cameras | Issues |",
        "|---:|---|---:|---|---|---:|---|",
    ]
    for item in inspections:
        parquet = item.get("parquet", {}) if isinstance(item.get("parquet", {}), Mapping) else {}
        state_shape = parquet.get("state_shape", "")
        action_shape = parquet.get("action_shape", "")
        cameras = item.get("videos", {}) if isinstance(item.get("videos", {}), Mapping) else {}
        camera_count = sum(bool(value.get("present")) for value in cameras.values() if isinstance(value, Mapping))
        issues = ", ".join(str(value) for value in item.get("issues", []))
        escaped_issues = issues.replace("|", "\\|")
        lines.append(f"| {item.get('episode_id', '')} | {bool(item.get('matched'))} | {parquet.get('rows', '')} | {state_shape} | {action_shape} | {camera_count} | {escaped_issues} |")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def audit_dataset(*, manifest_path: str | Path | None = None, excel_path: str | Path | None = None, dataset_root: str | Path | None = None, output: str | Path = "classify_audit", sheet_name: str = SHEET_NAME, start_episode: int = DEFAULT_START_EPISODE, episode_count: int = DEFAULT_EPISODE_COUNT, probe_videos: bool = False) -> dict[str, Any]:
    if manifest_path is None and excel_path is None:
        raise ValueError("Provide --manifest or --excel")
    initial_unmatched: list[dict[str, Any]] = []
    if manifest_path is not None:
        source_rows: Sequence[Mapping[str, Any]] = read_manifest(manifest_path)
    else:
        source_rows = load_classify_rows(excel_path, sheet_name=sheet_name, start_episode=start_episode)  # type: ignore[arg-type]
        mapped, initial_unmatched = build_manifest(source_rows, dataset_root=dataset_root, start_episode=start_episode, episode_count=episode_count, source_excel=excel_path)
        source_rows = mapped
    normalized, issues, validation, inspections = audit_manifest(source_rows, dataset_root=dataset_root, start_episode=start_episode, episode_count=episode_count, probe_videos=probe_videos)
    all_issues = [*initial_unmatched, *issues]
    output_dir = Path(output).expanduser()
    if output_dir.suffix.lower() == ".csv":
        output_dir = output_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_target = output_dir / "classify_episode_manifest.csv"
    unmatched_target = output_dir / "classify_unmatched_episodes.csv"
    audit_target = output_dir / "classify_dataset_audit.md"
    write_csv(manifest_target, normalized, MANIFEST_FIELDS)
    write_csv(unmatched_target, all_issues, UNMATCHED_FIELDS)
    # Short aliases make the tool convenient in ad-hoc audits while the
    # descriptive names above remain the experiment's canonical artifacts.
    short_manifest = output_dir / "manifest.csv"
    short_unmatched = output_dir / "unmatched.csv"
    short_audit = output_dir / "audit.md"
    if short_manifest != manifest_target:
        write_csv(short_manifest, normalized, MANIFEST_FIELDS)
    if short_unmatched != unmatched_target:
        write_csv(short_unmatched, all_issues, UNMATCHED_FIELDS)
    write_audit_markdown(audit_target, manifest=normalized, unmatched=all_issues, start_episode=start_episode, episode_count=episode_count, dataset_root=dataset_root)
    _append_inspection_markdown(audit_target, inspections)
    if short_audit != audit_target:
        write_audit_markdown(short_audit, manifest=normalized, unmatched=all_issues, start_episode=start_episode, episode_count=episode_count, dataset_root=dataset_root)
        _append_inspection_markdown(short_audit, inspections)
    inspection_target = output_dir / "classify_episode_inspection.json"
    inspection_target.write_text(json.dumps(inspections, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**validation, "manifest": str(manifest_target.resolve()), "unmatched": str(unmatched_target.resolve()), "audit": str(audit_target.resolve()), "inspection": str(inspection_target.resolve()), "short_manifest": str(short_manifest.resolve()), "short_unmatched": str(short_unmatched.resolve()), "short_audit": str(short_audit.resolve()), "ok": not all_issues}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--excel", "--xlsx", type=Path)
    parser.add_argument("--dataset-root", "--data-root", type=Path, default=None)
    parser.add_argument("--output", "--output-dir", "-o", type=Path, default=Path("classify_audit"))
    parser.add_argument("--sheet-name", default=SHEET_NAME)
    parser.add_argument("--start-episode", type=int, default=DEFAULT_START_EPISODE)
    parser.add_argument("--episodes", "--episode-count", dest="episode_count", type=int, default=DEFAULT_EPISODE_COUNT)
    parser.add_argument("--probe-videos", action="store_true")
    return parser


audit = audit_dataset


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = audit_dataset(manifest_path=args.manifest, excel_path=args.excel, dataset_root=args.dataset_root, output=args.output, sheet_name=args.sheet_name, start_episode=args.start_episode, episode_count=args.episode_count, probe_videos=args.probe_videos)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
