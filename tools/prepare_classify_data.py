#!/usr/bin/env python3
"""Prepare Classify Objects manifests for Fast-WAM and Efficient-WAM.

The matcher writes an episode-level CSV while ``filter_classify_windows.py``
writes a JSON report containing candidate/selected windows.  This module joins
those two artifacts, validates the common task contract, and creates
episode-disjoint train/validation manifests.  The split is deliberately made
from episode ids *before* windows are expanded, so overlapping windows from one
episode can never end up in different splits.

The implementation is intentionally independent of either model.  The output
CSV contains the source paths and window coordinates understood by downstream
adapters, and ``prepare_metadata.json`` records the same contract for both
Fast-WAM and Efficient-WAM.

Typical use::

    python tools/prepare_classify_data.py \
        --manifest runs/classify/manifest.csv \
        --window-manifest runs/classify/combined_windows.json \
        --dataset-root data/nero_classify_objects \
        --output-dir runs/classify/prepared \
        --strategy combined --seed 20261111 --episodes 106

The input readers also accept in-memory sequences, which keeps the public
functions useful from tests and notebooks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np

try:  # package import
    from tools.analyze_classify_actions import Episode, load_episodes
    from tools.classify_common import FIXED_PROMPT, MANIFEST_FIELDS
except ModuleNotFoundError:  # direct ``python tools/prepare_classify_data.py``
    from analyze_classify_actions import Episode, load_episodes  # type: ignore
    from classify_common import FIXED_PROMPT, MANIFEST_FIELDS  # type: ignore


EXPECTED_DIM = 16
DEFAULT_SEED = 20261111
DEFAULT_VALIDATION_FRACTION = 0.20
DEFAULT_START_EPISODE = 0
STRATEGIES = ("baseline", "quality", "action", "combined")
_STRATEGY_ALIASES = {
    "baseline": "baseline",
    "base": "baseline",
    "all": "baseline",
    "quality": "quality",
    "action": "action",
    "combined": "combined",
}


class PreparationError(ValueError):
    """Raised when preparation inputs violate the Classify Objects contract."""


def normalize_strategy(strategy: str | None) -> str:
    """Return a lower-case strategy name and reject typos early."""

    key = "baseline" if strategy is None else str(strategy).strip().casefold()
    try:
        return _STRATEGY_ALIASES[key]
    except KeyError as exc:
        choices = ", ".join(STRATEGIES)
        raise PreparationError(f"unknown strategy {strategy!r}; choose one of {choices}") from exc


def _as_int(value: Any, *, field: str = "value") -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError) as exc:
        raise PreparationError(f"{field} must be an integer, got {value!r}") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise PreparationError(f"{field} must be an integer, got {value!r}")
    return int(number)


def _as_bool(value: Any, *, default: bool = True) -> bool:
    if value is None or str(value).strip() == "":
        return default
    if isinstance(value, bool):
        return value
    token = str(value).strip().casefold()
    if token in {"1", "true", "yes", "y", "on", "selected", "keep", "kept"}:
        return True
    if token in {"0", "false", "no", "n", "off", "rejected", "drop", "dropped"}:
        return False
    return default


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"cannot encode {type(value).__name__}")


def _read_json_or_csv(source: str | Path | Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read a CSV/JSON manifest and retain top-level JSON metadata."""

    if isinstance(source, Mapping):
        for key in ("windows", "rows", "manifest", "episodes", "samples"):
            candidate = source.get(key)
            if isinstance(candidate, list):
                return [dict(row) for row in candidate], {str(k): v for k, v in source.items() if k != key}
        if "episode_id" in source or "episode_index" in source:
            return [dict(source)], {}
        raise PreparationError("manifest mapping has no rows/windows list")
    if not isinstance(source, (str, Path)):
        return [dict(row) for row in source], {}
    path = Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.casefold()
    if suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            return [dict(row) for row in rows], {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PreparationError(f"invalid JSON manifest {path}: {exc}") from exc
        if isinstance(value, list):
            return [dict(row) for row in value], {}
        if not isinstance(value, Mapping):
            raise PreparationError(f"JSON manifest must contain an object or list: {path}")
        for key in ("windows", "rows", "manifest", "episodes", "samples"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                metadata = {str(k): v for k, v in value.items() if k != key}
                return [dict(row) for row in candidate], metadata
        # A single record is useful in small hand-written manifests.
        if "episode_id" in value or "episode_index" in value:
            return [dict(value)], {}
        raise PreparationError(f"JSON manifest has no rows/windows list: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)], {}
    except OSError:
        raise


def read_matched_manifest(source: str | Path | Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> list[dict[str, Any]]:
    """Read and normalize the matcher CSV.

    Rows are not silently relabeled: ``episode_id`` is parsed explicitly and
    duplicate ids are rejected.  Filtering of non-``matched`` rows happens in
    :func:`prepare_classify_data`, where the excluded count is retained in the
    metadata.
    """

    rows, _ = _read_json_or_csv(source)
    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, original in enumerate(rows, 1):
        row = dict(original)
        episode_id = _as_int(row.get("episode_id", row.get("episode_index", row.get("id"))), field=f"episode_id (row {index})")
        if episode_id is None:
            raise PreparationError(f"matched manifest row {index} has no episode_id")
        if episode_id in seen:
            raise PreparationError(f"matched manifest contains duplicate episode_id {episode_id}")
        seen.add(episode_id)
        row["episode_id"] = episode_id
        normalized.append(row)
    return normalized


def read_window_manifest(source: str | Path | Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read a filtered-window report or CSV and return ``(rows, metadata)``."""

    rows, metadata = _read_json_or_csv(source)
    normalized: list[dict[str, Any]] = []
    for index, original in enumerate(rows, 1):
        row = dict(original)
        episode_id = _as_int(row.get("episode_id", row.get("episode_index")), field=f"window episode_id (row {index})")
        if episode_id is None:
            raise PreparationError(f"window manifest row {index} has no episode_id")
        row["episode_id"] = episode_id
        row["selected"] = _as_bool(row.get("selected"), default=True)
        if row.get("window_id", "") in (None, ""):
            ordinal = len([item for item in normalized if item["episode_id"] == episode_id])
            row["window_id"] = f"episode_{episode_id:06d}_window_{ordinal:06d}"
        for key in ("start", "end", "length"):
            if key in row and str(row[key]).strip() != "":
                row[key] = _as_int(row[key], field=f"{key} (window row {index})")
        source_rows = row.get("source_rows")
        if isinstance(source_rows, str) and source_rows.strip():
            try:
                parsed = json.loads(source_rows)
                if isinstance(parsed, list):
                    source_rows = parsed
            except json.JSONDecodeError:
                # CSV hand-authored manifests commonly use ``0,1,2``.
                source_rows = [int(part.strip()) for part in source_rows.split(",") if part.strip()]
        if source_rows is not None and not isinstance(source_rows, list):
            source_rows = list(source_rows)
        if source_rows is not None:
            row["source_rows"] = [int(value) for value in source_rows]
        if row.get("length") in (None, "") and row.get("start") is not None and row.get("end") is not None:
            row["length"] = int(row["end"] - row["start"])
        normalized.append(row)
    return normalized, metadata


def _manifest_status(row: Mapping[str, Any]) -> str:
    return str(row.get("match_status", "matched") or "matched").strip().casefold()


def _is_matched(row: Mapping[str, Any]) -> bool:
    return _manifest_status(row) in {"matched", "match", "ok", "valid", "true", "1"}


def _parse_array_field(value: Any) -> np.ndarray | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, np.ndarray):
        return np.asarray(value)
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    if isinstance(value, str):
        try:
            return np.asarray(json.loads(value))
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
    return None


def _row_dimension(row: Mapping[str, Any], names: Sequence[str]) -> int | None:
    for name in names:
        if name not in row or str(row.get(name, "")).strip() == "":
            continue
        value = row[name]
        if isinstance(value, str) and value.strip().startswith(("[", "{")):
            continue
        parsed = _as_int(value, field=name)
        if parsed is not None:
            return parsed
    return None


def _row_array(row: Mapping[str, Any], names: Sequence[str]) -> np.ndarray | None:
    for name in names:
        if name in row:
            value = _parse_array_field(row[name])
            if value is not None:
                return value
    return None


def _resolve_path(value: Any, *, manifest_path: Path | None, dataset_root: Path | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    candidate = Path(str(value)).expanduser()
    if candidate.exists():
        return candidate.resolve()
    if manifest_path is not None:
        relative = (manifest_path.parent / candidate)
        if relative.exists():
            return relative.resolve()
    if dataset_root is not None:
        relative = dataset_root / candidate
        if relative.exists():
            return relative.resolve()
        # A matcher may only retain a basename from a different checkout.
        episode_name = candidate.name
        matches = sorted(dataset_root.rglob(episode_name)) if dataset_root.exists() else []
        if matches:
            return matches[0].resolve()
    return None


def _episode_from_mapping(row: Mapping[str, Any]) -> Episode | None:
    state = _row_array(row, ("state", "observation.state", "state_values"))
    action = _row_array(row, ("action", "actions", "action_values"))
    if state is None or action is None:
        return None
    return Episode(
        episode_id=int(row["episode_id"]),
        state=state,
        action=action,
        timestamps=_row_array(row, ("timestamps", "timestamp")),
        frame_index=_row_array(row, ("frame_index", "frame_indices")),
        source=str(row.get("episode_path", "")) or None,
    )


def _load_episode_map(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset_root: str | Path | None,
    manifest_source: str | Path | Sequence[Mapping[str, Any]],
) -> tuple[dict[int, Episode], dict[int, str], bool]:
    """Load arrays from parquet/inline fields without assuming filename order."""

    root = Path(dataset_root).expanduser().resolve() if dataset_root is not None else None
    manifest_path = Path(manifest_source).expanduser().resolve() if isinstance(manifest_source, (str, Path)) else None
    episodes: dict[int, Episode] = {}
    source_paths: dict[int, str] = {}
    attempted = False

    # A dataset root is loaded once.  This handles both sharded LeRobot files
    # and one parquet file per episode.
    if root is not None and root.exists():
        attempted = True
        try:
            loaded = load_episodes(root)
        except (FileNotFoundError, KeyError, RuntimeError, ImportError, OSError, ValueError):
            loaded = []
        for episode in loaded:
            if episode.episode_id in episodes:
                # Duplicate ids can arise when a manifest accidentally points
                # to two shards.  Keep the first only if it is byte-for-byte
                # equivalent in shape; otherwise fail rather than mixing data.
                previous = episodes[episode.episode_id]
                if previous.state.shape != episode.state.shape or previous.action.shape != episode.action.shape:
                    raise PreparationError(f"dataset contains duplicate episode_id {episode.episode_id} with different shapes")
                continue
            episodes[int(episode.episode_id)] = episode
            source_paths[int(episode.episode_id)] = str(episode.source or "")

    for source_row in rows:
        episode_id = int(source_row["episode_id"])
        inline = _episode_from_mapping(source_row)
        if inline is not None:
            episodes[episode_id] = inline
            source_paths[episode_id] = str(source_row.get("episode_path", "") or "inline")
            attempted = True
            continue
        value = source_row.get("episode_path")
        path = _resolve_path(value, manifest_path=manifest_path, dataset_root=root)
        if path is None:
            # If a root was supplied, the bulk loader already tried all files.
            continue
        attempted = True
        try:
            loaded = load_episodes(path)
        except (FileNotFoundError, KeyError, RuntimeError, ImportError, OSError, ValueError) as exc:
            raise PreparationError(f"could not load episode {episode_id} from {path}: {exc}") from exc
        candidates = [item for item in loaded if item.episode_id == episode_id]
        if not candidates and len(loaded) == 1:
            candidates = loaded
        if candidates:
            episodes[episode_id] = candidates[0]
            source_paths[episode_id] = str(candidates[0].source or path)
    return episodes, source_paths, attempted


def _validate_episode(episode: Episode, *, expected_dim: int = EXPECTED_DIM) -> dict[str, Any]:
    state = np.asarray(episode.state)
    action = np.asarray(episode.action)
    if state.ndim != 2 or action.ndim != 2:
        raise PreparationError(f"episode {episode.episode_id} state/action must be 2-D; got {state.shape}/{action.shape}")
    if state.shape[1] != expected_dim or action.shape[1] != expected_dim:
        raise PreparationError(
            f"episode {episode.episode_id} expected 16D state/action, got {state.shape[1]}D/{action.shape[1]}D"
        )
    if state.shape[0] != action.shape[0]:
        raise PreparationError(f"episode {episode.episode_id} state/action row mismatch: {state.shape[0]} vs {action.shape[0]}")
    if state.shape[0] == 0:
        raise PreparationError(f"episode {episode.episode_id} has no state/action rows")
    if not np.issubdtype(state.dtype, np.number) or not np.issubdtype(action.dtype, np.number):
        raise PreparationError(f"episode {episode.episode_id} state/action must be numeric")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise PreparationError(f"episode {episode.episode_id} contains non-finite state/action values")
    return {
        "episode_id": int(episode.episode_id),
        "frames": int(state.shape[0]),
        "state_dim": int(state.shape[1]),
        "action_dim": int(action.shape[1]),
        "state_dtype": str(state.dtype),
        "action_dtype": str(action.dtype),
        "finite": True,
    }


def _empty_stats(dim: int = EXPECTED_DIM) -> dict[str, Any]:
    return {
        "rows": 0,
        "dim": int(dim),
        "finite": True,
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "per_channel_min": [],
        "per_channel_max": [],
        "per_channel_mean": [],
        "per_channel_std": [],
    }


def _matrix_stats(matrices: Iterable[np.ndarray], *, dim: int = EXPECTED_DIM) -> dict[str, Any]:
    count = 0
    total = np.zeros(dim, dtype=np.float64)
    total_sq = np.zeros(dim, dtype=np.float64)
    minimum = np.full(dim, np.inf, dtype=np.float64)
    maximum = np.full(dim, -np.inf, dtype=np.float64)
    for matrix in matrices:
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != dim:
            continue
        count += int(values.shape[0])
        total += values.sum(axis=0)
        total_sq += np.square(values).sum(axis=0)
        minimum = np.minimum(minimum, values.min(axis=0))
        maximum = np.maximum(maximum, values.max(axis=0))
    if count == 0:
        return _empty_stats(dim)
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    return {
        "rows": count,
        "dim": int(dim),
        "finite": True,
        "min": float(minimum.min()),
        "max": float(maximum.max()),
        "mean": float(mean.mean()),
        "std": float(np.sqrt(np.mean(variance))),
        "per_channel_min": minimum.tolist(),
        "per_channel_max": maximum.tolist(),
        "per_channel_mean": mean.tolist(),
        "per_channel_std": std.tolist(),
    }


def _stable_order(ids: Sequence[int], *, seed: int, mode: str) -> list[int]:
    unique = sorted({int(value) for value in ids})
    if mode == "sorted":
        return unique
    if mode == "hash":
        return sorted(unique, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest())
    if mode != "random":
        raise PreparationError("split_mode must be random, sorted, or hash")
    rng = np.random.default_rng(int(seed))
    return [int(value) for value in np.asarray(unique)[rng.permutation(len(unique))]] if unique else []


def split_episode_ids(
    episode_ids: Iterable[int],
    *,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: int = DEFAULT_SEED,
    mode: str = "random",
    group_by: Mapping[int, Any] | None = None,
    group_stratified: bool = False,
) -> dict[int, str]:
    """Assign each episode to train or validation deterministically."""

    fraction = float(validation_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise PreparationError("validation_fraction must be between 0 and 1")
    ids = sorted({int(value) for value in episode_ids})
    if not ids:
        return {}
    if fraction == 0.0:
        return {episode_id: "train" for episode_id in ids}
    if fraction == 1.0:
        return {episode_id: "validation" for episode_id in ids}
    validation_count = int(round(len(ids) * fraction))
    # A non-zero requested fraction should produce a validation episode when
    # possible, while preserving at least one training episode.
    validation_count = max(1, validation_count)
    validation_count = min(len(ids) - 1, validation_count) if len(ids) > 1 else len(ids)
    if group_stratified and group_by:
        # Stratify within each scene group.  The final adjustment ensures the
        # global validation count exactly matches the requested count.
        groups: defaultdict[str, list[int]] = defaultdict(list)
        for episode_id in ids:
            groups[str(group_by.get(episode_id, ""))].append(episode_id)
        selected: list[int] = []
        for group_name in sorted(groups):
            # Python's built-in hash is process-randomized.  Use a digest so
            # repeated runs in separate workers retain the same assignment.
            group_seed = int.from_bytes(hashlib.sha256(group_name.encode("utf-8")).digest()[:8], "big")
            ordered = _stable_order(groups[group_name], seed=seed ^ group_seed, mode=mode)
            target = int(round(len(ordered) * fraction))
            if fraction > 0 and len(ordered) > 1:
                target = max(1, target)
                target = min(len(ordered) - 1, target)
            selected.extend(ordered[:target])
        # Fill/trim by a stable global order to hit the exact count.
        global_order = _stable_order(ids, seed=seed, mode=mode)
        selected_set = set(selected)
        if len(selected_set) < validation_count:
            selected_set.update(item for item in global_order if item not in selected_set)
            selected_set = set(global_order[:validation_count])
        elif len(selected_set) > validation_count:
            selected_set = set(item for item in global_order if item in selected_set)  # stable trim below
            selected_set = set(item for item in global_order if item in selected_set)  # preserve global order
            selected_set = set(list(item for item in global_order if item in selected_set)[:validation_count])
        validation = selected_set
    else:
        validation = set(_stable_order(ids, seed=seed, mode=mode)[:validation_count])
    return {episode_id: ("validation" if episode_id in validation else "train") for episode_id in ids}


# Backwards/forwards-compatible aliases used by small training launchers.
split_episodes = split_episode_ids


def _canonical_row(row: Mapping[str, Any], *, episode_id: int, split: str, sample_type: str, strategy: str, episode_info: Mapping[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {key: row.get(key, "") for key in MANIFEST_FIELDS}
    result.update({
        "episode_id": int(episode_id),
        "episode_path": row.get("episode_path", ""),
        "prompt": row.get("prompt", FIXED_PROMPT),
        "group": row.get("group", ""),
        "group_index": row.get("group_index", ""),
        "split": split,
        "sample_type": sample_type,
        "strategy": strategy,
    })
    if episode_info:
        result.update({
            "frames": episode_info.get("frames", ""),
            "state_dim": episode_info.get("state_dim", ""),
            "action_dim": episode_info.get("action_dim", ""),
            "state_dtype": episode_info.get("state_dtype", ""),
            "action_dtype": episode_info.get("action_dtype", ""),
        })
    return result


def _window_row(base: Mapping[str, Any], window: Mapping[str, Any], *, split: str, strategy: str, episode_info: Mapping[str, Any] | None) -> dict[str, Any]:
    result = _canonical_row(base, episode_id=int(base["episode_id"]), split=split, sample_type="window", strategy=strategy, episode_info=episode_info)
    result.update({
        "window_id": window.get("window_id", ""),
        "start": window.get("start", ""),
        "end": window.get("end", ""),
        "length": window.get("length", ""),
        "source_rows": json.dumps(window.get("source_rows", []), ensure_ascii=False, separators=(",", ":")) if window.get("source_rows") is not None else "",
        "frame_indices": json.dumps(window.get("frame_indices", []), ensure_ascii=False, separators=(",", ":")) if window.get("frame_indices") is not None else "",
        "quality_pass": window.get("quality_pass", ""),
        "action_pass": window.get("action_pass", ""),
        "metrics": json.dumps(window.get("metrics", {}), ensure_ascii=False, separators=(",", ":"), default=_json_default) if window.get("metrics") not in (None, "") else "",
    })
    return result


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "episode_id", "window_id", "sample_type", "split", "strategy", "start", "end", "length", "source_rows", "frame_indices",
        "group", "group_index", "C1", "C2", "C3", "C4", "quality_raw", "quality_score", "quality_status", "note", "scene_metadata",
        "prompt", "match_status", "episode_path", "frames", "state_dim", "action_dim", "state_dtype", "action_dtype", "quality_pass", "action_pass", "metrics",
    ]
    present = {key for row in rows for key in row}
    fields = [key for key in preferred if key in present] + sorted(present - set(preferred))
    if not fields:
        fields = ["episode_id"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return path


def _path_string(path: str | Path | None) -> str | None:
    return None if path is None else str(Path(path).expanduser().resolve())


def prepare_classify_data(
    manifest: str | Path | Sequence[Mapping[str, Any]] | Mapping[str, Any],
    *,
    dataset_root: str | Path | None = None,
    window_manifest: str | Path | Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    output_dir: str | Path = "classify_prepared",
    seed: int = DEFAULT_SEED,
    strategy: str = "baseline",
    episode_count: int | None = None,
    episodes: int | None = None,
    start_episode: int = DEFAULT_START_EPISODE,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    split_mode: str = "random",
    group_stratified: bool = True,
    require_state_action: bool | None = None,
    strict_prompt: bool = True,
    train_output: str | Path | None = None,
    validation_output: str | Path | None = None,
    metadata_output: str | Path | None = None,
) -> dict[str, Any]:
    """Prepare train/validation manifests and return the metadata dictionary."""

    if episode_count is not None and episodes is not None and int(episode_count) != int(episodes):
        raise PreparationError("episode_count and episodes disagree")
    expected_count = int(episode_count if episode_count is not None else episodes) if (episode_count is not None or episodes is not None) else None
    if expected_count is not None and expected_count < 0:
        raise PreparationError("episode_count must be non-negative")
    strategy_name = normalize_strategy(strategy)
    source_rows = read_matched_manifest(manifest)
    excluded_rows = [row for row in source_rows if not _is_matched(row)]
    matched_rows = [row for row in source_rows if _is_matched(row)]
    if not matched_rows and expected_count:
        raise PreparationError("matched manifest contains no matched rows")
    by_episode = {int(row["episode_id"]): row for row in matched_rows}
    if expected_count is not None:
        expected_ids = list(range(int(start_episode), int(start_episode) + expected_count))
        missing = [episode_id for episode_id in expected_ids if episode_id not in by_episode]
        extra = [episode_id for episode_id in by_episode if episode_id not in set(expected_ids)]
        if missing or extra:
            raise PreparationError(f"manifest episode coverage mismatch; missing={missing}, extra={extra}")
        episode_ids = expected_ids
    else:
        episode_ids = sorted(by_episode)
        expected_count = len(episode_ids)
    if len(by_episode) != len(matched_rows):
        raise PreparationError("matched manifest contains duplicate episode ids")

    prompt_values = [str(row.get("prompt", "")).strip() for row in matched_rows]
    prompt_counts = Counter(prompt_values)
    prompt_bad = [episode_id for episode_id, row in by_episode.items() if str(row.get("prompt", "")).strip() != FIXED_PROMPT]
    missing_prompt = [episode_id for episode_id, row in by_episode.items() if not str(row.get("prompt", "")).strip()]
    if strict_prompt and prompt_bad:
        raise PreparationError(f"fixed prompt mismatch for episodes {sorted(prompt_bad)}; expected {FIXED_PROMPT!r}")

    episode_map, loaded_paths, arrays_attempted = _load_episode_map(source_rows, dataset_root=dataset_root, manifest_source=manifest)
    episode_info: dict[int, dict[str, Any]] = {}
    state_matrices: list[np.ndarray] = []
    action_matrices: list[np.ndarray] = []
    validation_missing: list[int] = []
    for episode_id in episode_ids:
        episode = episode_map.get(episode_id)
        if episode is None:
            # Manifest-provided dimensions are useful for dry-run preparation
            # when the actual dataset is mounted only on the training host.
            state_dim = _row_dimension(by_episode[episode_id], ("state_dim", "observation_state_dim"))
            action_dim = _row_dimension(by_episode[episode_id], ("action_dim", "observation_action_dim"))
            rows = _row_dimension(by_episode[episode_id], ("frames", "rows", "length"))
            if state_dim is not None or action_dim is not None:
                if state_dim != EXPECTED_DIM or action_dim != EXPECTED_DIM:
                    raise PreparationError(f"episode {episode_id} expected 16D state/action, got {state_dim}D/{action_dim}D")
                episode_info[episode_id] = {
                    "episode_id": episode_id, "frames": rows or "", "state_dim": state_dim, "action_dim": action_dim,
                    "state_dtype": "unknown", "action_dtype": "unknown", "finite": "unknown",
                }
                continue
            validation_missing.append(episode_id)
            continue
        info = _validate_episode(episode)
        episode_info[episode_id] = info
        state_matrices.append(np.asarray(episode.state))
        action_matrices.append(np.asarray(episode.action))
    if require_state_action is None:
        # Supplying a dataset root is an explicit request to validate the
        # mounted arrays; manifest-only preparation remains a useful dry run.
        require_state_action = dataset_root is not None
    if require_state_action and validation_missing:
        raise PreparationError(f"state/action arrays unavailable for episodes {validation_missing}")

    group_by = {episode_id: by_episode[episode_id].get("group", "") for episode_id in episode_ids}
    split_by_episode = split_episode_ids(
        episode_ids,
        validation_fraction=validation_fraction,
        seed=seed,
        mode=split_mode,
        group_by=group_by,
        group_stratified=group_stratified,
    )

    windows: list[dict[str, Any]] = []
    candidate_window_count = 0
    window_metadata: dict[str, Any] = {}
    if window_manifest is not None:
        windows, window_metadata = read_window_manifest(window_manifest)
        candidate_window_count = len(windows)
        report_strategy = window_metadata.get("strategy")
        if report_strategy not in (None, "") and normalize_strategy(str(report_strategy)) != strategy_name:
            raise PreparationError(f"window manifest strategy {report_strategy!r} does not match requested {strategy_name!r}")
        unknown = sorted({int(window["episode_id"]) for window in windows} - set(episode_ids))
        if unknown:
            raise PreparationError(f"window manifest contains episodes absent from matched manifest: {unknown}")
        windows = [window for window in windows if bool(window.get("selected", True))]

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    if window_manifest is not None:
        for window in windows:
            episode_id = int(window["episode_id"])
            split = split_by_episode[episode_id]
            output_row = _window_row(by_episode[episode_id], window, split=split, strategy=strategy_name, episode_info=episode_info.get(episode_id))
            (val_rows if split == "validation" else train_rows).append(output_row)
    else:
        for episode_id in episode_ids:
            split = split_by_episode[episode_id]
            output_row = _canonical_row(by_episode[episode_id], episode_id=episode_id, split=split, sample_type="episode", strategy=strategy_name, episode_info=episode_info.get(episode_id))
            (val_rows if split == "validation" else train_rows).append(output_row)

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_path = _write_rows(Path(train_output).expanduser().resolve() if train_output is not None else output / "train_manifest.csv", train_rows)
    val_path = _write_rows(Path(validation_output).expanduser().resolve() if validation_output is not None else output / "val_manifest.csv", val_rows)
    # Short aliases are convenient for launchers and retain compatibility with
    # early experiments that called these files train.csv/val.csv.
    train_alias = _write_rows(output / "train.csv", train_rows)
    val_alias = _write_rows(output / "val.csv", val_rows)

    # Keep the split assignment visible even when a filter rejects every
    # window from one episode.  The absence of samples then remains distinct
    # from an episode that was accidentally assigned to neither split.
    train_episode_ids = sorted(episode_id for episode_id, split in split_by_episode.items() if split == "train")
    val_episode_ids = sorted(episode_id for episode_id, split in split_by_episode.items() if split == "validation")
    all_stats_state = _matrix_stats(state_matrices)
    all_stats_action = _matrix_stats(action_matrices)
    train_stats_state = _matrix_stats([episode_map[e].state for e in train_episode_ids if e in episode_map])
    train_stats_action = _matrix_stats([episode_map[e].action for e in train_episode_ids if e in episode_map])
    dimensions = {
        "state_dim": EXPECTED_DIM,
        "action_dim": EXPECTED_DIM,
        "expected_state_dim": EXPECTED_DIM,
        "expected_action_dim": EXPECTED_DIM,
        "checked_episodes": len(episode_info),
        "missing_episodes": validation_missing,
        "all_finite": all(bool(info.get("finite") is True) for info in episode_info.values()) if episode_info else None,
        "arrays_attempted": bool(arrays_attempted),
    }
    metadata: dict[str, Any] = {
        "format": "classify_objects_prepared_v1",
        "task": "classify_objects",
        "prompt": FIXED_PROMPT,
        "fixed_prompt": FIXED_PROMPT,
        "prompt_counts": dict(prompt_counts),
        "prompt_validation": {"ok": not prompt_bad and not missing_prompt, "bad_episode_ids": sorted(prompt_bad), "missing_episode_ids": sorted(missing_prompt)},
        "strategy": strategy_name,
        "strategy_requested": str(strategy),
        "seed": int(seed),
        "split_seed": int(seed),
        "split_mode": split_mode,
        "validation_fraction": float(validation_fraction),
        "group_stratified": bool(group_stratified),
        "episode_count": int(expected_count),
        "episodes": int(expected_count),
        "expected_episode_count": int(expected_count),
        "matched_episode_count": len(matched_rows),
        "excluded_unmatched_count": len(excluded_rows),
        "episode_ids": episode_ids,
        "split": {
            "train_episode_ids": train_episode_ids,
            "validation_episode_ids": val_episode_ids,
            "train_episode_count": len(train_episode_ids),
            "validation_episode_count": len(val_episode_ids),
            "train_sample_count": len(train_rows),
            "validation_sample_count": len(val_rows),
            "episode_disjoint": set(train_episode_ids).isdisjoint(val_episode_ids),
        },
        "windows": {
            "provided": window_manifest is not None,
            "candidate_count": candidate_window_count,
            "selected_count": len(windows) if window_manifest is not None else 0,
            "sample_type": "window" if window_manifest is not None else "episode",
            "source_metadata": window_metadata,
        },
        "dimensions": dimensions,
        "state_dim": EXPECTED_DIM,
        "action_dim": EXPECTED_DIM,
        "state_action_dim": EXPECTED_DIM,
        "state_stats": all_stats_state,
        "action_stats": all_stats_action,
        "train_state_stats": train_stats_state,
        "train_action_stats": train_stats_action,
        "source": {
            "manifest": _path_string(manifest) if isinstance(manifest, (str, Path)) else None,
            "window_manifest": _path_string(window_manifest) if isinstance(window_manifest, (str, Path)) else None,
            "dataset_root": _path_string(dataset_root),
        },
        "outputs": {
            "train_manifest": str(train_path),
            "val_manifest": str(val_path),
            "validation_manifest": str(val_path),
            "train": str(train_path),
            "val": str(val_path),
            "train_alias": str(train_alias),
            "validation_alias": str(val_alias),
        },
        "fastwam": {
            "manifest": str(train_path),
            "validation_manifest": str(val_path),
            "instruction": FIXED_PROMPT,
            "state_dim": EXPECTED_DIM,
            "action_dim": EXPECTED_DIM,
            "strategy": strategy_name,
        },
        "efficient_wam": {
            "manifest": str(train_path),
            "validation_manifest": str(val_path),
            "instruction": FIXED_PROMPT,
            "state_dim": EXPECTED_DIM,
            "action_dim": EXPECTED_DIM,
            "strategy": strategy_name,
        },
    }
    # Keep both spellings because the upstream projects use different names
    # in their launchers.
    metadata["efficientwam"] = dict(metadata["efficient_wam"])
    metadata_path = Path(metadata_output).expanduser().resolve() if metadata_output is not None else output / "prepare_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_alias = output / "metadata.json"
    metadata["outputs"]["metadata"] = str(metadata_path)
    metadata["outputs"]["metadata_alias"] = str(metadata_alias)
    serialized_metadata = json.dumps(metadata, indent=2, ensure_ascii=False, default=_json_default) + "\n"
    metadata_path.write_text(serialized_metadata, encoding="utf-8")
    metadata_alias.write_text(serialized_metadata, encoding="utf-8")
    return metadata


# Common aliases used by callers that prefer a shorter name.
prepare_data = prepare_classify_data
prepare_dataset = prepare_classify_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", "--matched-manifest", required=True, type=Path, help="Matched episode CSV/JSON")
    parser.add_argument("--window-manifest", "--filtered-window-manifest", "--filtered-manifest", "--windows", type=Path, default=None)
    parser.add_argument("--dataset-root", "--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", "--output", "-o", type=Path, default=Path("classify_prepared"))
    parser.add_argument("--train-output", "--train-manifest", type=Path, default=None, help="Optional explicit train CSV path")
    parser.add_argument("--val-output", "--validation-output", "--val-manifest", type=Path, default=None, help="Optional explicit validation CSV path")
    parser.add_argument("--metadata-output", type=Path, default=None, help="Optional explicit metadata JSON path")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--strategy", default="baseline", type=normalize_strategy)
    parser.add_argument("--episodes", "--episode-count", "--expected-episodes", dest="episode_count", type=int, default=None)
    parser.add_argument("--start-episode", type=int, default=DEFAULT_START_EPISODE)
    parser.add_argument("--validation-fraction", "--val-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION)
    parser.add_argument("--split-mode", choices=("random", "sorted", "hash"), default="random")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--group-stratified", dest="group_stratified", action="store_true", default=True)
    group.add_argument("--no-group-stratified", dest="group_stratified", action="store_false")
    parser.add_argument("--require-state-action", action="store_true", default=None)
    parser.add_argument("--allow-prompt-drift", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata = prepare_classify_data(
        args.manifest,
        dataset_root=args.dataset_root,
        window_manifest=args.window_manifest,
        output_dir=args.output_dir,
        train_output=args.train_output,
        validation_output=args.val_output,
        metadata_output=args.metadata_output,
        seed=args.seed,
        strategy=args.strategy,
        episode_count=args.episode_count,
        start_episode=args.start_episode,
        validation_fraction=args.validation_fraction,
        split_mode=args.split_mode,
        group_stratified=args.group_stratified,
        require_state_action=args.require_state_action,
        strict_prompt=not args.allow_prompt_drift,
    )
    print(json.dumps({
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "strategy": metadata["strategy"],
        "episodes": metadata["episode_count"],
        "train_episodes": metadata["split"]["train_episode_count"],
        "validation_episodes": metadata["split"]["validation_episode_count"],
        "train_samples": metadata["split"]["train_sample_count"],
        "validation_samples": metadata["split"]["validation_sample_count"],
        "state_dim": metadata["state_dim"],
        "action_dim": metadata["action_dim"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_SEED",
    "DEFAULT_VALIDATION_FRACTION",
    "EXPECTED_DIM",
    "FIXED_PROMPT",
    "PreparationError",
    "normalize_strategy",
    "read_matched_manifest",
    "read_window_manifest",
    "split_episode_ids",
    "split_episodes",
    "prepare_classify_data",
    "prepare_data",
    "prepare_dataset",
    "build_parser",
    "main",
]
