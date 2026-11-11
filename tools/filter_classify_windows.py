#!/usr/bin/env python3
"""Filter Classify Objects action windows without stitching rejected frames.

Windows are generated independently inside each episode and, when frame
indices are available, independently inside each contiguous frame run.  A
selected window stores source row indices rather than a copied/concatenated
array.  This makes it impossible for a rejected middle interval or an episode
boundary to become a synthetic transition.

The four supported policies are:

``Baseline``
    Keep every structurally valid window.
``Quality``
    Apply motion-quality checks (speed, acceleration, jerk, spikes, pauses).
``Action``
    Apply action semantics checks (state/action alignment and gripper motion).
``Combined``
    Apply both sets of checks.

Example::

    python tools/filter_classify_windows.py DATASET --strategy Combined \
        --window-size 32 --stride 16 --output windows.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from .analyze_classify_actions import (  # type: ignore[import-not-found]
        AnalysisConfig,
        Episode,
        _json_default,
        analyze_episode,
        load_episodes,
        robust_upper_threshold,
    )
except ImportError:  # script execution from the repository root
    from analyze_classify_actions import (  # type: ignore[no-redef]
        AnalysisConfig,
        Episode,
        _json_default,
        analyze_episode,
        load_episodes,
        robust_upper_threshold,
    )


STRATEGIES = ("Baseline", "Quality", "Action", "Combined")
_STRATEGY_ALIASES = {name.lower(): name for name in STRATEGIES}


@dataclass
class WindowConfig:
    window_size: int = 32
    stride: int = 32
    include_tail: bool = False
    strategy: str = "Baseline"
    mad_sigma: float = 6.0
    # Absolute limits are optional.  None means derive a dataset-level robust
    # MAD limit from the candidate windows for the corresponding metric.
    velocity_threshold: float | None = None
    acceleration_threshold: float | None = None
    jerk_threshold: float | None = None
    spike_rate_threshold: float | None = None
    pause_fraction_threshold: float | None = None
    max_pause_frames: int | None = None
    max_gripper_transitions: int | None = None
    alignment_rmse_threshold: float | None = None
    alignment_max_error_threshold: float | None = None
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    split_seed: int = 0
    split_mode: str = "random"

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError("window_size must be positive")
        if self.stride < 1:
            raise ValueError("stride must be positive")
        self.strategy = normalize_strategy(self.strategy)
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("train/validation/test fractions must sum to 1")
        if min(self.train_fraction, self.validation_fraction, self.test_fraction) < 0:
            raise ValueError("split fractions cannot be negative")
        if self.split_mode not in {"random", "sorted", "hash"}:
            raise ValueError("split_mode must be random, sorted, or hash")


@dataclass
class WindowRecord:
    window_id: str
    episode_id: int
    start: int
    end: int
    source_rows: list[int]
    frame_indices: list[int] | None
    source: str | None
    metrics: dict[str, Any]
    quality_pass: bool
    action_pass: bool
    selected: bool
    split: str | None = None
    reason: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "episode_id": int(self.episode_id),
            "start": int(self.start),
            "end": int(self.end),
            "length": int(self.end - self.start),
            "source_rows": [int(value) for value in self.source_rows],
            "frame_indices": None if self.frame_indices is None else [int(value) for value in self.frame_indices],
            "source": self.source,
            "metrics": self.metrics,
            "quality_pass": bool(self.quality_pass),
            "action_pass": bool(self.action_pass),
            "selected": bool(self.selected),
            "split": self.split,
            "reason": list(self.reason),
        }


def normalize_strategy(strategy: str) -> str:
    try:
        return _STRATEGY_ALIASES[str(strategy).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"unknown strategy {strategy!r}; choose one of {STRATEGIES}") from exc


def _contiguous_runs(episode: Episode) -> list[tuple[int, int]]:
    """Return half-open local row ranges with no missing frame indices."""

    row_count = 0 if episode.row_indices is None else len(episode.row_indices)
    n = max(len(episode.state), len(episode.action), row_count)
    if n == 0:
        return []
    frame_index = None if episode.frame_index is None else np.asarray(episode.frame_index[:n], dtype=np.int64)
    row_indices = None if episode.row_indices is None else np.asarray(episode.row_indices[:n], dtype=np.int64)
    if (frame_index is None or len(frame_index) < n) and (row_indices is None or len(row_indices) < n):
        return [(0, n)]
    breaks_parts: list[np.ndarray] = []
    if frame_index is not None and len(frame_index) >= n and len(frame_index) > 1:
        breaks_parts.append(np.flatnonzero(np.diff(frame_index) != 1) + 1)
    # A sharded parquet can interleave episodes.  Even without frame_index,
    # source row gaps identify a missing middle interval and must not be
    # silently bridged into a synthetic training window.
    if row_indices is not None and len(row_indices) >= n and len(row_indices) > 1:
        row_breaks = np.diff(row_indices) != 1
        row_sources = episode.metadata.get("row_sources") if isinstance(episode.metadata, Mapping) else None
        if isinstance(row_sources, (list, tuple, np.ndarray)) and len(row_sources) >= n:
            source_values = np.asarray(row_sources[:n], dtype=object)
            # A shard boundary is not itself a missing frame.  Treat a source
            # row gap as a break only when both rows came from the same file.
            row_breaks &= source_values[:-1] == source_values[1:]
        breaks_parts.append(np.flatnonzero(row_breaks) + 1)
    breaks = np.unique(np.concatenate(breaks_parts)) if breaks_parts else np.empty((0,), dtype=np.int64)
    boundaries = [0, *[int(value) for value in breaks], n]
    return [(start, end) for start, end in zip(boundaries[:-1], boundaries[1:]) if end > start]


def iter_window_slices(
    episode: Episode,
    *,
    window_size: int,
    stride: int | None = None,
    include_tail: bool = False,
) -> Iterable[tuple[int, int]]:
    """Yield local ``[start, end)`` slices, never crossing a frame gap."""

    if window_size < 1:
        raise ValueError("window_size must be positive")
    stride = window_size if stride is None else int(stride)
    if stride < 1:
        raise ValueError("stride must be positive")
    for run_start, run_end in _contiguous_runs(episode):
        run_length = run_end - run_start
        if run_length < window_size:
            if include_tail:
                yield run_start, run_end
            continue
        start = run_start
        while start + window_size <= run_end:
            yield start, start + window_size
            start += stride
        if include_tail and start < run_end:
            tail_start = max(run_start, run_end - window_size)
            if tail_start != start - stride or tail_start + window_size > run_end:
                yield tail_start, run_end


def _metric_value(metrics: Mapping[str, Any], name: str) -> float:
    if name in {"velocity", "acceleration", "jerk"}:
        value = metrics.get(name, {}).get("p95", np.nan)
    elif name == "spike_rate":
        value = metrics.get("spike_rate", np.nan)
    elif name == "pause_fraction":
        value = metrics.get("pause_fraction", np.nan)
    elif name == "pause_frames":
        value = metrics.get("longest_pause_frames", np.nan)
    elif name == "gripper_transitions":
        value = metrics.get("gripper", {}).get("transition_count", np.nan)
    elif name == "alignment_rmse":
        value = metrics.get("alignment", {}).get("rmse", np.nan)
    elif name == "alignment_max_error":
        value = metrics.get("alignment", {}).get("max_abs_error", np.nan)
    else:
        value = metrics.get(name, np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _candidate_limits(records: Sequence[WindowRecord], config: WindowConfig) -> dict[str, float]:
    explicit: dict[str, float | None] = {
        "velocity": config.velocity_threshold,
        "acceleration": config.acceleration_threshold,
        "jerk": config.jerk_threshold,
        "spike_rate": config.spike_rate_threshold,
        "pause_fraction": config.pause_fraction_threshold,
        "pause_frames": float(config.max_pause_frames) if config.max_pause_frames is not None else None,
        "gripper_transitions": float(config.max_gripper_transitions) if config.max_gripper_transitions is not None else None,
        "alignment_rmse": config.alignment_rmse_threshold,
        "alignment_max_error": config.alignment_max_error_threshold,
    }
    limits: dict[str, float] = {}
    for name, value in explicit.items():
        if value is not None:
            limits[name] = float(value)
            continue
        values = np.asarray([_metric_value(record.metrics, name) for record in records], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size:
            limits[name] = float(robust_upper_threshold(values, mad_sigma=config.mad_sigma))
        else:
            limits[name] = float("inf")
    # Structural/action defaults should not reject every short window merely
    # because an optional alignment metric is unavailable.
    limits.setdefault("pause_frames", float("inf"))
    return limits


def _passes(
    record: WindowRecord,
    limits: Mapping[str, float],
) -> tuple[bool, bool, list[str]]:
    metrics = record.metrics
    flags = metrics.get("quality_flags", {})
    finite = bool(metrics.get("finite", False))
    alignment_ok = bool(metrics.get("alignment", {}).get("ok", False))
    quality_failures: list[str] = []
    action_failures: list[str] = []
    if not finite:
        quality_failures.append("non_finite")
    checks = (
        ("velocity", "velocity"),
        ("acceleration", "acceleration"),
        ("jerk", "jerk"),
        ("spike_rate", "spike_rate"),
        ("pause_fraction", "pause_fraction"),
        ("pause_frames", "longest_pause_frames"),
    )
    for limit_name, metric_name in checks:
        value = _metric_value(metrics, limit_name)
        if np.isfinite(value) and value > limits.get(limit_name, float("inf")):
            quality_failures.append(f"{metric_name}>{limits[limit_name]:g}")
        elif not np.isfinite(value) and metric_name in {"velocity", "acceleration", "jerk"}:
            quality_failures.append(f"missing_{metric_name}")
    if not alignment_ok:
        action_failures.append("state_action_alignment")
    for limit_name, metric_name in (
        ("gripper_transitions", "gripper.transition_count"),
        ("alignment_rmse", "alignment.rmse"),
        ("alignment_max_error", "alignment.max_abs_error"),
    ):
        value = _metric_value(metrics, limit_name)
        # NaN alignment is allowed only if dimensions contain no configured
        # gripper channels; structural alignment above still applies.
        if np.isfinite(value) and value > limits.get(limit_name, float("inf")):
            action_failures.append(f"{metric_name}>{limits[limit_name]:g}")
    # Explicit quality flags (for example a strict user threshold passed to
    # analyze_episode) remain authoritative.
    for name in ("velocity", "acceleration", "jerk", "spikes", "pauses"):
        if name in flags and not bool(flags[name]):
            quality_failures.append(name)
    quality_failures = list(dict.fromkeys(quality_failures))
    action_failures = list(dict.fromkeys(action_failures))
    return not quality_failures, not action_failures, quality_failures + action_failures


def _window_record(episode: Episode, start: int, end: int, *, config: AnalysisConfig, ordinal: int) -> WindowRecord:
    segment = episode.slice(start, end)
    metrics = analyze_episode(segment, config=config)
    source_rows = [] if segment.row_indices is None else [int(value) for value in segment.row_indices]
    frame_indices = None if segment.frame_index is None else [int(value) for value in segment.frame_index]
    return WindowRecord(
        window_id=f"episode_{episode.episode_id:06d}_window_{ordinal:06d}",
        episode_id=int(episode.episode_id),
        start=int(start),
        end=int(end),
        source_rows=source_rows,
        frame_indices=frame_indices,
        source=episode.source,
        metrics=metrics,
        quality_pass=False,
        action_pass=False,
        selected=False,
    )


def _assign_episode_splits(
    episode_ids: Iterable[int],
    *,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    mode: str,
) -> dict[int, str]:
    ids = sorted({int(value) for value in episode_ids})
    if mode == "sorted":
        ordered = ids
    elif mode == "hash":
        ordered = sorted(ids, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
    else:
        rng = np.random.default_rng(seed)
        ordered = [int(value) for value in np.asarray(ids)[rng.permutation(len(ids))]] if ids else []
    count = len(ordered)
    # Round only the first two bins and assign the remainder to test to avoid
    # dropping an episode due to floating point fractions.
    train_count = int(round(count * train_fraction))
    validation_count = int(round(count * validation_fraction))
    if train_count + validation_count > count:
        validation_count = max(0, count - train_count)
    result: dict[int, str] = {}
    for index, episode_id in enumerate(ordered):
        if index < train_count:
            result[episode_id] = "train"
        elif index < train_count + validation_count:
            result[episode_id] = "validation"
        else:
            result[episode_id] = "test"
    return result


def split_episodes(
    episode_ids: Iterable[int],
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 0,
    mode: str = "random",
) -> dict[int, str]:
    """Assign each episode to exactly one split.

    This function is deliberately episode-based; windows from one episode can
    never leak into another split.
    """

    total = train_fraction + validation_fraction + test_fraction
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError("split fractions must sum to 1")
    return _assign_episode_splits(
        episode_ids,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
        mode=mode,
    )


def filter_windows(
    episodes: Sequence[Episode],
    *,
    window_config: WindowConfig | None = None,
    analysis_config: AnalysisConfig | None = None,
    episode_quality: Mapping[int | str, Any] | None = None,
    quality_threshold: float | None = None,
) -> dict[str, Any]:
    """Generate and filter windows, returning a serializable manifest."""

    window_config = window_config or WindowConfig()
    analysis_config = analysis_config or AnalysisConfig(mad_sigma=window_config.mad_sigma)
    records: list[WindowRecord] = []
    for episode in episodes:
        for ordinal, (start, end) in enumerate(
            iter_window_slices(
                episode,
                window_size=window_config.window_size,
                stride=window_config.stride,
                include_tail=window_config.include_tail,
            )
        ):
            records.append(_window_record(episode, start, end, config=analysis_config, ordinal=ordinal))

    limits = _candidate_limits(records, window_config)
    for record in records:
        quality_pass, action_pass, reasons = _passes(record, limits)
        # Excel's human quality score is an episode-level gate.  It is only
        # applied by Quality/Combined; Baseline and Action remain useful as
        # independent ablations.  Unknown/missing scores are conservative
        # failures when an explicit threshold is requested.
        score_value: float | None = None
        score_status = "missing"
        if episode_quality is not None:
            raw_quality = episode_quality.get(record.episode_id, episode_quality.get(str(record.episode_id)))
            if isinstance(raw_quality, Mapping):
                raw_score = raw_quality.get("quality_score", raw_quality.get("score"))
                score_status = str(raw_quality.get("quality_status", raw_quality.get("status", "unknown")))
            else:
                raw_score = raw_quality
                score_status = "known" if raw_quality not in (None, "") else "missing"
            try:
                if raw_score not in (None, ""):
                    score_value = float(raw_score)
            except (TypeError, ValueError):
                score_value = None
                score_status = "unknown"
            record.metrics["excel_quality_score"] = score_value
            record.metrics["excel_quality_status"] = score_status
            if (
                quality_threshold is not None
                and window_config.strategy in {"Quality", "Combined"}
                and (score_value is None or not np.isfinite(score_value) or score_value < quality_threshold)
            ):
                quality_pass = False
                reasons.append(
                    "excel_quality_missing" if score_value is None else f"excel_quality<{quality_threshold:g}"
                )
                reasons = list(dict.fromkeys(reasons))
        record.quality_pass = quality_pass
        record.action_pass = action_pass
        strategy = window_config.strategy
        record.selected = {
            "Baseline": True,
            "Quality": quality_pass,
            "Action": action_pass,
            "Combined": quality_pass and action_pass,
        }[strategy]
        record.reason = reasons if not record.selected else []

    split_by_episode = _assign_episode_splits(
        (record.episode_id for record in records),
        train_fraction=window_config.train_fraction,
        validation_fraction=window_config.validation_fraction,
        test_fraction=window_config.test_fraction,
        seed=window_config.split_seed,
        mode=window_config.split_mode,
    )
    for record in records:
        record.split = split_by_episode.get(record.episode_id)

    selected = [record for record in records if record.selected]
    rejected = [record for record in records if not record.selected]
    split_summary = {
        split: {
            "episodes": sorted({record.episode_id for record in selected if record.split == split}),
            "windows": int(sum(record.split == split for record in selected)),
        }
        for split in ("train", "validation", "test")
    }
    return {
        "format": "classify_objects_window_filter_v1",
        "strategy": window_config.strategy,
        "window_config": asdict(window_config),
        "analysis_config": asdict(analysis_config),
        "quality_threshold": quality_threshold,
        "limits": {key: float(value) for key, value in limits.items()},
        "episode_splits": {str(episode_id): split for episode_id, split in sorted(split_by_episode.items())},
        "summary": {
            "episodes": len({record.episode_id for record in records}),
            "candidate_windows": len(records),
            "selected_windows": len(selected),
            "rejected_windows": len(records) - len(selected),
            "selected_fraction": float(len(selected) / len(records)) if records else 0.0,
            "splits": split_summary,
        },
        "windows": [record.as_dict() for record in records],
        # Convenience views keep consumers from having to reimplement the
        # selected/rejected predicate while preserving the full audit trail.
        "selected_windows": [record.as_dict() for record in selected],
        "rejected_windows": [record.as_dict() for record in rejected],
    }


def filter_dataset(
    source: str | Path | Sequence[Episode],
    *,
    window_config: WindowConfig | None = None,
    analysis_config: AnalysisConfig | None = None,
    episode_quality: Mapping[int | str, Any] | None = None,
    quality_threshold: float | None = None,
    **loader_kwargs: Any,
) -> dict[str, Any]:
    episodes = load_episodes(source, **loader_kwargs)
    return filter_windows(
        episodes,
        window_config=window_config,
        analysis_config=analysis_config,
        episode_quality=episode_quality,
        quality_threshold=quality_threshold,
    )


def compare_strategies(
    episodes: Sequence[Episode],
    *,
    window_config: WindowConfig | None = None,
    analysis_config: AnalysisConfig | None = None,
    episode_quality: Mapping[int | str, Any] | None = None,
    quality_threshold: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Run Baseline/Quality/Action/Combined on the same candidate windows.

    The function intentionally regenerates only the lightweight manifest for
    each policy; source episodes are never modified.  It is useful for a
    retention table in a quality report and makes policy comparisons use the
    exact same episode split.
    """

    base = window_config or WindowConfig()
    return {
        strategy: filter_windows(
            episodes,
            window_config=WindowConfig(**{**asdict(base), "strategy": strategy}),
            analysis_config=analysis_config,
            episode_quality=episode_quality,
            quality_threshold=quality_threshold,
        )
        for strategy in STRATEGIES
    }


# Explicit aliases make the small module convenient to import from notebooks
# without coupling callers to the CLI-oriented function names.
generate_windows = iter_window_slices
filter_classify_windows = filter_windows


def _parse_indices(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _read_quality_manifest(path: Path) -> dict[int, dict[str, Any]]:
    """Read the CSV produced by ``match_classify_excel.py``.

    A plain CSV reader is used here so filtering remains usable without the
    optional Excel/OOXML dependencies.  Unknown or blank scores are retained
    as statuses and are handled conservatively when a threshold is requested.
    """

    import csv

    result: dict[int, dict[str, Any]] = {}
    with path.expanduser().open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_id = row.get("episode_id", "")
            try:
                episode_id = int(float(str(raw_id).strip()))
            except (TypeError, ValueError):
                continue
            result[episode_id] = {
                "quality_score": row.get("quality_score", ""),
                "quality_status": row.get("quality_status", "unknown"),
                "quality_raw": row.get("quality_raw", ""),
                "group": row.get("group", ""),
                "C1": row.get("C1", ""),
                "C2": row.get("C2", ""),
                "C3": row.get("C3", ""),
                "C4": row.get("C4", ""),
                "episode_path": row.get("episode_path", ""),
                "prompt": row.get("prompt", ""),
            }
    return result


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, np.ndarray)):
        return ";".join(str(item) for item in np.asarray(value).reshape(-1).tolist())
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, default=_json_default, sort_keys=True)
    return value


def _window_csv_row(window: Mapping[str, Any], quality: Mapping[int, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    metrics = window.get("metrics", {})
    episode_id = int(window.get("episode_id", -1))
    excel = (quality or {}).get(episode_id, {})
    return {
        "window_id": window.get("window_id", ""),
        "episode_id": episode_id,
        "start": window.get("start", ""),
        "end": window.get("end", ""),
        "length": window.get("length", ""),
        "source_rows": _csv_value(window.get("source_rows", [])),
        "frame_indices": _csv_value(window.get("frame_indices", [])),
        "source": window.get("source", ""),
        "split": window.get("split", ""),
        "selected": window.get("selected", False),
        "quality_pass": window.get("quality_pass", False),
        "action_pass": window.get("action_pass", False),
        "reason": ";".join(str(value) for value in window.get("reason", [])),
        "velocity_p95": metrics.get("velocity", {}).get("p95", ""),
        "acceleration_p95": metrics.get("acceleration", {}).get("p95", ""),
        "jerk_p95": metrics.get("jerk", {}).get("p95", ""),
        "spike_rate": metrics.get("spike_rate", ""),
        "pause_fraction": metrics.get("pause_fraction", ""),
        "longest_pause_frames": metrics.get("longest_pause_frames", ""),
        "gripper_transitions": metrics.get("gripper", {}).get("transition_count", ""),
        "alignment_rmse": metrics.get("alignment", {}).get("rmse", ""),
        "alignment_max_error": metrics.get("alignment", {}).get("max_abs_error", ""),
        "excel_quality_score": excel.get("quality_score", metrics.get("excel_quality_score", "")),
        "excel_quality_status": excel.get("quality_status", metrics.get("excel_quality_status", "")),
        "group": excel.get("group", ""),
        "C1": excel.get("C1", ""),
        "C2": excel.get("C2", ""),
        "C3": excel.get("C3", ""),
        "C4": excel.get("C4", ""),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "window_id", "episode_id", "start", "end", "length", "source_rows", "frame_indices", "source", "split",
        "selected", "quality_pass", "action_pass", "reason", "velocity_p95", "acceleration_p95", "jerk_p95",
        "spike_rate", "pause_fraction", "longest_pause_frames", "gripper_transitions", "alignment_rmse",
        "alignment_max_error", "excel_quality_score", "excel_quality_status", "group", "C1", "C2", "C3", "C4",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def _write_quality_csv(path: Path, report: Mapping[str, Any], quality: Mapping[int, Mapping[str, Any]] | None) -> None:
    rows = [_window_csv_row(window, quality) for window in report.get("windows", [])]
    _write_csv(path, rows)


def _write_report(path: Path, report: Mapping[str, Any], *, strategy_reports: Mapping[str, Mapping[str, Any]] | None = None) -> None:
    summary = report.get("summary", {})
    lines = [
        "# Classify Objects Quality Report",
        "",
        f"- Strategy: **{report.get('strategy', '')}**",
        f"- Candidate windows: **{summary.get('candidate_windows', 0)}**",
        f"- Selected windows: **{summary.get('selected_windows', 0)}**",
        f"- Rejected windows: **{summary.get('rejected_windows', 0)}**",
        f"- Selected fraction: **{float(summary.get('selected_fraction', 0.0)):.4f}**",
        "",
        "## Episode Split",
        "",
        "| split | episodes | windows |",
        "|---|---:|---:|",
    ]
    for split, values in summary.get("splits", {}).items():
        lines.append(f"| {split} | {len(values.get('episodes', []))} | {values.get('windows', 0)} |")
    lines.extend(["", "## Thresholds", "", "```json", json.dumps(report.get("limits", {}), indent=2, ensure_ascii=False, default=_json_default), "```", ""])
    if strategy_reports:
        lines.extend(["## Strategy Retention", "", "| strategy | candidates | selected | fraction |", "|---|---:|---:|---:|"])
        for name in STRATEGIES:
            values = strategy_reports.get(name, {}).get("summary", {})
            lines.append(
                f"| {name} | {values.get('candidate_windows', 0)} | {values.get('selected_windows', 0)} | {float(values.get('selected_fraction', 0.0)):.4f} |"
            )
    lines.extend(["", "Rejected windows retain their source row/frame indices; no middle frames are concatenated.", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_output(path: Path, report: Mapping[str, Any], quality: Mapping[int, Mapping[str, Any]] | None) -> None:
    """Write JSON for the legacy API and CSV for launcher manifests."""

    if path.suffix.lower() == ".csv":
        _write_csv(path, [
            _window_csv_row(window, quality)
            for window in report.get("selected_windows", [])
        ])
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", type=Path, help="LeRobot root or parquet file (legacy positional)")
    parser.add_argument("--dataset-root", type=Path, help="LeRobot root (launcher-compatible alias)")
    parser.add_argument("--manifest", type=Path, help="Excel-derived episode manifest CSV")
    parser.add_argument("--strategy", type=normalize_strategy, default="Baseline", help="Baseline, Quality, Action, or Combined (case-insensitive)")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--include-tail", action="store_true")
    parser.add_argument("-o", "--output", type=Path, help="JSON report or selected-window CSV")
    parser.add_argument("--train-output", type=Path)
    parser.add_argument("--val-output", "--validation-output", dest="val_output", type=Path)
    parser.add_argument("--test-output", type=Path)
    parser.add_argument("--quality-csv", type=Path, help="Write per-window quality CSV")
    parser.add_argument("--report", type=Path, help="Write Markdown quality report")
    parser.add_argument("--quality-threshold", type=float, help="Minimum Excel quality score for Quality/Combined")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--time-unit", choices=("s", "ms", "frames"), default="s")
    parser.add_argument("--value-unit", default="native")
    parser.add_argument("--threshold-unit", default=None)
    parser.add_argument("--mad-sigma", type=float, default=6.0)
    parser.add_argument("--velocity-threshold", type=float)
    parser.add_argument("--acceleration-threshold", type=float)
    parser.add_argument("--jerk-threshold", type=float)
    parser.add_argument("--spike-rate-threshold", type=float)
    parser.add_argument("--pause-fraction-threshold", type=float)
    parser.add_argument("--max-pause-frames", type=int)
    parser.add_argument("--max-gripper-transitions", type=int)
    parser.add_argument("--alignment-rmse-threshold", type=float)
    parser.add_argument("--alignment-max-error-threshold", type=float)
    parser.add_argument("--pause-velocity-threshold", type=float, default=1e-3)
    parser.add_argument("--min-pause-frames", type=int, default=3)
    parser.add_argument("--gripper-transition-threshold", type=float, default=1e-3)
    parser.add_argument("--gripper-indices", default="7,15")
    parser.add_argument("--train-fraction", type=float)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.0)
    parser.add_argument("--split-seed", "--seed", dest="split_seed", type=int, default=0)
    parser.add_argument("--split-mode", choices=("random", "sorted", "hash"), default="random")
    args = parser.parse_args(argv)

    dataset = args.dataset_root or args.dataset
    if dataset is None:
        parser.error("a dataset path is required (positional DATASET or --dataset-root)")
    if args.output is None and args.train_output is None and args.val_output is None and args.quality_csv is None and args.report is None:
        parser.error("provide --output, --train-output, --val-output, --quality-csv, or --report")
    manifest_quality: dict[int, dict[str, Any]] = {}
    if args.manifest:
        if not args.manifest.is_file():
            parser.error(f"manifest does not exist: {args.manifest}")
        manifest_quality = _read_quality_manifest(args.manifest)
    # The workbook's supplied numeric scale is 0..1.  A conservative default
    # keeps Quality/Combined from silently accepting explicitly bad rows while
    # leaving runs without a manifest unaffected.
    quality_threshold = args.quality_threshold
    if quality_threshold is None and manifest_quality and args.strategy in {"Quality", "Combined"}:
        quality_threshold = 0.5
    stride = args.window_size if args.stride is None else args.stride
    validation_fraction = float(args.validation_fraction)
    test_fraction = float(args.test_fraction)
    if args.train_fraction is None:
        train_fraction = 1.0 - validation_fraction - test_fraction
    else:
        train_fraction = float(args.train_fraction)
        if args.test_fraction == 0.0:
            test_fraction = max(0.0, 1.0 - train_fraction - validation_fraction)
    window_config = WindowConfig(
        window_size=args.window_size,
        stride=stride,
        include_tail=args.include_tail,
        strategy=args.strategy,
        mad_sigma=args.mad_sigma,
        velocity_threshold=args.velocity_threshold,
        acceleration_threshold=args.acceleration_threshold,
        jerk_threshold=args.jerk_threshold,
        spike_rate_threshold=args.spike_rate_threshold,
        pause_fraction_threshold=args.pause_fraction_threshold,
        max_pause_frames=args.max_pause_frames,
        max_gripper_transitions=args.max_gripper_transitions,
        alignment_rmse_threshold=args.alignment_rmse_threshold,
        alignment_max_error_threshold=args.alignment_max_error_threshold,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        split_seed=args.split_seed,
        split_mode=args.split_mode,
    )
    analysis_config = AnalysisConfig(
        fps=args.fps,
        time_unit=args.time_unit,
        value_unit=args.value_unit,
        threshold_unit=args.threshold_unit,
        mad_sigma=args.mad_sigma,
        pause_velocity_threshold=args.pause_velocity_threshold,
        min_pause_frames=args.min_pause_frames,
        gripper_transition_threshold=args.gripper_transition_threshold,
        gripper_indices=_parse_indices(args.gripper_indices),
    )
    episodes = load_episodes(dataset)
    report = filter_windows(
        episodes,
        window_config=window_config,
        analysis_config=analysis_config,
        episode_quality=manifest_quality or None,
        quality_threshold=quality_threshold,
    )
    selected_rows = [_window_csv_row(window, manifest_quality) for window in report["selected_windows"]]
    if args.output:
        _write_output(args.output, report, manifest_quality or None)
    if args.train_output:
        _write_csv(args.train_output, [row for row in selected_rows if row.get("split") == "train"])
    if args.val_output:
        _write_csv(args.val_output, [row for row in selected_rows if row.get("split") == "validation"])
    if args.test_output:
        _write_csv(args.test_output, [row for row in selected_rows if row.get("split") == "test"])
    if args.quality_csv:
        _write_quality_csv(args.quality_csv, report, manifest_quality or None)
    strategy_reports = None
    if args.report:
        strategy_reports = compare_strategies(
            episodes,
            window_config=window_config,
            analysis_config=analysis_config,
            episode_quality=manifest_quality or None,
            quality_threshold=quality_threshold,
        )
        _write_report(args.report, report, strategy_reports=strategy_reports)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
