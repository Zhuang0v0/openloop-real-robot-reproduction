#!/usr/bin/env python3
"""Analyze action quality in a LeRobot/Parquet recording.

The module is intentionally independent from a training implementation.  It
loads both the usual LeRobot v3 sharded parquet layout and the per-episode
exports used by this repository, then computes motion, pause, gripper, and
state/action alignment diagnostics.  The public functions are useful from
Python as well as from the command line::

    episodes = load_episodes("data/nero_classify_objects")
    report = analyze_dataset(episodes, config=AnalysisConfig(fps=30))

All thresholds are expressed in the input value unit unless an explicit unit
conversion is requested.  Robust thresholds use the median and the normal
consistent MAD (``1.4826 * median(abs(x - median(x)))``).  Constant signals
use a small numerical floor rather than a standard-deviation fallback, so a
single excursion cannot inflate its own robust threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence

import numpy as np

try:  # pyarrow is optional when the module is imported for array-only use.
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - exercised only in minimal installs
    pq = None  # type: ignore[assignment]


EPISODE_RE = re.compile(r"episode[_-]?(\d+)", re.IGNORECASE)
_MISSING = object()


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


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _as_1d(values: Any, *, dtype: Any = np.float64) -> np.ndarray:
    """Convert an Arrow/pandas/list column to a one-dimensional array."""

    if values is None:
        return np.empty((0,), dtype=dtype)
    result = np.asarray(values, dtype=dtype)
    return result.reshape(-1)


def _as_matrix(values: Any, *, dtype: Any = np.float64) -> np.ndarray:
    """Convert nested list/Arrow values to ``[T, D]``.

    A scalar column is represented as one-dimensional data and promoted to a
    single feature.  Empty input is represented as ``(0, 0)``.
    """

    if values is None:
        return np.empty((0, 0), dtype=dtype)
    if isinstance(values, np.ndarray) and values.dtype != object:
        result = np.asarray(values, dtype=dtype)
    else:
        try:
            result = np.asarray(values, dtype=dtype)
        except (TypeError, ValueError):
            # Arrow can return a mixture of fixed-size and null list values.
            rows: list[np.ndarray] = []
            for value in values:
                if value is None:
                    rows.append(np.empty((0,), dtype=dtype))
                else:
                    rows.append(np.asarray(value, dtype=dtype).reshape(-1))
            width = max((row.size for row in rows), default=0)
            result = np.full((len(rows), width), np.nan, dtype=dtype)
            for row_index, row in enumerate(rows):
                result[row_index, : row.size] = row
    if result.size == 0:
        return np.empty((0, 0), dtype=dtype)
    if result.ndim == 1:
        return result.reshape(-1, 1)
    if result.ndim > 2:
        return result.reshape(result.shape[0], -1)
    return result


def _episode_id_from_path(path: Path, fallback: int) -> int:
    match = EPISODE_RE.search(path.stem)
    return int(match.group(1)) if match else int(fallback)


def _column_names(table: Any) -> list[str]:
    return [str(name) for name in getattr(table, "column_names", [])]


def _find_column(names: Sequence[str], requested: str | None, aliases: Sequence[str]) -> str | None:
    if requested and requested in names:
        return requested
    for candidate in aliases:
        if candidate in names:
            return candidate
    return None


def _flattened_columns(names: Sequence[str], prefixes: Sequence[str]) -> list[str]:
    """Find columns such as ``action.0`` or ``observation.state_15``."""

    found: list[tuple[int, str]] = []
    for name in names:
        lower = name.lower()
        for prefix in prefixes:
            p = prefix.lower()
            if lower.startswith(p) and lower[len(p) :].lstrip("._").isdigit():
                suffix = lower[len(p) :].lstrip("._")
                found.append((int(suffix), name))
                break
    return [name for _, name in sorted(found)]


def _table_column(table: Any, name: str) -> list[Any]:
    return table[name].to_pylist() if hasattr(table[name], "to_pylist") else list(table[name])


@dataclass
class Episode:
    """A single episode and its source row mapping.

    ``row_indices`` are row offsets in the source parquet.  Keeping them is
    important for window filtering: selected windows can point back to source
    rows without concatenating frames from separate windows.
    """

    episode_id: int
    state: np.ndarray
    action: np.ndarray
    timestamps: np.ndarray | None = None
    frame_index: np.ndarray | None = None
    row_indices: np.ndarray | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.state = _as_matrix(self.state)
        self.action = _as_matrix(self.action)
        n = max(len(self.state), len(self.action))
        if self.timestamps is not None:
            self.timestamps = _as_1d(self.timestamps)
        if self.frame_index is not None:
            self.frame_index = _as_1d(self.frame_index, dtype=np.int64)
        if self.row_indices is None:
            self.row_indices = np.arange(n, dtype=np.int64)
        else:
            self.row_indices = _as_1d(self.row_indices, dtype=np.int64)

    @property
    def length(self) -> int:
        row_count = 0 if self.row_indices is None else int(len(self.row_indices))
        return max(int(len(self.state)), int(len(self.action)), row_count)

    def slice(self, start: int, end: int) -> "Episode":
        """Return a view-like episode segment without changing source indices."""

        metadata = dict(self.metadata)
        if isinstance(metadata.get("row_sources"), (list, tuple, np.ndarray)):
            metadata["row_sources"] = list(metadata["row_sources"])[start:end]
        return Episode(
            episode_id=self.episode_id,
            state=self.state[start:end],
            action=self.action[start:end],
            timestamps=None if self.timestamps is None else self.timestamps[start:end],
            frame_index=None if self.frame_index is None else self.frame_index[start:end],
            row_indices=self.row_indices[start:end],
            source=self.source,
            metadata=metadata,
        )


@dataclass
class AnalysisConfig:
    """Numerical conventions and optional absolute quality limits."""

    fps: float = 30.0
    time_unit: str = "s"
    value_unit: str = "native"
    threshold_unit: str | None = None
    mad_sigma: float = 6.0
    min_scale: float = 1e-9
    velocity_threshold: float | None = None
    acceleration_threshold: float | None = None
    jerk_threshold: float | None = None
    spike_z_threshold: float = 6.0
    spike_rate_threshold: float | None = None
    pause_velocity_threshold: float = 1e-3
    min_pause_frames: int = 3
    max_pause_fraction: float | None = None
    max_pause_frames: int | None = None
    max_gripper_transitions: int | None = None
    gripper_transition_threshold: float = 1e-3
    gripper_indices: tuple[int, ...] = (7, 15)
    alignment_rmse_threshold: float | None = None
    alignment_max_error_threshold: float | None = None
    expected_state_dim: int | None = None
    expected_action_dim: int | None = None

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.mad_sigma <= 0:
            raise ValueError("mad_sigma must be positive")
        if self.min_pause_frames < 1:
            raise ValueError("min_pause_frames must be at least one")
        self.gripper_indices = tuple(int(index) for index in self.gripper_indices)


@dataclass
class RobustStats:
    median: float
    mad: float
    scale: float
    count: int

    @property
    def upper(self) -> float:
        return self.median + self.scale

    def as_dict(self) -> dict[str, float | int]:
        return {
            "median": float(self.median),
            "mad": float(self.mad),
            "scale": float(self.scale),
            "count": int(self.count),
        }


def robust_stats(values: Sequence[float] | np.ndarray, *, min_scale: float = 1e-9) -> RobustStats:
    """Return median, MAD, and a normal-consistent robust scale."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return RobustStats(float("nan"), float("nan"), float(min_scale), 0)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    scale = 1.4826 * mad
    if not math.isfinite(scale):
        scale = float(min_scale)
    elif scale < min_scale:
        # For a constant (or overwhelmingly repeated) signal, using the
        # standard deviation would let one very large spike inflate its own
        # threshold.  Keep the MAD semantics and use only the numerical floor.
        scale = float(min_scale)
    return RobustStats(median, mad, scale, int(array.size))


def robust_zscore(values: Sequence[float] | np.ndarray, *, min_scale: float = 1e-9) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    stats = robust_stats(array, min_scale=min_scale)
    if stats.count == 0:
        return np.full(array.shape, np.nan, dtype=np.float64)
    return (array - stats.median) / stats.scale


def robust_upper_threshold(
    values: Sequence[float] | np.ndarray,
    *,
    mad_sigma: float = 6.0,
    min_scale: float = 1e-9,
) -> float:
    """Compute ``median + mad_sigma * robust_scale`` for a signal."""

    stats = robust_stats(values, min_scale=min_scale)
    if stats.count == 0:
        return float("nan")
    return float(stats.median + mad_sigma * stats.scale)


_UNIT_TO_BASE = {
    "native": 1.0,
    "unit": 1.0,
    "rad": 1.0,
    "radian": 1.0,
    "radians": 1.0,
    "deg": math.pi / 180.0,
    "degree": math.pi / 180.0,
    "degrees": math.pi / 180.0,
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "mm": 1e-3,
    "millimeter": 1e-3,
    "millimeters": 1e-3,
    "cm": 1e-2,
    "centimeter": 1e-2,
    "centimeters": 1e-2,
}


def unit_factor(unit: str | None) -> float:
    if unit is None:
        return 1.0
    key = str(unit).strip().lower().replace("²", "^2").replace(" ", "")
    # Derivative thresholds are commonly written as ``deg/s``, ``rad/s2`` or
    # ``mm per second``.  The time denominator is already accounted for by
    # the finite-difference clock, so unit_factor converts the position part;
    # accepting the suffix here keeps CLI/configuration units ergonomic.
    key = key.replace("persecond", "").replace("/second", "").replace("/sec", "")
    key = key.replace("/s^2", "").replace("/s2", "").replace("/s", "")
    key = key.replace("pers", "").replace("_per_second", "").strip("_")
    if key not in _UNIT_TO_BASE:
        raise ValueError(f"unsupported value unit {unit!r}")
    return _UNIT_TO_BASE[key]


def convert_value(value: float, from_unit: str | None, to_unit: str | None) -> float:
    """Convert a scalar threshold between common position units."""

    if not from_unit or not to_unit or from_unit == to_unit:
        return float(value)
    return float(value) * unit_factor(from_unit) / unit_factor(to_unit)


def _time_seconds(timestamps: np.ndarray | None, length: int, config: AnalysisConfig) -> np.ndarray:
    if length <= 0:
        return np.empty((0,), dtype=np.float64)
    if timestamps is None or len(timestamps) != length:
        return np.arange(length, dtype=np.float64) / float(config.fps)
    values = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if values.size != length or not np.isfinite(values).all():
        return np.arange(length, dtype=np.float64) / float(config.fps)
    # LeRobot timestamps are seconds.  Accommodate explicit millisecond or
    # frame units without forcing callers to rewrite their parquet files.
    unit = config.time_unit.lower()
    if unit in {"ms", "millisecond", "milliseconds"}:
        values = values / 1000.0
    elif unit in {"frame", "frames"}:
        values = values / float(config.fps)
    values = values - values[0]
    # Repeated/non-monotonic timestamps cannot define a derivative.  Fall
    # back to a regular clock while retaining the episode length.
    if values.size > 1 and np.any(np.diff(values) <= 0):
        return np.arange(length, dtype=np.float64) / float(config.fps)
    return values


def derivative(values: np.ndarray, timestamps: np.ndarray | None, *, order: int, config: AnalysisConfig) -> np.ndarray:
    """Finite differences with the same number of rows as the input."""

    array = _as_matrix(values)
    if array.shape[0] == 0:
        return np.empty_like(array, dtype=np.float64)
    if array.shape[0] == 1:
        return np.zeros_like(array, dtype=np.float64)
    times = _time_seconds(timestamps, array.shape[0], config)
    result = array.astype(np.float64, copy=False)
    for _ in range(max(0, int(order))):
        # np.gradient handles the endpoints and does not drop a frame, which
        # keeps frame/window indices aligned with the source parquet.
        result = np.gradient(result, times, axis=0, edge_order=1)
    return np.asarray(result, dtype=np.float64)


def _norm(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        return np.abs(array)
    if array.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    return np.linalg.norm(array, axis=1)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"p50": float("nan"), "p95": float("nan"), "p99": float("nan"), "max": float("nan")}
    return {
        "p50": float(np.quantile(finite, 0.50)),
        "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
    }


def _run_lengths(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        elif not active and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _resolve_limit(
    values: np.ndarray,
    configured: float | None,
    config: AnalysisConfig,
    *,
    source_unit: str | None = None,
) -> tuple[float, str]:
    if configured is not None:
        converted = convert_value(configured, config.threshold_unit, source_unit or config.value_unit)
        return float(converted), "absolute"
    return robust_upper_threshold(values, mad_sigma=config.mad_sigma, min_scale=config.min_scale), "mad"


def _gripper_metrics(action: np.ndarray, state: np.ndarray, config: AnalysisConfig) -> dict[str, Any]:
    indices = [index for index in config.gripper_indices if 0 <= index < action.shape[1]]
    if not indices or action.shape[0] == 0:
        return {
            "indices": indices,
            "transition_count": 0,
            "transition_rate": 0.0,
            "movement_fraction": 0.0,
            "saturation_fraction": 0.0,
            "state_action_rmse": float("nan"),
            "state_action_max_error": float("nan"),
        }
    values = action[:, indices]
    delta = np.abs(np.diff(values, axis=0)) if len(values) > 1 else np.empty((0, len(indices)))
    transitions = (delta > config.gripper_transition_threshold).any(axis=1)
    # Saturation is estimated from each episode's observed range.  This works
    # for both [0, 1] and robot-specific joint limits without hard-coded units.
    low = np.nanmin(values, axis=0)
    high = np.nanmax(values, axis=0)
    span = np.maximum(high - low, config.min_scale)
    saturation = ((values - low <= 0.01 * span) | (high - values <= 0.01 * span)).any(axis=1)
    if state.shape[0] and state.shape[1] > max(indices):
        aligned = state[: len(values), indices] - values[: len(state), :]
        aligned = aligned[np.isfinite(aligned)]
    else:
        aligned = np.empty((0,), dtype=np.float64)
    return {
        "indices": indices,
        "transition_count": int(np.count_nonzero(transitions)),
        "transition_rate": float(np.mean(transitions)) if transitions.size else 0.0,
        "movement_fraction": float(np.mean(np.any(delta > config.gripper_transition_threshold, axis=1))) if delta.size else 0.0,
        "saturation_fraction": float(np.mean(saturation)) if saturation.size else 0.0,
        "state_action_rmse": float(np.sqrt(np.mean(aligned**2))) if aligned.size else float("nan"),
        "state_action_max_error": float(np.max(np.abs(aligned))) if aligned.size else float("nan"),
    }


def _alignment_metrics(state: np.ndarray, action: np.ndarray) -> dict[str, Any]:
    lengths_equal = len(state) == len(action)
    dims_equal = state.ndim == action.ndim == 2 and state.shape[1] == action.shape[1]
    n = min(len(state), len(action))
    if n == 0 or not dims_equal:
        return {
            "lengths_equal": bool(lengths_equal),
            "dims_equal": bool(dims_equal),
            "aligned_rows": int(n),
            "rmse": float("nan"),
            "mean_abs_error": float("nan"),
            "max_abs_error": float("nan"),
            "finite": bool(np.isfinite(state).all() and np.isfinite(action).all()),
            "ok": False,
        }
    difference = np.asarray(state[:n] - action[:n], dtype=np.float64)
    finite = bool(np.isfinite(state).all() and np.isfinite(action).all() and np.isfinite(difference).all())
    values = np.abs(difference[np.isfinite(difference)])
    return {
        "lengths_equal": bool(lengths_equal),
        "dims_equal": bool(dims_equal),
        "aligned_rows": int(n),
        "rmse": float(np.sqrt(np.mean(difference**2))) if finite else float("nan"),
        "mean_abs_error": float(np.mean(values)) if values.size else float("nan"),
        "max_abs_error": float(np.max(values)) if values.size else float("nan"),
        "finite": finite,
        # The numerical limit is applied by analyze_episode after config is
        # known; this field only captures structural alignment.
        "ok": bool(lengths_equal and dims_equal and finite),
    }


def _quality_flags(
    metrics: Mapping[str, Any],
    config: AnalysisConfig,
    *,
    limits: Mapping[str, float],
) -> dict[str, bool]:
    finite = bool(metrics.get("finite", False))
    flags = {
        "finite": finite,
        "alignment": bool(metrics.get("alignment", {}).get("ok", False)),
        "velocity": bool(metrics.get("velocity", {}).get("p95", np.inf) <= limits.get("velocity", np.inf)),
        "acceleration": bool(metrics.get("acceleration", {}).get("p95", np.inf) <= limits.get("acceleration", np.inf)),
        "jerk": bool(metrics.get("jerk", {}).get("p95", np.inf) <= limits.get("jerk", np.inf)),
        "spikes": bool(metrics.get("spike_rate", 1.0) <= limits.get("spike_rate", np.inf)),
        "pauses": bool(metrics.get("pause_fraction", 1.0) <= limits.get("pause_fraction", np.inf)
                       and metrics.get("longest_pause_frames", np.inf) <= limits.get("longest_pause_frames", np.inf)),
        "gripper": bool(metrics.get("gripper", {}).get("transition_count", 0) <= limits.get("gripper_transitions", np.inf)),
        "state_action": bool(metrics.get("alignment", {}).get("rmse", np.inf) <= limits.get("alignment_rmse", np.inf)
                              and metrics.get("alignment", {}).get("max_abs_error", np.inf) <= limits.get("alignment_max_error", np.inf)),
    }
    return flags


def analyze_episode(
    episode: Episode | Mapping[str, Any] | np.ndarray,
    action: np.ndarray | None = None,
    *,
    timestamps: np.ndarray | None = None,
    config: AnalysisConfig | None = None,
) -> dict[str, Any]:
    """Compute quality diagnostics for one episode.

    ``episode`` can be an :class:`Episode`, a mapping with ``state`` and
    ``action`` keys, or a state matrix when ``action`` is passed separately.
    The returned dictionary is JSON serializable (apart from no custom types).
    """

    config = config or AnalysisConfig()
    if isinstance(episode, Episode):
        item = episode
    elif isinstance(episode, Mapping):
        item = Episode(
            episode_id=int(episode.get("episode_id", episode.get("episode_index", 0))),
            state=np.asarray(episode.get("state", episode.get("observation.state", []))),
            action=np.asarray(episode.get("action", [])),
            timestamps=episode.get("timestamps", episode.get("timestamp")),
            frame_index=episode.get("frame_index"),
            row_indices=episode.get("row_indices"),
            source=episode.get("source"),
        )
    else:
        if action is None:
            raise TypeError("action is required when episode is an array")
        item = Episode(0, np.asarray(episode), np.asarray(action), timestamps=timestamps)

    state = _as_matrix(item.state)
    act = _as_matrix(item.action)
    n = max(len(state), len(act))
    finite = bool(np.isfinite(state).all() and np.isfinite(act).all())
    state_dim = int(state.shape[1]) if state.ndim == 2 and state.size else 0
    action_dim = int(act.shape[1]) if act.ndim == 2 and act.size else 0
    time_values = item.timestamps if item.timestamps is not None and len(item.timestamps) == len(act) else timestamps

    velocity_values = derivative(act, time_values, order=1, config=config)
    acceleration_values = derivative(act, time_values, order=2, config=config)
    jerk_values = derivative(act, time_values, order=3, config=config)
    state_velocity = derivative(state, item.timestamps, order=1, config=config)
    speed = _norm(velocity_values)
    acceleration = _norm(acceleration_values)
    jerk = _norm(jerk_values)

    velocity_limit, velocity_limit_mode = _resolve_limit(speed, config.velocity_threshold, config, source_unit=config.value_unit)
    acceleration_limit, acceleration_limit_mode = _resolve_limit(acceleration, config.acceleration_threshold, config, source_unit=config.value_unit)
    jerk_limit, jerk_limit_mode = _resolve_limit(jerk, config.jerk_threshold, config, source_unit=config.value_unit)

    zscores = robust_zscore(jerk, min_scale=config.min_scale)
    spike_mask = np.abs(zscores) > config.spike_z_threshold
    pause_mask = speed <= config.pause_velocity_threshold
    pause_runs = _run_lengths(pause_mask)
    long_pause_runs = [run for run in pause_runs if run[1] - run[0] >= config.min_pause_frames]
    alignment = _alignment_metrics(state, act)
    gripper = _gripper_metrics(act, state, config)

    # A default robust limit for a signal is episode-local.  Dataset-level
    # filtering can replace it with global MAD limits, while direct use of
    # analyze_episode remains deterministic and useful.
    limits: dict[str, float] = {
        "velocity": velocity_limit,
        "acceleration": acceleration_limit,
        "jerk": jerk_limit,
        "spike_rate": config.spike_rate_threshold if config.spike_rate_threshold is not None else 1.0,
        "pause_fraction": config.max_pause_fraction if config.max_pause_fraction is not None else 1.0,
        "longest_pause_frames": config.max_pause_frames if config.max_pause_frames is not None else float("inf"),
        "gripper_transitions": config.max_gripper_transitions if config.max_gripper_transitions is not None else float("inf"),
        "alignment_rmse": config.alignment_rmse_threshold if config.alignment_rmse_threshold is not None else float("inf"),
        "alignment_max_error": config.alignment_max_error_threshold if config.alignment_max_error_threshold is not None else float("inf"),
    }
    flags = _quality_flags(
        {
            "finite": finite,
            "alignment": alignment,
            "velocity": _quantiles(speed),
            "acceleration": _quantiles(acceleration),
            "jerk": _quantiles(jerk),
            "spike_rate": float(np.mean(spike_mask)) if spike_mask.size else 0.0,
            "pause_fraction": float(np.mean(pause_mask)) if pause_mask.size else 0.0,
            "longest_pause_frames": max((end - start for start, end in pause_runs), default=0),
            "gripper": gripper,
        },
        config,
        limits=limits,
    )
    quality_components = ("finite", "velocity", "acceleration", "jerk", "spikes", "pauses")
    action_components = ("alignment", "gripper", "state_action")
    quality_score = float(np.mean([flags[name] for name in quality_components]))
    action_score = float(np.mean([flags[name] for name in action_components]))

    result: dict[str, Any] = {
        "episode_id": int(item.episode_id),
        "source": item.source,
        "frames": int(n),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "finite": finite,
        "alignment": alignment,
        "velocity": _quantiles(speed),
        "acceleration": _quantiles(acceleration),
        "jerk": _quantiles(jerk),
        "velocity_values": speed,
        "acceleration_values": acceleration,
        "jerk_values": jerk,
        "state_velocity": _quantiles(_norm(state_velocity)),
        "spike_count": int(np.count_nonzero(spike_mask)),
        "spike_rate": float(np.mean(spike_mask)) if spike_mask.size else 0.0,
        "spike_indices": np.flatnonzero(spike_mask).astype(int),
        "spike_threshold_z": float(config.spike_z_threshold),
        "pause_fraction": float(np.mean(pause_mask)) if pause_mask.size else 0.0,
        "pause_count": int(len(pause_runs)),
        "long_pause_count": int(len(long_pause_runs)),
        "longest_pause_frames": int(max((end - start for start, end in pause_runs), default=0)),
        "pause_runs": [[int(start), int(end)] for start, end in pause_runs],
        "gripper": gripper,
        "limits": {
            **limits,
            "velocity_mode": velocity_limit_mode,
            "acceleration_mode": acceleration_limit_mode,
            "jerk_mode": jerk_limit_mode,
        },
        "quality_flags": flags,
        "quality_score": quality_score,
        "action_score": action_score,
        "config": asdict(config),
    }
    # Keep source indexing available to window code without forcing it into
    # reports written by users who only need aggregate numbers.
    if item.frame_index is not None:
        result["frame_index"] = np.asarray(item.frame_index, dtype=np.int64)
    if item.row_indices is not None:
        result["row_indices"] = np.asarray(item.row_indices, dtype=np.int64)
    return result


def discover_parquet_files(source: str | Path | Sequence[str | Path]) -> list[Path]:
    """Discover frame parquet files below a LeRobot root or explicit paths."""

    if isinstance(source, (str, Path)):
        paths = [Path(source)]
    else:
        paths = [Path(path) for path in source]
    found: list[Path] = []
    for path in paths:
        path = path.expanduser()
        if path.is_file():
            if path.suffix.lower() == ".parquet":
                found.append(path)
            continue
        if not path.exists():
            raise FileNotFoundError(path)
        data_root = path / "data"
        search_root = data_root if data_root.exists() else path
        candidates = sorted(search_root.rglob("*.parquet"))
        for candidate in candidates:
            if candidate.name.lower() in {"tasks.parquet", "episodes.parquet"}:
                continue
            # Metadata parquet files are not frame streams.  A cheap schema
            # check avoids accidentally loading them as an empty episode.
            if pq is not None:
                try:
                    names = _column_names(pq.read_schema(candidate))
                except Exception:
                    names = []
                has_action = any(
                    name.lower() in {"action", "actions", "observation.action", "observation_action"}
                    or name.lower().startswith(("action.", "action_", "actions.", "actions_", "observation.action.", "observation_action_"))
                    for name in names
                )
                if names and not has_action:
                    continue
            found.append(candidate)
    unique = sorted({path.resolve() for path in found})
    if not unique:
        raise FileNotFoundError(f"no frame parquet files found below {source}")
    return unique


def _read_table(path: Path) -> Any:
    if pq is None:  # pragma: no cover
        raise RuntimeError("pyarrow is required to read parquet data")
    return pq.read_table(path)


def _extract_matrix(table: Any, requested: str, aliases: Sequence[str], prefixes: Sequence[str]) -> np.ndarray:
    names = _column_names(table)
    column = _find_column(names, requested, aliases)
    if column:
        return _as_matrix(_table_column(table, column))
    flattened = _flattened_columns(names, prefixes)
    if flattened:
        columns = [_table_column(table, name) for name in flattened]
        return np.column_stack([_as_1d(values) for values in columns]).astype(np.float64)
    raise KeyError(f"parquet is missing {requested!r}; available columns: {names}")


def _concat_matrices(parts: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate episode shards, padding uncommon width mismatches with NaN."""

    nonempty = [np.asarray(part, dtype=np.float64) for part in parts if np.asarray(part).ndim == 2]
    if not nonempty:
        return np.empty((0, 0), dtype=np.float64)
    width = max((part.shape[1] for part in nonempty), default=0)
    padded: list[np.ndarray] = []
    for part in nonempty:
        if part.shape[1] == width:
            padded.append(part)
        else:
            target = np.full((part.shape[0], width), np.nan, dtype=np.float64)
            target[:, : part.shape[1]] = part
            padded.append(target)
    return np.concatenate(padded, axis=0) if padded else np.empty((0, width), dtype=np.float64)


def _merge_episode_parts(parts: Sequence[Episode]) -> Episode:
    """Merge shards carrying one episode id while retaining source mappings."""

    if len(parts) == 1:
        return parts[0]
    ordered = list(parts)
    # When global frame indices are available, restore chronological order;
    # otherwise discover_parquet_files' stable path order is the least
    # surprising ordering.
    if all(part.frame_index is not None and len(part.frame_index) for part in ordered):
        ordered.sort(key=lambda part: int(np.asarray(part.frame_index).reshape(-1)[0]))
    state = _concat_matrices([part.state for part in ordered])
    action = _concat_matrices([part.action for part in ordered])
    timestamps_parts = [part.timestamps for part in ordered if part.timestamps is not None]
    frame_parts = [part.frame_index for part in ordered if part.frame_index is not None]
    row_parts = [part.row_indices for part in ordered if part.row_indices is not None]
    timestamps = np.concatenate(timestamps_parts) if len(timestamps_parts) == len(ordered) else None
    frame_index = np.concatenate(frame_parts) if len(frame_parts) == len(ordered) else None
    row_indices = np.concatenate(row_parts) if row_parts else None
    row_sources: list[str] = []
    for part in ordered:
        source_values = part.metadata.get("row_sources")
        if isinstance(source_values, (list, tuple, np.ndarray)):
            row_sources.extend(str(value) for value in source_values)
        else:
            count = len(part.row_indices) if part.row_indices is not None else max(len(part.state), len(part.action))
            row_sources.extend([str(part.source or "") for _ in range(count)])
    unique_sources = list(dict.fromkeys(str(part.source or "") for part in ordered if part.source))
    metadata = {"row_sources": row_sources, "shard_count": len(ordered)}
    return Episode(
        episode_id=ordered[0].episode_id,
        state=state,
        action=action,
        timestamps=timestamps,
        frame_index=frame_index,
        row_indices=row_indices,
        source="|".join(unique_sources) if unique_sources else None,
        metadata=metadata,
    )


def load_episodes(
    source: str | Path | Sequence[str | Path] | Sequence[Episode],
    *,
    state_column: str = "observation.state",
    action_column: str = "action",
    episode_column: str = "episode_index",
    timestamp_column: str = "timestamp",
    frame_column: str = "frame_index",
) -> list[Episode]:
    """Load and group episode rows from parquet files.

    A sharded LeRobot file may contain many episodes; grouping is performed on
    ``episode_index``.  Per-episode files without that column infer the id from
    ``episode_000123.parquet``.  State/action row mappings remain local to each
    source parquet and are never reindexed by concatenating files.
    """

    if isinstance(source, Sequence) and not isinstance(source, (str, Path)) and all(isinstance(x, Episode) for x in source):
        return list(source)  # type: ignore[arg-type]
    files = discover_parquet_files(source)  # type: ignore[arg-type]
    parts_by_episode: dict[int, list[Episode]] = {}
    source_row_offset = 0
    for fallback_index, path in enumerate(files):
        table = _read_table(path)
        names = _column_names(table)
        state = _extract_matrix(
            table,
            state_column,
            ("state", "observation_state", "observation.state"),
            ("observation.state", "observation_state", "state"),
        )
        act = _extract_matrix(
            table,
            action_column,
            ("actions", "action", "observation.action"),
            ("action", "actions", "observation.action"),
        )
        nrows = max(len(state), len(act))
        if episode_column in names:
            episode_values = _as_1d(_table_column(table, episode_column), dtype=np.int64)
            ids = sorted({int(value) for value in episode_values if np.isfinite(value)})
        else:
            episode_values = np.full((nrows,), _episode_id_from_path(path, fallback_index), dtype=np.int64)
            ids = [int(episode_values[0])] if nrows else []
        timestamps = _as_1d(_table_column(table, timestamp_column)) if timestamp_column in names else None
        frame_index = _as_1d(_table_column(table, frame_column), dtype=np.int64) if frame_column in names else None
        for episode_id in ids:
            mask = episode_values == episode_id
            # If a malformed table has one column shorter than episode_index,
            # trim masks to the available rows and preserve the mismatch in the
            # resulting Episode for the alignment report.
            state_mask = mask[: len(state)]
            action_mask = mask[: len(act)]
            row_indices = np.flatnonzero(mask).astype(np.int64)
            part = Episode(
                    episode_id=int(episode_id),
                    state=state[state_mask],
                    action=act[action_mask],
                    timestamps=None if timestamps is None else timestamps[mask[: len(timestamps)]],
                    frame_index=None if frame_index is None else frame_index[mask[: len(frame_index)]],
                    row_indices=row_indices + int(source_row_offset),
                    source=str(path),
                    metadata={"row_sources": [str(path)] * len(row_indices)},
                )
            parts_by_episode.setdefault(int(episode_id), []).append(part)
        timestamp_count = 0 if timestamps is None else len(timestamps)
        frame_count = 0 if frame_index is None else len(frame_index)
        source_row_offset += max(nrows, len(episode_values), timestamp_count, frame_count)
    episodes = [_merge_episode_parts(parts_by_episode[episode_id]) for episode_id in sorted(parts_by_episode)]
    return episodes


def _aggregate_numeric(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def analyze_dataset(
    source: str | Path | Sequence[str | Path] | Sequence[Episode],
    *,
    config: AnalysisConfig | None = None,
    **loader_kwargs: Any,
) -> dict[str, Any]:
    """Analyze every episode and return an aggregate JSON-ready report."""

    config = config or AnalysisConfig()
    episodes = load_episodes(source, **loader_kwargs)
    reports = [analyze_episode(episode, config=config) for episode in episodes]
    # Keep array-valued details in the per-episode report useful to Python
    # callers; the report is JSON-ready through _json_default at write time.
    return {
        "format": "classify_objects_action_quality_v1",
        "config": asdict(config),
        "episodes": reports,
        "summary": {
            "episode_count": len(reports),
            "total_frames": int(sum(report["frames"] for report in reports)),
            "quality_score": _aggregate_numeric(report["quality_score"] for report in reports),
            "action_score": _aggregate_numeric(report["action_score"] for report in reports),
            "spike_rate": _aggregate_numeric(report["spike_rate"] for report in reports),
            "pause_fraction": _aggregate_numeric(report["pause_fraction"] for report in reports),
            "alignment_failures": int(sum(not report["alignment"]["ok"] for report in reports)),
        },
    }


# Compatibility aliases for callers that prefer explicit names.
load_lerobot_episodes = load_episodes
analyze_actions = analyze_dataset
compute_action_quality = analyze_episode


def _parse_indices(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _write_quality_artifacts(
    report: Mapping[str, Any],
    *,
    quality_csv: Path | None = None,
    anomaly_csv: Path | None = None,
    markdown: Path | None = None,
    plot_dir: Path | None = None,
) -> None:
    """Write review-friendly CSV/Markdown/PNG artifacts without changing data."""

    episodes = list(report.get("episodes", []))
    if quality_csv is not None:
        quality_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "episode_id", "source", "frames", "state_dim", "action_dim", "finite",
            "quality_score", "action_score", "velocity_p95", "acceleration_p95",
            "jerk_p95", "spike_count", "spike_rate", "pause_fraction",
            "longest_pause_frames", "gripper_transitions", "alignment_rmse",
            "alignment_max_error",
        ]
        with quality_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for item in episodes:
                writer.writerow({
                    "episode_id": item.get("episode_id", ""),
                    "source": item.get("source", ""),
                    "frames": item.get("frames", ""),
                    "state_dim": item.get("state_dim", ""),
                    "action_dim": item.get("action_dim", ""),
                    "finite": item.get("finite", False),
                    "quality_score": item.get("quality_score", ""),
                    "action_score": item.get("action_score", ""),
                    "velocity_p95": item.get("velocity", {}).get("p95", ""),
                    "acceleration_p95": item.get("acceleration", {}).get("p95", ""),
                    "jerk_p95": item.get("jerk", {}).get("p95", ""),
                    "spike_count": item.get("spike_count", ""),
                    "spike_rate": item.get("spike_rate", ""),
                    "pause_fraction": item.get("pause_fraction", ""),
                    "longest_pause_frames": item.get("longest_pause_frames", ""),
                    "gripper_transitions": item.get("gripper", {}).get("transition_count", ""),
                    "alignment_rmse": item.get("alignment", {}).get("rmse", ""),
                    "alignment_max_error": item.get("alignment", {}).get("max_abs_error", ""),
                })
    if anomaly_csv is not None:
        anomaly_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = ["episode_id", "kind", "start", "end", "length", "details"]
        with anomaly_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for item in episodes:
                episode_id = item.get("episode_id", "")
                for index in item.get("spike_indices", []):
                    writer.writerow({"episode_id": episode_id, "kind": "jerk_spike", "start": index, "end": int(index) + 1, "length": 1, "details": "robust jerk z-score exceeded threshold"})
                for start, end in item.get("pause_runs", []):
                    if int(end) - int(start) >= int(item.get("config", {}).get("min_pause_frames", 3)):
                        writer.writerow({"episode_id": episode_id, "kind": "long_pause", "start": start, "end": end, "length": int(end) - int(start), "details": "low action speed run"})
                if not item.get("finite", False):
                    writer.writerow({"episode_id": episode_id, "kind": "non_finite", "start": "", "end": "", "length": "", "details": "state/action contains NaN or infinity"})
    if markdown is not None:
        markdown.parent.mkdir(parents=True, exist_ok=True)
        summary = report.get("summary", {})
        lines = [
            "# Classify Objects action quality report", "",
            f"- Episodes: `{summary.get('episode_count', len(episodes))}`",
            f"- Total frames: `{summary.get('total_frames', '')}`",
            f"- Alignment failures: `{summary.get('alignment_failures', '')}`", "",
            "| Episode | Frames | Quality score | Action score | Spikes | Pause fraction | Alignment |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ]
        for item in episodes:
            lines.append(
                f"| {item.get('episode_id', '')} | {item.get('frames', '')} | {item.get('quality_score', '')} | "
                f"{item.get('action_score', '')} | {item.get('spike_count', '')} | {item.get('pause_fraction', '')} | "
                f"{item.get('alignment', {}).get('ok', False)} |"
            )
        markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if plot_dir is not None:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError:
            return
        plot_dir.mkdir(parents=True, exist_ok=True)
        for item in episodes:
            episode_id = int(item.get("episode_id", 0))
            velocity = np.asarray(item.get("velocity_values", []), dtype=float)
            acceleration = np.asarray(item.get("acceleration_values", []), dtype=float)
            jerk = np.asarray(item.get("jerk_values", []), dtype=float)
            if not velocity.size and not acceleration.size and not jerk.size:
                continue
            figure, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
            for axis, values, label in zip(axes, (velocity, acceleration, jerk), ("velocity", "acceleration", "jerk")):
                axis.plot(values)
                axis.set_ylabel(label)
                axis.grid(True, alpha=0.25)
            axes[-1].set_xlabel("frame")
            figure.suptitle(f"Classify Objects episode {episode_id}")
            figure.tight_layout()
            figure.savefig(plot_dir / f"episode_{episode_id:06d}_action_quality.png", dpi=120)
            plt.close(figure)


def _build_config(args: argparse.Namespace) -> AnalysisConfig:
    return AnalysisConfig(
        fps=args.fps,
        time_unit=args.time_unit,
        value_unit=args.value_unit,
        threshold_unit=args.threshold_unit,
        mad_sigma=args.mad_sigma,
        velocity_threshold=args.velocity_threshold,
        acceleration_threshold=args.acceleration_threshold,
        jerk_threshold=args.jerk_threshold,
        spike_z_threshold=args.spike_z_threshold,
        spike_rate_threshold=args.spike_rate_threshold,
        pause_velocity_threshold=args.pause_velocity_threshold,
        min_pause_frames=args.min_pause_frames,
        max_pause_fraction=args.max_pause_fraction,
        max_pause_frames=args.max_pause_frames,
        max_gripper_transitions=args.max_gripper_transitions,
        gripper_transition_threshold=args.gripper_transition_threshold,
        gripper_indices=_parse_indices(args.gripper_indices),
        alignment_rmse_threshold=args.alignment_rmse_threshold,
        alignment_max_error_threshold=args.alignment_max_error_threshold,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="LeRobot root or parquet file")
    parser.add_argument("-o", "--output", type=Path, help="write JSON report")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--time-unit", choices=("s", "ms", "frames"), default="s")
    parser.add_argument("--value-unit", default="native", help="input position unit (rad, deg, m, mm, native)")
    parser.add_argument("--threshold-unit", default=None, help="unit used by absolute thresholds")
    parser.add_argument("--mad-sigma", type=float, default=6.0)
    parser.add_argument("--velocity-threshold", type=float)
    parser.add_argument("--acceleration-threshold", type=float)
    parser.add_argument("--jerk-threshold", type=float)
    parser.add_argument("--spike-z-threshold", type=float, default=6.0)
    parser.add_argument("--spike-rate-threshold", type=float)
    parser.add_argument("--pause-velocity-threshold", type=float, default=1e-3)
    parser.add_argument("--min-pause-frames", type=int, default=3)
    parser.add_argument("--max-pause-fraction", type=float)
    parser.add_argument("--max-pause-frames", type=int)
    parser.add_argument("--max-gripper-transitions", type=int)
    parser.add_argument("--gripper-transition-threshold", type=float, default=1e-3)
    parser.add_argument("--gripper-indices", default="7,15")
    parser.add_argument("--alignment-rmse-threshold", type=float)
    parser.add_argument("--alignment-max-error-threshold", type=float)
    parser.add_argument("--quality-csv", "--output-csv", type=Path, help="per-episode quality CSV")
    parser.add_argument("--anomaly-csv", type=Path, help="spike/pause/non-finite segment CSV")
    parser.add_argument("--report", type=Path, help="Markdown quality report")
    parser.add_argument("--plot-dir", type=Path, help="optional per-episode PNG directory")
    args = parser.parse_args(argv)
    config = _build_config(args)
    report = analyze_dataset(args.dataset, config=config)
    payload = json.dumps(report, indent=2, ensure_ascii=False, default=_json_default)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    _write_quality_artifacts(
        report,
        quality_csv=args.quality_csv,
        anomaly_csv=args.anomaly_csv,
        markdown=args.report,
        plot_dir=args.plot_dir,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI smoke tested separately
    raise SystemExit(main())
