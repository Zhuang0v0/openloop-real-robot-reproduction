"""Focused regression tests for Classify Objects quality/window filtering."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tools.analyze_classify_actions import (
    AnalysisConfig,
    Episode,
    analyze_episode,
    load_episodes,
    robust_upper_threshold,
)
from tools.filter_classify_windows import (
    WindowConfig,
    filter_windows,
    iter_window_slices,
    split_episodes,
)


def _episode(episode_id: int, length: int = 12, *, gap: bool = False, spike: bool = False) -> Episode:
    time = np.arange(length, dtype=np.float64) / 10.0
    state = np.zeros((length, 16), dtype=np.float64)
    action = np.zeros((length, 16), dtype=np.float64)
    action[:, 0] = np.linspace(0.0, 0.5, length)
    action[:, 7] = np.where(np.arange(length) < length // 2, 0.0, 1.0)
    if spike:
        action[length // 2, 0] += 20.0
    frame_index = np.arange(length, dtype=np.int64)
    if gap and length > 6:
        frame_index[6:] += 2
    return Episode(episode_id, state, action, timestamps=time, frame_index=frame_index)


def test_robust_mad_ignores_single_large_spike() -> None:
    values = np.array([1.0] * 20 + [100.0])
    threshold = robust_upper_threshold(values, mad_sigma=6.0)
    assert threshold < 100.0


def test_derivative_metrics_and_alignment_are_reported() -> None:
    report = analyze_episode(_episode(3), config=AnalysisConfig(fps=10.0))
    assert report["episode_id"] == 3
    assert report["velocity"]["p95"] > 0
    assert report["acceleration"]["p95"] >= 0
    assert report["jerk"]["p95"] >= 0
    assert "spike_rate" in report
    assert "pause_fraction" in report
    assert report["alignment"]["ok"] is True
    assert report["gripper"]["transition_count"] >= 1


def test_windows_never_cross_a_frame_gap() -> None:
    episode = _episode(0, length=12, gap=True)
    slices = list(iter_window_slices(episode, window_size=4, stride=4))
    # The gap after row 5 splits [0, 6) and [6, 12); no slice contains both.
    assert slices == [(0, 4), (6, 10)]


def test_rejected_windows_are_not_stitched_into_new_windows() -> None:
    clean = _episode(0, length=16)
    bad = _episode(1, length=16, spike=True)
    config = WindowConfig(window_size=4, stride=4, strategy="Combined", train_fraction=0.5, validation_fraction=0.0, test_fraction=0.5)
    report = filter_windows([clean, bad], window_config=config, analysis_config=AnalysisConfig(fps=10.0))
    windows = report["windows"]
    assert all(window["episode_id"] in {0, 1} for window in windows)
    assert all(len(window["source_rows"]) == 4 for window in windows)
    # Every manifest window remains a source-local contiguous range.  There is
    # no synthetic window made from the tail of one window and head of another.
    for window in windows:
        rows = window["source_rows"]
        assert rows == list(range(rows[0], rows[-1] + 1))


def test_four_strategies_have_stable_names_and_baseline_keeps_candidates() -> None:
    episode = _episode(0, length=8)
    reports = {
        strategy: filter_windows(
            [episode],
            window_config=WindowConfig(window_size=4, stride=4, strategy=strategy),
            analysis_config=AnalysisConfig(fps=10.0),
        )
        for strategy in ("Baseline", "Quality", "Action", "Combined")
    }
    assert set(reports) == {"Baseline", "Quality", "Action", "Combined"}
    assert reports["Baseline"]["summary"]["selected_windows"] == reports["Baseline"]["summary"]["candidate_windows"]
    for strategy, report in reports.items():
        assert report["strategy"] == strategy


def test_episode_split_never_leaks_episode() -> None:
    assignment = split_episodes(range(10), train_fraction=0.6, validation_fraction=0.2, test_fraction=0.2, seed=7)
    assert set(assignment) == set(range(10))
    assert set(assignment.values()) == {"train", "validation", "test"}


def test_load_sharded_and_per_episode_parquet(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    frame_root = root / "data" / "chunk-000"
    frame_root.mkdir(parents=True)
    table = pa.table(
        {
            "episode_index": [0, 0, 1, 1],
            "frame_index": [0, 1, 0, 1],
            "timestamp": [0.0, 0.1, 0.0, 0.1],
            "observation.state": [[0.0] * 2, [0.1] * 2, [1.0] * 2, [1.1] * 2],
            "action": [[0.0] * 2, [0.1] * 2, [1.0] * 2, [1.1] * 2],
        }
    )
    pq.write_table(table, frame_root / "file-000.parquet")
    episodes = load_episodes(root)
    assert [episode.episode_id for episode in episodes] == [0, 1]
    assert [len(episode.action) for episode in episodes] == [2, 2]
    assert episodes[0].row_indices.tolist() == [0, 1]


def test_report_is_json_serializable(tmp_path: Path) -> None:
    report = filter_windows(
        [_episode(0, length=8)],
        window_config=WindowConfig(window_size=4, stride=4),
        analysis_config=AnalysisConfig(fps=10.0),
    )
    output = tmp_path / "windows.json"
    output.write_text(json.dumps(report, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)))
    assert output.stat().st_size > 0
