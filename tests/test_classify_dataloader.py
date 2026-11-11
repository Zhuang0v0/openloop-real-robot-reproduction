from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tools.classify_common import FIXED_PROMPT
from tools.prepare_classify_data import PreparationError, prepare_classify_data, split_episode_ids


def test_manifest_has_one_fixed_prompt(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    fields = ["episode_id", "group", "prompt", "match_status"]
    prompt = FIXED_PROMPT
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for episode in range(4):
            writer.writerow({"episode_id": episode, "group": f"C{episode + 1}", "prompt": prompt, "match_status": "matched"})
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["prompt"] for row in rows} == {prompt}
    assert {row["match_status"] for row in rows} == {"matched"}


def test_episode_split_is_disjoint() -> None:
    train = {0, 2, 4}
    val = {1, 3}
    assert train.isdisjoint(val)


def _write_manifest(path: Path, episodes: int = 4) -> Path:
    fields = ["episode_id", "group", "prompt", "match_status", "episode_path"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for episode in range(episodes):
            writer.writerow(
                {
                    "episode_id": episode,
                    "group": f"C{episode % 2 + 1}",
                    "prompt": FIXED_PROMPT,
                    "match_status": "matched",
                    "episode_path": "",
                }
            )
    return path


def _write_dataset(root: Path, episodes: int = 4, frames: int = 6) -> Path:
    for episode in range(episodes):
        target = root / "data" / f"episode_{episode:06d}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        state = (np.arange(frames * 16, dtype=np.float32).reshape(frames, 16) + episode).tolist()
        action = (np.arange(frames * 16, dtype=np.float32).reshape(frames, 16) + episode + 1).tolist()
        pq.write_table(
            pa.table(
                {
                    "episode_index": [episode] * frames,
                    "frame_index": list(range(frames)),
                    "timestamp": [index / 30.0 for index in range(frames)],
                    "observation.state": state,
                    "action": action,
                }
            ),
            target,
        )
    return root


def test_prepare_writes_16d_metadata_and_episode_disjoint_manifests(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "matched.csv")
    dataset = _write_dataset(tmp_path / "dataset")
    result = prepare_classify_data(
        manifest,
        dataset_root=dataset,
        output_dir=tmp_path / "prepared",
        episode_count=4,
        seed=17,
        validation_fraction=0.5,
        strategy="baseline",
        require_state_action=True,
    )
    assert result["state_dim"] == result["action_dim"] == 16
    assert result["dimensions"]["checked_episodes"] == 4
    assert result["state_stats"]["rows"] == result["action_stats"]["rows"] == 24
    assert result["split"]["episode_disjoint"] is True
    train = list(csv.DictReader((tmp_path / "prepared" / "train_manifest.csv").open(encoding="utf-8-sig")))
    validation = list(csv.DictReader((tmp_path / "prepared" / "val_manifest.csv").open(encoding="utf-8-sig")))
    assert {int(row["episode_id"]) for row in train}.isdisjoint({int(row["episode_id"]) for row in validation})
    metadata = json.loads((tmp_path / "prepared" / "prepare_metadata.json").read_text(encoding="utf-8"))
    assert metadata["fastwam"]["state_dim"] == 16
    assert metadata["efficient_wam"]["action_dim"] == 16


def test_prepare_filters_selected_windows_without_episode_leak(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "matched.csv", episodes=4)
    windows = {
        "format": "classify_objects_window_filter_v1",
        "strategy": "Combined",
        "windows": [
            {"window_id": "e0w0", "episode_id": 0, "start": 0, "end": 3, "source_rows": [0, 1, 2], "selected": True},
            {"window_id": "e0w1", "episode_id": 0, "start": 3, "end": 6, "source_rows": [3, 4, 5], "selected": False},
            {"window_id": "e1w0", "episode_id": 1, "start": 0, "end": 3, "source_rows": [0, 1, 2], "selected": True},
            {"window_id": "e2w0", "episode_id": 2, "start": 1, "end": 4, "source_rows": [1, 2, 3], "selected": True},
            {"window_id": "e3w0", "episode_id": 3, "start": 2, "end": 5, "source_rows": [2, 3, 4], "selected": True},
        ],
    }
    window_path = tmp_path / "windows.json"
    window_path.write_text(json.dumps(windows), encoding="utf-8")
    result = prepare_classify_data(
        manifest,
        window_manifest=window_path,
        output_dir=tmp_path / "prepared",
        episode_count=4,
        seed=3,
        validation_fraction=0.5,
        strategy="combined",
    )
    train = list(csv.DictReader((tmp_path / "prepared" / "train_manifest.csv").open(encoding="utf-8-sig")))
    validation = list(csv.DictReader((tmp_path / "prepared" / "val_manifest.csv").open(encoding="utf-8-sig")))
    assert all(row["sample_type"] == "window" for row in train + validation)
    assert all(row["window_id"] != "e0w1" for row in train + validation)
    assert {int(row["episode_id"]) for row in train}.isdisjoint({int(row["episode_id"]) for row in validation})
    assert result["windows"]["selected_count"] == 4


def test_prepare_rejects_prompt_drift_and_bad_dimensions(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "matched.csv", episodes=2)
    text = manifest.read_text(encoding="utf-8").replace(FIXED_PROMPT, "wrong prompt", 1)
    manifest.write_text(text, encoding="utf-8")
    try:
        prepare_classify_data(manifest, output_dir=tmp_path / "prompt", episode_count=2)
    except PreparationError as exc:
        assert "prompt" in str(exc)
    else:  # pragma: no cover - assertion form keeps the failure informative
        raise AssertionError("prompt drift should fail preparation")

    bad_manifest = tmp_path / "bad_dims.csv"
    with bad_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode_id", "prompt", "match_status", "state_dim", "action_dim"])
        writer.writeheader()
        writer.writerow({"episode_id": 0, "prompt": FIXED_PROMPT, "match_status": "matched", "state_dim": 14, "action_dim": 16})
    try:
        prepare_classify_data(bad_manifest, output_dir=tmp_path / "dims", episode_count=1)
    except PreparationError as exc:
        assert "16D" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("bad dimensions should fail preparation")


def test_split_seed_is_repeatable_and_changes_assignment() -> None:
    first = split_episode_ids(range(12), validation_fraction=0.25, seed=11)
    repeat = split_episode_ids(range(12), validation_fraction=0.25, seed=11)
    other = split_episode_ids(range(12), validation_fraction=0.25, seed=12)
    assert first == repeat
    assert set(first) == set(other) == set(range(12))
    assert {episode for episode, split in first.items() if split == "train"}.isdisjoint(
        {episode for episode, split in first.items() if split == "validation"}
    )
    assert first != other
