from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from adapters.efficient_wam.lerobot_to_robotwin_clean import _episode_records


def _write_shard(path: Path, episode_index: list[int], frame_index: list[int], offset: float) -> None:
    rows = len(episode_index)
    state = [[offset + row + 0.1 * col for col in range(16)] for row in range(rows)]
    action = [[offset + row + 0.2 * col for col in range(16)] for row in range(rows)]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": episode_index,
                "frame_index": frame_index,
                "observation.state": state,
                "action": action,
            }
        ),
        path,
    )


def test_episode_records_groups_v3_shards_and_restores_frame_order(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    # Episode 0 is split across two files and deliberately discovered out of
    # chronological row order. Episode 1 shares the first shard.
    _write_shard(
        root / "data" / "chunk-001" / "file-000.parquet",
        [0, 1],
        [2, 0],
        20.0,
    )
    _write_shard(
        root / "data" / "chunk-000" / "file-000.parquet",
        [0, 1],
        [1, 1],
        10.0,
    )
    records = _episode_records(root)

    assert sorted(records) == [0, 1]
    assert records[0]["action"].shape == (2, 16)
    assert records[1]["state"].shape == (2, 16)
    np.testing.assert_array_equal(records[0]["frame_index"], np.array([1, 2]))


def test_episode_records_keeps_per_episode_filename_fallback(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_shard(
        root / "data" / "episode_000007.parquet",
        [7, 7],
        [0, 1],
        0.0,
    )
    # Remove episode_index to exercise the legacy per-episode layout.
    table = pq.read_table(root / "data" / "episode_000007.parquet").drop(["episode_index"])
    pq.write_table(table, root / "data" / "episode_000007.parquet")
    records = _episode_records(root)
    assert list(records) == [7]
    assert records[7]["state"].shape == (2, 16)
