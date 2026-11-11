from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).parents[1] / "adapters" / "fastwam" / "manifest_loader.py"
    spec = importlib.util.spec_from_file_location("classify_fastwam_manifest_loader", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_episode_ids_deduplicates_selected_windows(tmp_path):
    module = _load_module()
    manifest = tmp_path / "windows.csv"
    manifest.write_text(
        "episode_id,selected,split\n"
        "3,true,train\n"
        "3,true,train\n"
        "7,false,train\n"
        "9,true,validation\n",
        encoding="utf-8",
    )
    assert module.manifest_episode_ids(manifest, split="train") == (3,)


def test_manifest_episode_ids_reads_json_rows(tmp_path):
    module = _load_module()
    manifest = tmp_path / "episodes.json"
    manifest.write_text(
        '{"rows": [{"episode_index": 0}, {"episode_index": 2, "selected": true}]}',
        encoding="utf-8",
    )
    assert module.manifest_episode_ids(manifest) == (0, 2)


def test_manifest_selection_preserves_window_starts_and_episode_rows(tmp_path):
    module = _load_module()
    manifest = tmp_path / "windows.csv"
    manifest.write_text(
        "episode_id,start,end,selected,split\n"
        "2,4,36,true,train\n"
        "2,40,72,true,train\n"
        "5,,,true,train\n"
        "9,1,33,true,validation\n",
        encoding="utf-8",
    )
    assert module.manifest_episode_selection(manifest, split="train") == {2: {4, 40}, 5: None}
    assert module.manifest_episode_selection(manifest, split="validation") == {9: {1}}
