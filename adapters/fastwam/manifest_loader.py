"""Restrict an upstream FastWAM LeRobot loader to a manifest.

FastWAM's stock loader selects a random validation fraction of episodes.  The
Classify Objects experiment instead supplies an explicit episode manifest.  We
install a narrow runtime wrapper around the upstream ``BaseLerobotDataset``:
the upstream reader, video decoder, transforms, and batching code remain
unchanged; only the ``episodes`` mapping handed to ``MultiLeRobotDataset`` is
intersected with IDs from the manifest.

The module is intentionally imported by ``run_with_manifest.py`` rather than
copied into the upstream checkout.  Set ``FASTWAM_EPISODE_MANIFEST`` to a CSV
or JSON manifest before importing it.  A window manifest is reduced to the
unique episodes with ``selected`` rows; exact window-level sampling still
requires a custom dataset implementation and is reported by the preparation
tools instead of being silently claimed here.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import math
from pathlib import Path
from typing import Any, Mapping

LOGGER = logging.getLogger("classify.fastwam_manifest")


def _as_bool(value: Any, default: bool = True) -> bool:
    if value is None or str(value).strip() == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() not in {"0", "false", "no", "off", "rejected", "drop", "dropped"}


def _rows_from_manifest(path: str | os.PathLike[str]) -> list[Mapping[str, Any]]:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"FastWAM manifest does not exist: {source}")
    if source.suffix.casefold() in {".json", ".jsonl"}:
        if source.suffix.casefold() == ".jsonl":
            return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        value = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(value, list):
            return value
        if isinstance(value, Mapping):
            for key in ("windows", "rows", "manifest", "episodes", "samples"):
                if isinstance(value.get(key), list):
                    return value[key]
            return [value]
        raise ValueError(f"FastWAM manifest JSON must contain rows: {source}")
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _episode_id(row: Mapping[str, Any], row_number: int) -> int:
    raw = row.get("episode_id", row.get("episode_index", row.get("id")))
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"manifest row {row_number} has invalid episode_id {raw!r}") from exc
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError(f"manifest row {row_number} has non-integral episode_id {raw!r}")
    return int(value)


def _row_split_matches(row: Mapping[str, Any], split: str | None) -> bool:
    row_split = str(row.get("split", "") or "").strip().casefold()
    return not split or not row_split or row_split == split.casefold()


def _parse_window_start(row: Mapping[str, Any]) -> int | None:
    """Return the local sample start for a filtered window, if present.

    ``start`` is intentionally preferred over ``source_rows``.  The latter
    can be a global parquet row number when an episode came from a flattened
    shard, while FastWAM's LeRobot reader indexes frames locally inside the
    episode.  A row without coordinates denotes the complete episode.
    """

    raw = row.get("start")
    if raw is not None and str(raw).strip() != "":
        try:
            value = float(str(raw).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid window start {raw!r}") from exc
        if not math.isfinite(value) or not value.is_integer() or value < 0:
            raise ValueError(f"window start must be a non-negative integer, got {raw!r}")
        return int(value)
    return None


def manifest_episode_selection(
    path: str | os.PathLike[str], *, split: str | None = None
) -> dict[int, set[int] | None]:
    """Read selected episode IDs and optional local window starts.

    The value ``None`` means every valid frame in that episode.  A set means
    that the manifest is a window manifest and only those local frame starts
    may be sampled.  Duplicate windows are collapsed deterministically.
    """

    selection: dict[int, set[int] | None] = {}
    for row_number, row in enumerate(_rows_from_manifest(path), 1):
        if not isinstance(row, Mapping):
            raise ValueError(f"manifest row {row_number} is not an object")
        if not _as_bool(row.get("selected"), default=True):
            continue
        if not _row_split_matches(row, split):
            continue
        episode_id = _episode_id(row, row_number)
        start = _parse_window_start(row)
        if episode_id not in selection:
            selection[episode_id] = None if start is None else {start}
        elif selection[episode_id] is not None and start is not None:
            selection[episode_id].add(start)
        elif start is None:
            # An episode-level row supersedes any narrower rows for the same
            # episode.  This is the least surprising interpretation for a
            # hand-edited manifest.
            selection[episode_id] = None
    if not selection:
        raise ValueError(f"FastWAM manifest contains no selected episode IDs: {path}")
    return selection


def manifest_episode_ids(path: str | os.PathLike[str], *, split: str | None = None) -> tuple[int, ...]:
    """Read unique selected episode IDs from a CSV/JSON manifest."""

    return tuple(sorted(manifest_episode_selection(path, split=split)))


def _selection_from_environment(*, is_training_set: bool) -> dict[int, set[int] | None] | None:
    env_name = "FASTWAM_EPISODE_MANIFEST" if is_training_set else "FASTWAM_VALIDATION_MANIFEST"
    path = os.environ.get(env_name, "").strip()
    if not path and not is_training_set:
        # A missing validation manifest is allowed when the launcher disables
        # the upstream validation dataset.  It must not silently reuse train
        # rows, which would turn a validation metric into training leakage.
        return None
    if not path:
        return None
    split = os.environ.get("FASTWAM_MANIFEST_SPLIT", "").strip() or None
    selection = manifest_episode_selection(path, split=split)
    # Training and validation manifests have different cardinalities.  Keep
    # the historical training variable for compatibility, but allow the
    # launcher to validate the held-out manifest independently instead of
    # accidentally comparing it with the training count.
    expected_name = "FASTWAM_EXPECTED_EPISODES" if is_training_set else "FASTWAM_EXPECTED_VALIDATION_EPISODES"
    expected = os.environ.get(expected_name, "").strip()
    if expected:
        expected_count = int(expected)
        if len(selection) != expected_count:
            raise ValueError(
                f"FastWAM manifest selected {len(selection)} episodes, expected {expected_count}: {path}"
            )
    return selection


def install_manifest_filter() -> tuple[int, ...] | None:
    """Install the filter once and return the selected IDs.

    The patch is deliberately idempotent so a launcher can import it more
    than once in a notebook or a retrying job.
    """

    try:
        from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
    except ImportError as exc:  # pragma: no cover - requires an installed FastWAM checkout
        raise RuntimeError("FastWAM must be installed before manifest_loader is imported") from exc

    if getattr(BaseLerobotDataset, "_classify_manifest_filter", False):
        # The monkey patch is process-global.  A second import should be a
        # harmless no-op; ``allowed`` only exists inside the constructor
        # closure and must not be referenced here.
        return None

    original_init = BaseLerobotDataset.__init__
    original_len = BaseLerobotDataset.__len__
    original_getitem = BaseLerobotDataset.__getitem__

    def filtered_init(self, *args, **kwargs):
        allowed = _selection_from_environment(is_training_set=bool(kwargs.get("is_training_set", False)))
        if allowed is None:
            # No patching is needed for a deliberately disabled validation
            # dataset or for a caller that did not request a manifest.
            return original_init(self, *args, **kwargs)

        # A zero validation fraction makes the upstream constructor enumerate
        # all episodes.  The proxy below then intersects that list with the
        # explicit manifest, avoiding its random split entirely.
        kwargs["val_set_proportion"] = 0.0
        kwargs["is_training_set"] = True
        original_multi_cls = self.multi_dataset_cls
        selected = set(allowed)

        class FilteredMultiDataset(original_multi_cls):
            def __init__(inner_self, dataset_dirs, episodes=None, **multi_kwargs):
                if episodes is None:
                    raise ValueError("manifest filtering requires an explicit episode mapping")
                filtered = {
                    repo_id: [int(index) for index in indices if int(index) in selected]
                    for repo_id, indices in episodes.items()
                }
                missing = {
                    repo_id: sorted(set(int(index) for index in indices) - selected)
                    for repo_id, indices in episodes.items()
                    if not any(int(index) in selected for index in indices)
                }
                if missing:
                    LOGGER.warning("manifest selected no episodes for dataset(s): %s", sorted(missing))
                if not any(filtered.values()):
                    raise ValueError("manifest does not overlap any dataset episode index")
                super().__init__(dataset_dirs=dataset_dirs, episodes=filtered, **multi_kwargs)

        self.multi_dataset_cls = FilteredMultiDataset
        try:
            original_init(self, *args, **kwargs)
        finally:
            self.multi_dataset_cls = original_multi_cls

        # Map local window starts to the flattened frame indices expected by
        # the upstream MultiLeRobotDataset.  This keeps action/video windows
        # contiguous without copying or stitching rejected frames.
        frame_indices: list[int] = []
        global_offset = 0
        for dataset in getattr(self.multi_dataset, "_datasets", ()): 
            episodes = list(getattr(dataset, "episodes", ()) or ())
            data_index = getattr(dataset, "episode_data_index", {})
            starts = data_index.get("from", ())
            ends = data_index.get("to", ())
            for nested_index, episode_id in enumerate(episodes):
                try:
                    episode_start = int(starts[nested_index]) + global_offset
                    episode_end = int(ends[nested_index]) + global_offset
                except (IndexError, TypeError, ValueError):
                    continue
                requested = allowed.get(int(episode_id))
                if requested is None:
                    frame_indices.extend(range(episode_start, episode_end))
                else:
                    frame_indices.extend(
                        episode_start + local
                        for local in sorted(requested)
                        if 0 <= int(local) < episode_end - episode_start
                    )
            global_offset += int(getattr(dataset, "num_frames", 0))
        if not frame_indices:
            raise ValueError("FastWAM manifest has no valid frame starts in the selected dataset")
        self._classify_frame_indices = tuple(sorted(set(frame_indices)))

        LOGGER.info(
            "FastWAM manifest filter selected %d episode IDs and %d frame starts",
            len(allowed),
            len(self._classify_frame_indices),
        )

    def filtered_len(self):
        indices = getattr(self, "_classify_frame_indices", None)
        return len(indices) if indices is not None else original_len(self)

    def filtered_getitem(self, index):
        indices = getattr(self, "_classify_frame_indices", None)
        if indices is None:
            return original_getitem(self, index)
        position = int(index)
        if position < 0:
            position += len(indices)
        if position < 0 or position >= len(indices):
            raise IndexError(f"Index {index} out of bounds {len(indices)}")
        return original_getitem(self, indices[position])

    BaseLerobotDataset.__init__ = filtered_init
    BaseLerobotDataset.__len__ = filtered_len
    BaseLerobotDataset.__getitem__ = filtered_getitem
    BaseLerobotDataset._classify_manifest_filter = True
    return True


__all__ = ["install_manifest_filter", "manifest_episode_ids", "manifest_episode_selection"]
