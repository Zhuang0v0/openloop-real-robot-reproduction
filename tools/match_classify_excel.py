#!/usr/bin/env python3
"""Map the ``classify objects`` workbook onto Classify Objects episodes.

The command writes three review-friendly artifacts: ``manifest.csv``,
``unmatched.csv``, and ``audit.md``.  It intentionally uses the pure OOXML
reader in :mod:`classify_common`; importing this script never imports
``openpyxl`` and therefore works around broken workbook style tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

try:  # Works both as ``python tools/...`` and as an imported package.
    from tools.classify_common import (
        DEFAULT_EPISODE_COUNT,
        DEFAULT_START_EPISODE,
        MANIFEST_FIELDS,
        SHEET_NAME,
        UNMATCHED_FIELDS,
        build_manifest,
        load_classify_rows,
        write_audit_markdown,
        write_csv,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script invocation
    from classify_common import (  # type: ignore
        DEFAULT_EPISODE_COUNT,
        DEFAULT_START_EPISODE,
        MANIFEST_FIELDS,
        SHEET_NAME,
        UNMATCHED_FIELDS,
        build_manifest,
        load_classify_rows,
        write_audit_markdown,
        write_csv,
    )


def _output_paths(output: str | Path | None, manifest: str | Path | None) -> tuple[Path, Path, Path]:
    if manifest is not None:
        manifest_path = Path(manifest).expanduser()
        output_dir = manifest_path.parent
    elif output is not None and Path(output).suffix.lower() == ".csv":
        manifest_path = Path(output).expanduser()
        output_dir = manifest_path.parent
    else:
        output_dir = Path(output or "classify_audit").expanduser()
        manifest_path = output_dir / "classify_episode_manifest.csv"
    return output_dir, manifest_path, output_dir / "classify_unmatched_episodes.csv"


def map_excel(
    excel: str | Path,
    *,
    dataset_root: str | Path | None = None,
    output: str | Path | None = None,
    manifest: str | Path | None = None,
    sheet_name: str = SHEET_NAME,
    start_episode: int = DEFAULT_START_EPISODE,
    episode_count: int = DEFAULT_EPISODE_COUNT,
) -> dict[str, object]:
    """Run the mapping and return a JSON-serializable summary."""

    excel_path = Path(excel).expanduser().resolve()
    rows = load_classify_rows(excel_path, sheet_name=sheet_name, start_episode=start_episode)
    manifest_rows, unmatched_rows = build_manifest(
        rows,
        dataset_root=dataset_root,
        start_episode=start_episode,
        episode_count=episode_count,
        source_excel=excel_path,
    )
    output_dir, manifest_path, unmatched_path = _output_paths(output, manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(manifest_path, manifest_rows, MANIFEST_FIELDS)
    write_csv(unmatched_path, unmatched_rows, UNMATCHED_FIELDS)
    # Keep short aliases for older local scripts while making the canonical
    # artifact names unambiguous in experiment reports.
    short_manifest = output_dir / "manifest.csv"
    short_unmatched = output_dir / "unmatched.csv"
    if short_manifest != manifest_path:
        write_csv(short_manifest, manifest_rows, MANIFEST_FIELDS)
    if short_unmatched != unmatched_path:
        write_csv(short_unmatched, unmatched_rows, UNMATCHED_FIELDS)
    # Keep the descriptive artifact names used by the experiment reports in
    # addition to the short names used by the CLI.  This is intentionally a
    # copy, not a symlink, so the outputs remain portable across filesystems.
    descriptive_unmatched = output_dir / "classify_unmatched_episodes.csv"
    if descriptive_unmatched != unmatched_path:
        write_csv(descriptive_unmatched, unmatched_rows, UNMATCHED_FIELDS)
    audit_path = output_dir / "classify_dataset_audit.md"
    write_audit_markdown(
        audit_path,
        manifest=manifest_rows,
        unmatched=unmatched_rows,
        start_episode=start_episode,
        episode_count=episode_count,
        dataset_root=dataset_root,
    )
    short_audit = output_dir / "audit.md"
    if short_audit != audit_path:
        write_audit_markdown(short_audit, manifest=manifest_rows, unmatched=unmatched_rows, start_episode=start_episode, episode_count=episode_count, dataset_root=dataset_root)
    summary: dict[str, object] = {
        "excel": str(excel_path),
        "sheet": sheet_name,
        "manifest": str(manifest_path.resolve()),
        "unmatched": str(unmatched_path.resolve()),
        "descriptive_unmatched": str(descriptive_unmatched.resolve()),
        "audit": str(audit_path.resolve()),
        "episode_count": episode_count,
        "manifest_rows": len(manifest_rows),
        "unmatched_rows": len(unmatched_rows),
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--excel", "--xlsx", required=True, type=Path, help="Source .xlsx/.xlsm workbook")
    parser.add_argument("--dataset-root", "--data-root", type=Path, default=None, help="Dataset root (optional; enables path matching)")
    parser.add_argument("--output", "--output-dir", "-o", type=Path, default=Path("classify_audit"), help="Output directory, or a manifest.csv path")
    parser.add_argument("--manifest", type=Path, default=None, help="Explicit manifest.csv destination")
    parser.add_argument("--sheet-name", default=SHEET_NAME, help=f"Worksheet name (must exactly equal {SHEET_NAME!r})")
    parser.add_argument("--start-episode", type=int, default=DEFAULT_START_EPISODE)
    parser.add_argument("--episodes", "--episode-count", dest="episode_count", type=int, default=DEFAULT_EPISODE_COUNT)
    return parser


# Public spelling used by a few notebooks; the CLI and ``map_excel`` remain
# the canonical interfaces.
match_excel = map_excel


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = map_excel(
        args.excel,
        dataset_root=args.dataset_root,
        output=args.output,
        manifest=args.manifest,
        sheet_name=args.sheet_name,
        start_episode=args.start_episode,
        episode_count=args.episode_count,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

