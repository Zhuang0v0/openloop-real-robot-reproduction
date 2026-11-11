"""Shared, dependency-light helpers for the Classify Objects audit tools.

The source workbook is intentionally read directly from OOXML.  A number of
Excel readers try to parse the optional ``styles.xml`` part eagerly and fail on
workbooks with a damaged style table, even though the cell values are intact.
The reader below only consumes workbook relationships, shared strings, and the
selected worksheet, so style corruption cannot prevent an audit.
"""

from __future__ import annotations

import csv
import json
import math
import posixpath
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree as ET


SHEET_NAME = "classify objects"
FIXED_PROMPT = (
    "Put all fruits into the large basket first, then put all blocks into the small basket."
)
# Descriptive aliases keep the contract easy to discover for downstream
# converters while preserving the canonical constant used by the launchers.
TASK_PROMPT = FIXED_PROMPT
DEFAULT_START_EPISODE = 0
DEFAULT_EPISODE_COUNT = 106
EXPECTED_EPISODES = DEFAULT_EPISODE_COUNT
EXPECTED_END_EPISODE = DEFAULT_START_EPISODE + DEFAULT_EPISODE_COUNT - 1

MANIFEST_FIELDS = [
    "episode_id",
    "group",
    "group_index",
    "C1",
    "C2",
    "C3",
    "C4",
    "quality_raw",
    "quality_score",
    "quality_status",
    "note",
    "scene_metadata",
    "prompt",
    "match_status",
    "episode_path",
    "excel_row",
    "source_excel",
]

UNMATCHED_FIELDS = [
    "episode_id",
    "reason",
    "excel_row",
    "episode_path",
    "details",
]

_EPISODE_RE = re.compile(r"(?:^|[^a-z])episode[_ -]?(\d+)(?:[^0-9]|$)", re.IGNORECASE)
_ID_RE = re.compile(r"(?:episode|ep|id|编号|序号)?\s*[_#:-]?\s*(\d+)", re.IGNORECASE)


class ExcelFormatError(ValueError):
    """Raised when an OOXML workbook cannot be read safely."""


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _attr_local(element: ET.Element, name: str, default: str | None = None) -> str | None:
    for key, value in element.attrib.items():
        if _local_name(key) == name:
            return value
    return default


def _parse_xml(blob: bytes, member: str) -> ET.Element:
    try:
        return ET.fromstring(blob)
    except ET.ParseError as exc:
        raise ExcelFormatError(f"Malformed OOXML part {member!r}: {exc}") from exc


def _relationship_target(workbook_member: str, target: str) -> str:
    # Relationship targets are relative to the part containing the .rels file.
    base = posixpath.dirname(workbook_member)
    target = target.replace("\\", "/")
    if target.startswith("/"):
        return target.lstrip("/")
    return posixpath.normpath(posixpath.join(base, target))


def _shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = _parse_xml(zf.read("xl/sharedStrings.xml"), "xl/sharedStrings.xml")
    strings: list[str] = []
    for item in root.iter():
        if _local_name(item.tag) != "si":
            continue
        # Rich strings contain multiple <r><t> nodes.  Concatenating text is
        # the value Excel displays and avoids any style/font interpretation.
        strings.append("".join((node.text or "") for node in item.iter() if _local_name(node.tag) == "t"))
    return strings


def _column_number(reference: str) -> int:
    match = re.match(r"([A-Za-z]+)", reference or "")
    if not match:
        return 0
    value = 0
    for char in match.group(1).upper():
        value = value * 26 + ord(char) - ord("A") + 1
    return value


def _cell_value(cell: ET.Element, shared: Sequence[str]) -> str:
    cell_type = _attr_local(cell, "t", "") or ""
    if cell_type == "inlineStr":
        return "".join((node.text or "") for node in cell.iter() if _local_name(node.tag) == "t")
    value_node = next((node for node in cell if _local_name(node.tag) == "v"), None)
    raw = "" if value_node is None or value_node.text is None else value_node.text
    if cell_type == "s":
        try:
            return shared[int(raw)]
        except (ValueError, IndexError):
            # Keep a damaged shared-string index visible in the audit instead
            # of silently dropping the cell.
            return raw
    if cell_type == "b":
        return "TRUE" if raw == "1" else "FALSE" if raw == "0" else raw
    # Formula results and ordinary numeric/string cells are represented by the
    # cached value.  Do not apply a style-dependent date/number conversion.
    return raw


def _find_worksheet(zf: zipfile.ZipFile, sheet_name: str) -> tuple[str, str]:
    names = set(zf.namelist())
    if "xl/workbook.xml" not in names:
        raise ExcelFormatError("Workbook is missing xl/workbook.xml")
    workbook = _parse_xml(zf.read("xl/workbook.xml"), "xl/workbook.xml")
    sheets = [node for node in workbook.iter() if _local_name(node.tag) == "sheet"]
    exact = [node for node in sheets if _attr_local(node, "name") == sheet_name]
    if not exact:
        available = [_attr_local(node, "name", "") or "" for node in sheets]
        raise ExcelFormatError(
            f"Worksheet {sheet_name!r} was not found (available sheets: {available!r}); "
            "the sheet name is case-sensitive"
        )
    if len(exact) > 1:
        raise ExcelFormatError(f"Worksheet {sheet_name!r} occurs more than once")
    sheet = exact[0]
    rel_id = _attr_local(sheet, "id")
    relation_member = "xl/_rels/workbook.xml.rels"
    if rel_id and relation_member in names:
        rels = _parse_xml(zf.read(relation_member), relation_member)
        for relation in rels.iter():
            if _local_name(relation.tag) != "Relationship":
                continue
            if _attr_local(relation, "Id") != rel_id:
                continue
            target = _attr_local(relation, "Target")
            if not target:
                break
            member = _relationship_target("xl/workbook.xml", target)
            if member in names:
                return member, _attr_local(sheet, "name", sheet_name) or sheet_name
            raise ExcelFormatError(f"Worksheet relationship points to missing part {member!r}")
    # A few minimal test workbooks omit relationships.  The sheet id normally
    # ends in its numeric worksheet index, so use that as a conservative
    # fallback while retaining the exact-name check above.
    index_match = re.search(r"(\d+)$", rel_id or "")
    candidates = []
    if index_match:
        candidates.append(f"xl/worksheets/sheet{int(index_match.group(1))}.xml")
    candidates.extend(sorted(name for name in names if name.startswith("xl/worksheets/") and name.endswith(".xml")))
    for member in candidates:
        if member in names:
            return member, _attr_local(sheet, "name", sheet_name) or sheet_name
    raise ExcelFormatError(f"Worksheet {sheet_name!r} has no readable worksheet part")


def read_ooxml_sheet(path: str | Path, sheet_name: str = SHEET_NAME) -> list[list[str]]:
    """Read one worksheet without importing openpyxl.

    The first list item is row 1 as represented by the worksheet XML.  Missing
    cells are empty strings and are padded to the widest row.  Styles are never
    read, which is deliberate: malformed ``styles.xml`` must not block data
    recovery.
    """

    workbook_path = Path(path).expanduser()
    if not workbook_path.exists():
        raise FileNotFoundError(f"Excel workbook does not exist: {workbook_path}")
    if workbook_path.suffix.lower() not in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        raise ExcelFormatError(
            f"Unsupported workbook type {workbook_path.suffix!r}; provide an OOXML .xlsx/.xlsm file"
        )
    try:
        zf = zipfile.ZipFile(workbook_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ExcelFormatError(f"Cannot open OOXML workbook {workbook_path}: {exc}") from exc
    with zf:
        worksheet_member, _ = _find_worksheet(zf, sheet_name)
        shared = _shared_strings(zf)
        root = _parse_xml(zf.read(worksheet_member), worksheet_member)
        parsed_rows: dict[int, dict[int, str]] = {}
        max_col = 0
        for row in root.iter():
            if _local_name(row.tag) != "row":
                continue
            row_number = int(_attr_local(row, "r", str(len(parsed_rows) + 1)) or "1")
            values: dict[int, str] = {}
            for cell in row:
                if _local_name(cell.tag) != "c":
                    continue
                column = _column_number(_attr_local(cell, "r", "") or "")
                if column <= 0:
                    column = max(values.keys(), default=0) + 1
                values[column] = _cell_value(cell, shared)
                max_col = max(max_col, column)
            parsed_rows[row_number] = values
        if not parsed_rows:
            return []
        max_row = max(parsed_rows)
        return [[parsed_rows.get(row, {}).get(col, "") for col in range(1, max_col + 1)] for row in range(1, max_row + 1)]


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def normalize_header(value: Any) -> str:
    """Normalize a header for matching while retaining original labels in metadata."""

    text = unicodedata.normalize("NFKC", _cell_text(value)).casefold()
    return "".join(char for char in text if char.isalnum())


def _canonical_headers(header_row: Sequence[Any]) -> list[str]:
    result: list[str] = []
    seen: Counter[str] = Counter()
    for index, value in enumerate(header_row, 1):
        label = _cell_text(value) or f"column_{index}"
        seen[label] += 1
        if seen[label] > 1:
            label = f"{label}_{seen[label]}"
        result.append(label)
    return result


_ID_ALIASES = {
    "id",
    "episode",
    "episodeid",
    "episodeindex",
    "ep",
    "编号",
    "序号",
    "对应编号",
    "对应id",
    "对应编号id",
    "对应序号",
    "episode no",
    "episode number",
}
_GROUP_ALIASES = {
    "group",
    "groupname",
    "组",
    "分组",
    "组别",
    "类别",
    "类别编号",
    "场景分类",
    "场景组",
    "任务组",
    "objectgroup",
}
_GROUP_INDEX_ALIASES = {
    "groupindex",
    "groupno",
    "groupnumber",
    "组序号",
    "组内序号",
    "同组编号",
    "组内编号",
    "组编号",
}
_QUALITY_ALIASES = {
    "quality",
    "qualityscore",
    "qualityrating",
    "rating",
    "grade",
    "质量",
    "数据质量",
    "质量分数",
    "质量评分",
    "质量等级",
    "评级",
    "总评",
}
_NOTE_ALIASES = {
    "note",
    "notes",
    "remark",
    "remarks",
    "comment",
    "comments",
    "备注",
    "备注信息",
    "说明",
}
_SCENE_ALIASES = {"scene", "sceneid", "scenename", "scenario", "environment", "场景", "场景id", "场景名称"}


def _header_matches(normalized: str, aliases: set[str]) -> bool:
    return normalized in {normalize_header(alias) for alias in aliases}


def _find_header(headers: Sequence[str], aliases: set[str]) -> int | None:
    for index, header in enumerate(headers):
        if _header_matches(normalize_header(header), aliases):
            return index
    return None


def _criterion_index(header: str) -> int | None:
    normalized = normalize_header(header)
    match = re.fullmatch(r"c([1-4])", normalized)
    if match:
        return int(match.group(1)) - 1
    # Accommodate labels such as "C1 score" or "criterion C1".
    match = re.search(r"c([1-4])", normalized)
    return int(match.group(1)) - 1 if match else None


def parse_episode_id(value: Any) -> int | None:
    text = _cell_text(value)
    if not text:
        return None
    try:
        number = float(text)
        if number.is_integer():
            return int(number)
    except ValueError:
        pass
    match = _ID_RE.search(text)
    return int(match.group(1)) if match else None


def quality_summary(value: Any) -> tuple[str, str, str]:
    """Return (raw, score, status), retaining the original quality value."""

    raw = _cell_text(value)
    if not raw:
        return "", "", "missing"
    token = unicodedata.normalize("NFKC", raw).casefold()
    score: str = ""
    status = "unknown"
    numeric = re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", token)
    if numeric:
        number = float(token)
        score = str(int(number)) if number.is_integer() else str(number)
        # The supplied workbook uses a 0..1 quality scale (0.5, 0.6, ...),
        # while generic workbooks may use a 1..5 rubric.  Keep the two scales
        # distinct so a 0.8 score is not mislabeled as a failure.
        status = "known" if 0.0 <= number <= 1.0 else "pass" if number >= 3 else "fail" if number <= 1 else "review"
    else:
        letter_scores = {"a": 4, "b": 3, "c": 2, "d": 1, "s": 4, "优": 4, "优秀": 4, "良": 3, "良好": 3, "中": 2, "一般": 2, "差": 1, "不合格": 1}
        for label, number in letter_scores.items():
            if token == label:
                score = str(number)
                status = "pass" if number >= 3 else "fail" if number <= 1 else "review"
                break
        if any(word in token for word in ("pass", "ok", "通过", "合格", "good")):
            status = "pass"
        elif any(word in token for word in ("fail", "bad", "失败", "不合格")):
            status = "fail"
    return raw, score, status


def parse_classify_rows(rows: Sequence[Sequence[Any]], *, start_episode: int = DEFAULT_START_EPISODE) -> list[dict[str, Any]]:
    """Parse the fixed workbook layout (title row, header row, records)."""

    if len(rows) < 2:
        raise ExcelFormatError("Worksheet must contain a title row and a header row")
    headers = _canonical_headers(rows[1])
    id_column = _find_header(headers, _ID_ALIASES)
    if id_column is None:
        raise ExcelFormatError(
            "Worksheet is missing the required episode ID column; expected a header such as '对应编号'"
        )
    group_column = _find_header(headers, _GROUP_ALIASES)
    group_index_column = _find_header(headers, _GROUP_INDEX_ALIASES)
    quality_column = _find_header(headers, _QUALITY_ALIASES)
    note_column = _find_header(headers, _NOTE_ALIASES)
    criterion_columns: dict[int, int] = {}
    for index, header in enumerate(headers):
        criterion = _criterion_index(header)
        if criterion is not None and criterion not in criterion_columns:
            criterion_columns[criterion] = index

    records: list[dict[str, Any]] = []
    group_counts: defaultdict[str, int] = defaultdict(int)
    for sheet_row, source_values in enumerate(rows[2:], 3):
        values = list(source_values) + [""] * max(0, len(headers) - len(source_values))
        if not any(_cell_text(value) for value in values):
            continue
        # IDs are authoritative.  Never substitute row order: the workbook can
        # be sorted or contain blank/deleted rows, and doing so would silently
        # attach scene labels to the wrong recording.
        episode_id = parse_episode_id(values[id_column]) if id_column < len(values) else None
        group = _cell_text(values[group_column]) if group_column is not None else ""
        if group and group_index_column is None:
            group_counts[group] += 1
            group_index = str(group_counts[group])
        else:
            group_index = _cell_text(values[group_index_column]) if group_index_column is not None else ""
        quality_value = values[quality_column] if quality_column is not None else ""
        quality_raw, quality_score, quality_status = quality_summary(quality_value)
        note = _cell_text(values[note_column]) if note_column is not None else ""
        criteria = {
            f"C{index + 1}": _cell_text(values[column]) if column < len(values) else ""
            for index, column in criterion_columns.items()
        }
        for criterion in ("C1", "C2", "C3", "C4"):
            criteria.setdefault(criterion, "")
        if not group:
            # Some exports encode the scene group as a one-hot marker in the
            # C1-C4 columns rather than having a separate ``group`` column.
            # Infer it only for unambiguous marker values; ordinary criterion
            # text is preserved and does not become a fabricated group label.
            marker_values = {"1", "true", "yes", "y", "x", "ok", "是", "√", "✓"}
            marked = [
                criterion
                for criterion, value in criteria.items()
                if _cell_text(value) and normalize_header(value) in {normalize_header(marker) for marker in marker_values}
            ]
            if len(marked) == 1:
                group = marked[0]
        recognized = {id_column, group_column, group_index_column, quality_column, note_column}
        recognized.discard(None)
        recognized.update(criterion_columns.values())
        scene_metadata = {
            headers[index]: _cell_text(value)
            for index, value in enumerate(values[: len(headers)])
            if index not in recognized and _cell_text(value) != ""
        }
        scene_column = _find_header(headers, _SCENE_ALIASES)
        scene = _cell_text(values[scene_column]) if scene_column is not None else ""
        records.append(
            {
                "episode_id": episode_id,
                "group": group,
                "group_index": group_index,
                **criteria,
                "quality_raw": quality_raw,
                "quality_score": quality_score,
                "quality_status": quality_status,
                "note": note,
                "scene": scene,
                "scene_metadata": scene_metadata,
                "excel_row": sheet_row,
                "raw": {headers[index]: _cell_text(value) for index, value in enumerate(values[: len(headers)])},
            }
        )
    return records


def load_classify_rows(path: str | Path, *, sheet_name: str = SHEET_NAME, start_episode: int = DEFAULT_START_EPISODE) -> list[dict[str, Any]]:
    if sheet_name != SHEET_NAME:
        raise ExcelFormatError(
            f"Classify Objects requires the exact worksheet name {SHEET_NAME!r}; got {sheet_name!r}"
        )
    return parse_classify_rows(read_ooxml_sheet(path, sheet_name), start_episode=start_episode)


def _path_episode_id(path: Path) -> int | None:
    # Search each path component so both episode_000001 directories and files
    # nested beneath them are recognized.
    for component in reversed(path.parts):
        match = _EPISODE_RE.search(component)
        if match:
            return int(match.group(1))
    return None


def discover_episode_paths(dataset_root: str | Path) -> tuple[dict[int, Path], dict[int, list[Path]]]:
    """Return selected episode paths and all candidates keyed by episode id."""

    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    candidates: defaultdict[int, list[Path]] = defaultdict(list)
    if root.is_dir():
        if (episode_id := _path_episode_id(root)) is not None:
            candidates[episode_id].append(root)
        for path in root.rglob("*"):
            episode_id = _path_episode_id(path)
            if episode_id is not None:
                candidates[episode_id].append(path)
    elif (episode_id := _path_episode_id(root)) is not None:
        candidates[episode_id].append(root)
    selected: dict[int, Path] = {}
    for episode_id, paths in candidates.items():
        unique = sorted(set(paths), key=lambda path: (0 if path.is_dir() else 1, len(path.parts), str(path)))
        candidates[episode_id] = unique
        selected[episode_id] = unique[0]
    # A standard LeRobot export stores many episodes in sharded parquet files
    # without episode numbers in the path.  Read only the episode_index column
    # to discover those IDs; the dataset root remains the authoritative path
    # and downstream tools filter rows by the matched ID.
    if root.is_dir():
        data_root = root / "data"
        parquet_files = sorted(data_root.rglob("*.parquet")) if data_root.is_dir() else []
        if parquet_files:
            try:
                import pyarrow.parquet as pq  # type: ignore

                flat_ids: set[int] = set()
                for parquet_path in parquet_files:
                    schema = pq.read_schema(parquet_path)
                    if "episode_index" not in schema.names:
                        continue
                    values = pq.read_table(parquet_path, columns=["episode_index"])["episode_index"].to_pylist()
                    flat_ids.update(int(value) for value in values if value is not None)
                for episode_id in sorted(flat_ids):
                    # Merge flat-shard IDs with any explicitly named
                    # per-episode candidates.  The latter remains preferred;
                    # the root is still retained for audit diagnostics.
                    candidates[episode_id] = list(candidates.get(episode_id, [])) + [root]
                    unique = sorted(set(candidates[episode_id]), key=lambda path: (0 if path.is_dir() and _path_episode_id(path) == episode_id else 1, len(path.parts), str(path)))
                    candidates[episode_id] = unique
                    selected[episode_id] = unique[0]
            except Exception:
                # The audit tool reports the missing mapping when parquet
                # support is unavailable; do not silently invent IDs.
                pass
    return selected, dict(candidates)


def _json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_manifest(
    excel_rows: Sequence[Mapping[str, Any]],
    *,
    dataset_root: str | Path | None = None,
    start_episode: int = DEFAULT_START_EPISODE,
    episode_count: int = DEFAULT_EPISODE_COUNT,
    source_excel: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Join worksheet rows to expected episode ids and filesystem paths."""

    if episode_count < 0:
        raise ValueError("episode_count must be non-negative")
    expected_ids = list(range(start_episode, start_episode + episode_count))
    selected_paths: dict[int, Path] = {}
    candidates: dict[int, list[Path]] = {}
    if dataset_root is not None:
        selected_paths, candidates = discover_episode_paths(dataset_root)

    by_id: defaultdict[int, list[Mapping[str, Any]]] = defaultdict(list)
    unmatched: list[dict[str, Any]] = []
    for row in excel_rows:
        episode_id = parse_episode_id(row.get("episode_id"))
        if episode_id is None:
            unmatched.append({"episode_id": "", "reason": "invalid_excel_id", "excel_row": row.get("excel_row", ""), "episode_path": "", "details": "Could not parse episode id"})
            continue
        by_id[episode_id].append(row)
        if episode_id not in expected_ids:
            unmatched.append({"episode_id": episode_id, "reason": "excel_id_out_of_range", "excel_row": row.get("excel_row", ""), "episode_path": "", "details": f"Expected ids {start_episode}..{start_episode + episode_count - 1}"})
    if dataset_root is not None:
        for episode_id, paths in candidates.items():
            if episode_id not in expected_ids:
                unmatched.append({"episode_id": episode_id, "reason": "dataset_id_out_of_range", "excel_row": "", "episode_path": str(paths[0]), "details": f"Expected ids {start_episode}..{start_episode + episode_count - 1}"})

    manifest: list[dict[str, Any]] = []
    for episode_id in expected_ids:
        rows_for_id = by_id.get(episode_id, [])
        row = rows_for_id[0] if rows_for_id else {}
        path = selected_paths.get(episode_id)
        if len(rows_for_id) > 1:
            for duplicate in rows_for_id[1:]:
                unmatched.append({"episode_id": episode_id, "reason": "duplicate_excel_id", "excel_row": duplicate.get("excel_row", ""), "episode_path": str(path) if path else "", "details": f"{len(rows_for_id)} rows share this id"})
        if row and path:
            status = "matched"
        elif row and dataset_root is None:
            status = "excel_only"
        elif row:
            status = "missing_path"
        elif path:
            status = "missing_excel"
        else:
            status = "missing_excel_and_path" if dataset_root is not None else "missing_excel"
        if not row:
            unmatched.append({"episode_id": episode_id, "reason": "missing_excel_row", "excel_row": "", "episode_path": str(path) if path else "", "details": "No worksheet record for expected episode"})
        if dataset_root is not None and path is None:
            unmatched.append({"episode_id": episode_id, "reason": "missing_dataset_path", "excel_row": row.get("excel_row", ""), "episode_path": "", "details": "No episode directory or file matched this id"})
        scene_metadata = row.get("scene_metadata", {}) if row else {}
        if not isinstance(scene_metadata, Mapping):
            scene_metadata = {"value": _cell_text(scene_metadata)}
        item = {
            "episode_id": episode_id,
            "group": row.get("group", "") if row else "",
            "group_index": row.get("group_index", "") if row else "",
            "C1": row.get("C1", "") if row else "",
            "C2": row.get("C2", "") if row else "",
            "C3": row.get("C3", "") if row else "",
            "C4": row.get("C4", "") if row else "",
            "quality_raw": row.get("quality_raw", "") if row else "",
            "quality_score": row.get("quality_score", "") if row else "",
            "quality_status": row.get("quality_status", "missing") if row else "missing",
            "note": row.get("note", "") if row else "",
            "scene_metadata": _json_text(scene_metadata),
            "prompt": FIXED_PROMPT,
            "match_status": status,
            "episode_path": str(path) if path else "",
            "excel_row": row.get("excel_row", "") if row else "",
            "source_excel": str(source_excel) if source_excel else "",
        }
        manifest.append(item)
    return manifest, unmatched


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    # Plain UTF-8 keeps the CSV interoperable with Python's default
    # ``encoding='utf-8'`` readers; callers targeting Excel can still open it
    # (modern Excel detects UTF-8), while avoiding a BOM in the first header.
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return target


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_audit_markdown(
    path: str | Path,
    *,
    manifest: Sequence[Mapping[str, Any]],
    unmatched: Sequence[Mapping[str, Any]],
    start_episode: int,
    episode_count: int,
    dataset_root: str | Path | None,
) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    status_counts = Counter(str(row.get("match_status", "")) for row in manifest)
    quality_counts = Counter(str(row.get("quality_status", "")) for row in manifest)
    group_counts = Counter(str(row.get("group", "")) for row in manifest)
    lines = [
        "# Classify Objects audit",
        "",
        f"- Expected episode IDs: `{start_episode}..{start_episode + episode_count - 1}` ({episode_count})",
        f"- Manifest rows: `{len(manifest)}`",
        f"- Unmatched/issues: `{len(unmatched)}`",
        f"- Dataset root: `{dataset_root if dataset_root is not None else '(not checked)'}`",
        f"- Fixed prompt: `{FIXED_PROMPT}`",
        "",
        "## Match status",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {key or '(empty)'} | {value} |" for key, value in sorted(status_counts.items()))
    lines.extend(["", "## Quality status", "", "| Status | Count |", "|---|---:|"])
    lines.extend(f"| {key or '(empty)'} | {value} |" for key, value in sorted(quality_counts.items()))
    lines.extend(["", "## Group counts", "", "| Group | Count |", "|---|---:|"])
    lines.extend(f"| {key or '(empty)'} | {value} |" for key, value in sorted(group_counts.items()))
    lines.extend(["", "## Issues", "", "| Episode | Reason | Excel row | Path | Details |", "|---:|---|---:|---|---|"])
    if unmatched:
        for item in unmatched:
            details = str(item.get("details", "")).replace("|", "\\|")
            lines.append(
                f"| {item.get('episode_id', '')} | {item.get('reason', '')} | {item.get('excel_row', '')} | {item.get('episode_path', '')} | {details} |"
            )
    else:
        lines.append("| | None | | | |")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


__all__ = [
    "DEFAULT_EPISODE_COUNT",
    "DEFAULT_START_EPISODE",
    "EXPECTED_EPISODES",
    "EXPECTED_END_EPISODE",
    "ExcelFormatError",
    "FIXED_PROMPT",
    "TASK_PROMPT",
    "MANIFEST_FIELDS",
    "SHEET_NAME",
    "UNMATCHED_FIELDS",
    "build_manifest",
    "discover_episode_paths",
    "load_classify_rows",
    "normalize_header",
    "parse_classify_rows",
    "parse_episode_id",
    "quality_summary",
    "read_manifest",
    "read_ooxml_sheet",
    "write_audit_markdown",
    "write_csv",
]

