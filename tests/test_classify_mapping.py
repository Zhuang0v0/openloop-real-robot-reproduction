from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import pytest

from tools.audit_classify_dataset import audit_dataset
from tools.classify_common import FIXED_PROMPT, ExcelFormatError, load_classify_rows, parse_classify_rows, read_ooxml_sheet
from tools.match_classify_excel import map_excel


def _xlsx(path: Path, *, sheet_name: str = "classify objects") -> Path:
    # A deliberately tiny OOXML workbook.  styles.xml is malformed on purpose:
    # the audit reader must never inspect it.
    workbook = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="{sheet_name}" sheetId="1" r:id="rId1"/></sheets>
</workbook>'''
    rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>'''
    worksheet = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>
<row r="1"><c r="A1" t="inlineStr"><is><t>Classify Objects review</t></is></c></row>
<row r="2">
 <c r="A2" t="inlineStr"><is><t>episode_id</t></is></c>
 <c r="B2" t="inlineStr"><is><t>group</t></is></c>
 <c r="C2" t="inlineStr"><is><t>group_index</t></is></c>
 <c r="D2" t="inlineStr"><is><t>C1</t></is></c><c r="E2" t="inlineStr"><is><t>C2</t></is></c>
 <c r="F2" t="inlineStr"><is><t>C3</t></is></c><c r="G2" t="inlineStr"><is><t>C4</t></is></c>
 <c r="H2" t="inlineStr"><is><t>质量</t></is></c><c r="I2" t="inlineStr"><is><t>备注</t></is></c>
 <c r="J2" t="inlineStr"><is><t>scene_id</t></is></c><c r="K2" t="inlineStr"><is><t>color</t></is></c>
</row>
<row r="3"><c r="A3" t="n"><v>0</v></c><c r="B3" t="inlineStr"><is><t>fruit</t></is></c><c r="C3" t="n"><v>1</v></c>
 <c r="D3" t="inlineStr"><is><t>ok</t></is></c><c r="E3" t="inlineStr"><is><t>ok</t></is></c><c r="F3" t="inlineStr"><is><t>ok</t></is></c><c r="G3" t="inlineStr"><is><t>ok</t></is></c>
 <c r="H3" t="inlineStr"><is><t>A</t></is></c><c r="I3" t="inlineStr"><is><t>clean</t></is></c><c r="J3" t="inlineStr"><is><t>S-0</t></is></c><c r="K3" t="inlineStr"><is><t>red</t></is></c></row>
<row r="4"><c r="A4" t="n"><v>1</v></c><c r="B4" t="inlineStr"><is><t>block</t></is></c><c r="C4" t="n"><v>1</v></c>
 <c r="D4" t="inlineStr"><is><t>ok</t></is></c><c r="E4" t="inlineStr"><is><t>ok</t></is></c><c r="F4" t="inlineStr"><is><t>review</t></is></c><c r="G4" t="inlineStr"><is><t>ok</t></is></c>
 <c r="H4" t="inlineStr"><is><t>C</t></is></c><c r="I4" t="inlineStr"><is><t>check grip</t></is></c><c r="J4" t="inlineStr"><is><t>S-1</t></is><c r="K4" t="inlineStr"><is><t>blue</t></is></c></row>
</sheetData></worksheet>'''
    # Fix the deliberately compact second data row's nested-cell typo while
    # retaining a malformed styles part.
    worksheet = worksheet.replace('</is><c r="K4"', '</is></c><c r="K4"')
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\"/>")
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", rels)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)
        archive.writestr("xl/styles.xml", "<styles>")
    return path


def test_ooxml_fallback_ignores_broken_styles_and_exact_sheet(tmp_path: Path) -> None:
    workbook = _xlsx(tmp_path / "labels.xlsx")
    rows = read_ooxml_sheet(workbook)
    assert rows[0][0] == "Classify Objects review"
    records = load_classify_rows(workbook)
    assert [row["episode_id"] for row in records] == [0, 1]
    assert records[0]["C1"] == "ok"
    assert records[0]["quality_score"] == "4"
    assert records[0]["quality_status"] == "pass"
    assert records[0]["note"] == "clean"
    assert records[0]["scene_metadata"] == {"color": "red", "scene_id": "S-0"}
    with pytest.raises(ExcelFormatError, match="case-sensitive"):
        read_ooxml_sheet(workbook, "Classify Objects")


def test_mapping_outputs_manifest_audit_and_unmatched(tmp_path: Path) -> None:
    workbook = _xlsx(tmp_path / "labels.xlsx")
    dataset = tmp_path / "dataset" / "per_episode"
    (dataset / "episode_000000").mkdir(parents=True)
    (dataset / "episode_000001").mkdir(parents=True)
    output = tmp_path / "audit"
    summary = map_excel(workbook, dataset_root=dataset.parent, output=output, episode_count=3)
    assert summary["manifest_rows"] == 3
    assert summary["unmatched_rows"] >= 1
    with (output / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["episode_id"] for row in rows] == ["0", "1", "2"]
    assert rows[0]["prompt"] == FIXED_PROMPT
    assert rows[0]["match_status"] == "matched"
    assert rows[2]["match_status"] == "missing_excel_and_path"
    assert (output / "audit.md").exists()
    assert (output / "unmatched.csv").exists()


def test_corresponding_id_is_authoritative_not_row_order() -> None:
    records = parse_classify_rows(
        [
            ["title"],
            ["对应编号", "组别", "质量", "备注"],
            ["7", "C3", "0.8", "kept"],
            ["2", "C1", "0.6", "review"],
        ]
    )
    assert [row["episode_id"] for row in records] == [7, 2]
    assert [row["group"] for row in records] == ["C3", "C1"]
    with pytest.raises(ExcelFormatError, match="required episode ID"):
        parse_classify_rows([["title"], ["group"], ["C1"]])


def test_audit_accepts_manifest_and_reports_prompt_drift(tmp_path: Path) -> None:
    workbook = _xlsx(tmp_path / "labels.xlsx")
    output = tmp_path / "mapped"
    map_excel(workbook, output=output, episode_count=2)
    manifest = output / "manifest.csv"
    text = manifest.read_text(encoding="utf-8-sig").replace(FIXED_PROMPT, "wrong prompt")
    manifest.write_text(text, encoding="utf-8-sig")
    checked = audit_dataset(manifest_path=manifest, output=tmp_path / "checked", episode_count=2)
    assert checked["ok"] is False
    assert checked["issue_count"] >= 2

