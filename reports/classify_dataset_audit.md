# Classify Objects Dataset Audit

This report is generated/updated by `tools/match_classify_excel.py` and
`tools/audit_classify_dataset.py`. The repository does not contain the dataset;
run the tools on the downloaded ModelScope export and keep the resulting report
with the experiment artifacts.

## Excel-only verification

The supplied workbook has a sheet named exactly `classify objects`. Its 106
data rows use `对应编号` values 0 through 105 exactly once. The verified group
counts are C1=27, C2=26, C3=27 and C4=26. The other workbook sheets are not
eligible for this experiment.

The parser reads the worksheet XML directly because this workbook contains a
malformed style record that makes some `openpyxl` versions fail while reading
styles. It does not use row order as an episode ID.

## Dataset verification

Run:

```bash
python tools/match_classify_excel.py \
  --excel /path/to/任务.xlsx \
  --output runs/classify/classify_episode_manifest.csv

python tools/audit_classify_dataset.py \
  --dataset-root /path/to/nero_classify_objects \
  --manifest runs/classify/classify_episode_manifest.csv \
  --output-dir runs/classify/audit
```

The audit must report the actual matched episode count, file paths, parquet
schema, action/state dimensions, finite-value checks, camera streams, frame
counts, timestamps and observed video FPS. Any missing, duplicate, damaged or
extra episode is written to `classify_unmatched_episodes.csv` and is excluded
from training until reviewed. A successful Excel match alone is not evidence
that the ModelScope episode exists or is readable.

## Prompt and metadata policy

Every matched episode receives the same task label:

`Put all fruits into the large basket first, then put all blocks into the small basket.`

C1-C4, quality score, raw score text and notes remain metadata for filtering
and stratified reporting. The English prompts in the workbook are not used as
training labels because they encode left/right scene layout and are blank for
some rows.

## Status

The report must be regenerated on the target GPU workspace after downloading
the dataset. Do not fill in counts or claim a clean audit without the generated
JSON/CSV evidence.
