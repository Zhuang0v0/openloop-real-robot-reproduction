#!/usr/bin/env bash
set -euo pipefail

# One-episode, one-step GPU smoke for the Classify Objects path.  This checks
# the real converter/packer, model initialization, loss/backward, and a saved
# checkpoint without claiming that a training run has completed.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs/classify_objects}"
SOURCE_MANIFEST="${CLASSIFY_MANIFEST:?Set CLASSIFY_MANIFEST to the Excel-derived manifest}"
SMOKE_ROOT="${CLASSIFY_SMOKE_ROOT:-$RUN_ROOT/smoke}"
SMOKE_MANIFEST="$SMOKE_ROOT/smoke_manifest.csv"

mkdir -p "$SMOKE_ROOT"
"$PYTHON_BIN" - "$SOURCE_MANIFEST" "$SMOKE_MANIFEST" <<'PY'
import csv
import sys
from pathlib import Path

source, target = map(Path, sys.argv[1:])
with source.open("r", encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
if not rows:
    raise SystemExit(f"manifest has no rows: {source}")
rows.sort(key=lambda row: int(float(row["episode_id"])))
fieldnames = list(rows[0])
target.parent.mkdir(parents=True, exist_ok=True)
with target.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(rows[0])
print(f"smoke_episode={rows[0]['episode_id']}")
PY

export CLASSIFY_MANIFEST="$SMOKE_MANIFEST"
export EXPECTED_EPISODES=1
export END_EPISODE="$("$PYTHON_BIN" -c 'import csv,sys; print(next(csv.DictReader(open(sys.argv[1], encoding="utf-8-sig", newline="")))["episode_id"])' "$SMOKE_MANIFEST")"
export CLASSIFY_VAL_FRACTION=0
export CLASSIFY_WINDOW_SIZE=1
export CLASSIFY_WINDOW_STRIDE=1
export EFFICIENT_MAX_STEPS="${EFFICIENT_MAX_STEPS:-1}"
export EFFICIENT_CHECKPOINT_INTERVAL="${EFFICIENT_CHECKPOINT_INTERVAL:-1}"
export EFFICIENT_GRAD_ACCUM="${EFFICIENT_GRAD_ACCUM:-1}"
export EFFICIENT_NUM_WORKERS="${EFFICIENT_NUM_WORKERS:-0}"
export EFFICIENT_REPORT_TO="${EFFICIENT_REPORT_TO:-none}"
export RUN_ROOT="$SMOKE_ROOT"

bash "$SCRIPT_DIR/train_efficientwam_classify.sh"

# Language embeddings and converted source tensors are also ``.pt`` files;
# require the trainer's exported Stage-2 checkpoint specifically so a failed
# optimizer run cannot be reported as a successful smoke test.
if ! find "$SMOKE_ROOT/efficient_checkpoints" -type f -path '*/exports/*stage2*_step_*.pt' -print -quit 2>/dev/null | grep -q .; then
  echo "Smoke did not produce an exported Stage-2 checkpoint under $SMOKE_ROOT/efficient_checkpoints" >&2
  exit 3
fi
echo "Efficient-WAM Classify Objects smoke completed: one episode, $EFFICIENT_MAX_STEPS step(s)."
