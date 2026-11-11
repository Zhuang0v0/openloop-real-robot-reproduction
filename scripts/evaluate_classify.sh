#!/usr/bin/env bash
set -euo pipefail

# Offline evaluation only; this script never labels prediction error as task
# success. Real-robot execution remains a separate, operator-controlled step.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to the Classify Objects export}"
MANIFEST="${CLASSIFY_MANIFEST:?Set CLASSIFY_MANIFEST to the matched manifest}"
CHECKPOINT="${CLASSIFY_EVAL_CHECKPOINT:?Set CLASSIFY_EVAL_CHECKPOINT to a model checkpoint}"
OUTPUT_DIR="${CLASSIFY_EVAL_OUTPUT:-$PROJECT_ROOT/runs/classify_objects/eval}"
mkdir -p "$OUTPUT_DIR"

"$PYTHON_BIN" "$PROJECT_ROOT/tools/audit_classify_dataset.py" \
  --dataset-root "$DATASET_ROOT" --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_DIR/audit" --probe-videos

echo "Offline audit complete. Model-specific prediction evaluation must be run"
echo "with the matching upstream runtime; report loss/error, not task success."
echo "checkpoint=$CHECKPOINT"
