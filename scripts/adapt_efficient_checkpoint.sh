set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_DIR="${MODEL_ROOT:-$PROJECT_ROOT/models}"
ADAPTER_DIR="${ADAPTER_ROOT:-$PROJECT_ROOT/adapters/efficient_wam}"
IN="${EFFICIENT_INPUT_CHECKPOINT:-$MODEL_DIR/Efficient-WAM-RT_stage3.pt}"
OUT="${EFFICIENT_OUTPUT_CHECKPOINT:-$MODEL_DIR/Efficient-WAM-RT_stage3_16d.pt}"
COMPACT="${EFFICIENT_COMPACT_CHECKPOINT:-$MODEL_DIR/Efficient-WAM-RT_compact_16d.pt}"
REPORT="${EFFICIENT_ADAPT_REPORT:-$MODEL_DIR/Efficient-WAM-RT_stage3_16d.json}"

mkdir -p "$MODEL_DIR"
command -v "$PYTHON_BIN" >/dev/null
test -f "$IN" || { echo "Missing input checkpoint: $IN" >&2; exit 2; }

"$PYTHON_BIN" "$ADAPTER_DIR/adapt_checkpoint_dims.py" \
  "$IN" "$OUT" --old-state-dim 14 --old-action-dim 14 \
  --index-map 0,1,2,3,4,5,7,8,9,10,11,12,13,15 \
  --compact-output "$COMPACT" --report "$REPORT"
sha256sum "$OUT" "$COMPACT"
"$PYTHON_BIN" "$ADAPTER_DIR/inspect_checkpoint.py" "$OUT"
