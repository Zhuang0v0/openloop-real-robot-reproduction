set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_DIR="${MODEL_ROOT:-$PROJECT_ROOT/models}"
ADAPTER="${FASTWAM_ADAPTER:-$SCRIPT_DIR/adapt_fastwam_checkpoint.py}"
IN="${FASTWAM_INPUT_CHECKPOINT:-$MODEL_DIR/robotwin_uncond_3cam_384.pt}"
OUT="${FASTWAM_OUTPUT_CHECKPOINT:-$MODEL_DIR/robotwin_uncond_3cam_384_16d.pt}"
REPORT="${FASTWAM_ADAPT_REPORT:-$MODEL_DIR/robotwin_uncond_3cam_384_16d.json}"
mkdir -p "$MODEL_DIR"
command -v "$PYTHON_BIN" >/dev/null
test -f "$IN" || { echo "Missing input checkpoint: $IN" >&2; exit 2; }
"$PYTHON_BIN" "$ADAPTER" \
  "$IN" \
  "$OUT" \
  --report "$REPORT"
sha256sum "$OUT"
