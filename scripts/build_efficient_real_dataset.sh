set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs}"
EFFICIENT_WAM_ROOT="${EFFICIENT_WAM_ROOT:-$PROJECT_ROOT/vendor/Efficient-WAM}"
ADAPTER_ROOT="${ADAPTER_ROOT:-$PROJECT_ROOT/adapters/efficient_wam}"
SRC="${DATASET_ROOT:-$DATA_ROOT/nero_classify_objects}"
OUT="${EFFICIENT_SOURCE_ROOT:-$RUN_ROOT/efficient_wam_source_16d}"
TASK_NAME="${TASK_NAME:-classify_objects}"
INSTRUCTION="${INSTRUCTION:-Put all fruits into the large basket first, then put all blocks into the small basket.}"
TASK="$OUT/clean/$TASK_NAME"
WAN="${WAN_MODEL_ROOT:-${MODEL_ROOT:-$PROJECT_ROOT/models}/Wan2.2-TI2V-5B}"
DEVICE="${EFFICIENT_DEVICE:-cuda:0}"
START_EPISODE=${START_EPISODE:-0}
END_EPISODE=${END_EPISODE:-105}
EXPECTED_EPISODES=${EXPECTED_EPISODES:-106}
EPISODE_IDS_FILE="${EPISODE_IDS_FILE:-}"

command -v "$PYTHON_BIN" >/dev/null
CONVERTER_ARGS=(
  --dataset-root "$SRC"
  --video-root "$SRC"
  --output-root "$OUT"
  --task-name "$TASK_NAME"
  --instruction "$INSTRUCTION"
  --start-episode "$START_EPISODE"
  --end-episode "$END_EPISODE"
)
if [[ -n "$EPISODE_IDS_FILE" ]]; then
  CONVERTER_ARGS+=(--episode-ids-file "$EPISODE_IDS_FILE")
fi
"$PYTHON_BIN" "$ADAPTER_ROOT/lerobot_to_robotwin_clean.py" "${CONVERTER_ARGS[@]}"


"$PYTHON_BIN" "$ADAPTER_ROOT/generate_umt5_wan.py" \
  --repo-root "$EFFICIENT_WAM_ROOT" \
  --wan-root "$WAN" \
  --task-root "$TASK" \
  --device "$DEVICE"

echo '=== clean source counts ==='
find "$TASK/qpos" -name '*.pt' | wc -l
find "$TASK/state" -name '*.pt' | wc -l
find "$TASK/videos" -name '*.mp4' | wc -l
find "$TASK/umt5_wan" -name '*.pt' | wc -l
TASK="$TASK" EXPECTED_EPISODES="$EXPECTED_EPISODES" "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import json, torch
import os
root=Path(os.environ['TASK'])
q=sorted((root/'qpos').glob('*.pt')); s=sorted((root/'state').glob('*.pt')); v=sorted((root/'videos').glob('*.mp4')); l=sorted((root/'umt5_wan').glob('*.pt'))
expected=int(__import__('os').environ.get('EXPECTED_EPISODES','100'))
assert len(q)==len(s)==len(v)==len(l)==expected, (len(q),len(s),len(v),len(l),expected)
qa=torch.load(q[0],map_location='cpu'); st=torch.load(s[0],map_location='cpu'); emb=torch.load(l[0],map_location='cpu')
assert tuple(qa.shape[1:])==(16,), qa.shape
assert tuple(st.shape[1:])==(16,), st.shape
assert isinstance(emb,list) and emb and tuple(emb[0].shape)[1]==4096, [tuple(x.shape) for x in emb]
print(json.dumps({'episodes':expected,'first_action_shape':list(qa.shape),'first_state_shape':list(st.shape),'first_lang_shapes':[list(x.shape) for x in emb]}, indent=2))
PY
