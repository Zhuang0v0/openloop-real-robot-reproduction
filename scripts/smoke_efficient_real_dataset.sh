set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
MODEL_ROOT="${MODEL_ROOT:-$PROJECT_ROOT/models}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs}"
EFFICIENT_WAM_ROOT="${EFFICIENT_WAM_ROOT:-$PROJECT_ROOT/vendor/Efficient-WAM}"
ADAPTER_ROOT="${ADAPTER_ROOT:-$PROJECT_ROOT/adapters/efficient_wam}"
SRC="${DATASET_ROOT:-$DATA_ROOT/building_block_towers}"
OUT="${EFFICIENT_SMOKE_SOURCE_ROOT:-$RUN_ROOT/efficient_wam_source_smoke}"
TASK="$OUT/clean/building_block_towers"
WAN="${WAN_MODEL_ROOT:-$MODEL_ROOT/Wan2.2-TI2V-5B}"
REPO="$EFFICIENT_WAM_ROOT"
CFG="$ADAPTER_ROOT/train_dataset_real_16d.yaml"
SMOKE_CFG="${EFFICIENT_SMOKE_CONFIG:-$RUN_ROOT/train_dataset_smoke_16d.yaml}"
PACKED="${EFFICIENT_SMOKE_PACKED_ROOT:-$RUN_ROOT/efficient_wam_train_dataset_smoke_16d}"

rm -rf "$OUT" "$PACKED"
"$PYTHON_BIN" "$ADAPTER_ROOT/lerobot_to_robotwin_clean.py" \
  --dataset-root "$SRC" --video-root "$SRC" --output-root "$OUT" \
  --start-episode 0 --end-episode 0
"$PYTHON_BIN" "$ADAPTER_ROOT/generate_umt5_wan.py" \
  --repo-root "$REPO" --wan-root "$WAN" --task-root "$TASK" --device cuda:0

"$PYTHON_BIN" - "$CFG" "$SMOKE_CFG" "$OUT" "$PACKED" <<'PY'
import sys, yaml
cfg=yaml.safe_load(open(sys.argv[1]))
cfg['output_root']=sys.argv[4]
cfg['source']['root']=sys.argv[3]
cfg['build']['batch_size']=1
cfg['build']['num_workers']=0
cfg['dataset']['validate_runtime']=True
yaml.safe_dump(cfg, open(sys.argv[2],'w'), sort_keys=False)
PY

cd "$REPO"
"$PYTHON_BIN" scripts/robotwin/build_train_dataset.py \
  --config "$SMOKE_CFG" --output-root "$PACKED" --device cuda:0 \
  --batch-size 1 --num-workers 0 --shard-size 512 --overwrite \
  --verify-condition-latent
"$PYTHON_BIN" scripts/robotwin/verify_train_dataset.py --root "$PACKED" --validate-runtime
