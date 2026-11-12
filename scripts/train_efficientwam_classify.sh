#!/usr/bin/env bash
set -euo pipefail

# Classify Objects only. The Excel-derived manifest is the authority for
# episode identity; this launcher never discovers labels from file ordering.
# A deterministic episode split is prepared before conversion so Efficient-
# WAM and FastWAM can use exactly the same train/validation IDs.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
DATASET_ROOT="${DATASET_ROOT:-$DATA_ROOT/nero_classify_objects}"
MODEL_ROOT="${MODEL_ROOT:-$PROJECT_ROOT/models}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs/classify_objects}"
EFFICIENT_WAM_ROOT="${EFFICIENT_WAM_ROOT:-$PROJECT_ROOT/vendor/Efficient-WAM}"
WAN_MODEL_ROOT="${WAN_MODEL_ROOT:-$MODEL_ROOT/Wan2.2-TI2V-5B}"
MANIFEST="${CLASSIFY_MANIFEST:-$RUN_ROOT/classify_episode_manifest.csv}"
STRATEGY="${CLASSIFY_FILTER_STRATEGY:-baseline}"
QUALITY_THRESHOLD="${CLASSIFY_QUALITY_THRESHOLD:-}"
INSTRUCTION="Put all fruits into the large basket first, then put all blocks into the small basket."
VAL_FRACTION="${CLASSIFY_VAL_FRACTION:-0.20}"
EXPECTED_EPISODES="${EXPECTED_EPISODES:-106}"
SEED="${CLASSIFY_SEED:-20261111}"

test -f "$MANIFEST" || { echo "Missing manifest: $MANIFEST" >&2; exit 2; }
test -d "$DATASET_ROOT" || { echo "Missing dataset: $DATASET_ROOT" >&2; exit 2; }
test -d "$EFFICIENT_WAM_ROOT" || { echo "Efficient-WAM source not found: $EFFICIENT_WAM_ROOT" >&2; exit 2; }
mkdir -p "$RUN_ROOT"

# Generate candidate windows for audit/reporting. The preparation step below
# reassigns the split from complete episode IDs, so overlapping windows from
# one episode cannot leak across train and validation.
WINDOW_MANIFEST="${CLASSIFY_WINDOW_MANIFEST:-$RUN_ROOT/${STRATEGY}_windows.json}"
FILTER_ARGS=(
  --dataset-root "$DATASET_ROOT" --manifest "$MANIFEST"
  --strategy "$STRATEGY" --window-size "${CLASSIFY_WINDOW_SIZE:-33}"
  --stride "${CLASSIFY_WINDOW_STRIDE:-16}" --output "$WINDOW_MANIFEST"
  --quality-csv "${CLASSIFY_QUALITY_CSV:-$RUN_ROOT/classify_action_quality.csv}"
  --seed "$SEED" --validation-fraction "$VAL_FRACTION" --test-fraction 0.0
  --report "$RUN_ROOT/${STRATEGY}_quality_report.md"
)
if [[ -n "$QUALITY_THRESHOLD" ]]; then
  FILTER_ARGS+=(--quality-threshold "$QUALITY_THRESHOLD")
fi
"$PYTHON_BIN" "$PROJECT_ROOT/tools/filter_classify_windows.py" "${FILTER_ARGS[@]}"

PREPARED_ROOT="${CLASSIFY_PREPARED_ROOT:-$RUN_ROOT/prepared}"
"$PYTHON_BIN" "$PROJECT_ROOT/tools/prepare_classify_data.py" \
  --manifest "$MANIFEST" --window-manifest "$WINDOW_MANIFEST" \
  --dataset-root "$DATASET_ROOT" --output-dir "$PREPARED_ROOT" \
  --strategy "$STRATEGY" --episodes "$EXPECTED_EPISODES" \
  --validation-fraction "$VAL_FRACTION" --split-mode "${CLASSIFY_SPLIT_MODE:-random}" \
  --seed "$SEED"

TRAIN_MANIFEST="${CLASSIFY_TRAIN_MANIFEST:-$PREPARED_ROOT/train_manifest.csv}"
VAL_MANIFEST="${CLASSIFY_VAL_MANIFEST:-$PREPARED_ROOT/val_manifest.csv}"
test -f "$TRAIN_MANIFEST" || { echo "Prepared train manifest missing: $TRAIN_MANIFEST" >&2; exit 2; }
test -f "$VAL_MANIFEST" || { echo "Prepared validation manifest missing: $VAL_MANIFEST" >&2; exit 2; }

# Count episodes, not windows. This count is passed to the converter as a
# guard against accidentally selecting a different split.
TRAIN_EPISODES=$("$PYTHON_BIN" -c 'import csv,sys; print(len({int(float(r["episode_id"])) for r in csv.DictReader(open(sys.argv[1], encoding="utf-8-sig", newline="")) if str(r.get("episode_id", "")).strip()}))' "$TRAIN_MANIFEST")
VAL_EPISODES=$("$PYTHON_BIN" -c 'import csv,sys; print(len({int(float(r["episode_id"])) for r in csv.DictReader(open(sys.argv[1], encoding="utf-8-sig", newline="")) if str(r.get("episode_id", "")).strip()}))' "$VAL_MANIFEST")
[[ "$TRAIN_EPISODES" -gt 0 ]] || { echo "Prepared train manifest has no episodes: $TRAIN_MANIFEST" >&2; exit 2; }

TRAIN_SOURCE_ROOT="${EFFICIENT_SOURCE_ROOT:-$RUN_ROOT/efficient_source_train}"
PROJECT_ROOT="$PROJECT_ROOT" DATA_ROOT="$DATA_ROOT" DATASET_ROOT="$DATASET_ROOT" \
RUN_ROOT="$RUN_ROOT" PYTHON_BIN="$PYTHON_BIN" TASK_NAME=classify_objects \
INSTRUCTION="$INSTRUCTION" START_EPISODE=0 END_EPISODE="${END_EPISODE:-105}" \
EXPECTED_EPISODES="$TRAIN_EPISODES" WAN_MODEL_ROOT="$WAN_MODEL_ROOT" \
EFFICIENT_SOURCE_ROOT="$TRAIN_SOURCE_ROOT" EPISODE_IDS_FILE="$TRAIN_MANIFEST" \
  bash "$SCRIPT_DIR/build_efficient_real_dataset.sh"

PROJECT_ROOT="$PROJECT_ROOT" DATA_ROOT="$DATA_ROOT" MODEL_ROOT="$MODEL_ROOT" \
RUN_ROOT="$RUN_ROOT" WAN_MODEL_ROOT="$WAN_MODEL_ROOT" EFFICIENT_SOURCE_ROOT="$TRAIN_SOURCE_ROOT" \
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/render_efficient_config.py" \
  "$PROJECT_ROOT/adapters/efficient_wam/train_dataset_classify_16d.yaml" \
  "$RUN_ROOT/train_dataset_classify_16d.yaml"

cd "$EFFICIENT_WAM_ROOT"
PYTHONPATH="$EFFICIENT_WAM_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" scripts/robotwin/build_train_dataset.py \
  --config "$RUN_ROOT/train_dataset_classify_16d.yaml" \
  --output-root "$RUN_ROOT/efficient_train_dataset" \
  --device "${EFFICIENT_DEVICE:-cuda:0}" --batch-size 1 --num-workers 2 \
  --shard-size 4096 --overwrite --verify-condition-latent

PYTHONPATH="$EFFICIENT_WAM_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" scripts/robotwin/verify_train_dataset.py \
  "$RUN_ROOT/efficient_train_dataset"

PROJECT_ROOT="$PROJECT_ROOT" DATA_ROOT="$DATA_ROOT" MODEL_ROOT="$MODEL_ROOT" \
RUN_ROOT="$RUN_ROOT" WAN_MODEL_ROOT="$WAN_MODEL_ROOT" \
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/render_efficient_config.py" \
  "$PROJECT_ROOT/adapters/efficient_wam/stage2_classify_16d.yaml" \
  "$RUN_ROOT/stage2_classify_16d.yaml"

echo "Prepared Classify Objects split: train episodes=$TRAIN_EPISODES, validation episodes=$VAL_EPISODES"
echo "Validation manifest: $VAL_MANIFEST"
echo "Efficient-WAM's upstream video-action trainer has no built-in validation loop;"
echo "use the validation manifest with scripts/evaluate_classify.sh or a matching"
echo "model-specific offline evaluator. It is never packed into the training root."

PYTHONPATH="$EFFICIENT_WAM_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -m train.stage2 \
  --config "$RUN_ROOT/stage2_classify_16d.yaml" \
  --device "${EFFICIENT_DEVICE:-cuda}" --per_device_batch_size "${EFFICIENT_BATCH_SIZE:-1}" \
  --num_workers "${EFFICIENT_NUM_WORKERS:-2}" \
  --gradient_accumulation_steps "${EFFICIENT_GRAD_ACCUM:-8}" \
  --max_steps "${EFFICIENT_MAX_STEPS:-1500}" \
  --checkpoint_interval "${EFFICIENT_CHECKPOINT_INTERVAL:-250}" \
  --report_to "${EFFICIENT_REPORT_TO:-tensorboard}"
