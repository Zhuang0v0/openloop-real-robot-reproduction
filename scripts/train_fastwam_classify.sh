#!/usr/bin/env bash
set -euo pipefail

# Classify Objects only. The caller supplies a downloaded dataset and a
# manifest produced from the exact Excel worksheet; no file-name ordering is
# used to assign labels.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
DATASET_ROOT="${DATASET_ROOT:-$DATA_ROOT/nero_classify_objects}"
MODEL_ROOT="${MODEL_ROOT:-$PROJECT_ROOT/models}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs/classify_objects}"
FASTWAM_ROOT="${FASTWAM_ROOT:-$PROJECT_ROOT/vendor/FastWAM}"
FASTWAM_PYTHON="${FASTWAM_PYTHON:-$FASTWAM_ROOT/.venv/bin/python}"
CHECKPOINT="${FASTWAM_CHECKPOINT:-$MODEL_ROOT/robotwin_uncond_3cam_384_16d.pt}"
MANIFEST="${CLASSIFY_MANIFEST:-$RUN_ROOT/classify_episode_manifest.csv}"
STRATEGY="${CLASSIFY_FILTER_STRATEGY:-baseline}"
QUALITY_CSV="${CLASSIFY_QUALITY_CSV:-$RUN_ROOT/classify_action_quality.csv}"
WINDOW_MANIFEST="${CLASSIFY_WINDOW_MANIFEST:-$RUN_ROOT/${STRATEGY}_windows.json}"
TRAIN_MANIFEST="${CLASSIFY_TRAIN_MANIFEST:-$RUN_ROOT/${STRATEGY}_train.csv}"
VAL_MANIFEST="${CLASSIFY_VAL_MANIFEST:-$RUN_ROOT/${STRATEGY}_val.csv}"
INSTRUCTION="Put all fruits into the large basket first, then put all blocks into the small basket."
VAL_FRACTION="${CLASSIFY_VAL_FRACTION:-0.20}"
EXPECTED_EPISODES="${EXPECTED_EPISODES:-106}"

test -f "$MANIFEST" || { echo "Missing manifest: $MANIFEST" >&2; exit 2; }
test -f "$CHECKPOINT" || { echo "Missing adapted 16D checkpoint: $CHECKPOINT" >&2; exit 2; }
test -d "$DATASET_ROOT" || { echo "Missing dataset: $DATASET_ROOT" >&2; exit 2; }
mkdir -p "$RUN_ROOT"

"$PYTHON_BIN" "$PROJECT_ROOT/tools/filter_classify_windows.py" \
  --dataset-root "$DATASET_ROOT" --manifest "$MANIFEST" \
  --strategy "$STRATEGY" --window-size "${CLASSIFY_WINDOW_SIZE:-33}" \
  --stride "${CLASSIFY_WINDOW_STRIDE:-16}" --output "$WINDOW_MANIFEST" \
  --quality-csv "$QUALITY_CSV" --seed "${CLASSIFY_SEED:-20261111}" \
  --validation-fraction "$VAL_FRACTION" --test-fraction 0.0 \
  --report "$RUN_ROOT/${STRATEGY}_quality_report.md"

PREPARED_ROOT="$RUN_ROOT/prepared"
"$PYTHON_BIN" "$PROJECT_ROOT/tools/prepare_classify_data.py" \
  --manifest "$MANIFEST" --window-manifest "$WINDOW_MANIFEST" \
  --dataset-root "$DATASET_ROOT" --output-dir "$PREPARED_ROOT" \
  --strategy "$STRATEGY" --episodes "$EXPECTED_EPISODES" \
  --validation-fraction "$VAL_FRACTION" --split-mode "${CLASSIFY_SPLIT_MODE:-random}" \
  --seed "${CLASSIFY_SEED:-20261111}"
TRAIN_MANIFEST="${CLASSIFY_TRAIN_MANIFEST:-$PREPARED_ROOT/train_manifest.csv}"
VAL_MANIFEST="${CLASSIFY_VAL_MANIFEST:-$PREPARED_ROOT/val_manifest.csv}"
test -f "$TRAIN_MANIFEST" || { echo "Prepared train manifest missing: $TRAIN_MANIFEST" >&2; exit 2; }

# Count the explicit episode IDs rather than deriving a count from window
# rows.  A single episode normally contributes many overlapping windows.
TRAIN_EPISODES=$("$PYTHON_BIN" -c 'import csv,sys; print(len({int(float(r["episode_id"])) for r in csv.DictReader(open(sys.argv[1], encoding="utf-8-sig", newline="")) if str(r.get("episode_id", "")).strip()}))' "$TRAIN_MANIFEST")
test -f "$VAL_MANIFEST" || { echo "Prepared validation manifest missing: $VAL_MANIFEST" >&2; exit 2; }
VAL_EPISODES=$("$PYTHON_BIN" -c 'import csv,sys; print(len({int(float(r["episode_id"])) for r in csv.DictReader(open(sys.argv[1], encoding="utf-8-sig", newline="")) if str(r.get("episode_id", "")).strip()}))' "$VAL_MANIFEST")
[[ "$TRAIN_EPISODES" -gt 0 ]] || { echo "Prepared train manifest has no episodes: $TRAIN_MANIFEST" >&2; exit 2; }

DATA_ROOT="$DATA_ROOT" DATASET_ROOT="$DATASET_ROOT" RUN_ROOT="$RUN_ROOT" \
PYTHON_BIN="$PYTHON_BIN" EXPECTED_EPISODES="$TRAIN_EPISODES" \
FASTWAM_STATS_MANIFEST="$TRAIN_MANIFEST" FASTWAM_STATS_FILE="$RUN_ROOT/fastwam/dataset_stats.json" \
  bash "$SCRIPT_DIR/generate_fastwam_stats.sh"

PROJECT_ROOT="$PROJECT_ROOT" FASTWAM_ROOT="$FASTWAM_ROOT" MODEL_ROOT="$MODEL_ROOT" \
RUN_ROOT="$RUN_ROOT" DATASET_ROOT="$DATASET_ROOT" PYTHON_BIN="$PYTHON_BIN" \
FASTWAM_TASK_NAME=classify_objects FASTWAM_DATA_CONFIG_NAME=classify_objects \
FASTWAM_TASK_CONFIG_NAME=classify_objects_3cam_384_5e-5 FASTWAM_MODEL_CONFIG_NAME=fastwam_classify \
FASTWAM_INSTRUCTION="$INSTRUCTION" FASTWAM_CHECKPOINT="$CHECKPOINT" \
FASTWAM_PYTHON="$FASTWAM_PYTHON" \
FASTWAM_ACTION_DIT_PRETRAINED_PATH="${FASTWAM_ACTION_DIT_PRETRAINED_PATH:-$RUN_ROOT/fastwam/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}" \
FASTWAM_RUN_ROOT="$RUN_ROOT/fastwam" \
  bash "$SCRIPT_DIR/prepare_fastwam.sh"

cd "$FASTWAM_ROOT"
export CLASSIFY_MANIFEST="$TRAIN_MANIFEST"
export FASTWAM_EPISODE_MANIFEST="$TRAIN_MANIFEST"
export FASTWAM_MANIFEST_SPLIT=""
if [[ -n "${FASTWAM_EXPECTED_TRAIN_EPISODES:-}" ]]; then
  export FASTWAM_EXPECTED_EPISODES="$FASTWAM_EXPECTED_TRAIN_EPISODES"
else
  export FASTWAM_EXPECTED_EPISODES="$TRAIN_EPISODES"
fi
if [[ "$VAL_EPISODES" -gt 0 ]]; then
  export FASTWAM_VALIDATION_MANIFEST="$VAL_MANIFEST"
  export FASTWAM_EXPECTED_VALIDATION_EPISODES="$VAL_EPISODES"
else
  unset FASTWAM_VALIDATION_MANIFEST
  unset FASTWAM_EXPECTED_VALIDATION_EPISODES
fi
export DIFFSYNTH_MODEL_BASE_PATH="${FASTWAM_MODEL_BASE_PATH:-$MODEL_ROOT/fastwam_models}"
HYDRA_VAL_ARGS=()
if [[ "$VAL_FRACTION" == "0" || "$VAL_FRACTION" == "0.0" || "$VAL_FRACTION" == "0.00" ]]; then
  # Explicitly disable upstream validation when the caller requests that all
  # episodes be used for training.  No train rows are reused as validation.
  HYDRA_VAL_ARGS+=("data.val=null" "eval_every=0")
fi
PYTHONPATH="$PROJECT_ROOT/adapters/fastwam${PYTHONPATH:+:$PYTHONPATH}" \
"$FASTWAM_PYTHON" "$PROJECT_ROOT/adapters/fastwam/run_with_manifest.py" \
  task=classify_objects_3cam_384_5e-5 \
  model=fastwam_classify data=classify_objects \
  batch_size="${FASTWAM_BATCH_SIZE:-1}" \
  num_workers="${FASTWAM_NUM_WORKERS:-2}" \
  gradient_accumulation_steps="${FASTWAM_GRAD_ACCUM:-8}" \
  max_steps="${FASTWAM_MAX_STEPS:-2000}" \
  save_every="${FASTWAM_SAVE_EVERY:-250}" \
  eval_every="${FASTWAM_EVAL_EVERY:-250}" \
  wandb.enabled=false \
  "${HYDRA_VAL_ARGS[@]}"
