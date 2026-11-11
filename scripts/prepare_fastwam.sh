set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FASTWAM_ROOT="${FASTWAM_ROOT:-$PROJECT_ROOT/vendor/FastWAM}"
MODEL_ROOT="${MODEL_ROOT:-$PROJECT_ROOT/models}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
DATASET_ROOT="${DATASET_ROOT:-$DATA_ROOT/nero_classify_objects}"
RUN="${FASTWAM_RUN_ROOT:-$RUN_ROOT/fastwam_real}"
CHECKPOINT="${FASTWAM_CHECKPOINT:-$MODEL_ROOT/robotwin_uncond_3cam_384_16d.pt}"
TASK_NAME="${FASTWAM_TASK_NAME:-classify_objects}"
DATA_CONFIG_NAME="${FASTWAM_DATA_CONFIG_NAME:-$TASK_NAME}"
TASK_CONFIG_NAME="${FASTWAM_TASK_CONFIG_NAME:-${TASK_NAME}_3cam_384_1e-4}"
MODEL_CONFIG_NAME="${FASTWAM_MODEL_CONFIG_NAME:-fastwam_real}"
INSTRUCTION="${FASTWAM_INSTRUCTION:-Put all fruits into the large basket first, then put all blocks into the small basket.}"
ACTION_DIT_PATH="${FASTWAM_ACTION_DIT_PRETRAINED_PATH:-$RUN/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
FASTWAM_PYTHON="${FASTWAM_PYTHON:-$PYTHON_BIN}"
FASTWAM_DEVICE="${FASTWAM_DEVICE:-cuda}"
FASTWAM_ACTION_DIT_DTYPE="${FASTWAM_ACTION_DIT_DTYPE:-bfloat16}"
FASTWAM_REDIRECT_COMMON_FILES="${FASTWAM_REDIRECT_COMMON_FILES:-true}"
test -d "$FASTWAM_ROOT" || { echo "FastWAM source not found: $FASTWAM_ROOT" >&2; exit 2; }
test -f "$CHECKPOINT" || { echo "FastWAM checkpoint not found: $CHECKPOINT" >&2; exit 2; }
DATA_TEMPLATE="${FASTWAM_DATA_TEMPLATE:-}"
if [[ -z "$DATA_TEMPLATE" ]]; then
  # Newer FastWAM checkouts use the read-only LeRobot v3 reader.  Prefer its
  # template when available; the older robotwin.yaml remains a fallback for
  # v2 datasets and older source revisions.
  if [[ -f "$FASTWAM_ROOT/configs/data/kuaidi2_lerobot_v30.yaml" ]]; then
    DATA_TEMPLATE="$FASTWAM_ROOT/configs/data/kuaidi2_lerobot_v30.yaml"
  else
    DATA_TEMPLATE="$FASTWAM_ROOT/configs/data/robotwin.yaml"
  fi
fi
test -f "$DATA_TEMPLATE" || { echo "FastWAM data template not found: $DATA_TEMPLATE" >&2; exit 2; }
TASK_TEMPLATE="${FASTWAM_TASK_TEMPLATE:-$FASTWAM_ROOT/configs/task/robotwin_uncond_3cam_384_1e-4.yaml}"
MODEL_TEMPLATE="${FASTWAM_MODEL_TEMPLATE:-$FASTWAM_ROOT/configs/model/fastwam.yaml}"
test -f "$TASK_TEMPLATE" || { echo "FastWAM task template not found: $TASK_TEMPLATE" >&2; exit 2; }
test -f "$MODEL_TEMPLATE" || { echo "FastWAM model template not found: $MODEL_TEMPLATE" >&2; exit 2; }
R="$FASTWAM_ROOT"
mkdir -p "$R/configs/data" "$R/configs/task" "$R/configs/model"
cp "$DATA_TEMPLATE" "$R/configs/data/$DATA_CONFIG_NAME.yaml"
cp "$TASK_TEMPLATE" "$R/configs/task/$TASK_CONFIG_NAME.yaml"
cp "$MODEL_TEMPLATE" "$R/configs/model/$MODEL_CONFIG_NAME.yaml"
PROJECT_ROOT="$PROJECT_ROOT" DATASET_ROOT="$DATASET_ROOT" R="$R" RUN="$RUN" CHECKPOINT="$CHECKPOINT" DATA_TEMPLATE="$DATA_TEMPLATE" \
TASK_NAME="$TASK_NAME" DATA_CONFIG_NAME="$DATA_CONFIG_NAME" TASK_CONFIG_NAME="$TASK_CONFIG_NAME" \
MODEL_CONFIG_NAME="$MODEL_CONFIG_NAME" INSTRUCTION="$INSTRUCTION" ACTION_DIT_PATH="$ACTION_DIT_PATH" "$FASTWAM_PYTHON" - <<'PY'
import os
from pathlib import Path
import yaml
base=Path(os.environ['PROJECT_ROOT']); dataset_root=Path(os.environ['DATASET_ROOT']); r=Path(os.environ['R']); run=Path(os.environ['RUN'])
checkpoint=Path(os.environ['CHECKPOINT'])
task_name=os.environ['TASK_NAME']; data_name=os.environ['DATA_CONFIG_NAME']; task_config_name=os.environ['TASK_CONFIG_NAME']; model_config_name=os.environ['MODEL_CONFIG_NAME']; instruction=os.environ['INSTRUCTION']; action_dit_path=Path(os.environ['ACTION_DIT_PATH'])
data=(r/f'configs/data/{data_name}.yaml').read_text()
data=data.replace('./data/robotwin2.0/robotwin2.0', str(dataset_root)).replace('${dataset_root}', str(dataset_root))
data=(data.replace('key: cam_high','key: camera_3').replace('key: cam_left_wrist','key: left_wrist').replace('key: cam_right_wrist','key: right_wrist')
     .replace('key: front','key: camera_3').replace('key: left\n','key: left_wrist\n').replace('key: right\n','key: right_wrist\n'))
data=data.replace('raw_shape: 14','raw_shape: 16').replace('shape: 14','shape: 16')
data=data.replace('action_output_dim: 14','action_output_dim: 16').replace('proprio_output_dim: 14','proprio_output_dim: 16')
old='pretrained_norm_stats: ./data/robotwin2.0/dataset_stats.json'
data=data.replace(old,'pretrained_norm_stats: '+str(run/'dataset_stats.json'))
data=data.replace('pretrained_norm_stats: null','pretrained_norm_stats: '+str(run/'dataset_stats.json'))
data=data.replace('text_embedding_cache_dir: ./data/text_embeds_cache/robotwin','text_embedding_cache_dir: '+str(run/'text_embeds'))
data=data.replace('text_embedding_cache_dir: ./data/text_embeds_cache/kuaidi2','text_embedding_cache_dir: '+str(run/'text_embeds'))
data=data.replace('context_len: 128','context_len: 128\n  video_backend: pyav')
# Disable the upstream random episode split.  The manifest adapter supplies
# the explicit train/validation IDs when the launcher requests a validation
# fraction; setting that fraction to zero disables validation altogether.
data=data.replace('val_set_proportion: 0.01','val_set_proportion: 0.0')
data=data.replace('val_set_proportion: 0.10','val_set_proportion: 0.0')
data=data.replace('val_set_proportion: 0.1','val_set_proportion: 0.0')
data=data.replace('  is_training_set: true','  is_training_set: true\n  override_instruction: '+instruction,1)
config=yaml.safe_load(data)
config['override_instruction'] = instruction
if isinstance(config.get('val'), dict):
    # Keep an explicit validation node available for the launcher.  The
    # manifest adapter, rather than the upstream random split, determines
    # which episode IDs enter this node.
    config['val']['is_training_set'] = False
    config['val']['val_set_proportion'] = 0.0
    config['val']['pretrained_norm_stats'] = str(run/'dataset_stats.json')
    config['val']['override_instruction'] = instruction
(r/f'configs/data/{data_name}.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
task=(r/f'configs/task/{task_config_name}.yaml').read_text()
task=task.replace('override /data: robotwin','override /data: '+task_name)
task=task.replace('override /model: fastwam','override /model: '+model_config_name)
task=task.replace('batch_size: 16','batch_size: 1').replace('num_workers: 8','num_workers: 2').replace('gradient_accumulation_steps: 1','gradient_accumulation_steps: 8').replace('save_every: 2500','save_every: 250').replace('eval_every: 500','eval_every: 0').replace('eval_every: 100','eval_every: 0').replace('resume: null','resume: '+str(checkpoint))
task += '\noutput_dir: '+str(run/'train')+'\n'
(r/f'configs/task/{task_config_name}.yaml').write_text(task)
model=(r/f'configs/model/{model_config_name}.yaml').read_text().replace('action_dit_pretrained_path: checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt','action_dit_pretrained_path: '+str(action_dit_path))
(r/f'configs/model/{model_config_name}.yaml').write_text(model)
print('prepared')
PY

if [[ ! -s "$ACTION_DIT_PATH" ]]; then
  test -f "$FASTWAM_ROOT/scripts/preprocess_action_dit_backbone.py" || {
    echo "FastWAM ActionDiT preprocessing script not found; set FASTWAM_ACTION_DIT_PRETRAINED_PATH to a compatible payload" >&2
    exit 2
  }
  mkdir -p "$(dirname "$ACTION_DIT_PATH")"
  DIFFSYNTH_MODEL_BASE_PATH="${FASTWAM_MODEL_BASE_PATH:-$MODEL_ROOT/fastwam_models}" \
    "$FASTWAM_PYTHON" "$FASTWAM_ROOT/scripts/preprocess_action_dit_backbone.py" \
      --model-config "$R/configs/model/$MODEL_CONFIG_NAME.yaml" \
      --output "$ACTION_DIT_PATH" --device "$FASTWAM_DEVICE" \
      --dtype "$FASTWAM_ACTION_DIT_DTYPE" \
      --redirect-common-files "$FASTWAM_REDIRECT_COMMON_FILES"
fi
test -s "$ACTION_DIT_PATH" || { echo "ActionDiT payload was not created: $ACTION_DIT_PATH" >&2; exit 2; }
echo '=== data config ==='
sed -n '1,180p' "$R/configs/data/$DATA_CONFIG_NAME.yaml"
echo '=== task config ==='
cat "$R/configs/task/$TASK_CONFIG_NAME.yaml"
