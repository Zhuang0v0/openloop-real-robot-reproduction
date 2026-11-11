set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FASTWAM_ROOT="${FASTWAM_ROOT:-$PROJECT_ROOT/vendor/FastWAM}"
MODEL_ROOT="${MODEL_ROOT:-$PROJECT_ROOT/models}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
DATASET_ROOT="${DATASET_ROOT:-$DATA_ROOT/building_block_towers}"
RUN="${FASTWAM_RUN_ROOT:-$RUN_ROOT/fastwam_real}"
CHECKPOINT="${FASTWAM_CHECKPOINT:-$MODEL_ROOT/robotwin_uncond_3cam_384_16d.pt}"
test -d "$FASTWAM_ROOT" || { echo "FastWAM source not found: $FASTWAM_ROOT" >&2; exit 2; }
test -f "$FASTWAM_ROOT/configs/data/robotwin.yaml" || { echo "FastWAM config layout is not recognized" >&2; exit 2; }
R="$FASTWAM_ROOT"
mkdir -p "$R/configs/data" "$R/configs/task" "$R/configs/model"
cp "$R/configs/data/robotwin.yaml" "$R/configs/data/building_block_towers.yaml"
cp "$R/configs/task/robotwin_uncond_3cam_384_1e-4.yaml" "$R/configs/task/building_block_towers_3cam_384_1e-4.yaml"
cp "$R/configs/model/fastwam.yaml" "$R/configs/model/fastwam_real.yaml"
PROJECT_ROOT="$PROJECT_ROOT" DATASET_ROOT="$DATASET_ROOT" R="$R" RUN="$RUN" CHECKPOINT="$CHECKPOINT" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
import yaml
base=Path(os.environ['PROJECT_ROOT']); dataset_root=Path(os.environ['DATASET_ROOT']); r=Path(os.environ['R']); run=Path(os.environ['RUN'])
checkpoint=Path(os.environ['CHECKPOINT'])
data=(r/'configs/data/building_block_towers.yaml').read_text()
data=data.replace('./data/robotwin2.0/robotwin2.0', str(dataset_root))
data=data.replace('cam_high','camera_3').replace('cam_left_wrist','left_wrist').replace('cam_right_wrist','right_wrist')
data=data.replace('raw_shape: 14','raw_shape: 16').replace('shape: 14','shape: 16')
data=data.replace('action_output_dim: 14','action_output_dim: 16').replace('proprio_output_dim: 14','proprio_output_dim: 16')
old='pretrained_norm_stats: ./data/robotwin2.0/dataset_stats.json'
data=data.replace(old,'pretrained_norm_stats: '+str(run/'dataset_stats.json'))
data=data.replace('text_embedding_cache_dir: ./data/text_embeds_cache/robotwin','text_embedding_cache_dir: '+str(run/'text_embeds'))
data=data.replace('context_len: 128','context_len: 128\n  video_backend: pyav')
# The new real-data experiment uses all 100 episodes for fine-tuning.  Keep
# validation disabled; real-robot evaluation is performed after training.
data=data.replace('val_set_proportion: 0.01','val_set_proportion: 0.0')
data=data.replace('val_set_proportion: 0.10','val_set_proportion: 0.0')
data=data.replace('  is_training_set: true','  is_training_set: true\n  override_instruction: Stack the blocks into a tower.',1)
config=yaml.safe_load(data)
config['val'] = None
config['override_instruction'] = 'Stack the blocks into a tower.'
(r/'configs/data/building_block_towers.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
task=(r/'configs/task/building_block_towers_3cam_384_1e-4.yaml').read_text()
task=task.replace('override /data: robotwin','override /data: building_block_towers').replace('batch_size: 16','batch_size: 1').replace('num_workers: 8','num_workers: 2').replace('gradient_accumulation_steps: 1','gradient_accumulation_steps: 8').replace('save_every: 2500','save_every: 250').replace('eval_every: 500','eval_every: 0').replace('eval_every: 100','eval_every: 0').replace('resume: null','resume: '+str(checkpoint))
task += '\noutput_dir: '+str(run/'train')+'\n'
(r/'configs/task/building_block_towers_3cam_384_1e-4.yaml').write_text(task)
model=(r/'configs/model/fastwam_real.yaml').read_text().replace('action_dit_pretrained_path: checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt','action_dit_pretrained_path: '+str(run/'ActionDiT_linear_interp_Wan22_16d.pt'))
(r/'configs/model/fastwam_real.yaml').write_text(model)
print('prepared')
PY
echo '=== data config ==='
sed -n '1,180p' "$R/configs/data/building_block_towers.yaml"
echo '=== task config ==='
cat "$R/configs/task/building_block_towers_3cam_384_1e-4.yaml"
