set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs}"
DATA="${DATASET_ROOT:-$DATA_ROOT/building_block_towers}"
OUT="${FASTWAM_STATS_FILE:-$RUN_ROOT/fastwam_real/dataset_stats.json}"
EXPECTED_EPISODES=${EXPECTED_EPISODES:-100}
mkdir -p "$(dirname "$OUT")"
DATA="$DATA" OUT="$OUT" EXPECTED_EPISODES="$EXPECTED_EPISODES" "$PYTHON_BIN" - <<'PY'
import json, os
from pathlib import Path
import re
import numpy as np
import pyarrow.parquet as pq

root = Path(os.environ['DATA'])
expected_episodes = int(os.environ['EXPECTED_EPISODES'])
files = sorted((root / 'data').rglob('episode_*.parquet')) or sorted(root.rglob('episode_*.parquet'))
if len(files) != expected_episodes:
    raise RuntimeError(f'expected {expected_episodes} episode parquet files, found {len(files)}')
def episode_id(path):
    match = re.search(r'episode_(\d+)\.parquet$', path.name)
    if not match:
        raise RuntimeError(f'cannot parse episode id from {path}')
    return int(match.group(1))
ids = sorted(episode_id(path) for path in files)
if ids != list(range(expected_episodes)):
    raise RuntimeError(f'expected episode ids 0..{expected_episodes - 1}, found {ids}')

arrays = {'action': [], 'state': []}
for path in files:
    table = pq.read_table(path, columns=['action', 'observation.state'])
    arrays['action'].append(np.asarray(table['action'].to_pylist(), dtype=np.float32))
    arrays['state'].append(np.asarray(table['observation.state'].to_pylist(), dtype=np.float32))

def stats(values):
    data = np.concatenate(values, axis=0)
    if data.ndim != 2 or data.shape[1] != 16 or not np.isfinite(data).all():
        raise ValueError(f'bad data shape/finite check: {data.shape}')
    return {
        'global_min': data.min(axis=0).tolist(),
        'global_max': data.max(axis=0).tolist(),
        'global_mean': data.mean(axis=0).tolist(),
        'global_std': data.std(axis=0).tolist(),
        'global_q01': np.quantile(data, 0.01, axis=0).tolist(),
        'global_q99': np.quantile(data, 0.99, axis=0).tolist(),
        'count': [int(data.shape[0])],
    }

payload = {
    'action': {'default': stats(arrays['action'])},
    'state': {'default': stats(arrays['state'])},
    'num_episodes': len(files),
    'training_episode_ids': list(range(expected_episodes)),
    'stats_scope': f'all {expected_episodes} episodes; no held-out evaluation split',
    'num_transition': int(sum(x.shape[0] for x in arrays['action'])),
}
target = Path(os.environ['OUT'])
tmp = target.with_suffix('.json.partial')
tmp.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
tmp.replace(target)
print(json.dumps({'output': str(target), 'episodes': len(files), 'transitions': payload['num_transition'], 'action_mean': payload['action']['default']['global_mean'], 'state_mean': payload['state']['default']['global_mean']}, indent=2))
PY
