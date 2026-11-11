set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs}"
DATA="${DATASET_ROOT:-$DATA_ROOT/nero_classify_objects}"
OUT="${FASTWAM_STATS_FILE:-$RUN_ROOT/fastwam_real/dataset_stats.json}"
EXPECTED_EPISODES=${EXPECTED_EPISODES:-106}
MANIFEST="${FASTWAM_STATS_MANIFEST:-}"
mkdir -p "$(dirname "$OUT")"
DATA="$DATA" OUT="$OUT" EXPECTED_EPISODES="$EXPECTED_EPISODES" MANIFEST="$MANIFEST" PROJECT_ROOT="$PROJECT_ROOT" "$PYTHON_BIN" - <<'PY'
import csv
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.environ['PROJECT_ROOT'])
from tools.analyze_classify_actions import load_episodes

root = Path(os.environ['DATA'])
expected_episodes = int(os.environ['EXPECTED_EPISODES'])
episodes = {int(item.episode_id): item for item in load_episodes(root)}
manifest = os.environ.get('MANIFEST', '').strip()
if manifest:
    selected = set()
    with Path(manifest).open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            raw = row.get('episode_id', row.get('episode_index', ''))
            if str(row.get('selected', 'true')).strip().lower() in {'0', 'false', 'no', 'off', 'rejected', 'drop'}:
                continue
            if raw is not None and str(raw).strip():
                selected.add(int(float(raw)))
    missing = sorted(selected - set(episodes))
    if missing:
        raise RuntimeError(f'manifest references missing episodes: {missing}')
    episodes = {episode_id: episodes[episode_id] for episode_id in sorted(selected)}
ids = sorted(episodes)
if expected_episodes > 0 and len(ids) != expected_episodes:
    raise RuntimeError(f'expected {expected_episodes} selected episodes, found {len(ids)}: {ids}')
if not ids:
    raise RuntimeError('no episodes available for FastWAM statistics')

arrays = {'action': [], 'state': []}
for episode_id in ids:
    item = episodes[episode_id]
    arrays['action'].append(np.asarray(item.action, dtype=np.float32))
    arrays['state'].append(np.asarray(item.state, dtype=np.float32))

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
    'num_episodes': len(ids),
    'training_episode_ids': ids,
    'stats_scope': f'all {len(ids)} selected episodes',
    'num_transition': int(sum(x.shape[0] for x in arrays['action'])),
}
target = Path(os.environ['OUT'])
tmp = target.with_suffix('.json.partial')
tmp.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
tmp.replace(target)
print(json.dumps({'output': str(target), 'episodes': len(ids), 'transitions': payload['num_transition'], 'action_mean': payload['action']['default']['global_mean'], 'state_mean': payload['state']['default']['global_mean']}, indent=2))
PY
