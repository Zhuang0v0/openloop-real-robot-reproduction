# Efficient-WAM data adapter

These utilities convert the 16D LeRobot export used by this project into the
intermediate and packed formats consumed by Efficient-WAM-RT. They do not
contain robot drivers and do not download model weights.

## Audit and stage

Run from the repository root (or replace the paths with your own data volume):

```bash
python adapters/efficient_wam/audit_convert.py audit \
  --dataset-root data/building_block_towers \
  --output runs/efficient_wam_audit \
  --probe-videos

python adapters/efficient_wam/audit_convert.py stage \
  --dataset-root data/building_block_towers \
  --output runs/efficient_wam_staged \
  --copy-mode hardlink \
  --write-torch
```

The adapter keeps `observation.state` and `action` separate. It maps the three
camera streams to `head`, `left_wrist`, and `right_wrist`, and records
episode-level shape and timing checks. The staged representation is an
intermediate format; the official Efficient-WAM packer still performs VAE,
UMT5, and safetensors packing.

## 16D checkpoint adaptation

The released RoboTwin RT checkpoint uses 14 channels. The real-robot layout is
`[L1..L7, Lgrip, R1..R7, Rgrip]` (16 channels). Use the project wrapper, which
calls `adapt_checkpoint_dims.py` with the verified index map:

```bash
MODEL_ROOT=/path/to/models \
PYTHON_BIN=python \
  bash scripts/adapt_efficient_checkpoint.sh
```

The wrapper writes the full 16D checkpoint, compact WAN payload, and a JSON
transformation report. Always inspect the report and run a one-batch model
load before a long training run.
