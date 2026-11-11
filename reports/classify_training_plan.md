# Classify Objects Training Plan

## Ordered execution

1. Parse the exact `classify objects` sheet and match `对应编号` to real
   ModelScope episode metadata.
2. Audit schema, dimensions, timestamps, videos and action/state units.
3. Analyze actions and produce the four filter manifests.
4. Split complete episodes into the same stratified train/validation manifest
   for both models.
5. Run one- or two-episode forward/loss/backward/checkpoint smoke tests for
   each model.
6. Start the baseline and filtered runs only after the smoke tests load the
   requested pretrained checkpoints.
7. Select checkpoints using validation loss and action error; report C1-C4
   metrics and real-robot success separately.

## Fair comparison

FastWAM and Efficient-WAM use the same episode IDs, split seed, filter
definition, fixed prompt and scene metadata. Model-specific normalization,
action head dimensions and learning rates remain separate. No action window
from an episode may cross the train/validation boundary.

## What is not claimed

No GPU training or real-robot success is claimed until the target workspace
produces logs and checkpoint hashes. Offline prediction error is not task
success rate.
