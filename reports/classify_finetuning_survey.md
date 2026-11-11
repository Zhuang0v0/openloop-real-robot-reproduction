# Classify Objects Fine-Tuning Survey

This document separates published/official evidence from engineering choices
that still need an experiment on the target data.

## Sources

- [FastWAM source](https://github.com/yuantianyuan01/FastWAM) and its
  [RoboTwin checkpoint](https://huggingface.co/yuanty/fastwam).
- [Efficient-WAM source](https://github.com/efficientwam/Efficient-WAM) and
  [Efficient-WAM-RT RoboTwin checkpoint](https://huggingface.co/jiajun0613/Efficient-WAM_RoboTwin/tree/main/Efficient-WAM-RT).
- [Efficient-WAM paper](https://arxiv.org/abs/2606.10040).
- [FastWAM paper](https://arxiv.org/abs/2603.16666).
- [Action Chunking with Transformers (ACT)](https://arxiv.org/abs/2304.13705).
- [Diffusion Policy](https://arxiv.org/abs/2303.04137).
- [DROID](https://arxiv.org/abs/2403.12945).
- [robomimic](https://arxiv.org/abs/2212.02944).

## Evidence from official implementations

1. Both requested initializations are pretrained RoboTwin checkpoints, so the
   first requirement is matching the checkpoint action/state dimensions and
   camera preprocessing before changing optimization.
2. Efficient-WAM-RT separates its RT inference schedule from training. The
   upstream code supports a continuous/1000-bin default, but this experiment
   explicitly sets **10 training flow-matching bins**. RT inference still uses
   ten total action solver steps with video refresh at steps 0 and 1. The
   10-bin training choice is an engineering configuration for this small-data
   run, not a claim that every paper or upstream checkpoint used that value.
3. Efficient-WAM exposes a staged action/video training path. The conservative
   first stage freezes the WAN/video expert and updates the action expert; a
   short joint stage is optional only after the action-only checkpoint loads.
4. FastWAM's official task configs use action chunking and a cosine optimizer
   schedule. The Classify Objects launcher keeps batch size small and uses
   gradient accumulation to fit a single GPU; it does not silently change the
   action horizon.

## Engineering choices for 106 episodes

- Start with action-expert-only adaptation: FastWAM freezes the video backbone;
  Efficient-WAM uses Stage 2 with frozen WAN. This reduces catastrophic
  forgetting on a small real-robot set.
- Use batch size 1 and accumulation 8. FastWAM starts at `5e-5`; Efficient-WAM
  starts at `5e-6` for action parameters. These are conservative starting
  points, not paper-certified optima.
- Compute normalization only from the selected training episodes. Never reuse
  building-block statistics, and never compute statistics from validation
  episodes.
- Split by complete episode, stratified by C1-C4. Action windows from one
  episode must not appear in both sets.
- Compare Baseline, Quality, Action and Combined filters. The unknown Excel
  score is never silently converted to zero or one.

## Open assumptions to validate

- The ModelScope export's action units, timestamp convention and actual FPS
  must be read from its metadata; the 30 Hz value in the config is a declared
  expectation, not a measurement.
- The checkpoint's text path must be inspected in a smoke run. A fixed prompt
  is only effective if it reaches the model's conditioning input.
- Physical velocity, acceleration and jerk limits depend on the robot
  controller. The analyzer therefore uses robust per-dimension statistics by
  default and accepts measured limits rather than imposing one global cutoff.
- Full joint/video fine-tuning, LoRA and longer schedules require measured
  validation loss and real-robot evidence; none is assumed to be better here.
