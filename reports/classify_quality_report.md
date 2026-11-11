# Classify Objects Quality Report

Run `tools/analyze_classify_actions.py` followed by
`tools/filter_classify_windows.py` to populate this report. The raw dataset is
never overwritten.

Required sections in the generated report:

- episode and window counts by C1-C4;
- quality-score distribution, including unknown values;
- robust velocity/acceleration/jerk and pause statistics;
- gripper-specific events separated from arm-controller oscillation;
- state/action/video alignment and FPS checks;
- Baseline, Quality, Action and Combined retention, removal reasons and group
  balance;
- episodes/windows requiring human review.

An anomaly is a review signal, not an automatic deletion rule. Complete action
windows are filtered; frames are never removed from the middle of a trajectory
and concatenated. Episodes with severe damage are excluded as whole episodes.
