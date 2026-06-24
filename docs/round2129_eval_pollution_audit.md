# Round 2129+ Eval Pollution Audit

Date: 2026-06-20

## Problem

`scripts/round2129_nwpu_posttrain_smoke.py` configured both the frozen baseline
model and the trainable model for rollout before running evaluation:

```text
roi_heads.score_thresh = rollout_score_threshold
roi_heads.detections_per_img = rollout_detections_per_img
```

For rescue runs this usually meant:

```text
rollout_score_threshold = 0.001
rollout_detections_per_img = 300
score_threshold = 0.05
```

The same mutated detector state was then reused by `evaluate(...)`. As a result,
`baseline_eval_metrics.json`, per-epoch AP/FP/ECE in `metrics_train.json`, and
`eval_metrics.json` measured rollout-mode detector outputs rather than clean
evaluation outputs.

## Affected Scope

The affected condition is:

```text
round2129_nwpu_posttrain_smoke.py
AND rescue_mode = true
AND rollout_score_threshold != score_threshold
```

A local scan found:

```text
103 top-level runs under runs/
30 trials under runs/round2207_adaptive_scene_fft/
```

The affected range includes most rescue/posttrain experiments from `round2129`
through `round2207`, especially:

```text
round2130-round2144
round2153
round2157-round2159
round2161-round2188
round2191-round2195
round2202-round2207
```

## Metrics To Downgrade

Treat these as polluted until re-evaluated with clean detector settings:

```text
baseline_eval_metrics.json
eval_metrics.json
metrics_train.json AP50/AP75/FP/ECE/num_predictions
checkpoint_best.pth selection decisions
adaptive_results.jsonl best/final AP summaries
```

## Still Useful With Caution

These artifacts remain useful for debugging signal and gradient behavior, but
must not be used as final detection performance evidence:

```text
rescue_reference_stats.json
verifier_offline_report.json
gradient diagnostics
loss curves
candidate/verifier coverage diagnostics
```

## Fix

Clean evaluation now uses `evaluate_clean_detector(...)`, which temporarily sets:

```text
roi_heads.score_thresh = score_threshold
roi_heads.detections_per_img = eval_detections_per_img
```

and restores the previous rollout state afterwards. Rollout candidate generation
still uses the rollout settings.

New CLI option:

```text
--eval-detections-per-img 100
```

## Required Re-Run Policy

Any future AP/FP/ECE claim for `round2129+` must come from a run after this fix,
or from explicit clean re-evaluation of an existing checkpoint.

Priority clean re-evaluation candidates:

```text
round2138 / round2139 / round2144
round2157 / round2159
round2188
round2203
round2206
round2207/015, 017, 027, 030
```
