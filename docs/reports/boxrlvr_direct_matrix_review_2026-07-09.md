# BoxRLVR And Direct Box Action Matrix

Date: 2026-07-09
Dataset: NWPU VHR-10 full validation
Seeds: 42, 2024, 999
Scope: remote `boxRLVR` and direct box-action runs completed before the
same-budget `box_head_only` control.

## Why This Matrix Matters

The supervised action-local line now has a cleaner candidate:

```bash
--box-base decoded \
--match-mode class_aware \
--energy-weight 0.0 \
--hwm-weight 0.0 \
--high-iou-preserve-weight 2.0
```

That candidate reaches about `+0.0507` mean AP75 with large precision and
false-positive-rate movement. The boxRLVR/direct matrix is reviewed separately
to decide whether RLVR still has a useful role or should remain diagnostic.

## Three-Seed Summary

All rows are mean deltas versus the corresponding seed baseline.

| Group | AP50 delta | AP75 delta | Best AP75 delta | Precision delta | Recall delta | FPR delta | Prediction delta | ECE delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `rlvr_zero_ltr256_3ep` | +0.0034 | +0.0269 | +0.0287 | +0.0022 | +0.0028 | -0.0022 | -6.0 | -0.0010 |
| `rlvr_random_ltr256_3ep` | +0.0042 | +0.0281 | +0.0285 | +0.0034 | +0.0042 | -0.0034 | -9.3 | -0.0021 |
| `rlvr_spectral_ltr256_3ep` | +0.0023 | +0.0260 | +0.0260 | +0.0015 | +0.0018 | -0.0015 | -4.0 | -0.0009 |
| `rlvr_spectral_shuffled_ltr256_3ep` | -0.0008 | +0.0024 | +0.0027 | -0.0003 | -0.0008 | +0.0003 | -1.3 | +0.0003 |
| `rlvr_random_ltr256_10ep` | +0.0058 | +0.0316 | +0.0336 | +0.0033 | +0.0043 | -0.0033 | -7.7 | -0.0022 |
| `direct_zero_ltr256_3ep` | +0.0000 | +0.0068 | +0.0068 | +0.0005 | +0.0003 | -0.0005 | -2.3 | -0.0002 |
| `direct_zero_fulltrain_3ep` | +0.0017 | +0.0185 | +0.0185 | +0.0008 | +0.0011 | -0.0008 | -1.7 | -0.0005 |
| `direct_zero_fulltrain_10ep` | +0.0047 | +0.0304 | +0.0308 | +0.0023 | +0.0039 | -0.0023 | -3.0 | -0.0014 |
| `rlvr_random_fulltrain_3ep` | +0.0059 | +0.0257 | +0.0287 | +0.0032 | +0.0051 | -0.0032 | -4.7 | -0.0021 |
| `rlvr_random_fulltrain_10ep` | +0.0063 | +0.0350 | +0.0359 | +0.0042 | +0.0051 | -0.0042 | -11.3 | -0.0031 |

## Interpretation

The RLVR groups are not pure noise. Real RLVR contexts improve AP75 by roughly
`+0.026` to `+0.032`, while the shuffled spectral control is near zero. This
means the IoU reward structure carries usable signal.

However, the signal is much weaker than supervised `box+preserve`. RLVR barely
moves precision, false-positive rate, prediction count, or ECE. It therefore
looks like weak local box or score ranking search rather than a detector-level
decision improvement.

The spectral context does not add value over random or zero context. Its main
use was as a control: real spectral is much better than shuffled, but no better
than random/zero. Keeping spectral in the active path would add complexity
without current evidence of benefit.

Direct box action improves with more data and longer training, reaching about
`+0.030` AP75 at full-train 10 epochs. That is similar to RLVR but still below
the supervised `box+preserve` result. This supports the view that reward-based
box movement has signal but lower information density than direct supervised
box refinement.

## DeepSeek V4 Pro Review

DeepSeek V4 Pro judged that:

- RLVR has weak but real signal.
- The signal is not RLVR-specific; it is mostly from the IoU reward structure.
- Spectral context should not be kept as an active method because random and
  zero contexts perform as well or better.
- Direct box action and RLVR converge toward a similar `+0.03` AP75 ceiling.
- The same-budget `box_head_only` fine-tune control is the next required causal
  experiment.

If RLVR is revisited after the box-head control, the minimum repair path is:

1. Run an offline reward-landscape scan around proposals.
2. Enable score actions, for example `score_scale=0.05`.
3. Add a small move penalty, for example `move_penalty=0.01`.

## Current Queue

The same-budget `box_head_only` control is queued on the remote server via:

```bash
runs/ctrl_boxhead_ft_20260709_launcher.log
```

It waits for currently running `round28_train_eval.py` jobs before starting.

A longer 10-epoch, three-seed box-head control is also queued:

```bash
runs/ctrl_boxhead_ft10_20260709_launcher.log
```

It waits for the 4-epoch box-head control outputs before starting, then runs
seeds 42, 2024, and 999.
