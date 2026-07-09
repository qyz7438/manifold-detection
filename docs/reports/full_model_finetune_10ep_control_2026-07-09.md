# Full-Model 10-Epoch Attribution Control

Date: 2026-07-09
Dataset: NWPU VHR-10 full validation
Model: Faster R-CNN MobileNetV3-320-FPN
Seeds: 42, 2024, 999

## Question

This attribution control tests whether the previous `box_head_only`, RLVR, and
action-local AP gains mainly came from a 12-epoch baseline that had not fully
converged.

All runs start from the corresponding 12-epoch NWPU baseline checkpoint.

## Mean Results

All rows are mean deltas over seeds 42, 2024, and 999 versus the corresponding
baseline checkpoint.

| Group | AP50 delta | AP75 delta | Best AP75 delta | Precision delta | Recall delta | FPR delta | ECE delta | Prediction delta | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `direct_zero_fulltrain_10ep` | +0.0047 | +0.0304 | +0.0308 | +0.0023 | +0.0039 | -0.0023 | -0.0014 | -3.0 | 9.3 |
| `rlvr_random_fulltrain_10ep` | +0.0063 | +0.0350 | +0.0359 | +0.0042 | +0.0051 | -0.0042 | -0.0031 | -11.3 | 9.0 |
| `box_head_only_10ep` | +0.0426 | +0.0792 | +0.0858 | +0.0641 | +0.0205 | -0.0641 | +0.0160 | -354.0 | 7.3 |
| `full_model_10ep` | +0.0688 | +0.1058 | +0.1133 | +0.0927 | +0.0437 | -0.0927 | +0.0313 | -426.3 | 7.7 |

## Per-Seed `full_model_10ep`

| Seed | AP75 delta | Best AP75 delta | Best epoch | ECE delta | Prediction delta |
|---:|---:|---:|---:|---:|---:|
| 42 | +0.0792 | +0.0798 | 6 | +0.0586 | -182 |
| 2024 | +0.1135 | +0.1262 | 9 | +0.0012 | -824 |
| 999 | +0.1248 | +0.1339 | 8 | +0.0342 | -273 |

## Interpretation

The full-model continuation is stronger than `box_head_only_10ep` and much
stronger than RLVR or direct box-action training. This means the previous gains
cannot be interpreted as evidence for the current independent action-head
structure. A large part of the apparent post-training improvement is explained
by the 12-epoch baseline not being fully trained.

`box_head_only_10ep` still explains a large share of the full-model gain, which
suggests ROI classification/regression adaptation is an important component.
However, the full model gains more AP50, AP75, precision, recall, and prediction
suppression, so the problem is broader than only the box head.

Calibration remains the main weakness of the stronger continuation runs:
`full_model_10ep` worsens ECE more than `box_head_only_10ep`. The AP-best epochs
also do not always coincide with epoch 10, especially for seeds 2024 and 999.

## DeepSeek V4 Pro Review

DeepSeek judged that:

- The 12-epoch baseline is the dominant confound behind earlier AP gains.
- `box_head_only_10ep` captures much of the full-model improvement, but the full
  detector continuation is stronger.
- RLVR and independent action-local heads should not remain the main
  AP-improvement path under the current evidence.
- The next benchmark must use a stronger, better-converged baseline before any
  transport, calibration, or residual adapter claim is tested.

DeepSeek also mentioned AFM as a possible architecture route, but that is not
adopted here because the active project guide currently keeps AFM/FPN-SM as a
historical or separately requested branch. The part retained for this line is
the baseline adequacy warning.

## Current Decision

Do not claim that the current action-local, HWM, energy, or RLVR variants are
the cause of AP improvement on NWPU. Treat them as diagnostics unless they can
beat a stronger full-model continuation baseline.

The immediate next target is a convergence/strong-baseline step:

1. Extend or reproduce the NWPU baseline continuation long enough to identify
   where AP50/AP75 saturate.
2. Re-test any local residual, calibration, or threshold-preservation method
   against that stronger baseline, not against the 12-epoch checkpoint.
