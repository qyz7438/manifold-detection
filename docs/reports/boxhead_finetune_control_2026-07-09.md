# Box-Head Fine-Tune Control For Action-Local Transport

Date: 2026-07-09
Dataset: NWPU VHR-10 full validation
Model: Faster R-CNN MobileNetV3-320-FPN
Seeds: 42, 2024

## Question

This control tests whether the current action-local candidate has information
gain beyond ordinary detector fine-tuning.

Current candidate before this control:

```bash
--box-base decoded \
--match-mode class_aware \
--energy-weight 0.0 \
--hwm-weight 0.0 \
--high-iou-preserve-weight 2.0
```

The equal-budget control fine-tunes the detector with:

```bash
--trainable-mode box_head_only \
--epochs 4 \
--lr 0.001 \
--batch-size 8
```

All runs start from the corresponding 12-epoch NWPU baseline checkpoint.

## Mean Results

All rows are mean deltas over seeds 42 and 2024 versus the corresponding
baseline checkpoint.

| Group | AP50 delta | AP75 delta | Best AP75 delta | Precision delta | Recall delta | FPR delta | ECE delta | Prediction delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `action class_aware boxonly` | +0.0146 | +0.0431 | +0.0460 | +0.0593 | -0.0013 | -0.0593 | -0.0092 | -407.5 |
| `action class_aware box+preserve` | +0.0132 | +0.0507 | +0.0507 | +0.0584 | -0.0005 | -0.0584 | -0.0093 | -401.5 |
| `box_head_only` fine-tune | +0.0376 | +0.0696 | +0.0698 | +0.0505 | +0.0200 | -0.0505 | +0.0076 | -316.0 |

## Per-Seed AP75

| Group | Seed 42 AP75 delta | Seed 2024 AP75 delta |
|---|---:|---:|
| `action class_aware box+preserve` | +0.0381 | +0.0632 |
| `box_head_only` fine-tune | +0.0528 | +0.0863 |

## Interpretation

The same-budget `box_head_only` control outperforms the action-local candidate
on AP50 and AP75. It also improves recall, while the action-local heads are
approximately recall-neutral or slightly negative.

The action-local heads still show a distinct precision/calibration profile:
they reduce prediction count more strongly, improve precision slightly more,
and improve ECE while `box_head_only` worsens ECE. This is a real behavioral
difference, but the current evidence suggests a conservative score/prediction
suppression bias rather than a stronger localization or transport mechanism.

This control weakens the claim that the current independent action head has
information gain beyond normal box-head adaptation. It should no longer be the
main validated method without a new causal advantage.

## DeepSeek V4 Pro Review

DeepSeek judged that the current action-local `box+preserve` configuration has
no evidence of information gain beyond ordinary fine-tuning:

- `box_head_only` is stronger on AP75 and AP50 under the same 4-epoch budget.
- The action-local value is mainly a precision/ECE tradeoff caused by more
  conservative prediction behavior.
- This conservative behavior may be useful for false-positive-sensitive
  settings, but is not enough to keep the independent action head as the main
  route.
- The ECE degradation of `box_head_only` is the main unresolved risk.

DeepSeek's next priorities:

1. Complete the 10-epoch, three-seed `box_head_only` control and compare it with
   `rlvr_random_fulltrain_10ep`.
2. Test calibration or threshold-preservation losses directly on `box_head_only`
   training before investing in a larger action-head redesign.

## Current Decision

Treat independent action-local `box+preserve` as a diagnostic/calibration
baseline, not as the active main method. The active question becomes whether a
simpler ROI/box-head correction can retain the AP gain of ordinary fine-tuning
while recovering the calibration and false-positive benefits that action-local
training showed.
