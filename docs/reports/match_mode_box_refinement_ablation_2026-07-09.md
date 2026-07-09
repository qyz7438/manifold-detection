# Match-Mode Box Refinement Ablation

Date: 2026-07-09
Dataset: NWPU VHR-10 full validation
Seeds: 42, 2024
Detector: frozen Faster R-CNN MobileNetV3 FPN checkpoints
Training: action-local ROI transport head only, 4 epochs
Shared settings: `box_base=decoded`, batch size 8, `num_workers=0`

## Goal

The loss-causal ablation showed that direct supervised box refinement plus
high-IoU preservation explains the useful AP75 gain. A remaining concern was
that `match_mode=class_agnostic` might relax proposal-to-GT matching and inflate
the result. This experiment compares `class_aware` against the previous
`class_agnostic` results for the two relevant configurations.

## Results

| Match mode | Config | AP50 delta | AP75 delta | Final AP75 | Best-final gap | Precision delta | Recall delta | FPR delta | Prediction delta | ECE delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `class_aware` | `boxonly` | +0.0146 | +0.0431 | 0.2127 | 0.0029 | +0.0593 | -0.0013 | -0.0593 | -407.5 | -0.0092 |
| `class_aware` | `box+preserve` | +0.0132 | +0.0507 | 0.2203 | 0.0000 | +0.0584 | -0.0005 | -0.0584 | -401.5 | -0.0093 |
| `class_agnostic` | `boxonly` | +0.0123 | +0.0416 | 0.2113 | 0.0022 | +0.0584 | -0.0030 | -0.0584 | -408.5 | -0.0101 |
| `class_agnostic` | `box+preserve` | +0.0130 | +0.0505 | 0.2202 | 0.0000 | +0.0588 | +0.0001 | -0.0588 | -401.5 | -0.0068 |

Per-seed AP75 deltas:

| Seed | Config | Class-agnostic | Class-aware |
|---:|---|---:|---:|
| 42 | `boxonly` | +0.0315 | +0.0316 |
| 42 | `box+preserve` | +0.0362 | +0.0381 |
| 2024 | `boxonly` | +0.0516 | +0.0545 |
| 2024 | `box+preserve` | +0.0649 | +0.0632 |

## Interpretation

The result does not support the matching-relaxation artifact hypothesis.
`class_aware` matching preserves the AP75 gain and is slightly stronger in three
of four per-seed comparisons. The safest candidate configuration is therefore:

```bash
--box-base decoded \
--match-mode class_aware \
--energy-weight 0.0 \
--hwm-weight 0.0 \
--high-iou-preserve-weight 2.0
```

The next critical control is an equal-budget standard `box_head_only` fine-tune.
If that baseline recovers the same AP75 gain, the current method should be
described as a training-capacity effect rather than a distinct refinement
mechanism. If it falls short, the decoupled action-local head has a stronger
causal claim.
