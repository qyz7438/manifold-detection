# HWM Loss-Causal Ablation

Date: 2026-07-09
Dataset: NWPU VHR-10 full validation
Seeds: 42, 2024
Detector: frozen Faster R-CNN MobileNetV3 FPN checkpoints
Training: action-local ROI transport head only, 4 epochs
Shared settings: `box_base=decoded`, `match_mode=class_agnostic`, batch size 8,
`num_workers=0`

## Goal

The previous HWM configuration improved AP75, AP50, precision, false-positive
rate, prediction count, and ECE. DeepSeek V4 Pro raised the concern that most of
the gain could come from direct supervised box refinement rather than the HWM
teacher. This ablation isolates:

- direct box refinement
- high-IoU preservation
- HWM anchoring
- the current combined configuration

## Configurations

| Tag | Energy | HWM | High-IoU preserve |
|---|---:|---:|---:|
| `boxonly` | 0.0 | 0.0 | 0.0 |
| `box+preserve` | 0.0 | 0.0 | 2.0 |
| `box+hwm` | 0.0 | 0.05 | 0.0 |
| `current` | 0.01 | 0.05 | 2.0 |

## Per-seed AP75

| Seed | Config | Default AP75 | Final AP75 | Delta AP75 | Best AP75 | Best epoch | Best-final gap |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | `boxonly` | 0.1923 | 0.2238 | +0.0315 | 0.2282 | 2 | 0.0044 |
| 42 | `box+preserve` | 0.1923 | 0.2285 | +0.0362 | 0.2285 | 4 | 0.0000 |
| 42 | `box+hwm` | 0.1923 | 0.2262 | +0.0339 | 0.2280 | 2 | 0.0018 |
| 42 | `current` | 0.1923 | 0.2270 | +0.0347 | 0.2275 | 1 | 0.0005 |
| 2024 | `boxonly` | 0.1471 | 0.1987 | +0.0516 | 0.1987 | 4 | 0.0000 |
| 2024 | `box+preserve` | 0.1471 | 0.2120 | +0.0649 | 0.2120 | 4 | 0.0000 |
| 2024 | `box+hwm` | 0.1471 | 0.1991 | +0.0520 | 0.1991 | 4 | 0.0000 |
| 2024 | `current` | 0.1471 | 0.2105 | +0.0634 | 0.2105 | 4 | 0.0000 |

## Mean metrics

| Config | AP50 delta | AP75 delta | Final AP75 | Best-final gap | Precision delta | Recall delta | FPR delta | Prediction delta | ECE delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `boxonly` | +0.0123 | +0.0416 | 0.2113 | 0.0022 | +0.0584 | -0.0030 | -0.0584 | -408.5 | -0.0101 |
| `box+preserve` | +0.0130 | +0.0505 | 0.2202 | 0.0000 | +0.0588 | +0.0001 | -0.0588 | -401.5 | -0.0068 |
| `box+hwm` | +0.0146 | +0.0429 | 0.2126 | 0.0009 | +0.0592 | -0.0016 | -0.0592 | -409.0 | -0.0088 |
| `current` | +0.0136 | +0.0490 | 0.2187 | 0.0002 | +0.0584 | +0.0005 | -0.0584 | -398.0 | -0.0077 |

## Loss magnitudes

| Config | Box loss | Preserve loss | HWM loss | Energy loss |
|---|---:|---:|---:|---:|
| `boxonly` | 0.04540 | 0.00000 | 0.000000 | 0.000000 |
| `box+preserve` | 0.04699 | 0.00443 | 0.000000 | 0.000000 |
| `box+hwm` | 0.04541 | 0.00000 | 0.000033 | 0.000000 |
| `current` | 0.04708 | 0.00435 | 0.000023 | 0.000146 |

## Interpretation

The main AP75 improvement is explained by direct supervised box refinement.
`boxonly` recovers about 85% of the current AP75 gain.

High-IoU preservation is the only useful auxiliary term in this ablation. It
adds about +0.009 mean AP75 over `boxonly` and removes the small best-final gap.
It should be treated as a consistency/stability regularizer.

HWM is not causal in the current configuration. Its loss is around 2e-05 to
3e-05, much smaller than the box and preserve terms, and `current` does not beat
`box+preserve`. The HWM/energy story should be downgraded to an exploratory or
negative-result branch unless later evidence changes this.

## Decision

Promote `box+preserve` to the next candidate configuration:

```bash
--box-base decoded \
--match-mode class_agnostic \
--energy-weight 0.0 \
--hwm-weight 0.0 \
--high-iou-preserve-weight 2.0
```

The next required check is `class_aware` versus `class_agnostic` matching. If the
gain depends heavily on `class_agnostic`, the method should be described more
conservatively as class-agnostic proposal refinement rather than a general
detector correction mechanism.
