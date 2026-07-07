# ROI Structure Information-Gain Check on NWPU Round 2148 Cache

## Purpose

Before using ROI dual-energy as a training loss or action policy, this check
asks whether the current ROI structure features contain incremental information
beyond the detector's own label probability.

The tested structure features are prototype-local diagnostics computed from
train-split AP75 prototypes:

```text
proto_true_cos
proto_best_other_cos
proto_margin
proto_sqdist
compact_norm
basin_energy
proto_true_rank
class_support_log1p
class_support_inv
```

Two controls are used:

- global shuffle: preserves feature marginal distribution but breaks ROI
  alignment;
- class-conditional shuffle: preserves the predicted class distribution and
  class-conditioned feature marginal, but breaks per-ROI alignment.

The logistic probe selects its regularization strength by train-fold
cross-validation, then reports validation metrics.

## Commands

```powershell
E:\anaconda\01\envs\RLimage\python.exe scripts\analyze_roi_structure_information_gain.py `
  --cache E:\CLIproject\RLimage\runs\round2148_final_head_dim_full_ap75\candidate_features.npz `
  --target ap75 `
  --feature-keys features_l2,final_head_l2 `
  --shuffle-reps 20 `
  --bootstrap-reps 500 `
  --output output\roi_structure_information_gain_round2148.json `
  --markdown-output output\roi_structure_information_gain_round2148.md

E:\anaconda\01\envs\RLimage\python.exe scripts\analyze_roi_structure_information_gain.py `
  --cache E:\CLIproject\RLimage\runs\round2148_final_head_dim_full_ap75\candidate_features.npz `
  --target risky030_low_iou `
  --feature-keys features_l2,final_head_l2 `
  --shuffle-reps 20 `
  --bootstrap-reps 500 `
  --skip-raw-feature-controls `
  --output output\roi_structure_information_gain_risky030_round2148.json `
  --markdown-output output\roi_structure_information_gain_risky030_round2148.md
```

The same command was also run for:

```text
rescue075
very_low_conf_high_iou
underestimated_rank
```

## AP75 / Rescue Target

In this cache, `rescue075` is equivalent to AP75 for validation because all
AP75-positive candidates are below the 0.50 score threshold.

| feature | class weight | comparison | dAUC | dAP | global-shuffle dAP | class-shuffle dAP | verdict |
|---|---|---|---:|---:|---:|---:|---|
| features_l2 | none | prior + structure vs prior | -0.2704 | -0.1066 | -0.0185 | +0.0134 | no_gain |
| features_l2 | balanced | prior + structure vs prior | -0.2557 | -0.1082 | -0.0133 | +0.0209 | no_gain |
| final_head_l2 | none | prior + structure vs prior | -0.2030 | -0.0712 | -0.0020 | +0.0325 | no_gain |
| final_head_l2 | balanced | prior + structure vs prior | -0.1665 | -0.0591 | -0.0081 | +0.0640 | no_gain |

Raw feature controls also did not help:

| feature | class weight | comparison | dAUC | dAP | verdict |
|---|---|---|---:|---:|---|
| features_l2 | none | prior + raw feature vs prior | -0.1702 | -0.0664 | no_gain |
| features_l2 | balanced | prior + raw feature vs prior | -0.2096 | -0.1063 | no_gain |
| final_head_l2 | none | prior + raw feature vs prior | -0.1279 | -0.1011 | no_gain |
| final_head_l2 | balanced | prior + raw feature vs prior | -0.1629 | -0.1097 | no_gain |

## Action-Local Targets

### Risky Score>=0.30 Low-IoU

| feature | class weight | dAUC | dAP | global-shuffle dAP | class-shuffle dAP | verdict |
|---|---|---:|---:|---:|---:|---|
| features_l2 | none | -0.0014 | -0.0292 | +0.0025 | +0.0027 | no_gain |
| features_l2 | balanced | -0.0004 | -0.0104 | -0.0059 | +0.0024 | no_gain |
| final_head_l2 | none | +0.0003 | +0.0114 | -0.0108 | +0.0000 | not_above_shuffle |
| final_head_l2 | balanced | +0.0004 | +0.0173 | -0.0114 | -0.0043 | not_above_shuffle |

There is a tiny positive final-head signal for risky low-IoU candidates, but it
does not beat the shuffle controls, so it cannot be treated as real
incremental information.

### Very-Low-Confidence High-IoU

| feature | class weight | dAUC | dAP | global-shuffle dAP | class-shuffle dAP | verdict |
|---|---|---:|---:|---:|---:|---|
| features_l2 | none | -0.2450 | -0.0259 | +0.0067 | +0.0156 | no_gain |
| features_l2 | balanced | -0.1858 | -0.0217 | +0.0032 | +0.0150 | no_gain |
| final_head_l2 | none | -0.2601 | -0.0237 | +0.0064 | +0.0264 | no_gain |
| final_head_l2 | balanced | -0.2097 | -0.0145 | +0.0074 | +0.0359 | no_gain |

### Underestimated Rank

| feature | class weight | dAUC | dAP | global-shuffle dAP | class-shuffle dAP | verdict |
|---|---|---:|---:|---:|---:|---|
| features_l2 | none | -0.0004 | -0.0022 | -0.0022 | -0.0055 | no_gain |
| features_l2 | balanced | -0.0002 | -0.0018 | -0.0021 | -0.0068 | no_gain |
| final_head_l2 | none | -0.0002 | -0.0134 | -0.0060 | -0.0161 | no_gain |
| final_head_l2 | balanced | -0.0000 | -0.0132 | -0.0062 | -0.0147 | no_gain |

## Interpretation

This check does not support the current ROI dual-energy features as a
per-candidate score-rescue signal.

The earlier group-level observation is still true: the 75-dimensional final
head state has lower dual energy for AP75 positives than negatives.  But this
new test shows that the current local structure features do not add reliable
candidate-level information beyond the detector score, and sometimes shuffled
structure features perform as well or better.

The current conclusion should therefore be:

```text
The structure signal exists as a batch/group diagnostic, but the current
prototype-local feature definition is not yet an information-bearing action
signal for AP75 rescue or low-IoU suppression.
```

## Design Consequence

Do not promote the current dual-energy features directly into a loss or score
residual.  The next useful experiment should change the endpoint definition:

- target the action state after score, bbox regression, and NMS context;
- include verifier signals that directly distinguish high-IoU rescue candidates
  from low-IoU risky candidates;
- keep dual-energy as a logged structural diagnostic until it passes the
  information-gain gate.

