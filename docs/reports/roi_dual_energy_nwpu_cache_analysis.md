# ROI Dual-Energy NWPU Cache Analysis

## Local Data

The current `manifold` checkout has no local `data/` or `runs/`, but the older
project folder does:

```text
E:\CLIproject\RLimage\data\NWPU VHR-10 dataset
E:\CLIproject\RLimage\data\NWPU_VHR10_coco.json
E:\CLIproject\RLimage\data\PennFudanPed
E:\CLIproject\RLimage\data\VOCdevkit
E:\CLIproject\RLimage\data\coco
E:\CLIproject\RLimage\runs
```

The analyzed ROI cache is:

```text
E:\CLIproject\RLimage\runs\round2148_final_head_dim_full_ap75\candidate_features.npz
```

It contains `1484` train and `641` val candidates with 1024-dim box-head
features, 75-dim final-head features, class ids, AP75 labels, IoU, label
probabilities, logits, and bbox regression features.

## Commands

```powershell
E:\anaconda\01\envs\RLimage\python.exe scripts\analyze_roi_dual_energy_cache.py `
  --cache E:\CLIproject\RLimage\runs\round2148_final_head_dim_full_ap75\candidate_features.npz `
  --feature-key features_l2 `
  --output output\roi_dual_energy_round2148_features_l2.json

E:\anaconda\01\envs\RLimage\python.exe scripts\analyze_roi_dual_energy_cache.py `
  --cache E:\CLIproject\RLimage\runs\round2148_final_head_dim_full_ap75\candidate_features.npz `
  --feature-key final_head_l2 `
  --output output\roi_dual_energy_round2148_final_head_l2.json
```

The script builds train-split reference prototypes, then evaluates each val
candidate group with:

```text
E_intra = E_compact_to_train_proto + E_basin_under_train_proto
E_inter = relation/current-prototype alignment to train prototypes + class-index anchor + separation
DualEnergy = 0.5 * E_intra + 0.5 * E_inter
```

## Results

### 1024-D Box-Head Feature Space

| val group | count | E_intra | E_inter | DualEnergy | mean IoU | mean label prob |
|---|---:|---:|---:|---:|---:|---:|
| AP75 positive | 41 | 0.5068 | 0.1381 | 0.3225 | 0.7989 | 0.2690 |
| AP75 negative | 600 | 0.5250 | 0.1168 | 0.3209 | 0.1563 | 0.1033 |
| low-conf high-IoU | 23 | 0.6317 | 0.1395 | 0.3856 | 0.7898 | 0.1601 |

The 1024-dim box-head features do not give a clean dual-energy separation:
AP75 positives and negatives have nearly identical DualEnergy.  Positives have
slightly better `E_intra`, but worse `E_inter`.

### 75-D Final-Head Feature Space

| val group | count | E_intra | E_inter | DualEnergy | mean IoU | mean label prob |
|---|---:|---:|---:|---:|---:|---:|
| AP75 positive | 41 | 0.3199 | 0.1569 | 0.2384 | 0.7989 | 0.2690 |
| AP75 negative | 600 | 0.4734 | 0.1914 | 0.3324 | 0.1563 | 0.1033 |
| low-conf high-IoU | 23 | 0.4016 | 0.1593 | 0.2805 | 0.7898 | 0.1601 |

The 75-dim final-head space separates AP75 positives from negatives much more
clearly.  Low-confidence high-IoU candidates sit between them: good enough
relation/Iou structure, but weaker intra/basin structure than already accepted
AP75 positives.

## Interpretation

This supports the concern that optimizing the raw 1024-dim box-head feature
manifold may be the wrong place to impose the full dual-energy objective.  In
this cache, the dual-energy signal becomes meaningful after the detector head
has mixed class logits, probabilities, bbox regression, and summary features
into the final-head representation.

For the next real experiment, treat the 1024-dim feature manifold as an
intermediate diagnostic, not the primary endpoint.  The more promising target is
the action/final-head state:

```text
ROI box-head feature -> final/action state -> dual-energy constrained local action
```

This also matches the earlier action-local framing: the detector endpoint is not
only a class prototype in hidden feature space, but a valid post-head detection
state with class identity, score calibration, box quality, and NMS survival.

## Effect Check

I also checked whether per-candidate energy can directly rank AP75 positives.
This is the stricter test for using the energy as a score-rescue or reranking
signal.

| feature space | score | AP75 AUC | AP75 AP |
|---|---:|---:|---:|
| detector baseline | `label_prob` | 0.8293 | 0.2095 |
| 1024-d box-head | `-E_intra` | 0.4017 | 0.0568 |
| 75-d final-head | `-E_intra` | 0.4411 | 0.0553 |
| 75-d final-head | `-E_basin` | 0.5248 | 0.0728 |

Simple score fusions such as:

```text
label_prob + alpha * (-E_intra)
label_prob + alpha * (-E_basin)
label_prob + alpha * margin
```

did not beat `label_prob` on validation.  The best validation row remained the
plain detector label probability.

So the current effect is:

- useful as a batch/group structure diagnostic;
- useful for locating which representation space carries the class structure;
- not yet useful as a direct per-candidate reranking score;
- not ready to claim AP improvement.

This matters for method design.  The next implementation should not simply add
`-E_intra` to the score.  If we use dual energy in training, it should be a
group/batch regularizer on the final/action state, guarded by AP/FP/ECE metrics,
not a standalone rescue scorer.

## Limits

This is still an offline cache analysis.  It does not prove AP improvement.
The next step should add `roi_dual_energy` logging to a small real train/eval
run and compare AP50, AP75, false positives, ECE, prediction count, and
DualEnergy before/after.
