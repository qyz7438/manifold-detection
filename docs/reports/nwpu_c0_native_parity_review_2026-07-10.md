# NWPU C0 Convergence and Native-Parity Review

Date: 2026-07-10
Dataset: NWPU VHR-10 full validation, seed/data-seed 42
Model: Faster R-CNN MobileNetV3-320-FPN at 480 px

## Decision

The 12-epoch detector is an under-adapted baseline. Ordinary full-model
fine-tuning explains more AP improvement than the historical action-local,
RLVR, HWM, or energy terms. In addition, the historical action evaluation
path fails zero-action parity and artificially suppresses predictions.

Do not use the historical action-local AP, precision, FPR, ECE, or prediction
count deltas as evidence for a learned transport action. Re-evaluation is
required through the native-parity path introduced in commit `6007db9`.

## C0: Full-Model Fine-Tune From 12-Epoch Best

Run: `ctrl_full_ft18_from12best_fullnw0_nwpu_s42_bs8_ep18`
Source checkpoint SHA256: `19845c4463d91b178c1289fc772a4bc0d34f8e9e0535951d7dd2b760fbb5f18a`
Recipe: optimizer reset, SGD, LR `0.001`, batch size 8, 18 epochs, no scheduler.

| Metric | Source | Final epoch 18 | Raw best |
|---|---:|---:|---:|
| AP50 | 0.518845 | 0.607227 | 0.607227 (epoch 18) |
| AP75 | 0.192277 | 0.281808 | 0.284649 (epoch 15) |
| Precision | 0.329755 | 0.426249 | - |
| Recall | 0.619597 | 0.680115 | - |
| ECE | 0.034371 | 0.093328 | - |
| Predictions | 1956 | 1661 | - |

Final deltas are AP50 `+0.088382` and AP75 `+0.089531`. The raw best AP75
delta is `+0.092372`. The AP gain therefore coexists with an ECE degradation
of `+0.058957`.

The robust convergence gate did not declare saturation:

- last three four-epoch AP75 medians: `0.267374`, `0.275207`, `0.280779`;
- AP75 block deltas: `+0.007833`, `+0.005572`;
- last-eight Theil-Sen AP75 slope: `+0.001714` per epoch;
- final-block AP75 IQR: `0.004058`.

This run is a low-LR fine-tune from a selected model checkpoint, not an
optimizer-continuous 30-epoch training run.

## P0: Historical Zero-Action Parity Failure

The action head was initialized with exact zero score and box actions. The
only change was routing predictions through the historical custom action
postprocessor.

| Checkpoint | Native AP75 | Zero-action AP75 | Prediction delta | Precision delta |
|---|---:|---:|---:|---:|
| 12-epoch best | 0.191656 | 0.193084 | -166 (-8.56%) | +0.029823 |
| C0 AP75 best | 0.281522 | 0.283097 | -58 (-3.54%) | +0.015668 |

Both aggregate parity checks failed on AP75 and prediction count. The main
causes were:

1. one argmax foreground class per proposal instead of all native foreground
   class candidates;
2. no native small-box filtering and different threshold semantics;
3. custom bbox decoding instead of the detector BoxCoder;
4. no `GeneralizedRCNNTransform.postprocess` coordinate restoration;
5. historical defaults of score threshold `0.001` and top-300 rather than
   native `0.05` and top-100.

The prediction suppression directly overlaps with the historical narrative of
higher precision, lower FPR, lower ECE, and fewer predictions. Those effects
cannot be assigned to learned actions.

## Native-Parity Repair

Commits `6007db9` and `5a15964` add:

- complete proposal groups, logits, box regression, and original image sizes
  to the action trace;
- detector BoxCoder decoding and full foreground-class expansion;
- native threshold, small-box removal, class-wise NMS, and top-K behavior;
- transform postprocessing back to original coordinates;
- aggregate and per-image strict zero-action parity diagnostics;
- separate `legacy` and default `native` postprocess modes.

Synthetic tests now match torchvision postprocessing exactly. Full NWPU
native-fix parity runs are queued as:

- `det_action_zero_parity_nativefix_baseline_s42`;
- `det_action_zero_parity_nativefix_fullft18best_s42`.

The action line remains blocked until both runs report exact-zero actions,
zero mismatched images, and passing aggregate and strict parity.

## Active Queue

1. `nwpu_mob_strong_cosine_s42_bs8_36ep`: from-pretrained 36-epoch full-model
   baseline, LR 0.005, two-epoch warmup, cosine decay to 0.00005.
2. Native-fix P0 parity runs, gated only by GPU2 free memory greater than 8 GiB.
3. `ctrl_full_refine8_from_ft18last_fullnw0_nwpu_s42_bs8_ep8_lr3e4`, serialized
   after the strong baseline.

Seeds 2024 and 999 are not expanded until the seed42 strong baseline and
native-parity gates are resolved.

## Claim Boundary

Supported:

- the old 12-epoch NWPU baseline was inadequate;
- ordinary detector training dominates prior AP gains;
- the legacy action postprocessor was a material confound;
- AP and calibration move in different directions under prolonged fine-tuning.

Unsupported:

- energy-guided transport independently improves detection AP;
- HWM supplies a useful causal teacher signal;
- historical precision/FPR/ECE gains were produced by learned actions;
- the current detector baseline is fully converged.
