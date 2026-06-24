# Spectral Quality Head Results - 2026-06-03

This run replaces image-level reward-weighted detector post-training with an offline Spectral Quality Head. The detector is fixed; frequency/structure evidence is used only for candidate quality estimation and reranking.

## Setup

- Detector checkpoint: `runs/mvp_pf_baseline/checkpoint_last.pth`
- Candidate source: final Faster R-CNN detections after detector NMS
- Candidate cache:
  - Train clean: 136 images, 446 candidates
  - Val clean: 34 images, 116 candidates
  - Val random patch: 34 images, 111 candidates
  - Val checkerboard patch: 34 images, 118 candidates
- Features per candidate:
  - TorchVision ROI-head box feature
  - ROI radial amplitude profile
  - Sobel/gradient structure feature with low-frequency phase summary
- Quality target:
  - TP: `IoU * normalized R_amp`
  - FP/unmatched: `0`
- Loss:
  - `BCEWithLogits(q_spec, q_target)`
  - same-image pairwise ranking loss between TP and FP candidates

## Quality Head Validation

| Head | Feature Mode | q_spec AUC TP vs FP | Mean q TP | Mean q FP | q-IoU Corr |
|---|---|---:|---:|---:|---:|
| ROI only | `roi` | 0.9559 | 0.7654 | 0.1417 | 0.7538 |
| ROI + spectral | `roi_amp_structure` | 0.9471 | 0.7370 | 0.0836 | 0.7692 |

Interpretation: both learned heads separate TP/FP well. ROI features are already very strong; adding amplitude/structure makes FP scores lower and improves IoU correlation slightly, but does not improve AUC on this small split.

## Clean Reranking

| Method | Combine | AP50 | Precision | Recall | High-conf FP | ECE |
|---|---|---:|---:|---:|---:|---:|
| Baseline | detector score | 0.8736 | 0.6983 | 0.8901 | 0.0282 | 0.3017 |
| Oracle R_amp | blend 0.7 | 0.8746 | 0.6983 | 0.8901 | 0.0405 | 0.2760 |
| ROI head | blend 0.7 | 0.8667 | 0.7143 | 0.8791 | 0.0282 | 0.1433 |
| ROI + spectral | blend 0.7 | 0.8583 | 0.7182 | 0.8681 | 0.0145 | 0.1663 |
| ROI + spectral | blend 0.9 | 0.8741 | 0.7043 | 0.8901 | 0.0282 | 0.1663 |

Interpretation: `alpha=0.9` is the safer reranking strength: it keeps clean AP50/Recall at baseline level while improving precision and ECE. `alpha=0.7` is more aggressive: it halves high-confidence FP, but costs some AP50 and recall.

## Patch Reranking

| Method | Patch | Combine | AP50 | Precision | Recall | High-conf FP | ECE |
|---|---|---|---:|---:|---:|---:|---:|
| Baseline | random | score | 0.8343 | 0.6937 | 0.8462 | 0.0282 | 0.3063 |
| ROI + spectral | random | blend 0.7 | 0.8263 | 0.7238 | 0.8352 | 0.0143 | 0.1517 |
| ROI + spectral | random | blend 0.9 | 0.8346 | 0.7130 | 0.8462 | 0.0282 | 0.1517 |
| Baseline | checkerboard | score | 0.8611 | 0.6780 | 0.8791 | 0.0286 | 0.3220 |
| ROI + spectral | checkerboard | blend 0.7 | 0.8538 | 0.7054 | 0.8681 | 0.0147 | 0.1736 |
| ROI + spectral | checkerboard | blend 0.9 | 0.8542 | 0.6870 | 0.8681 | 0.0286 | 0.1736 |

Interpretation: learned reranking does not yet improve patch AP50, but it avoids the earlier post-training collapse. It reduces ordinary FP and calibration error, and the stronger `alpha=0.7` setting reduces high-confidence FP with only a small recall drop.

## Conclusion

The new direction is validated at the engineering and diagnostic level:

- `R_amp` remains a useful oracle signal, but its absolute scale is too compressed to use as a direct detector reward.
- A learned quality head can separate TP/FP candidates with AUC around 0.95.
- Reranking improves precision and calibration without retraining the detector.
- Unlike image-level reward-weighted fine-tuning, this path does not collapse recall.

The result is not yet a robust AP improvement. The next step should tune score fusion and candidate source:

- Sweep `alpha` and final score threshold jointly.
- Train with hard patch candidates, not only clean train candidates.
- Cache pre-NMS proposals or RPN proposals instead of only final detections.
- Keep detector fixed until reranking consistently improves patch AP50.
