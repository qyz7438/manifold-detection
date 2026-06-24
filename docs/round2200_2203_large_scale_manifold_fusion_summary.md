# Round 2200-2203 Large-Scale Manifold/Fusion Summary

Date: 2026-06-19

## Offline Signal Sweep

Inputs:
- Full proposal cache: `runs/round2199_box_feature_classwise_iou_bucket_manifold/iou_bucket_box_features.npz`
- Raw-iFFT cache: `runs/round2150_raw_ifft_legacy_full_ap75/candidate_raw_ifft_features.npz`

Scripts:
- `scripts/large_scale_verifier_fusion_sweep.py`
- `scripts/large_scale_fusion_policy_sweep.py`

Key result:
- Full box-feature manifold has strong general IoU signal, but LC-HI remains an edge case.
- Raw-iFFT baseline: AP 0.393, rank R@P0.7 0.293; fixed P0.8 gives 12 TP / 3 FP on val.
- Best fusion ranking improves substantially:
  - `fusion_raw_hd/full_allneg/l2/classwise_knn/train_effect`: AP 0.471, rank R@P0.7 0.415.
  - Calibration is unstable: train-calibrated fixed threshold has low val precision.
- Policy sweep found usable fixed-threshold candidates:
  - `fusion_raw_hd_prob/full_matchedneg/l2_z_pca128/center/train_effect`: 14 TP / 2 FP, P=0.875, R=0.341.
  - `fusion_raw_hd_prob/lowconf_allneg/l2_z_pca96/logistic/train_effect`: 13 TP / 1 FP, P=0.929, R=0.317.

Interpretation:
- High-dimensional ROI features improve ranking and can improve fixed verifier selection.
- The gain is fragile under calibration and threshold transfer.

## Training Integration

Implemented:
- Added `raw_ifft_hd_fusion` verifier mode to `scripts/round2129_nwpu_posttrain_smoke.py`.
- Added NNI wrapper passthrough for:
  - `verifier_mode`
  - `hd_fusion_pca_components`
  - `hd_fusion_hd_scorer`
  - `hd_fusion_method`
- Added wrapper test coverage in `tests/test_nni_raw_ifft_posttrain_trial.py`.

Verification:
- `pytest tests/test_nni_raw_ifft_posttrain_trial.py tests/test_raw_ifft_verifier_calibration.py -q`
- Result: 13 passed.

## 15 Epoch Training Results

Baseline for both runs:
- AP50 0.654760
- AP75 0.293908
- ECE 0.085663
- Predictions 1425
- FP rate 0.477895

### Round 2202: HD Fusion

Run:
- `runs/round2202_hd_fusion_tp06_pca96_logistic_15ep`

Verifier:
- `raw_ifft_hd_fusion`
- raw features: `fft_edge_truncation@64`, `phase_edge@64`, `phase_abs_high@11`
- HD scorer: l2 ROI box features, PCA96, logistic
- fusion: train-effect over raw score + HD score + baseline label probability
- target precision: 0.6

Calibration:
- raw candidates: 1498, positives 41
- HD candidates: 2654, positives 41
- fusion calibration: 21 selected, 13 TP, 8 FP
- offline threshold: selected 22, precision 0.591, recall 0.317

Best epoch:
- Epoch 9
- AP75 0.296811
- Delta AP75 +0.002903
- AP50 0.652893
- ECE 0.085707
- Predictions 1412
- FP rate 0.475921
- verifier-positive LC-HI confidence delta +0.008404

Final:
- AP75 0.295543
- Delta AP75 +0.001635

### Round 2203: Raw-iFFT A3 Control

Run:
- `runs/round2203_raw_ifft_A3_maxprop100_15ep`

Verifier:
- `raw_ifft`
- raw features: `fft_edge_truncation@64`, `phase_edge@64`, `phase_abs_high@11`
- target precision: 0.8
- rescue delta/cap: 0.20 / 0.80
- max proposals: 100

Calibration:
- raw candidates: 1498, positives 41
- raw calibration: 10 selected, 8 TP, 2 FP
- offline threshold: selected 9, precision 0.889, recall 0.195

Best epoch:
- Epoch 9
- AP75 0.297455
- Delta AP75 +0.003546
- AP50 0.654054
- ECE 0.086029
- Predictions 1420
- FP rate 0.477465
- verifier-positive LC-HI confidence delta +0.004284

Final:
- AP75 0.296625
- Delta AP75 +0.002716

## Conclusion

High-dimensional fusion is useful offline, but current training integration does not beat raw-iFFT-only. The fusion verifier increases coverage but adds calibration noise; raw-iFFT's stricter high-precision gate transfers better into post-training.

For training, the current best direction is not to keep widening HD fusion. The next useful change is to use HD/manifold as a secondary ranking or analysis signal, while keeping raw-iFFT as the primary gate for rescue training.
