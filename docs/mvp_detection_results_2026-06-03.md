# MVP Detection Results - 2026-06-03

This run implements the minimum viable version requested for regional spectral evidence reward post-training.

## Setup

- Dataset: Penn-Fudan Pedestrian
- Task: person detection
- Model: TorchVision `fasterrcnn_mobilenet_v3_large_320_fpn`
- Initialization: COCO pretrained detector weights, ROI predictor changed to 2 classes
- Image size: dataset and detector resize capped at 320
- Baseline: 1 supervised epoch
- Post-training: 5 epochs, image-level reward-weighted fine-tuning
- Patch tests: clean, random patch, checkerboard patch
- Validation split: 34 images, 91 GT boxes

## Commands

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests -q
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.train.train_baseline --config spectral_detection_posttrain/configs/mvp.yaml --run-name mvp_pf_baseline --epochs 1
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.eval.eval_spectral_reward --config spectral_detection_posttrain/configs/mvp.yaml --checkpoint runs/mvp_pf_baseline/checkpoint_last.pth --run-name mvp_pf_reward_baseline
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.train.posttrain_reward_weighted --config spectral_detection_posttrain/configs/mvp.yaml --baseline runs/mvp_pf_baseline/checkpoint_last.pth --run-name mvp_pf_posttrain --epochs 5
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.eval.eval_detector --config spectral_detection_posttrain/configs/mvp.yaml --checkpoint runs/mvp_pf_posttrain/checkpoint_last.pth --run-name mvp_pf_eval_post_clean --patch-mode none
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.eval.eval_detector --config spectral_detection_posttrain/configs/mvp.yaml --checkpoint runs/mvp_pf_posttrain/checkpoint_last.pth --run-name mvp_pf_eval_post_random_patch --patch-mode random --patch-type random
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.eval.eval_detector --config spectral_detection_posttrain/configs/mvp.yaml --checkpoint runs/mvp_pf_posttrain/checkpoint_last.pth --run-name mvp_pf_eval_post_checker_patch --patch-mode random --patch-type checkerboard
```

## Training Output

| Stage | Epoch | Loss | Det Loss | Image Reward | Image Weight |
|---|---:|---:|---:|---:|---:|
| Baseline | 1 | 0.8275 | - | - | - |
| Post-train | 1 | 0.4381 | 0.4253 | 0.9532 | 1.0468 |
| Post-train | 2 | 0.4673 | 0.4489 | 0.9608 | 1.0392 |
| Post-train | 3 | 0.4793 | 0.4608 | 0.9464 | 1.0536 |
| Post-train | 4 | 0.4763 | 0.4601 | 0.9540 | 1.0460 |
| Post-train | 5 | 0.4761 | 0.4632 | 0.9685 | 1.0315 |

## R_amp Diagnostic

| Checkpoint | Mean R_amp TP | Mean R_amp FP | AUC TP vs FP | TP | FP |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.9989 | 0.9899 | 0.9311 | 80 | 37 |
| Post-train | 0.9982 | 0.9876 | 0.9411 | 42 | 19 |

Interpretation: `R_amp` can distinguish TP from FP in this MVP. The gap is small in absolute value because both ROI crops come from the same natural images, but the ranking AUC is clearly above 0.5.

## Detection Metrics

| Model | Patch | AP50 | Precision | Recall | High-conf FP Rate | Miss Rate | Predictions |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | clean | 0.8630 | 0.6838 | 0.8791 | 0.0282 | 0.1209 | 117 |
| Baseline | random | 0.8342 | 0.6875 | 0.8462 | 0.0282 | 0.1538 | 112 |
| Baseline | checkerboard | 0.8624 | 0.6557 | 0.8791 | 0.0286 | 0.1209 | 122 |
| Post-train | clean | 0.4349 | 0.6885 | 0.4615 | 0.0000 | 0.5385 | 61 |
| Post-train | random | 0.4334 | 0.6515 | 0.4725 | 0.0000 | 0.5275 | 66 |
| Post-train | checkerboard | 0.4532 | 0.6719 | 0.4725 | 0.0000 | 0.5275 | 64 |

## Output Files

- `runs/mvp_pf_baseline/checkpoint_last.pth`
- `runs/mvp_pf_posttrain/checkpoint_last.pth`
- `runs/mvp_pf_reward_baseline/r_amp_distribution.png`
- `runs/mvp_pf_reward_posttrain/r_amp_distribution.png`
- `runs/mvp_pf_roi_fft/roi_fft_comparison.png`

## Conclusion

The MVP is runnable and produces the requested evidence. The core verifier idea is feasible: TP boxes have higher `R_amp` than FP boxes, and the AUC is strong in this small Penn-Fudan run.

The first post-training strategy is not yet good enough. It removes high-confidence false positives, but it also suppresses many detections, causing a large recall and AP50 drop. The next implementation step should keep `R_amp` as a verifier, but replace coarse image-level weighting with a milder or ROI-level strategy, for example lower `reward_lambda`, warm-start only the head, or weight hard false-positive proposals instead of all image losses.
