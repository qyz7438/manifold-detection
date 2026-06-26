# Legacy Code Archive

This directory contains historical code assets that are no longer part of the active detection pipeline. They were archived during the 2026-06-26 detection-structure redesign.

## Why archived

Empirical results showed that the loss minimizers used by these methods were misaligned with the true detection objective (AP/AP75/ECE). Rather than continuing to maintain them in the main package, they are preserved here for reference and reproducibility of earlier experiments.

## Contents

| Path | Description |
|---|---|
| `methods/dpo/` | Direct Preference Optimization and action-verifier post-training |
| `methods/rlvr/` | RLVR policy-gradient components (detection verifier, ROI policy loss, confidence rescue) |
| `methods/segmentation/` | Segmentation experiments and OT-based segmentation losses |
| `methods/classification/` | Spectral/mixup/prototype classification experiments |
| `methods/defense/` | Adversarial patch and manifold defense experiments |
| `methods/manifold/` | High-dimensional manifold/transport prototypes used by BEM and DPO |
| `methods/multimodal/` | Cross-modal alignment and text-guided experiments |
| `methods/remote_sensing/` | Remote-sensing spectral head experiments |
| `methods/afm/` | Experimental AFM variants that did not beat the validated `micro_afm.py` baseline |
| `rlvr/` | Top-level RLVR helpers (roi_policy_loss, detection_verifier, action_verifier) |
| `signals/` | FFT/spectral reward signals used only by external verifiers |
| `models/` | Legacy model helpers (verifiers, spectral quality head, duplicate micro_afm) |
| `trainers/detection/` | Legacy trainers (RLVR, DPO, reward-weighted, manifold, quality head) |
| `trainers/segmentation/` | Segmentation trainers |
| `experiments/` | Old NNI trial wrappers |
| `configs/` | Segmentation and old task configs |
| `core/models/` | Segmentation adapters |
| `datasets/` | Segmentation dataset loader |
| `scripts/` | Old experiment drivers (DPO, BEM grid, Plan-C, segmentation) |
| `tests/` | Tests for archived components |
| `nni_configs/` | NNI search spaces for RLVR/DPO/quality/segmentation experiments |

## Status

These modules are **not maintained**. Import paths are broken after the move; if you need to run an old experiment, set `PYTHONPATH` to include both the project root and the relevant legacy subdirectory, or copy the module back into the main package.

## Active mainline

The current active pipeline is in `spectral_detection_posttrain/` and consists of:

- `methods/afm/micro_afm.py` — validated MPLSeg-style AFM baseline
- `methods/detection/pbg.py`, `tam.py`, `pah.py` — new structural components
- `core/models/build_detector.py` — detector builder
- `trainers/detection/train_baseline.py` — standard detection trainer
- `experiments/canonical_runner.py` — lightweight experiment runner
- `scripts/round28_train_eval.py` — main training/evaluation script
