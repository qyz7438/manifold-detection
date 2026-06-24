# Directory Refactor

Source document:

- [docs/refactor_directory_versioning_plan.md](../docs/refactor_directory_versioning_plan.md)

Status:

- First real migration batch completed.
- Historical scripts are still in place.
- Legacy import paths are compatibility shims.

Completed canonical paths:

- `core/matching`: IoU and prediction-GT matcher.
- `core/models`: detector builder, bbox adapter, spectral quality head.
- `methods/afm`: MicroAFM/MPLSeg AFM variants.
- `methods/dpo`: action verifier and DPO loss helpers.
- `methods/rlvr`: confidence rescue, detection verifier, ROI policy loss.
- `signals/fft`: raw iFFT feature extraction and verifier calibration.
- `trainers/detection`: baseline, RLVR, action verifier, quality head, rollout, reward-weighted trainers.

Compatibility policy:

- Old `models.*`, `matching.*`, `rlvr.*`, `analysis.raw_ifft_*`, and `train.*` paths remain import-compatible.
- New maintained code should use `core.*`, `methods.*`, `signals.*`, and `trainers.*`.
- Historical `scripts/round*.py` are not moved yet.

Verification:

- Import alias smoke test passed.
- Targeted migration tests: `143 passed`.

Next migration order:

1. Add registry and version metadata.
2. Migrate or archive `models/verifiers.py`.
3. Split `methods/rlvr/confidence_rescue.py`.
4. Move mature `spectral/` helpers into `signals/fft`.
5. Add segmentation trainers.
6. Archive historical scripts with manifest.
