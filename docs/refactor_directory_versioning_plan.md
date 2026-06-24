# Directory Refactor And Versioning Plan

## Goal

Move from a round-script-driven project to a canonical, versioned project layout without breaking reproducibility of existing AFM, RLVR, DPO, and segmentation experiments.

## Constraints

- Do not delete historical `round*.py` scripts.
- Preserve import compatibility until active results are revalidated or archived.
- Any real directory move must be paired with tests and compatibility shims.
- New maintained code should import from canonical paths, not from legacy shim paths.

## Target Package Responsibilities

```text
spectral_detection_posttrain/core/
```

Shared infrastructure: model builders, adapters, quality heads, matching, and eventually shared datasets/transforms.

```text
spectral_detection_posttrain/methods/afm/
```

In-network FFT modules, AFM variants, phase-only variants, multiscale variants, and AFM-specific initialization checks.

```text
spectral_detection_posttrain/methods/rlvr/
```

RLVR objectives, reward normalization, verifier-guided rescue, KL anchors, safety guards, and ROI policy losses.

```text
spectral_detection_posttrain/methods/dpo/
```

DPO pair mining, chosen/rejected construction, preference losses, baseline-relative logits/actions.

```text
spectral_detection_posttrain/methods/segmentation/
```

Segmentation-specific AFM/RLVR/DPO logic. It must not depend on detection proposals as a core assumption.

```text
spectral_detection_posttrain/signals/
```

FFT, raw iFFT, high-dimensional manifold, geometric, and calibration signals used by verifiers or diagnostics.

```text
spectral_detection_posttrain/trainers/
```

Canonical training and post-training entry points split by task.

```text
spectral_detection_posttrain/experiments/
```

Canonical runner, schema, metadata, registry, and historical alias mapping.

## Completed Migration Batch

Moved:

- `analysis/raw_ifft_features.py`, `analysis/raw_ifft_verifier.py` -> `signals/fft/`
- `models/micro_afm.py` -> `methods/afm/`
- `models/build_detector.py`, `models/bbox_adapter.py`, `models/spectral_quality_head.py` -> `core/models/`
- `matching/box_iou.py`, `matching/pred_gt_matcher.py` -> `core/matching/`
- `rlvr/action_verifier.py` -> `methods/dpo/`
- `rlvr/confidence_rescue.py`, `rlvr/detection_verifier.py`, `rlvr/roi_policy_loss.py` -> `methods/rlvr/`
- `train/action_verifier_posttrain.py`, `train/posttrain_rlvr.py`, `train/posttrain_reward_weighted.py`, `train/rollout.py`, `train/train_baseline.py`, `train/train_quality_head.py` -> `trainers/detection/`

Compatibility kept:

- Old `models.*`, `matching.*`, `rlvr.*`, `analysis.raw_ifft_*`, and `train.*` paths remain import-compatible.
- Old training modules preserve `python -m spectral_detection_posttrain.train.*` behavior.
- NNI launchers now call `spectral_detection_posttrain.trainers.detection.*`.

Validation:

- Import alias smoke test passed for old and new AFM, FFT, DPO, RLVR, matching, model, and trainer paths.
- Targeted migration suite passed: `143 passed`.

## Remaining Work

1. Add experiment registry indexing for migrated versioned runs.
2. Migrate or archive legacy `models/verifiers.py`.
3. Move mature spectral helpers from `spectral/` into `signals/fft` where appropriate.
4. Split the large `methods/rlvr/confidence_rescue.py` into smaller modules after tests are pinned.
5. Add concrete segmentation trainers under `trainers/segmentation`.
6. Archive historical scripts with a manifest after canonical replacements are verified.

## Historical Script Archive Plan

Move old scripts only after canonical aliases exist:

```text
scripts/archive/round2xx/
scripts/archive/round21xx/
scripts/archive/diagnostics/
```

Every archived script should have a manifest entry:

```json
{
  "script": "scripts/archive/round21xx/round2129_nwpu_posttrain_smoke.py",
  "historical_role": "NWPU canonical-ish posttrain runner before shared runner",
  "replacement": "spectral_detection_posttrain.trainers.detection.action_verifier_posttrain",
  "known_issues": ["large file", "many objective branches"]
}
```

## Still Not Moved

- `scripts/round*.py`
- CLI defaults for historical runners
- `spectral_detection_posttrain/rlvr/round211_spatial_verifier.py`
- `spectral_detection_posttrain/models/verifiers.py`
