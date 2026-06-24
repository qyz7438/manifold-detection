# Canonical Runner Hardening Note

Date: 2026-06-16

This note records a deferred engineering direction. It is not implemented yet.

## Motivation

The project now has a reusable core package under spectral_detection_posttrain, but a large fraction of experimental behavior still lives in one-off scripts/round*.py runners. That made sense while exploring quickly, but it now creates three risks:

- results can differ because runner plumbing differs, not because the method differs;
- old bugs can reappear through copied training/eval code;
- run metadata is inconsistent, making later evidence aggregation expensive.

The next engineering cleanup should converge experiment execution into a canonical runner path.

## Proposed Shape

Create one canonical experiment runner that owns:

- config validation;
- checkpoint validation;
- model construction;
- freeze/trainable policy setup;
- train/eval loop selection;
- metric calculation;
- metadata capture;
- run directory layout.

Old scripts/round*.py files should become archived launchers or thin config manifests, not independent training implementations.

## Required Guards

- Unknown model_name should fail fast instead of silently falling back to the first registry model.
- Formal experiments should disable random-initialization fallback unless explicitly marked as smoke/dev.
- AFM channel count should be inferred or strongly asserted against the actual insertion point.
- Every run should record:
  - git commit;
  - dirty-worktree flag;
  - config hash;
  - checkpoint path and hash;
  - torch/torchvision versions;
  - CUDA/device info;
  - metric implementation name, especially custom_project_ap vs COCO API.

## Suggested Files

- New: spectral_detection_posttrain/experiments/canonical_runner.py
- New: spectral_detection_posttrain/experiments/config_schema.py
- New: spectral_detection_posttrain/experiments/metadata.py
- Keep: spectral_detection_posttrain/experiments/runner_utils.py as migration support only.

## Migration Rule

When a new experiment is needed, add a config/manifest first. Only add a new Python runner if the canonical runner cannot express the experiment, and then move the missing capability back into the canonical path.
