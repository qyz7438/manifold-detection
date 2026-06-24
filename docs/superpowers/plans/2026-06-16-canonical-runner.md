# Canonical Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route formal detector experiments through one shared path for config validation, checkpoint validation, model construction, evaluation setup, and metadata recording.

**Architecture:** Add a small canonical experiment layer under `spectral_detection_posttrain/experiments/`. Existing package entrypoints (`train_baseline`, `eval_detector`, `posttrain_rlvr`) keep their training logic but delegate config/model/checkpoint/metadata setup to this layer. Old `scripts/round*.py` remain archival and should not receive new experiment logic.

**Tech Stack:** Python, PyTorch, TorchVision, pytest, YAML configs.

---

## Files

- Create: `spectral_detection_posttrain/experiments/schema.py` for config normalization and validation.
- Create: `spectral_detection_posttrain/experiments/metadata.py` for config/checkpoint hashes, git state, and runtime versions.
- Create: `spectral_detection_posttrain/experiments/canonical_runner.py` for shared experiment context and model/checkpoint helpers.
- Modify: `spectral_detection_posttrain/models/build_detector.py` to reject unknown model names and support safe AFM channel inference.
- Modify: `spectral_detection_posttrain/train/train_baseline.py` to use the canonical context and checkpoint metadata.
- Modify: `spectral_detection_posttrain/eval/eval_detector.py` to use the canonical context and checkpoint validation.
- Modify: `spectral_detection_posttrain/train/posttrain_rlvr.py` to use the canonical context and checkpoint metadata.
- Create: `tests/test_experiment_schema.py`.
- Create: `tests/test_experiment_metadata.py`.
- Create: `tests/test_canonical_runner.py`.

## Task 1: Config Schema

**Files:**
- Create: `spectral_detection_posttrain/experiments/schema.py`
- Modify: `spectral_detection_posttrain/models/build_detector.py`
- Test: `tests/test_experiment_schema.py`

- [ ] **Step 1: Write failing schema tests**

Required tests:
- unknown model name raises `ValueError`.
- conflicting `model.name` and `model.model_name` raises `ValueError`.
- formal config with `allow_random_init_fallback: true` raises `ValueError`.
- non-formal config can explicitly allow random-init fallback.
- AFM type without `afm_channels` infers 256 for supported Faster R-CNN FPN models.
- AFM type with wrong `afm_channels` raises `ValueError`.

- [ ] **Step 2: Run red tests**

Run: `python -m pytest tests/test_experiment_schema.py -q`
Expected: fail because `spectral_detection_posttrain.experiments.schema` does not exist yet.

- [ ] **Step 3: Implement schema**

Implement:
- `validate_experiment_config(config, formal=True) -> dict` returning a normalized copy.
- `resolve_model_name(model_cfg) -> str` using `model.model_name` or `model.name`.
- `infer_afm_channels(model_name) -> int` returning 256 for current FPN detector models.
- No unknown model fallback.
- Formal mode forbids random-init fallback.

- [ ] **Step 4: Run green tests**

Run: `python -m pytest tests/test_experiment_schema.py -q`
Expected: pass.

## Task 2: Metadata and Checkpoint Validation

**Files:**
- Create: `spectral_detection_posttrain/experiments/metadata.py`
- Create: `spectral_detection_posttrain/experiments/canonical_runner.py`
- Test: `tests/test_experiment_metadata.py`, `tests/test_canonical_runner.py`

- [ ] **Step 1: Write failing metadata tests**

Required tests:
- `sha256_file` returns the known hash for a temp file.
- metadata includes config hash, checkpoint hash, torch version, torchvision version, and git keys.
- missing checkpoint raises `FileNotFoundError`.
- empty checkpoint file raises `ValueError`.
- `prepare_experiment` writes normalized config and metadata JSON to the run directory.

- [ ] **Step 2: Run red tests**

Run: `python -m pytest tests/test_experiment_metadata.py tests/test_canonical_runner.py -q`
Expected: fail because metadata/canonical modules do not exist.

- [ ] **Step 3: Implement metadata and canonical context**

Implement:
- `sha256_file(path)`.
- `hash_config(config)` with deterministic JSON serialization.
- `collect_experiment_metadata(config, config_path=None, checkpoint_path=None)`.
- `validate_checkpoint_path(path, required=True)`.
- `prepare_experiment(config_path, run_name, phase, checkpoint_path=None, runs_root='runs', formal=True)`.
- `build_experiment_model(context, checkpoint_path=None, pretrained=None)`.
- `checkpoint_metadata(context, extra=None)`.

- [ ] **Step 4: Run green tests**

Run: `python -m pytest tests/test_experiment_metadata.py tests/test_canonical_runner.py -q`
Expected: pass.

## Task 3: Entrypoint Integration

**Files:**
- Modify: `spectral_detection_posttrain/train/train_baseline.py`
- Modify: `spectral_detection_posttrain/eval/eval_detector.py`
- Modify: `spectral_detection_posttrain/train/posttrain_rlvr.py`

- [ ] **Step 1: Integrate baseline training**

Replace local config/run-dir/model setup with `prepare_experiment` and `build_experiment_model`. Save checkpoints with `checkpoint_metadata(context, {'epoch': epoch, 'run_name': args.run_name})`.

- [ ] **Step 2: Integrate eval**

Use `prepare_experiment(..., phase='eval', checkpoint_path=args.checkpoint)` and `build_experiment_model(..., checkpoint_path=args.checkpoint, pretrained=False)`. Remove duplicate manual config saving.

- [ ] **Step 3: Integrate RLVR posttrain**

Use `prepare_experiment(..., phase='rlvr', checkpoint_path=args.baseline)` for config/run metadata. Build current and baseline models through canonical helpers, and save checkpoints with canonical metadata.

## Task 4: Verification

- [ ] **Step 1: Run focused tests**

Run: `python -m pytest tests/test_experiment_schema.py tests/test_experiment_metadata.py tests/test_canonical_runner.py tests/test_rlvr_verifier.py tests/test_rlvr_policy_objective.py tests/test_roi_policy_loss.py -q`
Expected: pass.

- [ ] **Step 2: Run import smoke checks**

Run: `python -m pytest tests/test_nni_quality_trial.py tests/test_nni_rlvr_round2.py tests/test_round23_freeze_state.py -q`
Expected: pass.

- [ ] **Step 3: Inspect changed files**

Run: `git diff -- spectral_detection_posttrain tests docs/superpowers/plans/2026-06-16-canonical-runner.md`
Expected: only intended canonical runner/schema/metadata/test/entrypoint changes.
