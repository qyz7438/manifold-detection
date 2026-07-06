# Manifold Detection Agent Guide

## Core Objective

Build and evaluate an **energy-guided ROI transport framework for object detection**. The goal is to move beyond ordinary manifold regularization and test whether verifiable signals can help a detector learn small local actions: which boxes are trustworthy, where evidence should come from, how scores or boxes should change, and when a proposal should remain below threshold.

The current active line is **action-local, non-AFM ROI transport** unless the user explicitly asks for AFM/FPN-SM. Treat AFM, FPN-SM, and old external FFT rewards as documented baselines or historical branches, not the default path for new work.

## Current Active Research Lines

### 1. Energy-Guided ROI Transport

The central problem is that the class-conditioned target manifold is not known.  Class prototypes and verifier scores are partial anchors, but the true target is defined by detector behavior after thresholding, score calibration, bbox regression, and NMS.

New maintained code should start from:

- `spectral_detection_posttrain/methods/energy_transport/`

The current primitives are:

- `ActionLocalTransportHead`
- `ROITransportActions`
- `transport_action_energy`
- `threshold_preservation_loss`
- `rescue_budget_loss`
- `apply_bounded_score_delta`

Design implication:

- Model transport as bounded local actions, not as direct prototype attraction.
- Keep score changes residual/additive, not score replacement.
- Add rescue budget, threshold preservation, and false-positive penalties.
- Judge success on clean full evaluation, not geometry metrics alone.

### 2. Detection RLVR / Score Rescue

The main detection setting is NWPU VHR-10 and Penn-Fudan smoke validation. The central problem is low-confidence high-IoU proposals: useful boxes exist, but the detector score keeps them below the final threshold or NMS ranking.

Recent status:

- Oracle score rescue on NWPU shows an AP75 upper bound of about `+0.052`, but naive additive rescue creates many false positives.
- GRPO score rescue currently moves AP75 only slightly on smoke runs.
- DPO score rescue learns pairwise preferences, but without absolute score constraints it increases predictions and hurts AP.

Design implication:

- Treat RLVR/GRPO/DPO as training mechanisms for action-local transport, not as the whole project identity.
- Preserve historical results as evidence for constraints and failure modes.

### 3. Verifier Signals

The project has several verifier families:

- FFT/raw-iFFT features under `spectral_detection_posttrain/signals/fft/`
- geometry/spatial signals under `spectral_detection_posttrain/signals/geometry/`
- interpretable fusion diagnostics from Round 2221
- manifold and prototype features under `spectral_detection_posttrain/methods/manifold/`

Current conclusion:

- Offline verifier signals can rank proposals better than chance.
- Online transfer into detector improvement is still weak.
- Any new verifier must pass shuffled/control comparisons and full clean eval.

### 4. Manifold Post-Training

The historical manifold path uses proposal-aligned feature training:

- `PrototypeBank`
- `SinkhornAssigner`
- `TransportHead`
- `train_manifold_posttrain.py`

Important implementation idea:

Apply manifold loss to the same RPN proposals used by the detector training path, not only to GT boxes. This keeps the manifold objective aligned with inference-time proposal distributions.

Current status:

- `manifold_posttrain_proposal_smoke` has a positive Penn-Fudan smoke result.
- It still needs clean full validation and NWPU testing before being treated as a real improvement.
- Prototype attraction alone is no longer the active thesis; it is a baseline/diagnostic unless connected to action-local constraints.

### 5. Adversarial Patch Defense

The defense line contains DPatch/RP2-style detector attacks and spectral/manifold defenses.

Current status:

- The patch attack can now reduce AP50 on smoke runs.
- Current defenses do not recover AP reliably.
- Treat defense as active but not yet successful.

## Historical Lines

### External FFT Reward

Early R_amp / phase / structure FFT rewards produced a stable RLVR shell but did not show robust causal gains. Real vs shuffled controls often matched. Keep the code and reports for lineage, but do not cite early R_amp as a validated reward.

### AFM / In-Network FFT

AFM showed positive Penn-Fudan localization results in earlier experiments, but it is an architecture/fine-tuning line, not the current RLVR verifier-reward path. Do not restart AFM work unless the user asks.

### MFVPT Classification MVP

`mfvpt/` is retained as historical classification work. The active detection project is `spectral_detection_posttrain/`.

## Canonical Package Layout

New maintained code should use these namespaces:

```text
spectral_detection_posttrain/core/          detector builders, matching, shared primitives
spectral_detection_posttrain/methods/energy_transport/
spectral_detection_posttrain/methods/rlvr/  RLVR losses, confidence rescue, verifier modules
spectral_detection_posttrain/methods/dpo/   action verifier and DPO utilities
spectral_detection_posttrain/methods/manifold/
spectral_detection_posttrain/methods/defense/
spectral_detection_posttrain/signals/fft/   FFT/raw-iFFT features and rewards
spectral_detection_posttrain/signals/geometry/
spectral_detection_posttrain/trainers/      training entry points
spectral_detection_posttrain/experiments/   canonical runner, schema, metadata
```

Compatibility shims remain for historical imports:

- `spectral_detection_posttrain/spectral/*`
- `spectral_detection_posttrain/models/*`
- `spectral_detection_posttrain/rlvr/*`
- `spectral_detection_posttrain/train/*`

Historical `scripts/round*.py` are experiment artifacts and should not be bulk-migrated unless the user requests it.

## Versioning And Experiment Hygiene

Use the new version format for clean future work:

```text
<task>.<method>.<stage>.<sequence>
```

Examples:

- `det.rlvr.clean.001`
- `det.dpo.smoke.001`
- `det.signal.clean.001`
- `shared.runner.validated.001`

Validated runs must have:

- clean full eval
- fixed config
- checkpoint hash
- git commit
- dirty status understood and preferably clean
- reproduction command

Do not promote smoke-only or polluted historical runs to validated results.

## Local Artifacts

These are local/generated and should not drive research conclusions:

- `runs/`
- `data/`
- `.omc/`
- `.agent_reports/`
- `.repochan/`
- `.pi/`
- `pipeline_state/`
- large `.npz` caches

Keep them available locally if useful, but avoid committing runtime state or generated caches.

## Useful Commands

Focused refactor smoke:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest `
  tests/test_canonical_runner.py `
  tests/test_experiment_schema.py `
  tests/test_experiment_metadata.py `
  tests/test_manifold_modules.py `
  tests/methods/test_pbg.py -q
```

Full test suite:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest -q
```

## Guidance For Future Agents

- Read current docs before interpreting old round numbers.
- Prefer canonical package paths over historical shim paths.
- Keep AFM out of the active story unless explicitly requested.
- Separate offline verifier quality from online detector improvement.
- Report AP50, AP75, precision, recall, false-positive rate, ECE, prediction count, and whether eval is full-val or limited smoke.
- Never compare `limit_val=32` smoke metrics against full-val metrics as if they are equivalent.

## Current Active Experiment Matrix (June 2026)

The immediate priority is the **VOC 20-class main matrix (M1)** followed by NWPU VHR-10 completion (M4) and COCO smoke validation (M3).

| Task | Dataset | Seeds | Models / Methods | Status |
|------|---------|-------|------------------|--------|
| M1   | VOC 07  | 42,2024,999 | MobileNet baseline, MobileNet+fpn_sm, ResNet50 baseline, ResNet50+fpn_sm | in progress on GPU2 |
| M2   | VOC 07  | 42,2024,999 | SE / FcaNet / ECA comparison + per-size AP | queued |
| M3   | COCO 17 | 42 | smoke run on 500/200 images | data ready, script ready |
| M4   | NWPU    | 42,2024,999 | baseline + fpn_sm | needs third seed |
| S1-S2| VOC 07  | all | statistical significance + aggregated tables | pending M1/M2 |
| A2-A10| VOC/PF | selected | fpn_sm ablations (latent dim, freq coords, level coords, gate activation, suppress dc, init alpha, etc.) | pending |

**GPU policy on the remote server**: only `CUDA_VISIBLE_DEVICES=2` may be used. Do not stop, restart, or otherwise interfere with the process currently running the VOC matrix. New jobs must wait until GPU2 is free or be launched with `nohup` after the active run finishes.

## Remote Server Quick Reference

```bash
ssh ps@122.51.19.136
cd /home/ps/lzz/RLimage
source /home/ps/anaconda3/etc/profile.d/conda.sh
conda activate RLimage
export PYTHONPATH=/home/ps/lzz/RLimage:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2
python scripts/round28_train_eval.py --help
```

Python interpreter: `/home/ps/anaconda3/envs/RLimage/bin/python` (Python 3.10, PyTorch 2.1.0+cu121).

## Code Conventions For Current Work

- **Single-run script**: all train/eval flows go through `scripts/round28_train_eval.py`.
- **Datasets**: use `--dataset {penn_fudan,voc,nwpu,coco}`; for VOC use `--voc-full` for the full 20-class train/val split.
- **FPN spectral manifold**: `--fpn-spectral-manifold` plus `--fpn-sm-*` flags.
- **FPN real adapter**: `--fpn-real-adapter` plus `--fpn-real-*` flags.
- **FPN attention baselines**: `--fpn-attention-type {se,fcanet,eca}` and `--fpn-attention-reduction N`.
- **Run naming**: use descriptive names such as `<dataset>_<model>_<method>_s<seed>_<epochs>ep`.
- **Run directory**: outputs are written under `runs/<run_name>/` (config, checkpoints, eval metrics, logs).

## Keeping The Repository Clean

- Active run scripts and analysis tools live in `scripts/`.
- Obsolete/one-off scripts and old round reports are archived to `scripts/legacy/` and `docs/reports/archive/`; these directories are ignored by git.
- Temporary backup files (`*.bak`, `temp_push/`) are deleted and ignored.
- Do not commit runtime artifacts (`runs/`, `data/`, checkpoints, logs).

## Updated Useful Commands

Focused refactor smoke (local or remote):

```bash
python -m pytest \
  tests/test_canonical_runner.py \
  tests/test_experiment_schema.py \
  tests/test_experiment_metadata.py \
  tests/test_manifold_modules.py \
  tests/methods/test_pbg.py -q
```

Full test suite before validation:

```bash
python -m pytest -q
```

## Note On Test Suite

The full `pytest tests/` collection currently reports a small number of historical test errors because the environment is missing `scikit-learn`, which is only required by legacy raw-iFFT verifier and dimensionality modules. The actively maintained smoke set (canonical runner, experiment schema/metadata, manifold modules, PBG, geometry metrics) passes. Do not install new packages on the remote server without explicit user approval.
