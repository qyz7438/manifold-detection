# RLIimage Agent Guide

## Core Objective

Build and evaluate an **RLVR-style post-training framework for object detection**. The goal is to move beyond ordinary ROI reranking and test whether verifiable signals can help a detector learn which boxes are trustworthy, where evidence should come from, and how scores or localization decisions should change.

The current active line is **non-AFM** unless the user explicitly asks for AFM. Treat AFM/in-network FFT as a documented historical branch, not the default path for new work.

## Current Active Research Lines

### 1. Detection RLVR / Score Rescue

The main detection setting is NWPU VHR-10 and Penn-Fudan smoke validation. The central problem is low-confidence high-IoU proposals: useful boxes exist, but the detector score keeps them below the final threshold or NMS ranking.

Recent status:

- Oracle score rescue on NWPU shows an AP75 upper bound of about `+0.052`, but naive additive rescue creates many false positives.
- GRPO score rescue currently moves AP75 only slightly on smoke runs.
- DPO score rescue learns pairwise preferences, but without absolute score constraints it increases predictions and hurts AP.

Design implication:

- Use residual/additive score changes, not score replacement.
- Add rescue budget, threshold preservation, and false-positive penalties.
- Judge success on full clean eval, not small smoke-only splits.

### 2. Verifier Signals

The project has several verifier families:

- FFT/raw-iFFT features under `spectral_detection_posttrain/signals/fft/`
- geometry/spatial signals under `spectral_detection_posttrain/signals/geometry/`
- interpretable fusion diagnostics from Round 2221
- manifold and prototype features under `spectral_detection_posttrain/methods/manifold/`

Current conclusion:

- Offline verifier signals can rank proposals better than chance.
- Online transfer into detector improvement is still weak.
- Any new verifier must pass shuffled/control comparisons and full clean eval.

### 3. Manifold Post-Training

The new manifold path uses proposal-aligned feature training:

- `PrototypeBank`
- `SinkhornAssigner`
- `TransportHead`
- `train_manifold_posttrain.py`

Important implementation idea:

Apply manifold loss to the same RPN proposals used by the detector training path, not only to GT boxes. This keeps the manifold objective aligned with inference-time proposal distributions.

Current status:

- `manifold_posttrain_proposal_smoke` has a positive Penn-Fudan smoke result.
- It still needs clean full validation and NWPU testing before being treated as a real improvement.

### 4. Adversarial Patch Defense

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
  tests/methods/test_adversarial_defense.py -q
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
