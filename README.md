# RLIimage

RLIimage is a research codebase for **RLVR-style post-training for object detection**. The current active question is not whether a detector can rerank already finished boxes, but whether a detector can be improved with verifiable signals about which proposals are trustworthy, where the model should look, and how score or localization decisions should change.

The project started from ROI Fourier rewards on Penn-Fudan. That early MVP is now historical. The active codebase has been refactored into method families, reusable signal modules, canonical experiment metadata, and cleaner experiment runners.

## Current Status

The current non-AFM research line focuses on:

- **Detection RLVR / score rescue**: KL-anchored policy or preference optimization for low-confidence high-IoU proposals.
- **DPO score rescue**: pairwise preference learning over detector proposals.
- **Verifier signals**: FFT/raw-iFFT, geometry, edge alignment, and manifold signals used as offline or training-time evidence.
- **Manifold post-training**: prototype-bank, Sinkhorn assignment, and transport-head guidance on detector proposal features.
- **Adversarial patch defense**: DPatch/RP2-style detector attack plus spectral/manifold defenses.

Recent non-AFM findings:

- Oracle score rescue on NWPU shows a real AP75 ceiling: a perfect LC-HI signal can lift AP75 by about `+0.052`, but naive rescue greatly increases false positives.
- GRPO score rescue currently produces only tiny AP75 movement on smoke runs.
- DPO can learn proposal preferences, but without absolute threshold and rescue-budget constraints it increases predictions and hurts AP.
- Interpretable verifier fusion has useful offline signal, but online training transfer remains weak.
- Manifold proposal post-training has a positive Penn-Fudan smoke result, but still needs full clean validation.

AFM/in-network FFT remains documented as a separate historical line, but it is not the default active path unless explicitly requested.

## Repository Layout

```text
spectral_detection_posttrain/
  core/                 detector builders, matching, shared model primitives
  methods/
    rlvr/               ROI policy losses, confidence rescue, detector verifiers
    dpo/                action verifier and preference-learning helpers
    manifold/           prototype banks, Sinkhorn assignment, transport heads
    defense/            detector patch attacks and spectral/manifold defenses
    segmentation/       segmentation prototypes and signals
  signals/
    fft/                FFT/raw-iFFT features and spectral rewards
    geometry/           spatial and box-geometry signals
    pixel_classification/
  trainers/             detection and segmentation training entry points
  experiments/          canonical runner, schema, metadata, version records
```

Compatibility shims keep historical imports working:

- `spectral_detection_posttrain/spectral/*` forwards to `signals/fft/*`
- `spectral_detection_posttrain/models/*` forwards to `core/models/*` or method modules
- `spectral_detection_posttrain/rlvr/*` forwards to `methods/rlvr/*`
- `spectral_detection_posttrain/train/*` forwards to `trainers/detection/*`

Historical `scripts/round*.py` files are intentionally kept as experiment artifacts. New maintained code should use canonical package paths.

## Environment

Use the existing conda environment:

```powershell
conda activate RLimage
```

Direct Python path on this machine:

```text
E:\anaconda\01\envs\RLimage\python.exe
```

## Useful Checks

Run the focused refactor smoke tests:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest `
  tests/test_canonical_runner.py `
  tests/test_experiment_schema.py `
  tests/test_experiment_metadata.py `
  tests/test_manifold_modules.py `
  tests/methods/test_adversarial_defense.py -q
```

Run all tests when preparing a validated experiment branch:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest -q
```

## Current Important Artifacts

- `docs/versioning_scheme.md`: new version naming scheme.
- `docs/package_migration_signal_layout_2026-06-21.md`: canonical package layout and compatibility policy.
- `docs/round2226_nwpu_oracle_rerank_summary.md`: oracle NWPU score-rescue ceiling.
- `docs/round2227_nwpu_grpo_score_rescue_summary.md`: GRPO score-rescue smoke result.
- `docs/round2228_nwpu_dpo_score_rescue_summary.md`: DPO score-rescue smoke result.
- `docs/round2221_interpretable_reward_signal_diagnostics.md`: offline verifier fusion diagnostics.
- `obsidian/RLIimage Map.md`: human-readable project map.

## Reproducibility Rules

- Use canonical experiment metadata for new clean runs.
- Record config hash, checkpoint hash, git commit, and dirty status.
- Treat `runs/`, `data/`, local agent state, and generated analysis caches as local artifacts.
- Promote an experiment to validated only after full clean eval and a fixed commit.
