# Manifold Detection Architecture

This project studies post-training and architecture changes for visual detection.  The maintained framing is (research-line status is governed by the registries; see [`docs/research/current_status.md`](research/current_status.md)):

- Energy-guided ROI transport: action-local feature, score, bbox, and keep/reject corrections under low-energy and detection-environment constraints.

Historical method families remain available for comparison:

- AFM / FPN-SM: in-network FFT or spectral adapters trained through standard detection losses.
- RLVR: verifier-guided post-training with reward or rescue signals.
- DPO: pairwise preference optimization over proposals, actions, or logits.
- Prototype manifold losses: class prototype banks, Sinkhorn assignment, and low-energy residual fields.

The current codebase still contains substantial historical experiment code. The architecture goal is not a cosmetic directory move. The priority is to create stable canonical entry points, migrate implementation modules behind those entry points, and preserve reproducibility of historical runs.

## Current Layout

```text
spectral_detection_posttrain/
|-- core/              # canonical matching and shared model components
|-- methods/
|   |-- energy_transport/  # action-local ROI transport primitives
|   |-- afm/           # canonical AFM modules
|   |-- dpo/           # canonical DPO/action-preference utilities
|   |-- rlvr/          # canonical RLVR policy/verifier/rescue utilities
|   |-- manifold/      # prototype/spectral manifold diagnostics and baselines
|   `-- segmentation/  # segmentation method namespace
|-- signals/
|   `-- fft/           # canonical raw iFFT/FFT verifier feature code
|-- trainers/
|   |-- detection/     # canonical detection train/posttrain entry points
|   `-- segmentation/  # segmentation trainer namespace
|-- analysis/          # diagnostics and compatibility shims for migrated signal code
|-- configs/           # partial config files and version configs
|-- datasets/          # Penn-Fudan, VOC, patch transforms
|-- eval/              # detector metrics, clean eval, rerank and rescue/oracle eval
|-- experiments/       # canonical runner, schema, metadata, NNI trial utilities
|-- matching/          # compatibility shims to core/matching
|-- models/            # compatibility shims plus legacy verifier code
|-- rlvr/              # compatibility shims plus legacy round211 verifier
|-- spectral/          # FFT features, spectral rewards, ROI crops, radial profiles
|-- train/             # compatibility CLI modules to trainers/detection
|-- utils/             # config, io, seed, gradient diagnostics
`-- visualization/     # visual diagnostics
```

Root-level `scripts/round*.py` files are historical experiment runners. They should be treated as reproducibility artifacts, not as the preferred interface for new work.

## Migrated Modules

The following modules have been physically moved. The old paths now act as compatibility shims or module aliases:

- `analysis/raw_ifft_features.py` -> `signals/fft/raw_ifft_features.py`
- `analysis/raw_ifft_verifier.py` -> `signals/fft/raw_ifft_verifier.py`
- `models/micro_afm.py` -> `methods/afm/micro_afm.py`
- `models/build_detector.py` -> `core/models/build_detector.py`
- `models/bbox_adapter.py` -> `core/models/bbox_adapter.py`
- `models/spectral_quality_head.py` -> `core/models/spectral_quality_head.py`
- `matching/box_iou.py` -> `core/matching/box_iou.py`
- `matching/pred_gt_matcher.py` -> `core/matching/pred_gt_matcher.py`
- `rlvr/action_verifier.py` -> `methods/dpo/action_verifier.py`
- `rlvr/confidence_rescue.py` -> `methods/rlvr/confidence_rescue.py`
- `rlvr/detection_verifier.py` -> `methods/rlvr/detection_verifier.py`
- `rlvr/roi_policy_loss.py` -> `methods/rlvr/roi_policy_loss.py`
- `train/*.py` detection entry points -> `trainers/detection/*.py`

Current maintained code should use the new canonical paths. Historical scripts and older tests can keep using old import paths while the archive pass is pending.

## Data Flow

Energy-guided detection post-training should follow this high-level flow:

1. Load resolved config.
2. Validate model name, dataset, checkpoint, and eval mode.
3. Build frozen baseline detector and trainable policy detector.
4. Generate proposals from rollout or RPN.
5. Extract ROI logits, box regression, ROI features, proposal geometry, and verifier features.
6. Predict local actions: feature delta, score delta, bbox delta, and/or keep/reject.
7. Apply action constraints: low transport energy, threshold preservation, rescue budget, false-positive penalties, and optional KL/reference anchors.
8. Evaluate after detector post-processing, including NMS and clean score thresholds.
9. Record metadata, resolved config, checkpoint hash, git state, action diagnostics, metrics, and safety-guard decisions.

Segmentation should follow the same canonical runner principles, but the training signal is dense mask supervision rather than proposal-level matching.

## Method Boundaries

Energy-guided ROI transport is the active bridge between representation geometry and action learning.  It treats prototype/FFT/geometry signals as partial evidence for local actions, not as complete target states.

AFM is an architecture path. It changes the forward computation and receives gradients through standard task losses.

RLVR is a reward/rescue path. It updates selected trainable parameters using verifiable reward signals, KL anchors, and safety guards.

DPO is a pairwise preference path. It compares chosen and rejected proposals/actions/logits and optimizes relative preference against a frozen baseline.

Prototype manifold modules are diagnostics and baselines unless an experiment explicitly connects them to action-local constraints and clean full-val detector improvement.

Segmentation is a task path. It should support AFM, RLVR, and DPO variants, but should not reuse detection-specific proposal assumptions.

## Compatibility

Compatibility shims are intentionally kept in:

- `spectral_detection_posttrain.analysis.raw_ifft_*`
- `spectral_detection_posttrain.matching.*`
- `spectral_detection_posttrain.models.*`
- `spectral_detection_posttrain.rlvr.*`
- `spectral_detection_posttrain.train.*`

They preserve historical imports and command-line entry points. New code should avoid adding new dependencies on those legacy paths.

## Design Decisions

Architecture decisions are tracked in [docs/decisions](decisions/index.md). The current migration plan is [docs/refactor_directory_versioning_plan.md](refactor_directory_versioning_plan.md).
