# Energy-Guided ROI Transport

## Why This Refactor Exists

The previous manifold path treated detector features as if the target class
manifold were already known:

```text
ROI feature -> class prototype cloud -> pull feature closer to prototype
```

The experiments do not support that as the full objective.  Existing score
rescue, GRPO, DPO, and verifier-fusion results show a different bottleneck:

- oracle additive rescue has an AP75 ceiling, but flooding scores creates many
  false positives;
- GRPO score rescue barely moves because the action signal is weak and heavily
  constrained;
- DPO can learn pairwise preferences while still hurting AP when low-quality
  boxes cross thresholds;
- offline verifier signals can rank proposals, but online transfer into NMS and
  clean AP remains fragile.

That means the target state is not just "closer to a prototype."  The target is
an unknown, class-conditioned detection state that must be discovered through
local actions and evaluated after thresholding, NMS, calibration, and bbox
quality.

## Current Thesis

The project should be framed as:

```text
unknown class-conditioned target manifold
  -> local feature/score/bbox actions
  -> low-energy constrained transport
  -> detection-environment reward and diagnostics
```

This keeps the ChordEdit analogy intact:

```text
ChordEdit:
  noisy/source visual state -> low-energy transport -> target visual manifold

Detection:
  uncertain ROI/proposal state -> low-energy local action -> valid detection state
```

The important difference is that detection does not have a text prompt that
fully specifies the target image manifold.  Class labels and prototypes are only
partial anchors.  NMS survival, AP75, false-positive budget, and threshold
preservation define the missing part of the objective.

## Maintained Entry Point

New code should start in:

```text
spectral_detection_posttrain/methods/energy_transport/
```

The first primitives are intentionally small:

- `ActionLocalTransportHead`: predicts bounded local feature, score, bbox, and
  keep/reject actions from ROI features;
- `ROIActionState`: proposal-aligned state before action;
- `ActionOutcome`: post-action state and threshold-crossing diagnostics;
- `PreferenceBatch`: explicit chosen/rejected action pairs with validity masks;
- `ConstraintConfig`: shared safety constraints for action experiments;
- `transport_action_energy`: penalizes large feature/score/bbox moves;
- `threshold_preservation_loss`: prevents low-quality candidates from crossing
  eval thresholds;
- `rescue_budget_loss`: softly caps the number of rescued proposals;
- `apply_bounded_score_delta`: keeps score updates residual and bounded.
- `apply_box_delta` / `clip_boxes_to_image`: apply bbox actions with standard
  center-size decoding and image clipping.
- `build_top_bottom_preferences`: constructs conservative top-vs-bottom
  preference pairs with quality margin, IoU floor, IoU-gap checks, and explicit
  invalid masks.

The proposal-state bridge lives in:

```text
spectral_detection_posttrain/trainers/detection/roi_state.py
```

It exposes:

- `build_roi_action_state_from_predictions`: builds a `ROIActionState` from
  per-image proposal dictionaries and concatenated ROI features;
- `extract_proposal_roi_action_state`: runs a detector's transform, backbone,
  RPN, ROI pooling, box head, and box predictor to produce proposal-aligned
  ROI state.

These are not a full trainer yet.  They define the state, action, preference,
and constraint surface that future RLVR/DPO/manifold trainers should share.

## Relationship To Existing Modules

`methods/manifold/` remains useful, but its role changes:

- `PrototypeBank`: class-conditioned anchors and diagnostics;
- `SinkhornAssigner`: soft assignment for proposal manifolds;
- `TransportHead`: historical low-energy residual field;
- `ManifoldCorrectionPredictor`: legacy active-correction wrapper and baseline.

The old manifold loss is no longer the main story.  It is a diagnostic or
baseline for whether prototype attraction alone is enough.

`methods/rlvr/` and `methods/dpo/` remain compatibility paths.  New policy work
should use their lessons, not inherit their historical failure modes directly.

## Required Evaluation Gates

A run cannot be promoted by geometry metrics alone.  It must report:

- AP50 and AP75;
- precision, recall, false-positive rate, and prediction count;
- ECE or another calibration diagnostic;
- LC-HI score shift and coverage;
- threshold-crossing counts for low-quality proposals;
- rescue budget usage;
- shuffled/control verifier comparison;
- whether the eval is full-val or smoke.

## Current Implementation Step

The current refactor adds action-local primitives, proposal state extraction,
and tests.  The next trainer should apply actions and record post-NMS outcomes:

```text
frozen/current detector proposals
  -> ROI feature state
  -> ActionLocalTransportHead
  -> bounded score/bbox/feature update
  -> NMS-aware diagnostics
  -> constrained RLVR/DPO or supervised action loss
```

The first formal experiment should compare:

```text
det-only same trainable scope
energy_transport score-only
energy_transport score + threshold preservation
energy_transport score + bbox + budget
shuffled verifier/control
```
