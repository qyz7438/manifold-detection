# Basic Project Mapping: ROI Cone-Internal Manifold Optimization

## What Basic Adds

The `basic/nc-rf-scsi` project is not mainly an RL score-rescue line. Its useful idea for this repository is the cone-internal projection view:

- Neural Collapse or class prototypes describe whether a feature enters the correct class cone.
- They do not determine the feature's tangential residual direction inside that cone.
- Generalization can depend on whether that residual direction is stable, low-energy, and globally consistent.

In basic's language:

- DPA measures whether residual directions are aligned across augmentations or references.
- DPOG measures whether the current residual direction is close to a locally lower-energy residual direction.

This is the missing manifold-structure piece in the detector project.

## Detection Translation

For an ROI feature `z` and class prototype `mu_y`, decompose:

```text
z = a * mu_y + r
```

where `a * mu_y` is the class-axis projection and `r` is the cone-internal residual.

The endpoint is therefore not simply a class prototype. The endpoint is:

```text
z* = normalize(a * mu_y + r*)
```

where `r*` is a low-energy tangent residual found under detector classifier energy while preserving the class-axis coefficient and residual norm.

## Two Complementary Loops

Outer detection-action loop:

```text
default detection state -> dense candidate endpoint -> score/box action
```

This is implemented by the energy action search and `default_replace_or_insert` evaluation mode.

Inner ROI-manifold loop:

```text
ROI feature -> class cone decomposition -> low-energy residual endpoint
```

This is implemented in `spectral_detection_posttrain.methods.energy_transport.cone_projection`.

## New Maintained Primitives

- `decompose_cone_features`
- `cone_residual_alignment_loss`
- `local_tangent_energy_endpoint`
- `cone_dpog_regularizer`
- `cross_entropy_energy`

These are intentionally independent of the legacy prototype attraction head. They can be used as diagnostics first, then added as a small regularizer during ROI post-training.

## Next Experiment

When remote data is available again:

1. Measure RoI-DPOG for TP, localization-error, classification-error, and background-FP groups.
2. Check whether AP75 failures have higher RoI-DPOG than AP50-only successes.
3. Add a small `lambda_roi_dpog` regularizer only on foreground proposal-aligned ROI features.
4. Compare against the action-only endpoint search to see whether feature manifold correction reduces the need for post-hoc score/box actions.
