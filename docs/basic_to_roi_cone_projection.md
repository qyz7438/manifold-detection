# Basic Project Mapping: ROI Intra/Inter Dual Energy

## Corrected Reading

The useful lesson from `basic/nc-rf-scsi` is not that a single new loss, a
one-dimensional class order, or a cone-residual endpoint is the whole story.
Its core formulation is an intra-class / inter-class dual-energy plane:

```text
E_intra = compactness energy + basin stability energy
E_inter = class-relation energy + class identity anchors + class separation
DualEnergy = weighted combination of E_intra and E_inter
```

`E_intra` is about whether samples of the same class collapse into a stable
basin.  `E_inter` is about whether different classes keep a valid relational
structure instead of simply becoming compact in isolation.  For detection it
also needs a light class-index anchor, because classifier output channels give
each category an identity.  The training transition matters because useful
generalization appears when both sides move together.

The important warning is also inherited from `basic`: a geometry metric can
improve while task accuracy gets worse.  Therefore detection structure metrics
are diagnostics and gating signals first.  They should only become losses after
they correlate with clean AP, AP75, precision, recall, false positives, ECE,
and prediction count.

## Detection Translation

For detection, the unknown endpoint is not simply:

```text
ROI feature -> class prototype
```

and it is not only:

```text
ROI feature -> lower-energy cone residual endpoint
```

The endpoint we can defend is a verifiable foreground ROI state:

```text
proposal-aligned ROI feature
  -> compact within its class
  -> retained in the correct class basin under small perturbations
  -> separated from and relationally aligned with other classes
  -> still improves detector behavior after score thresholding, bbox regression, and NMS
```

This keeps the project aligned with the active energy-guided ROI transport
story: local actions are allowed, but only when they preserve threshold behavior,
respect rescue budgets, and do not create false positives.

In detection terms:

```text
E_intra_roi:
  foreground RoIs of class y should be close to mu_y
  and should keep a positive margin against mu_not_y under perturbation

E_inter_roi:
  class prototypes should not collapse into each other
  and their relation matrix should align with available anchors:
    frozen/reference prototypes,
    classifier weights,
    or later semantic/text class-relation matrices
  while preserving class-index identity when detector classifier axes are fixed
```

## What DPOG Means Here

The cone-projection module remains useful, but its role is narrower than the
previous version of this note implied.

It asks a local question:

```text
after an ROI has entered a class cone, is its tangential residual already close
to a lower-energy residual direction?
```

That is a residual stability diagnostic.  It is not the whole manifold
optimization target.  A detector can have low RoI-DPOG and still be badly
calibrated, unstable near threshold, or full of false positives.

## Maintained Dual-Energy Metrics

The detection-side dual-energy implementation starts from:

```text
spectral_detection_posttrain/methods/energy_transport/structure_metrics.py
```

Maintained primitives:

- `roi_compactness_energy`: normalized distance from foreground ROI features to
  their class prototypes;
- `roi_basin_energy`: softplus energy for leaving the correct class basin;
- `inter_class_relation_energy`: TCRA-like inter-class relation alignment,
  class-index anchor energy, and class separation;
- `roi_dual_energy`: differentiable `E_intra / E_inter / DualEnergy` bundle;
- `roi_basin_retention`: score-form diagnostic for basin retention;
- `prototype_basin_geometry`: graph agreement among class prototypes, current
  sample prototypes, optional classifier weights, and optional basin leakage;
- `simplex_energy`: optional diagnostic deviation from an equiangular class
  prototype layout.

The older cone primitives remain available:

- `decompose_cone_features`
- `cone_residual_alignment_loss`
- `local_tangent_energy_endpoint`
- `cone_dpog_regularizer`

but they should be treated as auxiliary diagnostics or small ablations, not as
the main claim.

## Revised Experiment Logic

The next local/remote experiments should answer two separate questions.

First, dual-energy representation question:

```text
Does the post-training window reduce E_intra without damaging E_inter?
```

Track at least:

- `e_compact` lower is better;
- `e_basin` lower is better;
- `e_intra` lower is better;
- `e_inter` lower is better when a valid relation anchor exists;
- `dual_energy` lower is better only if both halves are interpretable;
- `roi_basin_retention` higher is a score-form diagnostic;
- RoI-DPOG and action energy as local residual diagnostics.

Second, detector question:

```text
Does that transition survive the detector's real output pipeline?
```

Track AP50, AP75, precision, recall, false-positive rate, ECE, prediction
count, and full-val versus smoke status.

Only if both move in the right direction should the structure signature become
a training objective.  Otherwise it stays a diagnostic telling us where the ROI
head or action policy is failing.
