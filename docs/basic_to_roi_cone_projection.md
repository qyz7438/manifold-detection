# Basic Project Mapping: ROI Structure Transition, Not Just Cone Projection

## Corrected Reading

The useful lesson from `basic/nc-rf-scsi` is not that a single new loss, a
one-dimensional class order, or a cone-residual endpoint is the whole story.
Its strongest CIFAR evidence is a transition signature:

- class features become more compact relative to class spacing;
- the decision basin becomes more stable under small perturbations;
- prototype, probe-sample, classifier-weight, and basin graphs become more
  compatible;
- the effect depends on the training/reset/schedule window, not on an isolated
  RF-style regularizer alone.

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
  -> compatible with class prototype / sample / classifier / leakage graphs
  -> still improves detector behavior after score thresholding, bbox regression, and NMS
```

This keeps the project aligned with the active energy-guided ROI transport
story: local actions are allowed, but only when they preserve threshold behavior,
respect rescue budgets, and do not create false positives.

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

## Maintained Structure Metrics

The detection-side structure signature now starts from:

```text
spectral_detection_posttrain/methods/energy_transport/structure_metrics.py
```

Maintained primitives:

- `roi_compactness_energy`: normalized distance from foreground ROI features to
  their class prototypes;
- `roi_basin_retention`: prototype-margin retention under small feature
  perturbations;
- `prototype_basin_geometry`: graph agreement among class prototypes, current
  sample prototypes, optional classifier weights, and optional basin leakage;
- `simplex_energy`: diagnostic deviation from an equiangular class prototype
  layout;
- `roi_structure_signature`: one bundle for logging the basic-style detection
  structure transition.

The older cone primitives remain available:

- `decompose_cone_features`
- `cone_residual_alignment_loss`
- `local_tangent_energy_endpoint`
- `cone_dpog_regularizer`

but they should be treated as auxiliary diagnostics or small ablations, not as
the main claim.

## Revised Experiment Logic

The next local/remote experiments should answer two separate questions.

First, representation question:

```text
Does the post-training window produce a basic-style ROI structure transition?
```

Track at least:

- `roi_compactness_energy` lower is better;
- `roi_basin_retention` higher is better;
- `roi_pbg` higher is better;
- `roi_simplex_energy` lower is usually better, but only diagnostic;
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
