# Re-ROI Counterfactual Evidence Protocol

## Status

Researcher-adaptive train-only protocol. No detector validation read, no action
policy training, no AP claim, and no GPU experiment is authorized until this
document, the nested split manifest, and the TDD contract tests are committed.

This protocol does not revive any frozen research line:

- `energy_transport.native_actions.c1_e1`
- `energy_transport.dense_absolute_endpoint`
- `energy_transport.local_delta_q`
- `energy_transport.residual_content_protocol`

Any gate failure freezes this direction permanently.

## Scientific Question

For a fixed detector checkpoint, fixed detector-only candidate generator, fixed
action family `A`, and fixed native postprocessing, does the post-action ROI
visual evidence provide additional, transferable information about the
counterfactual utility of an action, beyond what can be explained by:

1. the action-family prior `mu_f`;
2. the pre-action static ROI feature `h_pre`;
3. chance correlations?

Formally, let `a in A` act on proposal set `S` to produce `T_a(S)`. Let
`h_post(S, a)` be the re-ROI visual feature computed on the boxes of `T_a(S)`.
The target is

```text
Q_teacher(S, a) = g(Y_teacher(S, a))
```

where `Y_teacher` is computed from train GT and native postprocessing, and `g`
is a locked scalarization. The student `Q_theta(S, a)` uses only detector-only
information. The core test is whether a model with access to `h_post` predicts
`Q_teacher` better than a model with only `h_pre`, after residualizing the
action-family prior.

## Why This Is Not a Revival of C3-E1

C3b, D1, D2b, D3, D4, and E1 (`docs/autonomous_exploration_report.md:14` and
`spectral_detection_posttrain/configs/registry/research_lines.json:4`) tested
whole-image Delta-U, set context, native NMS topology, fine-grained actions,
adaptive candidates, and post-NMS suppression using only pre-action detector
features. They did not re-extract visual evidence from the counterfactual box
positions. The present protocol adds one new variable: `h_post(S, a)`.

## Data Boundary

- Dataset: NWPU VHR-10 train images only.
- Do not reuse images from any prior train-only split or experiment:
  - `spectral_detection_posttrain/configs/splits/nwpu_dense_endpoint_s42_nested.json`
  - `det.energy.dense_endpoint.absolute.001`
  - `det.energy.dense_endpoint.cleanval.001`
  - `det.energy.dense_endpoint.geometry_control.001`
  - `det.energy.dense_endpoint.shift_audit.001`
  - `det.energy.dense_local_delta_learner.001`
  - `det.energy.dense_local_delta_family_audit.001`
  - `det.energy.dense_local_delta_family_prior.001`
  - `det.energy.dense_local_delta_stats.002`
  - `det.energy.set_policy.m1.001`
  - `det.energy.set_oracle.m0.001`
- Lock a new deterministic multilabel class-presence and object-density
  stratified split: fit / tune / calibration / outer-heldout. Every class must
  have at least three images in tune, calibration, and outer-heldout.
- Lock image IDs, class support, object counts, manifest hash, annotation hash,
  detector checkpoint hash, action family hash, and Git commit before any cache
  generation.
- Detector validation remains untouched until all train-only gates pass.

## Action Family

Freeze the action family to the following nine non-identity primitives, with
exactly one magnitude each:

```text
score_down, score_up
translate_left, translate_right, translate_up, translate_down
scale_down, scale_up
drop
```

- Score step: `0.02` absolute.
- Box translation step: `0.02` of box width/height in the corresponding axis.
- Box scale step: `0.02` log-space relative to box area.
- `drop` removes the detection entirely.
- `identity_permutation` is the exact no-op control.

Candidate generation is detector-only. In the first phase, act on the top
`K=3` native post-NMS detections per image, with `B=1` action per image.

Identity permutation must produce exact `Q_teacher = 0` and the full model's
predicted residual must be zero within `1e-7`.

## Re-ROI Evidence

For each `(image, candidate, action)`:

1. Run the detector once to obtain FPN features and native proposals.
2. Cache FPN features and the initial proposal list.
3. Apply action `a` to the candidate box/score to form a counterfactual set
   `T_a(S)`.
4. On the counterfactual boxes, perform a second ROIAlign against the cached
   FPN features to obtain `h_post(S, a)`.
5. Do not rerun the full backbone.

For `drop`, the acted-upon box no longer exists. Its `h_post` is therefore a
zero vector with the same shape and dtype as `h_pre`, accompanied by
`h_post_present=false`. All other actions set `h_post_present=true`. Reusing
the next surviving detection as the dropped candidate's feature is forbidden.

The re-ROI layer matches the detector's `box_roi_pool` settings
(`output_size`, `sampling_ratio`, `aligned`). Features are cached in fp16 on
disk with per-file SHA-256.

The existing code in
`spectral_detection_posttrain/trainers/detection/action_local_transport.py:407-408`
applies `box_coder.decode(box_regression, proposals)` once and then optionally
applies an external `box_delta`. That external delta is not a valid candidate
for this protocol because it corresponds to a second decode. All actions here
must be defined on the **already-decoded** native output boxes.

## Teacher Structured Result

For each `(S, a)`, run native bbox decode, thresholding, small-box removal,
class-aware NMS, and top-K on the counterfactual set. Using train GT, compute

```text
Y_teacher(S, a) = {
    native_changed: bool,
    delta_tp75: int,
    delta_fp75: int,
    delta_fp50: int,
    delta_duplicate: int,
    delta_score_margin: float,
    delta_localization_quality: float,
    action_energy: float,
}
```

`native_changed` is deterministic from detector logits, box regression, action,
and postprocessing (`action_local_transport.py:494`). It is used as a hard
filter and diagnostic, not as a learned target. Only if future work needs to
avoid re-running postprocessing would an approximate `native_changed` predictor
be justified.

## Teacher Utility

Lock the scalarization before fitting:

```text
Q_teacher(S, a) =
    + 1.00 * delta_tp75
    - 0.25 * delta_fp75
    - 0.10 * delta_fp50
    - 0.50 * delta_duplicate
    + 0.10 * delta_score_margin
    + 0.10 * delta_localization_quality
    - 1.00 * action_energy
```

The cache stores the raw locked scalarization and all structured terms. For the
re-ROI student only, the resulting non-identity scalar `Q_teacher` is
robust-standardized using the inner-fit median and IQR before fitting the family
prior; identity remains exactly zero. The weights are fixed; no weight may be selected from tune,
calibration, or heldout data. Consumers such as the
independent box-head weighting protocol may explicitly lock the raw cache value
instead, but must record that utility definition in metadata.

## Fit-Only Family Prior

For each non-identity action family `f`, compute on fit images only:

```text
mu_f = candidate_row_mean_fit(Q_teacher | family=f)
```

Then define residual targets:

```text
r(S, a) = Q_teacher(S, a) - mu_f
```

`mu_identity = 0`; identity rows never enter prior fitting, residual loss,
rank loss, or primary metrics. All nine non-identity families must have fit
support. A missing family is a cache-contract failure.

## Arms

All learned arms share the same base architecture, optimizer, epochs, and
regularization. The only permitted experimental difference is the input feature
set.

A. `family_prior`: predict `Q_teacher` using only `mu_f` (equivalent to zero
   residual).
B. `static_roi`: predict residual `Q_teacher - mu_f` using pre-action static
   ROI features `h_pre`.
C. `re_roi`: predict residual using `h_pre` plus post-action re-ROI features
   `h_post`.
D. `re_roi_bundle_shuffle`: same inputs as C, but shuffle the
   `(action, h_post)` bundle so that each candidate's `h_post` is replaced by
   another action's `h_post` from the same image.

Fixed non-learned baselines:

- `zero_residual`: predict `mu_f` only.
- `utility_shuffle`: permute `Q_teacher` within each family.
- `matched_rate_random`: sample actions at the same rate as the oracle.
- `always_noop`: always predict identity.

The primary information-gain test is **C versus B**. C must also beat D to
rule out bundle-correlation artifacts.

## Model And Objective

Use a minimal permutation-equivariant set model for `Q_theta(S, a)`:

```text
Q_theta(S, a) = bias + mean_i u_theta(z_i) + mean_(i,j) v_theta(e_ij)
```

where `z_i` are node features and `e_ij` are sparse pair features. The model
sees:

- Arm B: pre-action ROI features, box geometry, score, class.
- Arm C: B plus post-action re-ROI features for the acted-upon candidate.

Train the residual endpoint with a fixed hybrid objective:

```text
L = SmoothL1(r_hat, r, beta=0.05)
  + lambda_rank * softplus(-(r_hat_i - r_hat_j) * sign(r_i - r_j))
```

Ranking pairs use only the same image and same action family. Exact and near
ties with `|r_i - r_j| <= 1e-6` are excluded from the ranking term.

## Primary Metrics

Report residual metrics (`Q_teacher - mu_f`) and reconstructed total metrics.

Residual metrics:

- MAE and relative MAE gain over zero residual.
- Image-equal, same-family pairwise accuracy.
- Residual sign AUROC.
- Fit-to-tune gap.

Reconstructed total metrics (`mu_f + residual_hat`):

- MAE versus family prior.
- Within-image pairwise accuracy versus family prior.
- Sign AUROC versus family prior.

All full-control comparisons use identical supporting-image intersection and a
seed-42, 10,000-resample image bootstrap 95% CI.

## Locked Gates

1. `support`: at least 60 tune images, 40 calibration images, all nine
   non-identity families represented on at least 24 tune images, and at least
   500 non-identity tune rows.
2. `identity`: teacher and full-model identity errors at most `1e-7`.
3. `re_roi_gain`: Arm C beats Arm B on image-equal residual pairwise accuracy
   by at least `0.03`, with paired bootstrap LCB above zero.
4. `bundle_integrity`: Arm C beats Arm D by at least `0.03` on the same metric.
5. `static_baseline`: Arm C beats family prior by at least `0.03` pairwise and
   10% relative MAE.
6. `calibration`: on the independent calibration split, the top-1 action chosen
   by a grouped split-conformal LCB has positive oracle utility with 90%
   coverage.
7. `generalization`: fit-to-tune residual pairwise gap at most `0.10` and
   relative MAE-gain gap at most `0.15`.

No gate may be relaxed after cache or tune results are read.

## Branch Rule

- If all gates pass, retain a researcher-adaptive train-only signal and
  preregister one confirmation on a fresh outer heldout set. Do not read
  detector validation or train an action policy.
- If any gate fails, freeze the re-ROI counterfactual evidence direction.
  Do not add more actions, do not increase model capacity, do not change the
  re-ROI layer, do not retune weights, do not expand seeds, and do not convert
  the probe into native actions.

A passing result does not establish detector AP improvement, coordinated action
selection, manifold correction, or deployment value.
