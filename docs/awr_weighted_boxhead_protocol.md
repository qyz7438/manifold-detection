# Oracle-Utility-Weighted Box-Head Fine-Tuning Protocol

## Status

Researcher-adaptive train-only protocol. No detector validation read, no
action policy deployment, no inference-path modification, and no GPU
experiment is authorized until this document, its executable contracts, and
the re-ROI cache manifest are committed. The historical `awr` filename is kept
for traceability, but this experiment is weighted supervised empirical-risk
minimization, not an implementation of AWR, AWAC, CRR, or RL.

This protocol is parallel to and independent of
`docs/re_roi_counterfactual_evidence_protocol.md`. It may consume a passing
phase-1 re-ROI cache, but it does not share re-ROI gates. It does not revive
`rlvr_grpo_dpo`, `energy_transport.native_actions.c1_e1`, or
`energy_transport.dense_absolute_endpoint`.

## Scientific Question

For a fixed detector checkpoint, fixed fit image pool, fixed optimizer budget,
and fixed training seeds, does weighting box-head fine-tuning images by a
train-only, no-op-relative oracle utility produce higher heldout AP75 than
uniform box-head fine-tuning? If so, is the gain attributable to the assignment
of oracle utilities to images rather than to the weight distribution or to
positive-only filtering?

Historical motivation is limited to method selection. Three-seed full-split
controls reported mean AP75 deltas of `rlvr_random_fulltrain +0.0350`,
`box_head_only_10ep +0.0792`, and `full_model_10ep +0.1058`. Those numbers do
not predict the result on this protocol's 250/68 fit/tune split. Arm `U` must be
rerun on the exact locked split and is the only primary baseline.

## Claim Boundary

A passing result establishes only that this oracle utility transfers through
image-level weighted supervised box-head fine-tuning on the locked NWPU split.
It does not establish proposal-level weighting, re-ROI evidence gain, online
action value, detector-only action selection, manifold correction, RL, or
generalization beyond NWPU.

## Data Boundary

- Dataset: the 454 positive NWPU VHR-10 train images recorded by
  `spectral_detection_posttrain/configs/splits/nwpu_re_roi_counterfactual_s42_nested.json`.
- Splits: fit 250, tune 68, calibration 68, outer_heldout 68. Training reads
  fit only. Exploratory gates read tune only.
- Calibration and outer_heldout remain unread during the exploratory group.
  A passing tune result requires a new read-once confirmation document before
  either split is opened.
- Detector validation remains untouched until train-only confirmation passes.
- The source detector may previously have seen these train images. Therefore
  tune means held out from the weighted continuation, not held out from the
  source detector.
- Exact image IDs, split hash, annotation hash, detector checkpoint hash,
  transform size, native postprocessing settings, cache hash, and Git commit
  must be recorded before the first training process starts.

## Oracle Utility

The only permitted source is a committed phase-1 re-ROI cache built on the fit
partition. There is no fallback utility source. A missing or incompatible cache
blocks the experiment; it does not authorize switching to IoU-score rescue.

For each fit image `S`, detector candidate `i`, and non-identity action `a`, the
cache provides the no-op-relative raw locked scalarization
`Q_teacher_raw(S, i, a)`. This is the weighted sum of the structured detector
deltas before the re-ROI ranker's optional fit-only target standardization.
The cache metadata must declare `raw_locked_scalarization_v1`; silently mixing
raw and standardized utilities is a contract failure. Identity must have exact
utility zero and never enters the maximum.

```text
candidate_utility(S, i) = max_{a != identity} Q_teacher_raw(S, i, a)
image_utility(S)        = max(0, max_i candidate_utility(S, i))
```

An image with no valid non-identity action receives `image_utility = 0`. Every
cache row must belong to fit, and duplicate `(image, candidate, action)` rows,
non-finite utilities, unknown images, or mismatched hashes invalidate the
cache.

This maximum can encode candidate-count effects. Phase-0 diagnostics therefore
report valid candidate count per image and utility correlations with candidate
count, object count, class presence, and baseline loss. These are diagnostics,
not post-hoc gates.

## Locked Weights

The weighting transform is frozen before cache inspection:

```text
raw_w(S) = min(exp(image_utility(S) / lambda), w_max)
lambda   = 1.0
w_max    = 20.0
w(S)     = raw_w(S) / mean_fit(raw_w)
```

The denominator is computed once over all 250 fit images and stored in the run
manifest. Global fit-mean normalization is mandatory so `W` and `U` have the
same average loss scale. Batch-local normalization is forbidden. Report the
raw and normalized min, max, mean, standard deviation, saturation fraction,
and effective sample size `ESS=(sum w)^2/sum(w^2)`.

## Arms

- `Z` `zero_train`: source checkpoint evaluated without fine-tuning. `Z` is not
  part of the optimizer-budget equality contract.
- `U` `uniform`: every fit image has weight one.
- `W` `oracle_weighted`: use the locked normalized `w(S)`.
- `F` `positive_filter`: sample uniformly from images with
  `image_utility(S) > 0`, with replacement, to consume the exact U image and
  optimizer-step schedule.
- `S` `utility_shuffle`: deterministically permute normalized W weights across
  all fit image IDs separately for each locked training seed. The exact weight
  multiset is preserved.

The primary transfer test is `W versus U`; the causal assignment test is
`W versus S`; `F` tests whether positive-only filtering explains the result.

## Locked Training Configuration

- Architecture: Faster R-CNN MobileNetV3-Large-320-FPN with 11 classes.
- Transform and postprocessing must be byte-for-byte compatible with the
  accepted re-ROI cache manifest. The intended NWPU transform is
  `min_size=max_size=480`; a cache generated at 320 is incompatible.
- Trainable parameters: `roi_heads.box_head` and
  `roi_heads.box_predictor` only. Backbone and RPN parameters and buffers must
  retain their pre-training state hashes.
- Optimizer: SGD, learning rate `0.001`, momentum `0.9`, weight decay `0.0005`,
  no LR scheduler.
- Training seeds: `42`, `2024`, `999`; all arms use common-random-number
  schedules for initial state, fit ordering, augmentation, and proposal
  sampling wherever arm semantics permit.
- Epochs: 10 fixed epochs. Each epoch consumes exactly 250 image exposures.
- Logical batch size: 8 images. If the implementation uses one-image forwards
  to recover per-image detector losses, gradients are accumulated and
  normalized over each logical batch before one optimizer step.
- Checkpoint selection: final epoch only. Tune may not select an epoch, stop a
  run, change a hyperparameter, or choose between `checkpoint_last` and
  `checkpoint_best`.
- U/W/F/S must have identical per-seed image-exposure counts, logical batch
  counts, optimizer-step counts, and scheduler-step counts. Runtime code must
  assert equality from counters, not infer it from requested epochs.

The weighted objective for one logical batch `B` is:

```text
L_B = sum_{S in B} w(S) * L_box(S) / |B|
```

The denominator is the logical batch image count, not the batch weight sum.
All arms use the same per-image-loss path. Implementing W by multiplying an
already batch-aggregated TorchVision loss is forbidden. Importance resampling
may be studied later but is not equivalent to this protocol.

## Evaluation And Bootstrap

Evaluate the final epoch on the exact tune IDs with native detector inference,
`score_threshold=0.05`, `nms_threshold=0.50`, and
`detections_per_image=100`. Report AP50, AP75, precision, recall,
`false_positive_rate`, `high_conf_fp_rate`, ECE, prediction count, and all
support counts for every arm and seed.

Pairwise AP75 uncertainty uses the same resampled tune image indices for both
arms and recomputes global AP75 after every resample. Averaging per-image AP is
forbidden. Use 10,000 image resamples with seed 42. Report every per-seed paired
interval and a hierarchical seed-and-image paired interval for the three-seed
mean.

## Locked Gates

0. `provenance`: exact split, cache, annotation, checkpoint, transform,
   postprocessing, config, and Git hashes match; the worktree is clean.
1. `support`: at least 15% of fit images have positive image utility and at
   least 500 fit candidates have positive candidate utility.
2. `weight_health`: normalized mean equals one within `1e-7`, saturation is at
   most 10%, and ESS is at least 50% of fit image count.
3. `budget_match`: U/W/F/S runtime image, logical-batch, optimizer-step, and
   scheduler-step counters are exactly equal for every seed.
4. `advantage_causality`: the three-seed mean tune AP75 delta `W-S` has paired
   hierarchical-bootstrap LCB above zero; at least two of three per-seed point
   deltas are positive.
5. `uniform_gain`: the three-seed mean tune AP75 delta `W-U` is at least
   `+0.002` with paired hierarchical-bootstrap LCB above zero; no per-seed
   point delta is below `-0.002`.
6. `safety`: for every seed, W versus U increases false-positive rate by at
   most `0.01` and prediction count by at most 10%.
7. `calibration`: for every seed, W versus U increases ECE by at most `0.01`.
8. `generalization`: fit and tune are both evaluated at the fixed final epoch;
   the W fit-to-tune AP75 gap exceeds the U gap by at most `0.05` for every
   seed.

No threshold may be relaxed after cache diagnostics or training output is
read.

## Failure And Branch Rules

A provenance, cache, implementation, frozen-state, or budget mismatch is a
contract failure. It invalidates the run and must be repaired and rerun from
scratch; it does not count as scientific evidence for or against weighting.

If contracts pass and any scientific gate fails, freeze this image-level
oracle-utility weighting direction. Do not tune `lambda` or `w_max`, switch
utility source, expand to full-model fine-tuning, add actions, add seeds, use
proposal-level weights, or convert the result into an inference-time action.

If every tune gate passes, write and commit a separate confirmation document
before reading calibration or outer_heldout. A passing tune group alone is not
a detector-validation or AP improvement claim.
