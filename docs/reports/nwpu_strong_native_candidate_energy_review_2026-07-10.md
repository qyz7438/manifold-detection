# NWPU Strong Native Candidate-Energy Review

Date: 2026-07-10

Dataset: NWPU VHR-10 full validation, seed/data-seed 42

Model: Faster R-CNN MobileNetV3-320-FPN at 480 px

## Decision

The action-local line now has a valid strong baseline, exact native zero-action
parity, and a bounded discrete action space with substantial greedy GT-policy
headroom. The original direct residual head and 73-way listwise energy head do
not produce usable detector corrections.

Dense identity-relative gain supervision changes the result qualitatively. It
learns the sign of local candidate IoU changes and can select a majority of
locally beneficial actions. The best full-val detector result is spatial dense
gain at epoch 2:

- AP50 `0.664589`, unchanged from identity;
- AP75 `0.318879`, versus identity `0.317619`, delta `+0.001260`;
- 122 selected actions, 57.38% beneficial, 25.41% harmful;
- mean realized proposal IoU gain `+0.005414`;
- context-preserving spatial-feature shuffle AP75 `0.317981`.

This is a weak information-gain signal, not a promoted detector improvement.
It fails the preregistered AP75 promotion threshold of `+0.0015`, occurs at one
epoch and one seed, and is sensitive to the energy-drop threshold. Seed
expansion and further architecture growth are therefore stopped.

## Reproducibility Identity

- Detector checkpoint SHA256:
  `de126708d063a82dd5c721799cd124fc4e52d28866139e103b1cd65427742027`
- Annotation SHA256:
  `dde27d9362d9c6aa358a1cab160c9e075e57405d3fe876de049973c6e6150f0e`
- Seed-42 split SHA256:
  `96801b04fcabeeac175248f1c3641f6cb1cdf1411b48997081471e57a65d9a04`
- Positive images: 650; train: 454; validation: 196.
- Active implementation commits: `a87b727`, `de27c4a`, `0a1b2f4`,
  `08a3b50`, `e9b7412`, `d1f6191`, `34e023e`.
- Remote workspace:
  `/home/ps/lzz/manifold-detection-energy-transport`.
- Every reported experiment below is full-val unless explicitly marked smoke.

## Strong Baseline And Native Parity

The common native identity evaluation from the strong checkpoint is:

| AP50 | AP75 | Precision | Recall | FPR | ECE | Predictions |
|---:|---:|---:|---:|---:|---:|---:|
| 0.664589 | 0.317619 | 0.562500 | 0.717579 | 0.437500 | 0.116116 | 1328 |

Both repaired zero-action parity runs passed all required gates:

- aggregate parity passed;
- `strict_zero_action_parity.passed=true`;
- `mismatched_images=0` over 196 images;
- exact-zero actions;
- zero count, label, box, and score error.

The historical custom postprocessor remains invalid evidence. All candidate
results in this report use the native BoxCoder, full foreground expansion,
thresholding, small-box filtering, class-wise NMS, top-K, and transform
postprocessing.

## Direct Residual No-Go

The strong-checkpoint native matrix rejected the old learned residual action:

| Policy | Best AP75 | Identity delta |
|---|---:|---:|
| Direct box-only | 0.167666 | -0.149953 |
| Preserve-2 | 0.290477 | -0.027142 |
| Preserve-2, smallest tested scale | 0.314247 | -0.003372 |

No tested global action scale exceeded identity, and learned small-scale
directions did not reliably exceed within-image permutation controls. Even a
perfect GT acceptance gate applied to the old actions reached only AP75
`0.346452` for box-only and `0.328327` for preserve-2. This was insufficient
to justify adding a learned stop gate to the old direction field.

## Discrete Candidate Reachability

The replacement action set contains identity plus 24 symmetric translation,
size, and joint directions at step sizes `0.05`, `0.10`, and `0.20`, for 73
actions total.

| Greedy GT policy | AP50 | AP75 |
|---|---:|---:|
| Max 32 actions per image | 0.730278 | 0.519619 |
| Unlimited actions | 0.732473 | 0.553664 |

These policies choose actions from locally matched GT quality but are evaluated
through native postprocessing and NMS. They demonstrate candidate-space
reachability, but they are not mathematical AP upper bounds because action
selection is local and independent before global NMS and matching effects.

## One-Hot Ranking Failure

Treating the 73 actions as a hard listwise class is destructive:

| Feature source | Best epoch | Loss | Learned AP75 | Feature-shuffle AP75 |
|---|---:|---:|---:|---:|
| 1024-D box-head vector | 3 | 4.2364 | 0.051452 | not available in original run |
| 256x7x7 spatial pooled ROI | 1 | 4.2866 | 0.025362 | 0.027976 |

For reference, `ln(73)=4.2905`. The box-head run's best epoch selected 3662
actions, of which only 2.18% exactly matched the hard target, 12.15% were
beneficial, and 83.94% were harmful. A full policy sweep did not rescue it:

- budget 1 AP75: `0.298036`;
- margin 0.10 AP75: `0.290825`;
- all tested budget and margin settings remained below identity.

The hard target discards the fact that many neighboring candidates have
similar quality. Replacing only the 1024-D feature with spatial ROI evidence
does not fix this objective.

## Dense Gain Supervision

Dense gain supervision fits

`E(identity) - E(action) ~= quality(action) - quality(identity)`

for every candidate. It is gauge-invariant and gives all 73 actions continuous
supervision rather than one discontinuous target index.

| Feature / target | Epoch | AP50 | AP75 | AP75 delta | Moves | Beneficial | Harmful | Mean IoU gain | Feature shuffle AP75 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Box / raw IoU gain | 3 | 0.662853 | 0.317494 | -0.000125 | 220 | 60.91% | 29.09% | +0.002347 | 0.317494 |
| Spatial / raw IoU gain | 2 | 0.664589 | 0.318879 | +0.001260 | 122 | 57.38% | 25.41% | +0.005414 | 0.317981 |
| Box / AP75 utility | 3 | 0.664670 | 0.318100 | +0.000481 | 81 | 79.01% | 16.05% | +0.010908 | 0.318100 |
| Spatial / AP75 utility | 2 | 0.664609 | 0.317620 | +0.000001 | 80 | 53.75% | 15.00% | +0.002599 | 0.317620 |
| Context-only / raw IoU gain | 4 | 0.664829 | 0.318447 | +0.000828 | 136 | 72.79% | 18.38% | +0.013640 | n/a |

The box raw-gain epoch 3 candidate-sign accuracy was `0.9047`; spatial raw-gain
epoch 2 reached `0.8769`. Dense supervision therefore learns local geometric
ordering even when exact hard-target accuracy is low. Local IoU improvement is
not equivalent to detector AP improvement.

## Spatial Marginal Information

Three controls bound the spatial feature contribution:

1. Spatial raw-gain learned AP75 is `0.318879`; context-preserving spatial
   feature shuffle is `0.317981`, a learned-minus-shuffle difference of
   `+0.000898`.
2. Context-only raw-gain reaches AP75 `0.318447`, explaining most of the
   spatial run's `+0.001260` identity delta.
3. Box and spatial AP75-utility runs match their feature-shuffle controls.

The correct conclusion is not that ROI features contain exactly zero
information. They contain, at most, a weak marginal action signal under this
single-seed setup. That signal is below the promotion threshold and cannot yet
support a manifold-information claim.

## Policy Robustness

The best spatial raw-gain epoch-2 checkpoint was swept without retraining.

| Policy | AP75 | Identity delta | Moves |
|---|---:|---:|---:|
| Budget 1, drop 0.002 | 0.318094 | +0.000475 | 39 |
| Budget 4, drop 0.002 | 0.318094 | +0.000475 | 88 |
| Budget 8, drop 0.002 | 0.318879 | +0.001260 | 107 |
| Budget 16/32, drop 0.002 | 0.318879 | +0.001260 | 122 |
| Budget 32, drop 0.001 | 0.308389 | -0.009230 | 803 |
| Budget 32, drop >=0.003 | 0.317619 | +0.000000 | <=47 |

The budget behavior is stable from 8 to 32 actions, but the energy margin has
a narrow useful window. This is not a robust operating point.

## DeepSeek Review And Codex Arbitration

DeepSeek agreed that:

- local geometric sign learning is real;
- detector AP gain is not established;
- context-only is the decisive marginal-information control;
- seed expansion and architecture growth should stop.

Codex rejected DeepSeek's stronger claim of exactly zero spatial information.
The spatial learned-minus-shuffle difference (`+0.000898`) and spatial versus
context-only difference (`+0.000432`) are positive, but too small and
single-seed to promote. Codex also corrected earlier reviewer mistakes: eligible
identity energies are learned, oracle predictions do pass through native NMS,
and the detector checkpoint is the NWPU strong baseline.

## Claim Boundary

Supported:

- the bounded candidate action space has substantial locally supervised
  reachability;
- 73-way one-hot ranking is structurally misaligned with smooth candidate
  quality;
- dense identity-relative energy learns local gain signs and abstention;
- spatial pooled ROI evidence has a weak positive marginal signal under the
  raw dense-gain objective;
- context and candidate-delta priors explain most of the observed AP behavior.

Unsupported:

- robust AP50 or AP75 improvement over the strong detector;
- a seed-general spatial-manifold information gain;
- the claim that lower intrinsic dimension alone makes 256x7x7 features a
  better correction representation;
- expanding seeds 2024/999 or growing the current energy-head architecture;
- treating greedy GT-policy AP as a true global upper bound.

## Next Research Step

Do not continue tuning this proposal-independent candidate head on the same
validation split. If the line is resumed, the next experiment should change
the problem boundary: train a set-level, NMS-aware action selector or evaluate
detector-native regression suggestions as candidates. The target must represent
which coordinated actions survive thresholding, matching, and NMS, rather than
only which individual proposal gains local IoU.

## Verification

- Local maintained suite: 277 passed, 2 known unrelated tests deselected.
- Remote focused candidate suite: 16 passed.
- All launchers passed remote `bash -n`.
- Spatial full-pipeline smoke: one validation image, implementation-only; not
  used for scientific conclusions.
- Remote worktree was clean at each `--require-clean-git` experiment launch.
