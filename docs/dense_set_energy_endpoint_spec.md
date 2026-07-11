# Dense Global Set-Energy Endpoint Specification

## Objective

Replace the current hard whole-image Delta-U count lattice with a dense train-time teacher energy over a detector prediction set. The learned endpoint must estimate set quality from detector-only evidence at inference and must contain an explicit identity basin before any action learner is attached.

This is an endpoint project, not a new action-head experiment.

## Prediction Set

For image `x`, let the detector produce a native candidate set

```text
S = {(b_i, p_i, y_i, h_i)}
```

where `b_i` is a box, `p_i` class confidence, `y_i` predicted class, and `h_i` detector feature. Candidate generation is detector-only. Ground truth may be used on the train split only to construct teacher energy and never to filter `S`.

## Dense Teacher Energy

For ground-truth object `g`, define soft class-aware coverage

```text
c_g(S) = tau * logsumexp_i [
    log(p_i + eps)
    + kappa * (IoU(b_i, g) - 0.75)
    + class_mask(y_i, y_g)
]
```

where `class_mask` is zero for a class match and a large negative constant otherwise. The smooth localization term preserves information on both sides of IoU 0.75.

Define all-class and same-class overlap

```text
m_any_i  = max_g IoU(b_i, g)
m_same_i = max_{g: y_g = y_i} IoU(b_i, g)
```

and separate background and classification risks

```text
r_bg_i(S)  = p_i * sigmoid((0.50 - m_any_i) / tau_bg)
r_cls_i(S) = p_i * sigmoid((m_any_i - 0.50) / tau_bg)
                  * sigmoid((0.50 - m_same_i) / tau_cls)
```

and pairwise duplicate energy

```text
d_ij(S) = 1[y_i = y_j] * p_i * p_j
          * sigmoid((IoU(b_i, b_j) - nms_ref) / tau_dup)
```

The teacher quality is

```text
Q_teacher(S) =
    lambda_cov * sum_g softplus(c_g(S))
    - lambda_bg * sum_i r_bg_i(S)
    - lambda_cls * sum_i r_cls_i(S)
    - lambda_dup * sum_{i<j} d_ij(S)
    - lambda_cal * calibration_error(S)
```

Use a soft same-class correctness target

```text
q_i = sigmoid((m_same_i - 0.50) / tau_cal)
calibration_error(S) = sum_i (p_i - q_i)^2
```

This is a dense Brier-style term, not binned ECE. Raw coverage, background, class-error, duplicate, and calibration terms are robustly standardized using median and IQR from inner-fit only. Temperatures are fixed before fitting; no temperature or weight may be selected from detector validation AP. The first implementation uses unit weights after robust standardization and treats alternative weights as a new endpoint version.

## Identity Basin

The endpoint must represent identity explicitly. For any candidate perturbation `a`, define

```text
Delta Q_teacher(S, a) = Q_teacher(T_a(S)) - Q_teacher(S) - lambda_energy * ||a||^2
```

No-op is preferred unless the lower confidence bound of predicted `Delta Q` is positive. A learned endpoint that systematically assigns positive action energy to identity-equivalent changes fails before detector evaluation.

## Learned Detector-Only Endpoint

Train a permutation-equivariant set model `Q_theta(S)` to reproduce teacher set energy and local energy differences. The first experiment contains no learned actions.

The minimum architecture is an explicitly additive DeepSets energy:

```text
Q_theta(S) = sum_i u_theta(z_i) + sum_{i<j} v_theta(z_i, z_j)
```

where `u_theta` is a regularized unary MLP and `v_theta` is a symmetric pair MLP evaluated only on a detector-defined sparse edge graph. This decomposition makes node/pair contributions sum exactly to global energy. Uncertainty is calibrated from absolute heldout residual quantiles on inner-tune; MC-dropout uncertainty is not used in the first version.

Required outputs:

- absolute set quality `Q_theta(S)`;
- per-node quality contribution;
- pairwise duplicate contribution;
- residual-quantile calibrated uncertainty for `Delta Q`;
- an identity/no-change probability.

## Train-Only Validation Protocol

Use a new nested split entirely within NWPU train images:

```text
manifest: spectral_detection_posttrain/configs/splits/nwpu_dense_endpoint_s42_nested.json
SHA256: ce19316aeaef1cbdf85f2c9668c5ae2c8443da8e848de2d22ed0f0f3a687e080
source train: 454 images, SHA256 7abe3c8370985f49698dcc3c42ca5917f1e17941fa024643147a58479c7cd9bd
inner-fit: 318 images, SHA256 3314b7d59dccf8df83a11662883b822c874757f7dd6b62356cb91335b434e54f
inner-tune: 68 images, SHA256 48dc60dacdcfa3e532bc61ca87e08c2dad36ded90ff5dcc82b7a6764188174d0
outer train-heldout: 68 images, SHA256 4e1acfd60ffdc4a528adeb9fc942d4c1867ff9b2117e5e86089e8f84240fa771
```

The manifest uses deterministic multilabel stratification over class presence plus object-count bins. Every class has at least three images in inner-tune and outer heldout. Detector validation IDs are absent.

1. inner-fit trains `Q_theta`;
2. inner-tune selects regularization and a conservative uncertainty threshold;
3. outer train-heldout is read once for endpoint validation;
4. detector validation remains untouched until all endpoint gates pass.

Controls use identical capacity and optimization:

- within-image node-feature shuffle;
- teacher-energy shuffle;
- pairwise-edge topology shuffle;
- zero-energy and always-identity baselines.

## Endpoint Gates

All gates are mandatory:

- equal-image and candidate-weighted pairwise accuracy above `0.60`;
- minimum `0.03` gain over every shuffle control;
- absolute energy MAE below zero-energy baseline;
- positive-vs-nonpositive AUROC above `0.65` after image-centering;
- exact native-identity-equivalent action false-positive rate below `5%`;
- selected positive-energy subset with image-paired bootstrap LCB above zero;
- fit-to-heldout gap below `0.10`;
- support on at least 60 heldout images for a full-scale endpoint test.

Failure freezes the endpoint formulation. It does not authorize a larger model or action learner.

Identity-equivalent examples must be verified, not assumed. Node permutation is exactly equivalent by construction. A box/score perturbation is admitted to the identity control only when native decode, clipping, thresholding, NMS, and top-k produce exactly identical boxes, scores, and labels under the strict parity comparator. Unverified pixel or confidence jitter is not an identity control.

## Action Reintroduction Gate

Only after endpoint validation may one bounded action family be tested. The action experiment must:

- use detector-only candidates at inference;
- preserve exact no-op native parity;
- compare against feature, energy, and action-rate-matched controls;
- use a fresh detector validation split once;
- report AP50, AP75, precision, recall, FPR, ECE, and prediction count;
- stop on any detector, safety, parity, or control failure.

## Claim Boundary

A successful endpoint probe establishes only that dense set quality is learnable from detector evidence. It does not establish detector AP improvement, manifold correction, or useful actions. Those claims require the later native detector experiment.
