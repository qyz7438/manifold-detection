# Dense Local Delta-Q Residual Protocol Draft

## Status

Blocked design. No cache, model, detector inference, validation read, or action run is authorized by this document.

The one-shot split builder at commit `df400296...` was executed once with seed `52042` and failed before writing a manifest: the 32-image tune split contained only one image for class 3 and one for class 8, below the locked minimum of three. Fit class 8 support was also only three images. Density-bin support was complete. Per the no-alternative-manifest rule, the seed, capacities, and threshold were not changed and no second split was generated. This protocol is not executable.

Any future residual study must be proposed as a new researcher-adaptive protocol, not as a repair or continuation of this blocked design.

The preceding learner is frozen. Its tune pairwise accuracy was `0.57469` against a locked `0.60` gate, and a fit-only perturbation-family mean baseline outperformed it on pairwise accuracy, MAE, and sign AUROC. This protocol must not reinterpret that result.

## Scientific Question

For perturbation family `f`, decompose the dense teacher difference as

```text
Delta-Q(S, a_f) = mu_f + r(S, a_f)
```

where `mu_f` is estimated only from fit images. The sole question is whether detector-only set features predict the content-conditioned residual `r` on newly locked train-only images.

The family term is an explicit fixed baseline, not a learned categorical embedding. The endpoint is successful only if its residual prediction adds information beyond `mu_f`.

## Data Boundary

- Use NWPU train images only.
- Exclude all 64 images used by `det.energy.dense_local_delta_learner.001`.
- Select 128 images from the remaining 254 images in the original 318-image `inner_fit` pool.
- Lock a deterministic multilabel class-presence and object-density stratified split before detector inference: 96 fit images and 32 tune images. The future manifest must record the algorithm version, seed `42`, density-bin edges, deterministic tie-break, complete ID lists, and hashes; no alternative manifest may be selected after annotation statistics are inspected.
- Do not read the old 68-image `inner_tune`, old 68-image train-heldout outer split, or the 196-image detector validation split.
- Lock image IDs, class support, object counts, manifest hash, annotation hash, detector checkpoint hash, and Git commit before cache generation.

This remains researcher-adaptive train-only evidence because the method was designed after observing earlier train-only results.

## Perturbations

Keep the existing fixed families and magnitudes unchanged:

```text
identity_permutation
score_down, score_up
translate_left, translate_right, translate_up, translate_down
scale_down, scale_up
drop
```

Use the top three native post-NMS detections per image, score step `0.02`, and box step `0.02` of box width/height. Do not rerun native NMS after the synthetic set perturbation. These remain endpoint probes, not detector actions.

Identity permutation must produce exact teacher `Delta-Q=0` and the `local_residual_full` prediction must be zero within `1e-7`. Shuffle-arm identity errors are control-integrity diagnostics, not success gates, because the existing strong geometry control is not permutation equivariant.

## Fit-Only Family Prior

For each non-identity family, compute

```text
mu_f = candidate_row_mean_fit(Delta-Q | family=f)
```

Then define fit and tune targets with the frozen fit statistic:

```text
r = Delta-Q - mu_f
```

The candidate-row mean matches the completed family-prior control. Set `mu_identity=0`; identity rows never enter prior fitting, residual loss, rank loss, or primary MAE/rank metrics. All nine non-identity families must have fit support. A missing family is a cache-contract failure; there is no fallback.

Tune labels must never influence `mu_f`, feature normalization, optimization, early stopping, thresholds, or model selection.

## Model And Objective

Use the same 22-D node features, 5-D sparse pair features, hidden dimension 32, and mean-additive unary/pair endpoint as the frozen learner. This first residual test changes the target decomposition, not capacity.

Use AdamW for exactly 100 epochs with learning rate `0.001`, weight decay `0.001`, Smooth-L1 beta `0.05`, and `lambda_rank=1.0`. Feature statistics come only from fit rows. Enumerate every eligible same-image, same-family fit pair once per epoch in stable row-ID order; do not sample pairs. The executable config must lock deterministic PyTorch/CUDA settings and accumulation details before any cache is built.

Train the residual endpoint with the fixed hybrid objective:

```text
L = SmoothL1(r_hat, r, beta=0.05)
  + lambda_rank * softplus(-(r_hat_i-r_hat_j) * sign(r_i-r_j))
```

Ranking pairs use only the same image and perturbation family. Exact and near ties with `|r_i-r_j| <= 1e-6` are excluded from the ranking term. If an image-family cell has no eligible pair, it contributes no rank term and remains in support accounting.

## Arms

All learned arms use identical initialization, capacity, optimizer, epochs, and pair sampling.

1. `local_residual_full`: detector-only set features predict `r`.
2. `strong_geometry_shuffle`: preserve non-geometric values while breaking node geometry and pair topology exactly as in the prior strong control.
3. `within_family_utility_shuffle`: with seed `742`, permute residual target rows globally within each family using one deterministic non-identity permutation; preserve row features and family distribution.
4. `within_family_image_feature_shuffle`: with seed `842`, permute images separately within each `(family, candidate_rank)` cell and move the complete `(base_node, base_pair, perturbed_node, perturbed_pair)` bundle while keeping the destination residual target fixed. Each mapping must be non-identity when support exceeds one.

The cache schema must include `stable_row_id`, `image_id`, `family`, `candidate_rank`, target, identity flag, and all full/control feature bundles. Every control mapping is stored and hashed.

Fixed non-learned baselines:

1. `family_prior`: predict total Delta-Q as `mu_f`, equivalent to zero residual.
2. `global_prior`: predict the global fit mean Delta-Q.
3. `zero_delta`: predict zero total Delta-Q.

## Primary Metrics

Report both residual and reconstructed total Delta-Q metrics.

Residual metrics:

- MAE and relative MAE gain over zero residual.
- Same-family, within-image pairwise accuracy: within each image pool all same-family pairs satisfying `|r_i-r_j|>1e-6`, compute that image's accuracy, then average equally over images with at least one eligible pair.
- Candidate-weighted pairwise accuracy as secondary context.
- Residual sign AUROC.
- Fit-to-tune gaps.

Reconstructed total metrics using `mu_f + r_hat`:

- MAE versus the fixed family prior.
- Within-image pairwise accuracy versus the family prior, using every non-identity within-image pair with `|Delta-Q_i-Delta-Q_j|>1e-6` and then averaging equally over supporting images.
- Sign AUROC versus the family prior.

Always report supporting images, eligible pairs, exact/near ties, per-family support, and candidate-weighted secondary metrics. For every full-control comparison, compute differences on the identical supporting-image intersection and a seed-42, 10,000-resample image bootstrap 95% confidence interval. Training loss is never a success metric.

## Locked Gates

Every gate must pass:

1. `support`: 32 tune images, at least 800 non-identity tune rows, all nine families represented on at least 24 tune images, and every family has an eligible same-family rank pair on at least 24 tune images.
2. `identity`: teacher and `local_residual_full` identity errors are at most `1e-7`; shuffle-arm identity is reported only.
3. `residual_mae`: at least 10% relative MAE gain over zero residual.
4. `within_family_rank`: image-equal same-family pairwise accuracy at least `0.60`.
5. `rank_controls`: for the primary residual same-family image-equal metric, full minus each learned control is at least `0.03`, and each image-paired bootstrap lower bound is above zero.
6. `mae_controls`: compute candidate MAE within each image, then compare images equally; full residual MAE must be lower than every learned control and each paired improvement lower bound must exceed zero.
7. `family_prior_gain`: on reconstructed total Delta-Q, per-image candidate MAE improves over the fixed family prior by at least 10%, and all-family within-image image-equal pairwise accuracy improves by at least `0.03`; both paired lower bounds must exceed zero on fixed common image support.
8. `generalization`: fit-to-tune residual pairwise gap is at most `0.15` and relative MAE-gain gap is at most `0.20`.

No gate may be relaxed after cache or tune results are read. The executable config may only be created while this document remains the sole evidence input; once the manifest/config is committed, further protocol edits invalidate that run rather than modifying it.

## Branch Rule

- If all gates pass, retain only a researcher-adaptive train-only content-residual signal and preregister one larger train-only confirmation. Do not read detector validation or run actions.
- If any gate fails, freeze the synthetic post-NMS local Delta-Q endpoint line. Do not add family embeddings, increase endpoint capacity, change pooling, retune the loss, expand seeds, or convert the probe into native actions.

Even a passing result would not establish manifold correction, coordinated action selection, native postprocessing gain, AP improvement, or detector deployment value.
