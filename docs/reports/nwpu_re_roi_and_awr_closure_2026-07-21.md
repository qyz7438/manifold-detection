# NWPU Re-ROI Evidence and Oracle-Utility Box-Head Closure

## Decision Summary

Two train-only protocols are closed without opening `outer_heldout` or
detector validation:

- `det.energy.re_roi_counterfactual_evidence.001` is a completed scientific
  negative result. The pipeline was operational and non-degenerate, but the
  post-action re-ROI arm failed the locked matched-static and bundle-integrity
  gates, and it also failed calibration and generalization. This exact
  mechanism is frozen.
- `det.energy.oracle_utility_boxhead.001` stopped at its locked Z support
  preflight. Positive-image support passed, but positive-candidate support did
  not. Arms U/W/F/S were never authorized to start. The preregistered version
  config remains unchanged.

Neither result is detector-unseen evidence, a detector AP result, an action
policy, or validation of a target manifold. No new experiment is authorized by
this closure.

## Provenance and Scope

| Item | Value |
|---|---|
| Local review branch | `codex/repository-research-state-refactor` |
| Implementation baseline | `ce49b5747800e0a7b76e1f5be8e2b46f6500dbea` |
| Handoff/review baseline | `657334127a291e87e65c9704174dbb864e5b2dbd` |
| Remote workspace alias | `remote:manifold` |
| Remote run | `remote:manifold/runs/nwpu_re_roi_evidence_s42_strongbest_ce49b57_v1` |
| Re-ROI scope | `nwpu_train_only_fit_tune_calibration` |
| Fit / tune / calibration images | 250 / 68 / 68 |
| `outer_heldout_read` | `false` |
| `detector_validation_read` | `false` |
| Seed | 42 |

The local review confirmed that `ce49b57` is an ancestor of the handoff tip.
The remote worktree remained at `ce49b57`, clean, with no active re-ROI
process before this report was written. The remote split file is registered by
its remote logical alias because its run-bound SHA-256 reflects the Linux
checkout bytes; the Windows checkout has different line-ending bytes.

## AWR-Weighted Box-Head Support Stop

The historical AWR name refers here to supervised image-weighted empirical
risk minimization, not AWR/AWAC/CRR or reinforcement learning. Its locked
preflight produced:

| Diagnostic | Observed | Gate |
|---|---:|---:|
| Positive fit images | 62 / 250 = 0.248 | at least 0.15, pass |
| Positive candidates | 93 | at least 500, fail |
| Effective sample size | 197.0039 | at least 125, pass |
| Weight saturation | 0 | at most 0.10, pass |
| Normalized mean weight | 1.0 | 1 within `1e-7`, pass |

This is a valid support-gate failure rather than a normalization defect. The
training arms U/W/F/S did not run, no tune/calibration/outer split was opened,
and no AP metric was produced. Relaxing the 500-candidate floor, changing
`lambda` or `w_max`, switching utility source, or moving to proposal/full-model
weighting would be a new post-hoc protocol and is not authorized.

The only surviving runtime file is
`remote:manifold/runs/nwpu_oracle_utility_boxhead_strongbest_e58806b_s42_Z_launcher.log`
(591 bytes, SHA-256
`ab95634556eba177c1c6ec39c478171dba6b14daf8c7fbddb88a2236739159c5`).
It records the fail-closed `support gate failed before training` stop. The
numeric diagnostics survive in the reviewed session handoff, not in that log;
the artifact manifest records this limitation explicitly. The immutable
version config remains `status=preregistered_not_run` because the blocked
preflight is recorded in the registries rather than rewriting the protocol.

## Formal Re-ROI Result

The four arms were:

- A: family prior;
- B: matched static-ROI learner;
- C: post-action re-ROI learner;
- D: deterministic within-image re-ROI bundle derangement.

Identity was excluded from non-identity statistics and was exact for the
teacher and all learned arms. All nine non-identity action families covered
every tune image. The principal results were:

| Metric | Fit | Tune |
|---|---:|---:|
| B residual pairwise accuracy | 0.59005 | 0.57820 |
| C residual pairwise accuracy | 0.54843 | 0.53822 |
| D residual pairwise accuracy | 0.56672 | 0.52583 |
| A reconstructed pairwise accuracy | 0.59931 | 0.59474 |
| B reconstructed pairwise accuracy | 0.71505 | 0.71066 |
| C reconstructed pairwise accuracy | 0.73349 | 0.71969 |
| D reconstructed pairwise accuracy | 0.69330 | 0.67922 |
| C reconstructed relative MAE gain | 0.55308 | 0.28024 |

### Locked gate outcome

| Gate | Result | Evidence |
|---|---:|---|
| support | pass | 68 tune, 68 calibration, 1,557 tune non-identity rows, all nine families |
| identity | pass | teacher and model maximum absolute identity error 0.0 |
| re_roi_gain | fail | C 0.53822 vs B 0.57820; delta -0.03999; LCB -0.08438 |
| bundle_integrity | fail | C 0.53822 vs D 0.52583; delta 0.01238; LCB -0.02924 |
| static_baseline | pass | C beats zero; reconstructed relative MAE gain 0.28024 |
| calibration | fail | positive rate 0.02941 below 0.90; coverage 1.0; correction 268.4032 |
| generalization | fail | C residual-MAE-gain fit/tune gap about 0.27284 above 0.15 |

Paired image bootstrap controls reinforce the decision:

| Comparison | Point delta | 95% interval |
|---|---:|---:|
| C vs B static ROI | -0.04145 | [-0.08438, 0.00145] |
| C vs D bundle shuffle | 0.00976 | [-0.02924, 0.04837] |
| C vs utility shuffle | 0.06599 | [0.01204, 0.11942] |

The utility-shuffle comparison shows that the labels contain learnable
structure. It does not establish the causal increment of post-action
re-pooling because C fails against the matched static learner and the
bundle-shuffle control. Likewise, the reconstructed metric is helped by the
family prior; it cannot replace the residual matched-arm gates.

## Immutable Re-ROI Artifacts

| Artifact | Size (bytes) | SHA-256 |
|---|---:|---|
| `eval_metrics.json` | 13,436 | `4ac0da6a065bc58486006326abbcf32fbe25ca4df0152d76ce075a9511affd85` |
| `manifest.json` | 4,939 | `d011653bc9997b2d0f5a0d206229417cf1c76c6a8bc5baa0b3db46f85d154591` |
| `arm_B_final.pt` | 640,185 | `9d6985798bcbfcbb54fc7ad092679097c8a6abb881e0bdaa2a23f437b7d1df47` |
| `arm_C_final.pt` | 902,329 | `68fc54c68e2a56982da49c8fd8789c2e58c7aa385ceaafa73c4e8c8ceed3ac2f` |
| `arm_D_final.pt` | 902,329 | `f2880626ec7595a8c0aba7a71056a3f2d590d50f5f50ce22577e1d327722373e` |

Bound inputs include the strong checkpoint
`de126708d063a82dd5c721799cd124fc4e52d28866139e103b1cd65427742027`,
fit/tune/calibration caches `7fcb3774...288f7`,
`6110a25c...07db7`, and `6421f83c...13c8c`, and remote split SHA-256
`5c4222e033f7ed333835eab7f7786498b84bba11856c2d362ec8895845942244`.
The reviewed manifest stores all hashes in full.

## Claim Boundary and Frozen Decision

Permitted wording:

> On the locked NWPU train-only fit/tune/calibration split, post-action re-ROI
> evidence did not add the required transferable residual information beyond
> a matched static-ROI learner or the bundle-shuffle control. A separate
> oracle-utility box-head protocol stopped before training because its locked
> positive-candidate support floor was not met.

Forbidden promotion:

- no detector-unseen, full-validation, AP, deployment, or action-policy claim;
- no statement that re-ROI features are universally useless;
- no seed/action/capacity expansion or gate relaxation for this version;
- no restoration of C1-E1, dense-absolute, local-Delta-Q, GT-filtered oracle
  pools, prototype attraction, FFT reward, or AFM/FPN-SM as the active thesis.

## Project Retrospective

- Terminal state: re-ROI scientific NO-GO; oracle-utility box-head blocked at
  a preregistered support gate.
- First causes: absent matched-arm re-ROI increment and insufficient positive
  candidate support, respectively.
- Evidence validity: valid within train-only boundaries; `outer_heldout` and
  detector validation remain sealed.
- Compute avoided: AWR U/W/F/S and re-ROI outer/validation/seed expansion were
  stopped before unnecessary GPU work.
- Retained practice: cheap support gates before training, identity and shuffled
  controls, matched static capacity, explicit residual versus reconstructed
  metrics, and fail-closed cohort sealing.
- Next decision: only after registry/test/review/sync closure may a genuinely
  prospective mechanism be designed under a new version and fresh gates.

## Independent Review

One bounded DeepSeek v4 Pro read-only evidence review returned `CONFIRM` with
no findings. Codex accepted its checks that:

- the re-ROI negative decision follows the locked matched-arm failures despite
  static-baseline success;
- the oracle-utility box-head result is a valid preregistered support stop
  before U/W/F/S rather than a post-hoc training decision;
- the report preserves the train-only and sealed-cohort claim boundaries; and
- the AWR launcher log does not contain the numeric preflight diagnostics, a
  limitation already disclosed in the report and reviewed manifest.

No review point was rejected. DeepSeek did not authorize a new experiment and
is not treated as the final authority; Codex retains the integrated scientific
decision above.
