# Autonomous Research Exploration Report

## Status

- Window: 2026-07-12 03:06 to 11:06 Asia/Shanghai.
- Current status: action line and dense absolute endpoint are frozen; the first shift-invariant local Delta-Q learner also failed its locked pairwise gate and is frozen.
- Automation: `manifold-autonomous-8h-20260712`, two-hour heartbeat.
- Project: `/home/ps/lzz/manifold-detection-energy-transport`, branch `codex/energy-guided-roi-transport`.
- Dataset/scope: NWPU VHR-10, seed 42; historical 32/32 action smoke plus 454-image train-only endpoint development and a one-time 196-image detector-unseen absolute endpoint validation.
- Identity smoke reference: AP50 `0.7399366`, AP75 `0.4636476`, precision `0.572165`, recall `0.776224`, FPR `0.427835`, ECE `0.119736`, 194 predictions.

## Codex Judgment

The explored action line is negative. Detector-only candidates and native postprocessing are implemented correctly, but the old whole-image Delta-U endpoint does not support a safe, attributable per-proposal action policy.

The main problem is not merely network capacity. The endpoint is an integer event-count utility:

```text
U = tp75 - 0.25 fp75 - 0.10 fp50 - action energy - action count cost
```

For bbox motion, almost every action leaves native output unchanged and receives only action cost. For post-NMS deletion, utility becomes a four-valued TP/FP lattice that encourages suppression without providing a transferable identity basin. The intended low-energy manifold flow is therefore not represented by that supervision.

The dense set-energy pivot is more informative than the hard action utility, but it is not validated as an absolute or local endpoint. Detector-unseen absolute ranking transfers, yet its preregistered gap gate fails under a quality-level shift. Local Delta-Q labels are continuous, bidirectional, and exactly identity-invariant, but the first differential learner reaches only `0.5747` pooled within-image pairwise accuracy against a locked `0.60` gate. This preserves a weak sign/MAE signal while freezing the learner and all downstream action claims.

## Experiment Matrix

| Run | Changed variable | Train support | Full action rate | Delta AP50 | Delta AP75 | Control gate | Decision |
|---|---|---:|---:|---:|---:|---|---|
| C3b | balanced global loss | 12 action / 20 noop | 0.250 | +0.00955 | -0.00869 | fail | freeze same-cache loss tuning |
| D1 | proposal-set mean/max context | 12 / 20 | 0.281 | 0.00000 | 0.00000 | fail | freeze set-context |
| D2 | top-1 pairwise topology proxy | 12 / 20 | 0.281 | 0.00000 | 0.00000 | invalid audit | proxy only; not native claim |
| D2b | exact class-expanded native-NMS topology | 12 / 20 | 0.281 | -0.00015 | -0.01589 | fail | freeze topology |
| D3 | fixed axes, 0.05 to 0.02 | 10 / 22 | 0.219 | 0.00000 | 0.00000 | fail | freeze fixed-grid actions |
| D4 | adaptive proposal-graph consensus bbox action | 1 / 31 | not trained | n/a | n/a | label support fail | freeze all bbox actions |
| E1 | post-NMS suppress/no-op | 17 / 15 | 1.000 | -0.11981 | -0.10096 | pass | freeze: detector/safety/abstention fail |

All metric rows are 32-image smoke results and are not full-val AP claims. D2 is retained only as an implementation diagnostic because review found it was not a true native topology test; D2b is the corrected experiment.

## Endpoint Audit

Artifact: `runs/autonomous_action_lattice_summary.json`, SHA256 `30a764e7f06d24136f8ac1908d20d68e5a9fab23a2a8af5facefe63a96f46d8c`.

| Cache | Candidates | Positive candidates | Positive images | Dominant structure |
|---|---:|---:|---:|---|
| C3 0.05 fixed bbox | 7824 | 53 (0.677%) | 12/32 | 7680 values equal `-0.023125` |
| D3 0.02 fixed bbox | 7824 | 23 (0.294%) | 10/32 | 7778 values equal `-0.0205` |
| D4 graph consensus | 803 | 1 (0.125%) | 1/32 | continuous action cost, no set benefit |
| E1 post-NMS suppress | 276 | 153 (55.4%) | 17/32 | only four values: `-1.02`, `-0.92`, `0.23`, `0.33` |

This reconciles the apparent contradiction between bbox inaction and suppression over-action. Bbox supervision is nearly flat; suppression supervision is coarse and action-heavy. Neither resembles a smooth low-energy transport field.

## Engineering Corrections

- Strict zero-action native parity passes with zero mismatched images and zero box/score error.
- D2b reproduces torchvision class expansion, thresholding, small-box removal, class-aware NMS, and top-k.
- Candidate index 0 reuses the native baseline trace exactly; `apply_box_delta(0)` is not used because it introduces coordinate drift around `1.5e-5`.
- Topology shuffle remains shuffled in both training and validation.
- All maintained runs use detector-only candidates; train GT is utility-only after candidate fixation.
- Cache manifests lock config, commit, clean state, detector/annotation hashes, and split manifests.
- GPU work used physical GPU2 only and always passed the estimated-peak plus 8192 MiB reserve rule. No pre-existing process was stopped.

## DeepSeek Review

DeepSeek agreed with each final branch freeze and with stopping score modulation on the same endpoint. Its strongest useful point is that D4's candidate-support failure is structurally earlier than a learning failure, while E1 proves that even a deterministic post-NMS action cannot learn safe abstention from this target.

Codex rejected several reviewer misreads:

- D2b controls selected the same number of actions as full; they did not collapse to no-op.
- The C1 table contains only single-axis magnitude 0.05, not mixed 0.05/0.10 actions.
- D3 feature shuffle changed one prediction, so 0.02 actions were not universally bitwise inert.
- Historical `/home/ps/lzz/RLimage` guidance is superseded by the explicit current manifold-only workspace contract.

## Frozen Claims

- No AP gain is established for action-local transport.
- No current evidence supports prototype attraction, native topology, fixed-grid bbox motion, graph-consensus bbox motion, or post-NMS suppression as a deployable correction.
- Offline or train-cache utility, ranking, geometry, or loss convergence must not be described as detector improvement.
- Seeds 2024/999, larger train, and full validation are not justified for these failed smoke branches.

## Next Legitimate Direction

Future work must change the endpoint before changing the policy again. A candidate endpoint should be a dense global set energy with:

1. continuous localization quality around IoU 0.75 rather than a hard TP count;
2. calibrated class-confidence quality;
3. pairwise duplicate/exclusion energy;
4. coverage preservation so removing a unique detection is expensive;
5. an explicit identity basin and train-only abstention controls;
6. native full-set evaluation after the endpoint passes feature/utility shuffles on a fresh nested split.

Until that endpoint is mathematically specified and preregistered, the correct action is to stop, not to add another head, action table, threshold, or seed.

The endpoint draft is now recorded in `docs/dense_set_energy_endpoint_spec.md` and has received a DeepSeek read-only critique. Calibration was changed from an undefined term to soft Brier quality; background and wrong-class risks were separated; identity controls now require strict native-output equivalence. This is a design artifact only. No positive endpoint or detector claim follows from it.

The next endpoint's train-only validation boundary is also locked. The 454 NWPU train images are split into 318 inner-fit, 68 inner-tune, and 68 outer train-heldout images using deterministic multilabel class and object-density stratification. All classes have at least three images in tune and outer. The manifest SHA256 is `ce19316aeaef1cbdf85f2c9668c5ae2c8443da8e848de2d22ed0f0f3a687e080`; detector validation remains untouched.

See `docs/autonomous_exploration_ledger.md` for commands, commits, gates, retries, hashes, and reviewer corrections.

## Dense Endpoint Progress

The endpoint work has now moved beyond design. Fixed native post-NMS teacher statistics on all 318 inner-fit images are finite and non-degenerate, with 2652 predictions and 1420 sparse duplicate edges. This is the first evidence in this exploration that the proposed target is not a hard, nearly flat Delta-U lattice.

That positive support did not authorize immediate fitting. A locked count-confounding audit failed: residualized penalty correlation was `0.8879` and support-normalized correlation was `0.8530`, both above the `0.85` limit. The five-term target was therefore reduced without fitting weights: coverage is normalized per GT object, background and class risks are merged and normalized per prediction, duplicate risk is normalized per edge, and calibration remains diagnostic-only.

A minimal absolute endpoint probe is now running on the train-only `318/68/68` nested split. It uses a normalized additive unary/pair DeepSets model and three complete controls. The outer train-heldout set is loaded only after hyperparameters and controls are frozen; detector validation remains untouched. Even a passing result will establish only absolute detector-only teacher learnability, not local Delta-Q, identity safety, actions, AP, or full validation gain.

The absolute probe has since completed with a scientifically mixed result. Full heldout quality prediction is strong (MAE `0.3489`, Pearson `0.9785`, pairwise `0.8889`, AUROC `0.9614`) and beats constants, teacher shuffle, and topology shuffle by large margins. However, it missed both `0.03` gains over the feature-alignment control, so the preregistered endpoint claim remains frozen.

A fit/tune-only ablation then showed why that control was too weak: it shuffled raw box geometry but retained conflict geometry. Removing all geometry from the frozen full model reduced pairwise by `0.0935` and increased MAE by `0.7172`; disabling the pair branch reduced pairwise by `0.0461` and increased MAE by `0.6805`. A fresh 254/64 resplit strong-control experiment is running inside the old inner-fit pool. It jointly breaks node geometry and pair topology and includes a score+class-only control. No old outer or detector validation is reused.

That strong-control resplit passed every gate. Full holdout pairwise was `0.8785` and MAE `0.4283`; it beat joint geometry/topology shuffle by `+0.0565` pairwise and `0.3684` MAE, and beat score+class-only by `+0.0655` and `0.5280`. This supports independent set-geometry information inside the researcher-adaptive train-only pool.

The final endpoint experiment now running is a separately locked detector-unseen validation. It trains fixed full/control models on all 454 train-cache images, then reads the 196-image NWPU detector validation set once. A pass would validate only absolute dense set-quality prediction. Local energy differences, identity safety, useful actions, and detector AP remain untested.

The detector-unseen validation did not pass every gate. Full ranking remained strong (pairwise `0.8093`, AUROC `0.9358`, Pearson `0.8846`) and beat all three controls by the required margins, but the train-to-validation pairwise gap was `0.1169`, above the locked `0.10`. The absolute endpoint is therefore frozen despite substantial transferable information.

A cache-only diagnosis found calibration shift rather than broad feature OOD. Validation coverage/GT fell `1.17` train standard deviations and error/prediction rose `1.38`, while duplicate energy stayed stable. The model's mean residual was `+1.707`; oracle mean-centering reduced MAE by `39.8%`, yet only `11.7%` of validation sets exceeded the train 95th-percentile feature distance. This is evidence to change the endpoint objective, not to calibrate on validation.

The local candidate objective was `Q_theta(S') - Q_theta(S)`. Its support audit passed after fixing exact permutation identity: among 1557 bounded perturbations, `94.2%` of labels were nonzero with median absolute magnitude `0.0266`. The subsequent minimal learner did not pass. On the 16-image train-only tune split it improved MAE over zero by `21.9%` and reached sign AUROC `0.6901`, but pooled within-image pairwise accuracy was `0.5747`, below the locked `0.60` gate.

A zero-training family audit reproduced that value and found true image-equal accuracy `0.5715`, so unequal pair counts were not responsible. Accuracy was `0.6429/0.4894/0.6246` across near-zero/medium/large target-margin bands, refuting the idea that tiny target differences alone caused the failure. The image-paired gain over strong geometry shuffle was only `+0.0250` with 95% CI `[-0.0126, 0.0665]`. The learner is therefore frozen rather than enlarged or retuned. The remaining evidence is limited to weak train-only differential predictability; it is not an endpoint validation, action policy, manifold correction, or AP result.

The decisive final control is a fit-only perturbation-family prior. A nine-value lookup table, estimated only on the 48 fit images, achieves tune pairwise `0.6724`, MAE `0.0772`, and sign AUROC `0.7084`; the neural endpoint reaches only `0.5747`, `0.0858`, and `0.6901`. The dominant `drop` family has mean Delta-Q `-0.478`, while score changes are near zero and geometric families occupy intermediate fixed ranges. Current predictability is therefore better explained by action-family identity than by learned image-conditioned set geometry.

This closes the present local learner branch. A legitimate future test must residualize the fit-only family prior and ask whether detector features predict `Delta-Q - mean_family` on newly locked train-only images. It must beat a zero-residual baseline and both feature/utility shuffles with image-paired uncertainty. Reusing the current 16-image tune split, adding family one-hot features, increasing endpoint capacity, or changing the failed `0.60` gate would be post-hoc optimization and is not authorized.
