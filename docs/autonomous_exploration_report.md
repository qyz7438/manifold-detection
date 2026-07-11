# Autonomous Research Exploration Report

## Status

- Window: 2026-07-12 03:06 to 11:06 Asia/Shanghai.
- Current status: evidence synthesis; no further current-endpoint action experiment is authorized.
- Automation: `manifold-autonomous-8h-20260712`, two-hour heartbeat.
- Project: `/home/ps/lzz/manifold-detection-energy-transport`, branch `codex/energy-guided-roi-transport`.
- Dataset/scope: NWPU VHR-10, locked 32 train / 32 validation smoke, seed 42.
- Identity smoke reference: AP50 `0.7399366`, AP75 `0.4636476`, precision `0.572165`, recall `0.776224`, FPR `0.427835`, ECE `0.119736`, 194 predictions.

## Codex Judgment

The explored method line is negative. Detector-only candidates and native postprocessing are now implemented correctly, but the current whole-image Delta-U endpoint does not support a safe, attributable per-proposal action policy.

The main problem is not merely network capacity. The endpoint is an integer event-count utility:

```text
U = tp75 - 0.25 fp75 - 0.10 fp50 - action energy - action count cost
```

For bbox motion, almost every action leaves native output unchanged and receives only action cost. For post-NMS deletion, utility becomes a four-valued TP/FP lattice that encourages suppression without providing a transferable identity basin. The intended low-energy manifold flow is therefore not represented by the current supervision.

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

See `docs/autonomous_exploration_ledger.md` for commands, commits, gates, retries, hashes, and reviewer corrections.
