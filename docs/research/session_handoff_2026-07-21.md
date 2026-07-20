# Manifold Detection Session Handoff

Last verified: 2026-07-21 (Asia/Shanghai)

This document is the single initialization entry for moving the current
research session to a new Codex window. It records the current objective,
scientific claim boundary, repository and remote state, completed evidence,
frozen routes, pending work, and exact startup checks.

## 1. Project Identity

The project is `manifold-detection`, not `RLimage` and not `manifoldRL`.

The maintained thesis is an **energy-guided ROI transport framework for object
detection**. The unknown class-conditioned endpoint is not assumed to be a
known prototype. The model is instead asked to learn bounded local actions
whose value is judged after detector-native score thresholding, box decoding,
matching, and NMS.

The long-term scientific question is:

> Can detector-only evidence identify a coordinated, low-energy ROI action or
> no-op decision that produces a causal improvement after native
> post-processing, while beating matched feature, label, topology, and random
> controls?

No current result supports an AP-gain claim for this mechanism.

## 2. Current Objective

### Immediate engineering objective

Close the completed `det.energy.re_roi_counterfactual_evidence.001` experiment
as immutable negative evidence:

1. add a reviewed report and artifact manifest;
2. update the research-line and experiment registries without rewriting old
   history;
3. regenerate derived research documentation;
4. run registry, focused, and full tests;
5. commit, push, and synchronize the remote research workspace.

### Immediate scientific objective

Freeze the present re-ROI formulation because its preregistered causal gates
failed. Do not tune capacity, action families, thresholds, seeds, or splits to
rescue it. Before another GPU experiment, define a genuinely new mechanism and
protocol that addresses the observed failure: post-action ROI evidence did not
add transferable information beyond static ROI features or shuffled bundles,
and its lower-confidence calibration was unusable.

### Claim boundary

The latest experiment is **NWPU train-only fit/tune/calibration evidence**. It
is not full validation, not detector-unseen confirmation, not an action policy,
and not an AP experiment. The locked `outer_heldout` and detector validation
were not read and must remain unread for this failed protocol.

## 3. Authoritative Workspaces

### Local development worktree

```text
E:\CLIproject\.worktrees\manifold-research-state-refactor
```

Branch:

```text
codex/repository-research-state-refactor
```

Verified implementation baseline:

```text
ce49b5747800e0a7b76e1f5be8e2b46f6500dbea
```

GitHub branch `origin/codex/repository-research-state-refactor` was verified at
that implementation commit on 2026-07-19. The published branch tip is later
because it also contains this docs-only handoff commit. In a new window, use
the latest branch tip and verify that `ce49b57` is its ancestor. The local
worktree was clean before publishing this handoff.

### Remote experiment workspace

```text
ps@122.51.19.136:/home/ps/lzz/manifold-detection-energy-transport
```

Verified remote HEAD:

```text
ce49b5747800e0a7b76e1f5be8e2b46f6500dbea
```

The remote worktree was clean on 2026-07-21. Its tracking branch reports
`ahead 2` only because the server-side GitHub remote-tracking ref is stale
after GitHub TLS failures; this is not an uncommitted change. The code was
safely fast-forwarded from a one-commit Git bundle.

### Forbidden workspace

Do not write project code or run this project's experiments in:

```text
/home/ps/lzz/RLimage
```

Some older `AGENTS.md` sections still mention that path for historical VOC
matrix work. For this research stream, the explicit workspace above is the
authoritative one.

## 4. GPU and Runtime Contract

- All training and GPU inference must use `CUDA_VISIBLE_DEVICES=2`.
- Before launch, GPU2 `memory.free` must be strictly greater than `8192 MiB`.
- Do not stop, restart, signal, or otherwise interfere with other processes.
- The presence of another process is not itself a reason to wait if the memory
  gate passes.
- Do not install packages on the remote server.
- Check for an existing run and process before every launch; never duplicate a
  formal experiment.
- Use `/home/ps/anaconda3/envs/RLimage/bin/python` with
  `PYTHONPATH=/home/ps/lzz/manifold-detection-energy-transport`.
- Distinguish `smoke`, `train_only`, `detector_unseen`, and `full_val` in every
  report.

## 5. Refactor State

The repository research-state refactor is complete through commit `7ed4798`.
The main maintained code lives under:

```text
spectral_detection_posttrain/core/
spectral_detection_posttrain/methods/energy_transport/
spectral_detection_posttrain/methods/rlvr/
spectral_detection_posttrain/methods/dpo/
spectral_detection_posttrain/methods/manifold/
spectral_detection_posttrain/signals/
spectral_detection_posttrain/trainers/
spectral_detection_posttrain/experiments/
```

Historical import shims remain intentionally. Old `round*.py` files are
lineage artifacts and should not be bulk-migrated.

Recent commits:

```text
ce49b57 feat: validate re-roi evidence protocol
e58806b fix: resolve cache builder device config
7dd4fd8 feat: add oracle-utility box-head experiment
7ed4798 docs: complete repository research-state refactor
cb7b5a6 docs: add guarded GPU2 execution contract
```

The re-ROI protocol implementation added or materially changed:

```text
scripts/experiments/re_roi_counterfactual/build_cache.py
scripts/experiments/re_roi_counterfactual/gates.py
scripts/experiments/re_roi_counterfactual/ranker.py
scripts/experiments/re_roi_counterfactual/run_protocol.py
spectral_detection_posttrain/trainers/detection/awr_boxhead.py
tests/test_re_roi_counterfactual_cache.py
tests/test_re_roi_counterfactual_gates.py
tests/test_re_roi_counterfactual_ranker.py
tests/test_re_roi_counterfactual_protocol_runner.py
tests/test_awr_boxhead_runner.py
```

Key implementation contracts:

- Arms B, C, and D use matched initialization, capacity, schedule, and
  image-family grouped rank batches.
- Identity is excluded from q statistics, regression targets, and primary
  metrics.
- Arm C uses post-action re-ROI evidence; arm D uses deterministic within-image
  bundle derangement.
- The runner validates cache image size, `h_pre`/`h_post` shape and finiteness,
  action alignment, drop semantics, split hash, cache provenance, and source
  commit.
- Calibration uses reconstructed total utility and a max-per-image one-sided
  finite-sample conformal error.
- Metrics include residual and reconstructed views, utility shuffle,
  family/static controls, no-op, matched-rate random, oracle top-1, and paired
  10,000-sample bootstrap intervals.

## 6. Verification Already Completed

Local verification before commit:

```text
focused expanded tests: 137 passed
full suite: 1659 passed, 3 skipped, 2 xfailed, 1 warning
repository-state validation: 1084 files checked, 0 violations
dependency-boundary check: clean
generated research docs: up to date
git diff --check: passed
```

Remote verification at `ce49b57`:

```text
68 passed in 14.59s
```

Remote focused command covered re-ROI cache, gates, ranker, protocol runner,
and AWR box-head runner tests.

## 7. Strong Baseline and Native Parity

Strong NWPU checkpoint:

```text
runs/nwpu_mob_strong_cosine_s42_bs8_36ep/checkpoint_best.pth
SHA256 de126708d063a82dd5c721799cd124fc4e52d28866139e103b1cd65427742027
```

Reviewed strong-baseline metrics:

```text
best epoch 33
best AP50 0.6721873811
best AP75 0.3177292383
final AP50 0.6608888572
final AP75 0.3083840534
precision 0.5658796649
recall 0.7137367915
FPR 0.4341203351
ECE 0.1122461377
predictions 1313
GT 1041
scope full_val, 196 images
```

Strict native zero-action parity was previously repaired and passed for both
baseline and full-finetune checkpoints with zero mismatched images and exact
zero actions. This validates the identity path, not action gain.

## 8. Frozen and Retained Research Lines

### Prototype attraction

Status: diagnostic only.

Class prototypes, Sinkhorn assignment, ETF/PAH endpoints, and low-dimensional
geometry did not establish a known correct endpoint or causal detector gain.
The components remain useful for diagnostics, not as the active objective.

### Intrinsic dimension

Status: diagnostic only.

The `256 x 7 x 7` ROI maps can have lower estimated intrinsic dimension than
the flattened `1 x 1024` box-head representation because spatial/channel
correlation, sample-size limits, estimator scale, and nonlinear mixing change
the measured geometry. Lower intrinsic dimension alone did not predict AP or
action utility and must not be optimized as a standalone target.

### Dual intra-class/inter-class energy

Status: diagnostic only.

The loss is mathematically non-degenerate on synthetic data, but real NWPU
candidate-level information did not beat detector score or shuffle controls.

### External FFT reward

Status: invalidated as a causal reward.

Real and shuffled FFT rewards were repeatedly inseparable. FFT/raw-iFFT signals
remain offline diagnostics and controls.

### RLVR, GRPO, and DPO

Status: historical training mechanisms.

They did not independently beat matched ordinary fine-tuning. Their useful
legacy is the signed objective, KL anchor, valid-pair masks, rescue budget, and
false-positive constraints.

### AFM / FPN spectral manifold

Status: comparison architecture baseline.

There is positive evidence on selected backbone/dataset combinations, but it
is unstable and is not the active manifold transport thesis. Do not restart it
without an explicit new protocol.

### Native action C1-E1 line

Status: frozen.

The C3b, D1, D2b, D3, D4, and E1 branches failed their locked structural or
control gates. Do not repackage them as a new line.

### Dense absolute endpoint

Status: frozen.

The detector-unseen endpoint beat controls but missed the preregistered
train-to-validation pairwise gap gate. It supports no action or AP claim.

### Local Delta-Q endpoint

Status: frozen.

The local learner missed its locked pairwise gate and was beaten by a fit-only
perturbation-family prior. Do not enlarge or retune it.

## 9. Offline Oracle-Pool Audit Sequence

These experiments reused train-cache candidate pools and were diagnostics, not
detector-native AP tests.

### A. Linear identifiability

Commit `bcd527e`. Failed because heldout pairwise was
`0.47619 / 0.49310`, while feature shuffle was `0.53172 / 0.51860`.

### B1. Step-0.05 stable strata

Commit `899910d`. The pool contained 1,243 candidates over 321 images, but no
fit+tune stratum had both positive raw LCB and positive same-image
uniform-control gain. No restricted 0.05 probe was authorized.

### B2. Spatial counterfactual features

Commit `be5096c`. Structured pairwise was `0.43721 / 0.45257`, below the
delta-alignment shuffle at `0.47330 / 0.48893`; frozen LCB was `-0.00486`.

### B3. Sign, rank, and abstention decomposition

Commit `39dc4e5`. Aggregate rank information was strong, but candidate-local
sign gain was negligible, rank argmax selected a negative mean utility, and
the frozen abstention support gate missed its minimum count.

### B3a. Top-focused audit

Commit `2a58ead`. Median-sign AUROC was worse than controls. Oracle-top-vs-rest
ranking remained an exploratory signal, but cross-boundary rank and locked
control gain failed. The positive always-top1 observation is reused-outer and
cannot be promoted.

### B4. Direct listwise plus no-op

Commit `7c8bd1a`. The full arm selected 14/71 images with mean Delta-U
`0.08250` and LCB `0.01587`, but its paired gain over feature shuffle had LCB
`-0.01351`. The offline oracle-pool action-learning route is frozen.

## 10. AWR-Weighted Box-Head Branch

Protocol version:

```text
det.energy.oracle_utility_boxhead.001
```

This is supervised weighted empirical risk minimization, despite the historical
`AWR` label. The Z preflight stopped before detector evaluation or training:

```text
positive images 62 / 250 = 0.248, passed
positive candidates 93, required at least 500, failed
ESS 197.0039, healthy
weight saturation 0, healthy
normalized mean weight 1, healthy
```

Interpretation: this is a valid scientific support-gate failure, not a weight
normalization bug. Arms U/W/F/S must not run, and the threshold must not be
relaxed post hoc.

## 11. Formal Re-ROI Counterfactual Evidence Experiment

Version:

```text
det.energy.re_roi_counterfactual_evidence.001
```

Run:

```text
/home/ps/lzz/manifold-detection-energy-transport/runs/
nwpu_re_roi_evidence_s42_strongbest_ce49b57_v1
```

Scope:

```text
nwpu_train_only_fit_tune_calibration
outer_heldout_read=false
detector_validation_read=false
seed=42
runner commit=ce49b5747800e0a7b76e1f5be8e2b46f6500dbea
```

Locked cache inputs:

```text
fit: 7fcb3774467aeae7b6592e20485ce93565946c1b96fc4cd7469ce63ddae288f7
tune: 6110a25c6f666c3d86e1ce4635ce8d66046c3483452d5541a273171234a07db7
calibration: 6421f83c2b6dd68e2862bfc8702586c7e042ebf01a7300d773c68f34bee13c8c
split manifest: 5c4222e033f7ed333835eab7f7786498b84bba11856c2d362ec8895845942244
```

Support:

```text
fit: 5,787 non-identity rows, 250 images
tune: 1,557 non-identity rows, 68 images
calibration: 1,548 non-identity rows, 68 images
all nine non-identity families cover every tune image
```

Artifact hashes:

```text
eval_metrics.json 4ac0da6a065bc58486006326abbcf32fbe25ca4df0152d76ce075a9511affd85
manifest.json d011653bc9997b2d0f5a0d206229417cf1c76c6a8bc5baa0b3db46f85d154591
arm_B_final.pt 9d6985798bcbfcbb54fc7ad092679097c8a6abb881e0bdaa2a23f437b7d1df47
arm_C_final.pt 68fc54c68e2a56982da49c8fd8789c2e58c7aa385ceaafa73c4e8c8ceed3ac2f
arm_D_final.pt f2880626ec7595a8c0aba7a71056a3f2d590d50f5f50ce22577e1d327722373e
```

### Locked gate outcome

| Gate | Result | Evidence |
|---|---:|---|
| support | pass | tune/calibration counts and nine-family support pass |
| identity | pass | teacher and model identity max error `0.0` |
| re_roi_gain | fail | C residual pairwise `0.53822`, B `0.57820`, delta `-0.03999`, LCB `-0.08438` |
| bundle_integrity | fail | C `0.53822`, D `0.52583`, delta `0.01238`, LCB `-0.02924` |
| static_baseline | pass | C beats zero and reconstructed relative MAE gain is `0.28024` |
| calibration | fail | positive rate `0.02941`, required `>=0.9`; coverage `1.0`; correction `268.4032` |
| generalization | fail | pairwise gap is small, but MAE-gain fit/tune gap is about `0.27284`, above `0.15` |

`all_gates_passed=false`.

### Main fit/tune metrics

Residual pairwise accuracy:

```text
fit:  B 0.59005, C 0.54843, D 0.56672
tune: B 0.57820, C 0.53822, D 0.52583
```

Reconstructed total pairwise accuracy:

```text
fit:  A 0.59931, B 0.71505, C 0.73349, D 0.69330
tune: A 0.59474, B 0.71066, C 0.71969, D 0.67922
```

Tune reconstructed relative MAE gain:

```text
B 0.29557
C 0.28024
D 0.29004
```

The reconstructed metric remains decent because it includes the family prior.
The primary residual and matched-arm gates show that post-action re-ROI evidence
does not provide the required additional transferable signal.

### Controls

```text
matched-rate random: 17 selections, action rate 0.25,
  mean raw utility -0.03534, selected mean -0.14137
oracle top1: 17 selections, action rate 0.25,
  mean raw utility 0.13565, selected mean 0.54259
C vs B paired interval: point -0.04145, [-0.08438, 0.00145]
C vs D paired interval: point 0.00976, [-0.02924, 0.04837]
C vs utility shuffle: point 0.06599, [0.01204, 0.11942]
```

The C-vs-utility-shuffle result only shows that the labels contain structure.
It does not rescue the causal re-ROI claim because C fails against the matched
static and bundle-shuffle controls.

### Scientific decision

Freeze this exact re-ROI evidence mechanism. The pipeline is operational and
non-degenerate, but the causal increment attributed to post-action re-pooling
is absent under the locked controls. Do not read outer heldout, expand seeds,
increase model capacity, add actions, or alter gates for this protocol.

## 12. What Is Still Pending

1. Create a reviewed report under `docs/reports/` for the AWR support-gate
   failure and formal re-ROI result.
2. Add immutable artifact registry entries with the hashes above.
3. Add or update research-line and experiment registry records. Keep the AWR
   version config's preregistered state intact; record the blocked run in the
   registry rather than rewriting the protocol.
4. Regenerate `docs/research/current_status.md`, `experiment_index.md`, and
   `artifact_index.md` through the repository generator.
5. Run focused registry tests, generated-doc checks, the full suite, and
   `git diff --check`.
6. Perform one bounded DeepSeek v4 Pro read-only evidence review of the final
   completed experiment group, then critically accept or reject its points.
7. Commit, push the branch, and fast-forward the remote workspace. If server
   GitHub TLS fails, use a minimal Git bundle again.
8. Only after closure, design a new prospective mechanism. The registry lists
   no next GPU route at this handoff.

The previous Luna session ID is no longer available (`not_found` on
2026-07-21). A new window must create a fresh bounded Luna sidecar if log or
registry field extraction is needed.

## 13. Multi-Agent Routing Contract

Use the repository/user routing instructions, especially:

```text
C:\Users\青云志\AGENTS.md
E:\CLIproject\AGENTS.md
C:\Users\青云志\.codex\skills\ai-orchestrator-routing\SKILL.md
```

Routing:

- Main Codex/Sol owns scientific decisions, leakage judgment, architecture,
  claim boundaries, final integration, and user-facing conclusions.
- Terra High handles bounded implementation, remote-runner diagnosis, and
  read-only code/protocol review.
- Luna High handles deterministic log/JSON extraction, manifest checks,
  generated docs, and low-risk test reporting.
- Thinking-enabled Kimi Code is the required substantive implementation or
  protocol sidecar when available. It was attempted once for the present
  implementation and timed out; Terra performed the allowed fallback review.
- DeepSeek v4 Pro is used only for the final scientific evidence review or a
  distinct unresolved evidence question. It is not the final authority.

Do not spawn multiple Sol agents to reread the same experiment. Delegate
bounded sidecars with exact files and acceptance criteria.

## 14. New Window Initialization Checklist

Run these steps in order:

1. Set the local working directory to:

   ```text
   E:\CLIproject\.worktrees\manifold-research-state-refactor
   ```

2. Read:

   ```text
   C:\Users\青云志\AGENTS.md
   E:\CLIproject\AGENTS.md
   E:\CLIproject\manifold\AGENTS.md
   C:\Users\青云志\.codex\skills\ai-orchestrator-routing\SKILL.md
   docs/research/session_handoff_2026-07-21.md
   docs/re_roi_counterfactual_evidence_protocol.md
   docs/awr_weighted_boxhead_protocol.md
   ```

3. Verify local state:

   ```powershell
   git rev-parse HEAD
   git status --short --branch
   git log -5 --oneline
   ```

4. Verify remote state read-only:

   ```powershell
   ssh ps@122.51.19.136 "git -C /home/ps/lzz/manifold-detection-energy-transport rev-parse HEAD; git -C /home/ps/lzz/manifold-detection-energy-transport status --short --branch; pgrep -af '[r]e_roi_counterfactual/run_protocol.py' || true"
   ```

5. Read the completed artifacts only from the run path in section 11. Do not
   read `outer_heldout` or detector validation.

6. Resume at section 12. Do not launch a new experiment merely because the GPU
   is free.

## 15. Copy-Paste Initialization Prompt

```text
You are continuing the manifold-detection research project, not RLimage and
not manifoldRL. Use the local worktree
E:\CLIproject\.worktrees\manifold-research-state-refactor and the only remote
workspace ps@122.51.19.136:/home/ps/lzz/manifold-detection-energy-transport.

First read C:\Users\青云志\AGENTS.md, E:\CLIproject\AGENTS.md,
E:\CLIproject\manifold\AGENTS.md,
C:\Users\青云志\.codex\skills\ai-orchestrator-routing\SKILL.md, and
docs/research/session_handoff_2026-07-21.md. Verify local and remote HEAD and
clean status before editing.

Current goal: formally close and register the completed
det.energy.re_roi_counterfactual_evidence.001 experiment and the blocked
oracle-utility box-head preflight, regenerate research docs, run tests, obtain
one bounded DeepSeek v4 Pro read-only scientific review, commit/push/sync, and
only then frame a genuinely new prospective mechanism. The current re-ROI
formulation is frozen because re_roi_gain, bundle_integrity, calibration, and
generalization gates failed. Do not tune it, expand seeds/actions/capacity,
read outer_heldout or detector validation, restart C1-E1, return to GT-filtered
oracle pools, or claim AP gain.

Use multi-agent routing by task: main Codex for scientific decisions and final
integration, Terra High for bounded code/remote review, Luna High for
deterministic JSON/log/registry checks. All GPU work, if a future protocol is
explicitly authorized, must use only CUDA_VISIBLE_DEVICES=2 after confirming
GPU2 memory.free >8192 MiB. Never stop other processes and never write this
project in /home/ps/lzz/RLimage.
```

## 16. Definition of a Correct Handoff Completion

The migrated session is initialized correctly only when it can state all of
the following without guessing:

- project identity and unknown-endpoint energy-transport thesis;
- current local and remote workspaces and the `ce49b57` implementation baseline;
- strong baseline and native parity status;
- why AWR stopped before training;
- why formal re-ROI evidence failed despite static-baseline success;
- the exact no-outer/no-validation claim boundary;
- all frozen historical routes that must not be renamed as new work;
- pending registry/report/test/sync tasks;
- GPU2 and multi-agent routing constraints.
