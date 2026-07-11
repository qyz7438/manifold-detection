# Autonomous Research Exploration Ledger

## Window

- Start: 2026-07-12 03:06 Asia/Shanghai
- Deadline: 2026-07-12 11:06 Asia/Shanghai
- Project root: `E:\CLIproject\manifold`
- Remote root: `ps@122.51.19.136:/home/ps/lzz/manifold-detection-energy-transport`
- Branch: `codex/energy-guided-roi-transport`
- Initial remote HEAD: `835503af4a9af2f37e098f1d1e1f95a9afa8bafd`
- Wake interval: two hours in the current thread
- Automation ID: `manifold-autonomous-8h-20260712`
- Local requirement: the computer, project storage, network, and Codex App must remain available.

## Operating Rules

- GPU work uses physical GPU2 only through `CUDA_VISIBLE_DEVICES=2`.
- Before every launch, estimate the new process peak and require `current_free_mib - estimated_peak_mib > 8192`.
- Check GPU2 immediately after launch and at steady load. If the reserve fails, stop only the newly launched process.
- Never stop or modify pre-existing processes. Do not write in `/home/ps/lzz/RLimage`. Do not install packages.
- Candidate generation must be detector-only. Train GT may define utility labels only after the candidate set is fixed.
- Smoke, reused-validation, and single-seed results never establish an AP improvement claim.
- Every direction requires identity parity, equal-capacity feature/utility controls, fixed split manifests, hashes, and a unique run directory.
- A failed preregistered direction is not revived without genuinely new evidence.

## Research Scope

### Mainline

Learn bounded detector actions when the class-conditioned endpoint is unknown. The maintained endpoint is native detector behavior after bbox decode, score thresholding, class-aware matching, and NMS. The active question is whether proposal-set context can map detector-visible evidence to a low-energy top-1/no-op action that improves localization without false-positive cost.

### Dataset And Protocol

- Dataset: NWPU VHR-10, seed/data seed 42.
- Strong detector: Faster R-CNN MobileNetV3 320 FPN, 480x480 input.
- Detector checkpoint SHA256: `de126708d063a82dd5c721799cd124fc4e52d28866139e103b1cd65427742027`.
- Annotation SHA256: `dde27d9362d9c6aa358a1cab160c9e075e57405d3fe876de049973c6e6150f0e`.
- Full split: 454 train / 196 validation.
- Locked smoke split: 32 train / 32 validation.
- Smoke train manifest: `37912bbb682f370f551c80aeae8e6dd600122fbec6a2450360ec2fdd54a81d69`.
- Smoke validation manifest: `bb6bb1764892bcc5b496c8ccbfca60c3d1e745a2901349429af3927a5770f85c`.
- Primary metric: AP75.
- Secondary metrics: AP50, precision, recall, FPR, ECE, prediction count.
- Diagnostic metrics: action/no-op support, selected count, action image rate, action family, training accuracy, parity errors, control deltas.
- Resource metrics: GPU2 free memory before/after launch and steady state.

### Reproducible References

- Full-val strong baseline reference: AP50 `0.660889`, AP75 `0.308384`. It is not compared numerically to smoke results.
- Locked 32-val identity: AP50 `0.7399366`, AP75 `0.4636476`, precision `0.572165`, recall `0.776224`, FPR `0.427835`, ECE `0.119736`, predictions `194`.
- C1 native contract artifact SHA256: `71022f08acf22b0c982929089d9799d0f5f19941cc31428d6ec7d9f795f53b44`.
- C3 global Delta-U cache SHA256: `1fa2184945bf5d38b1c47e5b732ffce4bcdd1b57064f4f2eb71d76611304a53e`.

## Stopped Directions

- A: linear local Delta-U identifiability. Failed ranking and shuffle-control gates.
- B1: step-0.05 strata. No stable train/tune stratum.
- B2: ROI spatial counterfactual descriptors. Worse than alignment shuffle.
- B3/B3a: decomposed sign/rank/abstention. Aggregate ranking did not produce safe top-1 actionability.
- B4: offline oracle-pool direct listwise/no-op. Failed feature-control attribution.
- C2/C2a: local-IoU nine-action policy. Move gate collapsed; forced action had no AP75 gain.
- C3/C3a/C3b: current single-ROI representation plus native C1 actions plus whole-image Delta-U. Balanced optimization became non-degenerate but AP75 fell `0.008688` and both controls beat full. No further same-cache loss tuning.

## Ranked Hypothesis Queue

1. **D1: proposal-set context representation.** Replace independent ROI scoring with a permutation-equivariant set-context encoder while keeping the C3 endpoint, action table, cache, balanced loss, and controls fixed. High information gain because C3b proved optimization and the action path work, while single-ROI generalization failed.
2. **D2: action-conditioned detector-native NMS topology.** For every proposal-action pair, encode post-action higher-score same-class IoU, NMS survival margin, and topology change versus identity. Add a topology-shuffle control. Do not alter utility, action table, split, or loss.
3. **D3: action-space change.** Only if D1/D2 establish set-level information but the fixed 0.05 translation/scale table remains limiting, compare a locked smaller localization action set against identity and controls.
4. **Stop condition.** If D1 fails non-degeneracy, AP75, or both controls under the locked C3 cache, freeze set-context on this cache and do not expand train size. If D1 passes every smoke gate, run one locked larger-train/full-val confirmation without tuning.

## Cycle 0 State

- Time: 2026-07-12 03:07 Asia/Shanghai
- Remote HEAD: `835503af4a9af2f37e098f1d1e1f95a9afa8bafd`
- Remote tracked worktree: clean.
- Related active PIDs: none.
- GPU2 free memory: 43376 MiB.
- Decision: start D1 immediately. Reuse the locked C3 cache only after SHA/provenance verification. No GPU cache regeneration is needed.
- DeepSeek review: C3b freeze accepted with Codex qualification. Causal action path was active; the failure was the AP50/AP75 tradeoff and control attribution, not zero detector influence.

## Experiment Log

### D1 Pending

- Hypothesis: whole-image Delta-U is not learnable from independent ROI features, but becomes learnable when every candidate receives detector-visible proposal-set context.
- Changed variable: representation only.
- Fixed variables: detector, split, seed, C1 actions, Delta-U cache, balanced loss, optimizer, epochs, controls, native postprocessing, and gates.
- Success: non-degenerate action rate, AP75 strictly above identity, AP50 non-negative, and AP75 strictly above both feature- and utility-shuffle controls.
- Failure: any success gate fails.
- Expected artifacts: version config, tests, commit, three policy checkpoints, `eval_metrics.json`, hashes, launcher log, DeepSeek read-only review.

### D1 Implementation Milestone

- Recorded in autonomous cycle 1; use the run artifact and git commits, rather than conversational timestamps, as the reproducibility clock.
- Config: `det.energy.set_context.d1.001`, SHA256 `441dfca02683b63d8408d81c3919dd970b4ca8718685b63cc89de363acedf357`.
- Representation: permutation-equivariant local proposal encoding plus observable-set mean/max context; global no-op remains image-level.
- Loss: unchanged C3b balanced actionability plus conditional hard-negative rank.
- Cache: locked C3 global Delta-U cache, no regeneration.
- Controls: local full, within-image ROI feature shuffle, within-image utility shuffle; same initialization seed.
- Focused verification: 15 tests passed, including proposal permutation equivariance and invariant global no-op.
- Next action: commit/sync, remote tests, GPU2 reserve gate, launch D1 smoke.
- Estimated D1 peak: 7168 MiB, based on the same detector/cache/evaluation path as C3b plus a small set-context encoder. Launcher requires `free_mib - 7168 > 8192`.
- First launch failed before model construction because the config SHA was transcribed as `f441...` instead of `441...`. No cache/checkpoint/result was created. Root cause is locked by a direct `load_config()` test; retry count 1/2.

### D1 Result

- Completed in autonomous cycle 1; the authoritative ordering is commit/artifact provenance below.
- Commits: `a560325`, `733793a`, `4816a81`.
- Command: `CUDA_VISIBLE_DEVICES=2 python scripts/train_nwpu_d1_set_context.py --run-dir runs/nwpu_d1_set_context_smoke_s42 --require-clean-git`.
- Launcher PID: `1606773` (completed).
- GPU2: 43372 MiB before launch, 34291 MiB immediately after/steady; estimated 7168 MiB peak reserve gate passed.
- Run: `runs/nwpu_d1_set_context_smoke_s42`.
- Artifact SHA256: `2ce9aee19783e5f0d5ae61024df03675f400e4a186d787420405c4c8a7fff722`.
- Checkpoints: local full `6dba66347f30893bd4f8333170e1e0216efd487bd70e7cc2fd3cfb0c00bac70f`; feature shuffle `271cc78baff1f8d1d4449979a5d988f1db36e786477feac14369c29da0b7c1e5`; utility shuffle `e7126ba91d24857183da215da26798a37fb179b9a5ffc8f63f6a347a75ebe073`.
- Label support: 12 action / 20 no-op images.
- Full diagnostics: selected 9/32, action rate 0.28125, all selected actions were candidate 2 (`-dx`).
- Identity/full metrics were exactly equal: AP50 `0.7399366`, AP75 `0.4636476`, precision `0.572165`, recall `0.776224`, FPR `0.427835`, ECE `0.119736`, predictions `194`.
- AP75 full minus feature shuffle: `0.0`; full minus utility shuffle: `+0.0000914`.
- Gates: native parity and non-degeneracy passed; detector and control gates failed.
- Decision: freeze D1 set mean/max context on the locked C3 cache. Do not expand train size.
- Interpretation: set context changed policy decisions but selected actions were postprocessing-inert on full/feature arms; the representation still collapsed to a global direction prior rather than proposal-conditioned localization.
- DeepSeek review: agreed that D1 must freeze and that the next experiment must target the representation-to-postprocessing interface.
- Codex qualification: DeepSeek proposed higher-score max-IoU distance and same-class count. D1 already receives max/mean/thresholded/higher-score same-class IoU statistics, so a static threshold-distance transform is not genuinely new evidence. D2 is refined to action-conditioned topology, which D1 cannot infer directly from its pre-action local conflict vector.
- Next action: implement D2 action-conditioned topology plus a topology-shuffle control. Keep the same C3 cache, balanced loss, action table, split, seed, and gates. If D2 fails, freeze set-context on this cache and move to D3 action-space change.

### D2 Implementation Milestone

- Time: 2026-07-12 03:27 Asia/Shanghai.
- Config: `det.energy.action_topology.d2.001`, SHA256 `2ef50b3152d0aac6349fb0c52e110332420f3419bbf275b4baf829355f3dffe6`.
- Changed variable: each detector proposal/action pair receives post-action maximum IoU with a higher-score same-class peer, NMS survival margin, topology change from identity, and higher-score peer count.
- Detector-only boundary: features use native predicted boxes, labels, scores, image size, the fixed C1 action table, and the detector NMS threshold. GT and Delta-U are used only by the locked training target.
- Control: `topology_shuffle` preserves the per-image topology-feature multiset but breaks proposal/action alignment during training. It has the same base encoder, topology head, initialization, optimizer, and epochs as `local_full`.
- Fixed variables: detector, 32/32 manifests, seed 42, C1 actions, C3 cache, whole-image Delta-U labels, balanced loss, optimizer, eight epochs, and native postprocessing.
- Gates: native parity; 3 or more selected actions and action rate in `[0.10, 0.90]`; AP75 strictly above identity with non-negative AP50 delta; AP75 strictly above topology- and utility-shuffle controls.
- Verification: 46 focused and regression tests passed; Python compilation and `git diff --check` passed.
- Estimated GPU peak: 7680 MiB. Launcher requires `free_mib - 7680 > 8192` and records immediate/steady reserve checks.
- Next action: independent Terra read-only review, commit/sync, remote tests, then launch the locked D2 smoke if GPU2 reserve passes.

### D2 Pairwise-Proxy Result And Audit

- Remote completion: 2026-07-12 03:33 Asia/Shanghai, commit `fd57cba347cdc5b92a4c581bc4fc66cf5e0ecce6`.
- Run: `runs/nwpu_d2_action_topology_smoke_s42`.
- Artifact SHA256: `cf7075d5b0764a4b8ad7e10bff7b6eca3a8af8b46def4dc3e95b6ec92a595f7d`.
- Checkpoints: local full `69d9c76c349e5a22b036a6f6bed19c5c9d9c333fe259ddbbd0cce3f52d7d371f`; topology shuffle `f8a826beb7133a2ec4d9b92097a68d2fb9213bd15af65127dd9279ef66cacbb8`; utility shuffle `836a43d4cbbe48dcea638cef00c3ca605d2c66edb1f73020f2ccdb1e8c5e317a`.
- GPU2: 43372 MiB free before launch and after completion. Estimated peak 7680 MiB; reserve gate passed. No other process was stopped or modified.
- Support: 12 action / 20 no-op images. Full selected 9/32 actions; topology and utility controls also selected 9/32.
- Identity, full, topology-shuffle, and utility-shuffle all produced AP50 `0.7399366`, AP75 `0.4636476`, precision `0.572165`, recall `0.776224`, FPR `0.427835`, ECE `0.119736`, and 194 predictions.
- Native zero-action parity passed with zero mismatched images and zero box/score error. Non-degeneracy passed; detector and control gates failed.
- Initial gate decision: no expansion and no positive topology claim.
- Terra post-run audit found two interpretation defects. First, the topology-shuffle arm received shuffled topology during training but true topology during evaluation. Second, the feature was a top-1-label pairwise overlap proxy, not torchvision-equivalent class-expanded greedy NMS topology. It omitted non-top-1 class boxes, score/small-box filtering, and the fact that a higher-score peer can itself be suppressed.
- Scientific correction: this run freezes only the pairwise top-1 topology proxy. It does not constitute a valid negative test of native NMS topology. No GT/Delta-U feature leakage was found.
- Next action: D2b must compute detector-only class-expanded native kept-set topology under the same 32 images, C1 actions, Delta-U endpoint, loss, and capacity. Its shuffle control must remain shuffled at train and evaluation. This is a correctness repair, not a capacity or offline-probe expansion. If D2b fails, move to D3.

### D2b Native Kept-Set Implementation Milestone

- Time: 2026-07-12 autonomous cycle 1.
- Config: `det.energy.native_topology.d2b.001`, SHA256 `1b07a3f3862cc49945b0a972a8ed96298c1924ced23638f0d09b7debb7836f92`.
- Native topology exactly follows the detector boundary: class-expanded decoded boxes, clipping, background removal, score threshold, small-box removal, class-aware `batched_nms`, and detections-per-image cap.
- Per proposal/action observables: acted candidate kept, identity kept, kept-state delta, max IoU to the actual final kept higher-score same-class set, NMS margin, max-IoU delta, normalized kept rank, and rank delta.
- Identity action index 0 is required to be exactly zero. Tests cover manual native-NMS kept parity, non-top-1 class competition, clipping/degenerate removal, and proposal permutation equivariance.
- The same locked C3 Delta-U records are copied unchanged. A detector-only enrichment pass on the same 32 train images recomputes only native topology and rejects any image-order, label, score, box, or logit misalignment above `1e-4`.
- The topology-shuffle arm remains shuffled during both training and validation with deterministic per-image seeds. Utility shuffle retains true topology and shuffles only locked utility labels.
- Architecture, balanced loss, eight epochs, action table, splits, detector, optimizer, and gates remain fixed from D2. No GT field other than image identity is read; candidate extraction explicitly uses `targets=None`.
- Verification at this milestone: 27 focused tests passed, direct config hash load passed, and Python compilation plus `git diff --check` passed.
- Estimated peak remains 7680 MiB because D2b enumerates native postprocessing serially and retains only one detector plus three small policy heads. Launcher requires more than 8192 MiB remaining after that estimate.
- Terra pre-launch review found that calling `apply_box_delta` with the zero vector is not bitwise identity (observed max coordinate drift `1.5258789e-05`). Candidate index 0 now directly reuses the baseline trace; a mock-guarded regression test proves no box transform is called. The launcher also validates existing artifacts against current HEAD, clean status, source-cache hash, config hash, and scope before idempotent exit.
- Post-fix focused verification: 19 tests passed. Terra otherwise confirmed torchvision operation ordering, no GT candidate leakage, persistent train/eval topology shuffle, and the GPU2 reserve gate.
