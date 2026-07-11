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

- Time: 2026-07-12 03:25 Asia/Shanghai
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

- Time completed: 2026-07-12 03:31 Asia/Shanghai.
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
