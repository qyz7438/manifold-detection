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
- First remote launch failed before writing cache/checkpoint/result because the D2b alignment threshold was looked up in the inherited C3 gate namespace. The threshold is now an explicit function argument sourced from the locked D2b config; retry count is 1/2.

### D2b Native Kept-Set Result

- Completed on retry 1/2 at commit `6a3634d2ebce010ed3dad9b7d13f66e5087586cc`.
- Run: `runs/nwpu_d2b_native_topology_smoke_s42`.
- Artifact SHA256: `c35e7e20492b3a39a985f2e8b09a77b0eed853bbacd60c3f9f929d2984989bf0`.
- Native topology cache SHA256: `629eda04bcbaf5e6f41bbb9a9b683a7e9bec1631af4989adf9e5b47382f81ca8`.
- Checkpoints: full `c6680a7346e1cc3100a035d975659af78fa20475e6e69fdaea40da62430b27ea`; topology shuffle `0fc0313e2facd3ad776ffa895af3ce0bb7e61fe253e6fbf873db9d73fa3aa7d5`; utility shuffle `8b84436c4abf1107f0a698e843f0ec00dd770743fb6dbb2e18fb121ac8e4071d`.
- Cache alignment: all 32 images aligned with max absolute detector/cache error `0.0`; no GT used for topology; locked Delta-U copied unchanged.
- Support: 12 action / 20 no-op images. All arms selected 9/32 actions. Full chose candidate 3 on all nine acted images; topology shuffle chose candidates 2/3; utility shuffle chose 2/3/4.
- Identity: AP50 `0.7399366`, AP75 `0.4636476`, precision `0.572165`, recall `0.776224`, FPR `0.427835`, ECE `0.119736`, predictions `194`.
- Full: AP50 `0.7397871` (`-0.0001495`), AP75 `0.4477563` (`-0.0158913`), precision `0.569231`, recall `0.776224`, FPR `0.430769`, ECE `0.118883`, predictions `195`.
- Topology shuffle AP75 `0.4635562`; utility shuffle AP75 `0.4636476`. Full minus controls: `-0.0157998` and `-0.0158913`.
- Gates: cache alignment, native parity, and non-degeneracy passed; detector and control gates failed.
- Decision: freeze native kept-set topology on the locked 0.05 C1 action endpoint. Do not expand train size, seeds, topology capacity, or topology features.
- DeepSeek read-only review agreed this is a strong bounded negative and recommended D3. Codex rejects two review details: controls did not collapse to no-op (both selected exactly 9 actions), and the C1 table contains only single-axis magnitude `0.05`, not `0.05/0.10`. Its broader conclusion remains valid because full harmed AP75 and lost to both equal-budget controls.
- Next action: D3 changes only the action magnitude from `0.05` to a locked finer `0.02`, recomputes native whole-image Delta-U on the same 32 detector-only train images, and otherwise restores the C3b balanced policy/loss/controls. If D3 fails, stop this action-table branch rather than tune step size repeatedly.

### D3 Fine-Action Implementation Milestone

- Config: `det.energy.fine_action.d3.001`, SHA256 `0d76a612045dfb2be24c131a6672566f0dfb7a19c5d1c7c46b3dd21112425ea9`.
- Changed variable: the eight single-axis translation/scale actions use magnitude `0.02` instead of `0.05`; identity remains exact index 0 and the one-action-per-image budget is unchanged.
- Whole-image Delta-U is recomputed through native decode, threshold, class-aware matching, and NMS on the same locked 32 detector-only train images. GT is used only after candidate fixation to score train utility.
- Fixed from C3b: independent ROI policy architecture, balanced-margin objective, eight epochs, optimizer, feature/utility controls, initialization, detector, train/validation manifests, and strict detector/control gates.
- Provenance requires the frozen C3b artifact and corrected frozen D2b artifact. Cache metadata locks commit, detector/annotation hashes, train manifest, candidate builder, and `candidate_step=0.02`.
- Decision is single-shot: any label-support, parity, non-degeneracy, detector, or control failure freezes the fixed-grid action-table branch. No additional step sweep or same-cache tuning.
- Focused verification: 18 tests pass; config SHA loads directly; Python compilation and `git diff --check` pass.
- Estimated GPU peak: 7680 MiB with strict `free - peak > 8192 MiB` launcher gate.
- Luna's quick review raised the historical `AGENTS.md` RLimage path as a blocker. Codex rejects that finding: the user's repeated current instruction and active autonomous contract explicitly require `/home/ps/lzz/manifold-detection-energy-transport` and forbid writing `/home/ps/lzz/RLimage`. The launcher path is therefore correct.

### D3 Fine-Action Result

- Commit: `c6d4860d3edca247f09bd62547dcfb079cc6742e`.
- Run: `runs/nwpu_d3_fine_action_smoke_s42`.
- Artifact SHA256: `dd4d00d2a37e168cecc26482eb715e12fbd846bd71856c3ff323a54e086d1a11`.
- Recomputed 0.02 Delta-U cache SHA256: `92fe5403a6e6ead3cf7240e1df5e1979ab5bb819ab33eb1f3e759c3c866b0155`.
- Checkpoints: full `0e5fdb2f17140cfa054ba6000f56fb2d320f597eb8f01cbe46a12d505d56ba86`; feature shuffle `d9e17b1a20a4349fa75fcf98287a81c5e60a436aeaed73346a0a48a7a9155326`; utility shuffle `14f6a7bb7feccc2b9a42b3a6a4d911862867bf11e24c59d8e9019bf93619fe39`.
- GPU2: 43372 MiB free before launch, 42605 MiB during cache construction, and 43372 MiB after completion. Reserve gate passed; no other process was stopped.
- Support: 10 action / 22 no-op images. Full selected 7/32 actions, feature shuffle 7/32, utility shuffle 6/32. Full action rate `0.21875`.
- Full training loss fell `1.0396 -> 0.2073` and accuracy reached `0.9375`; optimization was not degenerate.
- Identity/full metrics were exactly equal: AP50 `0.7399366`, AP75 `0.4636476`, precision `0.572165`, recall `0.776224`, FPR `0.427835`, ECE `0.119736`, predictions `194`.
- Feature shuffle: AP50 `0.7404088`, AP75 `0.4637405`, predictions `193`; utility shuffle was exactly identity. Full AP75 minus controls: `-0.0000929` and `0.0`.
- Gates: label support, native parity, and non-degeneracy passed; detector and control gates failed.
- Decision: freeze the fixed-grid single-axis action branch. No additional step sweep, seed expansion, or same-cache tuning.
- DeepSeek agreed with the freeze and proposed one final genuinely different adaptive-consensus action test. Codex rejects its statement that 0.02 actions changed no output at all: the feature-shuffle arm changed prediction count and metrics, while full happened to be metric-inert. The more defensible conclusion is that learned full actions did not create attributable validation benefit.
- Next hypothesis D4: replace the global fixed action table with one deterministic detector-only graph-consensus delta per proposal, bounded at max absolute `0.05`; learn only top-1/no-op selection under native whole-image Delta-U. This tests an adaptive manifold-flow action rather than another magnitude or representation sweep. One smoke failure freezes the bbox-adjustment action line.

### D4 Adaptive Consensus Implementation Milestone

- Config: `det.energy.adaptive_consensus.d4.001`, SHA256 `17e9d28770f5fab9fe7fe067d20c559c7c4d11c91251472414ef2ffb0de23642`.
- Candidate generator is detector-only and permutation-equivariant. For each observable proposal, it forms edges to strictly higher-score, same-predicted-class peers with IoU at least `0.30`; the score-times-IoU weighted peer box is the local manifold anchor.
- Each proposal receives at most one standard center/scale delta toward that anchor, clipped to max absolute `0.05`. Proposals without a valid peer have no action candidate. The image budget remains top-1 action or explicit no-op.
- A small adaptive-consensus policy wraps the C3 global policy, encodes the actual proposal-specific delta, applies its true energy penalty, and masks unavailable actions. It does not regress a new action vector.
- Train Delta-U is newly enumerated through native whole-image postprocessing on the same 32 detector-only train images. GT remains utility-only after the graph candidate set is fixed.
- Controls preserve adaptive deltas: feature shuffle breaks ROI/action alignment, utility shuffle breaks Delta-U/action alignment. Train and validation recompute the same graph rule and restrict the policy observable set to proposals with nonzero adaptive actions.
- Hard stop: any support, parity, action-rate, detector, or control gate failure permanently freezes bbox adjustment. No consensus threshold, weight, magnitude, representation, or seed tuning.
- Focused verification: 23 tests pass, including graph motion, exclusions, bound/permutation equivariance, adaptive selection alignment, generic train/eval integration, config hash, and launcher contract. Compilation and `git diff --check` pass.
- Terra found no GT candidate leakage and accepted the graph bound, but noted that cache provenance did not distinguish a dirty run at the same commit. `git_dirty` is now part of the cache manifest, so a clean launcher cannot reuse a dirty-generated cache. Terra again cited the historical RLimage path; Codex rejects that stale instruction in favor of the user's explicit current manifold-only workspace contract.

### D4 Adaptive Consensus Result

- Commit: `678c8f4cf6874ab91f923b5f951c1f8b2944433e`.
- Run: `runs/nwpu_d4_adaptive_consensus_smoke_s42`.
- Artifact SHA256: `6f437f7c9e05c718f80dda8ec5f9d1d4e230288f0d2f73207d8e452356c92223`.
- Cache SHA256: `e8bc77a4ab0b966a47c2705a1440998e753bd62e4cff84e39dd11b08ca44a84c`.
- Candidate support: 803 detector-only graph actions across all 32 train images; gate passed.
- Label support: only 1 action-positive image and 31 no-op images; gate failed. No policy was trained, no checkpoint was written, and validation was not read.
- GPU2 remained above reserve; run ended with 43342 MiB free. Remote tracked worktree remained clean.
- Decision: permanently freeze pre-NMS bbox-adjustment actions for this detector/checkpoint/endpoint. This includes fixed axes, finer steps, graph consensus, topology variants, and single-box spatial deltas. No threshold/magnitude/consensus/seed/capacity tuning.
- DeepSeek agreed the early stop is the most informative negative because the candidate set itself is utility-barren. Codex qualifies the strongest wording: D4 disproves this specified higher-score graph-consensus generator, while the combined D1-D4 evidence is what supports freezing the broader single-box bbox interface.
- Next structural pivot E1: leave native boxes and NMS untouched, then learn at most one post-NMS suppress/no-op action over the stable native kept set. This is a discrete global set-energy decision with deterministic consequences, not another bbox action or NMS-threshold adjustment.

### E1 Post-NMS Suppress Implementation Milestone

- Config: `det.energy.post_nms_suppress.e1.001`, SHA256 `72405a2fdaf19b1b5d9b3ae28b57e21c5c33343e177f8e181273ba4d53ef25bb`.
- Candidate set is the native detector's final kept set after decode, threshold, small-box removal, class-aware NMS, and top-k. GT never filters candidates.
- Action is deterministic: suppress exactly one kept detection or no-op. It changes no box, score, threshold, or NMS decision and has no cascade.
- Detector-only node features contain confidence, normalized box, log area/aspect, predicted-class one-hot code, and kept-set conflict statistics. A small permutation-equivariant set head scores one global suppression versus no-op.
- Train labels enumerate native whole-image Delta-U for dropping each kept detection on the same 32 train images. Controls separately shuffle node features or utility alignment within each image.
- Gates require candidate/label support, exact native no-op parity, bounded non-degenerate suppression rate, positive AP75 with non-negative AP50, non-increasing FPR, recall drop at most `0.005`, and AP75 strictly above both controls.
- Focused verification: seven E1 module/runner tests pass; empty kept sets, permutation equivariance, exact single removal, tie-to-noop, control isolation, config hash, compilation, and launcher provenance are covered.
- Estimated peak: 6144 MiB; launcher requires `free - 6144 > 8192 MiB` on physical GPU2.
- Broad maintained regression: 106 tests pass. Terra found no other P0/P1 and confirmed detector-only kept candidates, train-utility-only GT, exact suppress/no-op behavior, controls, loss/eval, and hash chain. Its repeated historical RLimage-path warning is rejected under the explicit current manifold-only workspace contract.
- First remote launch built the clean cache but failed before a completed training arm because the new runner logged balanced-loss accuracy using `row["correct"]` instead of the canonical `row["accuracy"]`. The metric key is now locked by a source regression test; retry count is 1/2.
