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

### E1 Post-NMS Suppress Result

- Completed on retry 1/2 at commit `35ece286a1d356f9624f0dc702ec9cde5c4ae408`.
- Run: `runs/nwpu_e1_post_nms_suppress_smoke_s42`.
- Artifact SHA256: `f2e1a65d2fe54f8e85db8db9fc7ab629654514b211f856b04480e3dabf2e7e46`.
- Cache SHA256: `62c2e6b37414289e925a4904659d74d8ef8ecfd169d75175a1d2b7f59eda27b0`.
- Checkpoints: full `1d97843427aefaa208632d8e5c57966aa303ba6575908d221eb8b2533abc19cf`; feature shuffle `652429b73a3b555c92949bf158ce59d008ab4e178741860dfafab2879f293f83`; utility shuffle `b0d3bcb71b77914bff3ffa967d761e0e94862ad41d7cce07c5c55a15dcfedc6b`.
- Candidate support: 276 native post-NMS detections across all 32 train images. Label support: 17 action / 15 no-op images.
- All three validation policies suppressed on 32/32 images; non-degeneracy failed because action rate was `1.0` versus the locked maximum `0.50`.
- Full: AP50 `0.620131` (`-0.119806`), AP75 `0.362689` (`-0.100959`), precision `0.574074`, recall `0.650350` (`-0.125874`), FPR `0.425926` (`-0.001909`), ECE `0.125600`, predictions `162` (`-32`).
- Feature shuffle AP75 `0.352333`; utility shuffle AP75 `0.332512`. Full beat controls by `+0.010356` and `+0.030177`, so the control gate passed despite catastrophic detector/safety failure.
- Full training loss only fell `1.21793 -> 1.15473`; accuracy ended `0.25`, versus `0.1875/0.15625` controls. This is weak suppress-which information with failed action/no-op separation, not a successful policy.
- Gates: candidate support, label support, native parity, and control attribution passed; non-degeneracy, detector, and safety gates failed.
- Decision: freeze post-NMS suppress-only transport. Do not calibrate thresholds, retune loss/capacity, expand seeds, or reinterpret the small FPR improvement as a detector gain.
- DeepSeek agreed the information is dominated by fatal abstention collapse and recommended stopping rather than opening score modulation on the same whole-image Delta-U endpoint. Codex agrees: the full arm's control advantage is real numerically but scientifically insufficient inside a universal-suppression regime.
- Research conclusion for this cycle: stop all per-proposal actions driven by the current whole-image Delta-U endpoint. Any future restart must change the endpoint or supervision semantics, not the action representation.

## Endpoint Lattice Audit

- Tool: `scripts/analyze_action_cache_lattice.py`; output `runs/autonomous_action_lattice_summary.json`, SHA256 `30a764e7f06d24136f8ac1908d20d68e5a9fab23a2a8af5facefe63a96f46d8c`.
- C3 0.05 fixed actions: 7824 candidates, only 53 positive (`0.6774%`) across 12 images. `7680/7824` actions share exactly `Delta-U=-0.023125`, meaning native output was unchanged and only action cost remained. There are only 11 rounded utility values.
- D3 0.02 fixed actions: 7824 candidates, only 23 positive (`0.2940%`) across 10 images. `7778/7824` share `Delta-U=-0.0205`; only seven rounded utility values.
- D4 graph consensus: 803 candidates, one positive (`0.1245%`) on one image. Continuous energy cost creates 751 rounded values, but almost all remain negative because the set outcome does not improve.
- E1 post-NMS suppression: 276 candidates, 153 positive (`55.43%`) across 17 images, but Delta-U has only four rounded values: `-1.02`, `-0.92`, `0.23`, and `0.33`. This is a coarse TP/FP count lattice, not a smooth low-energy field.
- Root cause synthesis: the current endpoint combines integer `tp75/fp75/fp50` counts with action penalties. For bbox actions it is nearly flat; for suppression it is highly discontinuous and over-rewards deleting any counted FP without supplying a transferable abstention geometry.
- Future endpoint requirement: replace hard count events with a dense set-quality energy that preserves candidate independence and safety. It should include continuous localization quality near IoU 0.75, calibrated class confidence, explicit duplicate/coverage terms, and an identity basin. Until such an endpoint is specified and train-only controls are preregistered, no additional action learner should run.

## Dense Endpoint Specification Review

- Draft: `docs/dense_set_energy_endpoint_spec.md`.
- DeepSeek correctly identified undefined calibration, overlap semantics, temperature normalization, architecture decomposition, and identity-control definitions.
- Accepted corrections: calibration is now a dense soft Brier term; background and wrong-class risks are separate; unary/pair contributions add exactly to global DeepSets energy; uncertainty comes from inner-tune residual quantiles; identity controls require strict native-output equality.
- Rejected reviewer suggestions: binned ECE is not a smooth per-set teacher; all-class IoU alone misses classification errors; `tau=1/std(log p)` is dimensionally arbitrary; MC dropout is not calibrated evidence; fixed two-pixel/0.02-confidence jitter is not identity-equivalent near detector boundaries.
- Implementation remains blocked until temperatures, robust standardization statistics, and edge sparsification are concretely versioned and hashed. The nested split is now locked below. No endpoint model has been trained.

## Dense Endpoint Nested Split

- Builder: `scripts/build_dense_endpoint_nested_manifest.py`.
- Locked manifest: `spectral_detection_posttrain/configs/splits/nwpu_dense_endpoint_s42_nested.json`, SHA256 `ce19316aeaef1cbdf85f2c9668c5ae2c8443da8e848de2d22ed0f0f3a687e080`.
- Source is exactly the existing 454-image NWPU train split, SHA256 `7abe3c8370985f49698dcc3c42ca5917f1e17941fa024643147a58479c7cd9bd`; scope is `full_train_only_never_detector_validation`.
- Inner-fit: 318 images / 1952 objects, hash `3314b7d59dccf8df83a11662883b822c874757f7dd6b62356cb91335b434e54f`.
- Inner-tune: 68 images / 457 objects, hash `48dc60dacdcfa3e532bc61ca87e08c2dad36ded90ff5dcc82b7a6764188174d0`.
- Outer train-heldout: 68 images / 471 objects, hash `4e1acfd60ffdc4a528adeb9fc942d4c1867ff9b2117e5e86089e8f84240fa771`.
- The first random-hash split was rejected because rare classes had only 2-3 outer images. The final split uses deterministic multilabel class-presence plus object-density stratification; all ten classes have at least three images in tune and outer, with mean objects/image `6.14/6.72/6.93`.
- Six tests lock source hash, manifest hash, disjointness, full coverage, class support, determinism, and density balancing. No GPU or detector validation was used.

## Dense Teacher Component Statistics

- Version: `det.energy.dense_teacher_stats.001`; implementation commit `2ddb0da792b624adf6df502d6f7f5742dc8b057b`.
- Scope: native post-NMS predictions on the 318-image `inner_fit` split only. `inner_tune`, outer train-heldout, and detector validation were not read.
- Artifact: `runs/nwpu_dense_teacher_stats_s42_innerfit/eval_metrics.json`, SHA256 `6fd77a0e70b8a2365f441a98fcc19ca3c6260a6727579185807c798e913d2d10`.
- Provenance: config `c8ff0e29a3d191b2aed0db2883830171c8f60ce379df1a44493342c5b9133496`; checkpoint `de126708...42027`; manifest `ce19316...e080`; clean Git; `CUDA_VISIBLE_DEVICES=2`.
- Support: 318/318 images have predictions; 2652 predictions, 1952 GT objects, 1420 duplicate edges on 152 images. All five components are finite and non-degenerate.
- IQRs: coverage `5.19767`; background `0.85394`; class `0.39578`; duplicate `0.19703`; calibration `0.25434`.
- Important diagnostic: background/class/calibration raw correlations are `0.94028/0.95102/0.88989`. Component support passed, but learnability was not claimed.
- GPU2 free memory: `42751 MiB` before launch, about `42533 MiB` under detector load, `43340 MiB` after completion. The reserve gate passed and no other process was stopped.
- DeepSeek agreed that support justifies only a confounding audit, not model fitting; it correctly identified prediction/GT count as a possible common scale factor. Its proposed SciPy implementation was rejected because the project forbids package installation and the same residualization is available in PyTorch.

## Dense Teacher Confounding Result

- Version and commit: `det.energy.dense_teacher_confound.001`, `8a379308...`; source-artifact-only, no images or new split read.
- Artifact SHA256: `50da6f35fff24de3cdce7be34c5194d9a9a9aaf56e2b06c12903b95e424de181`.
- Count-only `R^2`: background `0.60060`, class `0.69997`, calibration `0.54877`.
- After residualizing `log1p(prediction_count)` and `log1p(ground_truth_count)`, maximum penalty correlation remained `0.88793`, above the locked `0.85` gate.
- Natural-support normalization remained non-degenerate but maximum penalty correlation was `0.85300`, also above the gate.
- Decision: reject the five-term unit-weight target. Do not relax the threshold or fit lambdas. Calibration becomes diagnostic-only; background and class risks are combined into total error; the first model target uses normalized coverage, total error, and duplicate coordinates.

## Reduced Absolute Endpoint Launch

- Version: `det.energy.dense_endpoint.absolute.001`; implementation commit `80abed6d...`, provenance correction `52c82685...`.
- Model: 22-D native post-NMS node features, 5-D same-class sparse pair features, 32-D mean-additive unary/pair DeepSets. No ROI action or detector parameter is trained.
- Teacher: inner-fit median/IQR standardization of `coverage/GT`, `(background+class)/prediction`, and `duplicate/edge`; calibration remains diagnostic-only.
- Protocol: fit `318`, tune `68`, outer train-heldout `68`; detector validation forbidden. Weight decay is selected on tune for full only, then frozen for feature-alignment, teacher-shuffle, and genuine box-to-node topology-shuffle controls. Outer cache is built once after all models are frozen.
- Tests: 114 maintained energy-transport tests pass; synthetic two-epoch train/predict path passes.
- First launch stopped before GPU/data work because the preregistered confounding SHA string was accidentally 62 characters. The artifact itself and all scientific settings were unchanged; commit `52c82685...` corrects only the source hash and adds a 64-character regression assertion.
- Retry started on physical GPU2 with `43308 MiB` free and estimated peak `6144 MiB`; steady free memory is about `42533 MiB`. Current PID `3155362`; log `runs/nwpu_dense_endpoint_absolute_s42_nested/launcher.log`.

## Reduced Absolute Endpoint Result

- Completed at clean commit `52c826859e5edaecbf7f6121fff7bc2feb6816ee`; artifact `runs/nwpu_dense_endpoint_absolute_s42_nested/eval_metrics.json`, SHA256 `d0e4afb68bd9c6f7636672e501c3638400efe82c91ec24ef5c68740b30043c2f`.
- Cache provenance: fit+tune `42d72c4...a1d2c`; one-time outer `241ec83...47fa`. `outer_read_count=1`; detector validation was not read. Selected weight decay was `0.001` using tune only.
- Full outer: MAE `0.34888`, Pearson `0.97847`, pairwise `0.88894`, positive AUROC `0.96140`; best-constant MAE improvement `79.26%`; fit-to-outer pairwise gap `0.04049`.
- Full beat teacher shuffle by `+0.30158` pairwise / `+1.24496` MAE and topology shuffle by `+0.04697` / `+0.35801`.
- Full beat feature-alignment shuffle by only `+0.01932` pairwise / `+0.01632` MAE, below both locked `0.03` attribution gates. All absolute/gap/baseline gates passed; both control-attribution gates failed.
- Decision: `reduced_absolute_endpoint_frozen`. The absolute teacher signal is strongly learnable, but the preregistered experiment did not attribute enough net value to the full feature alignment. No local Delta-Q, identity, action, AP, or detector-validation claim is restored.
- DeepSeek agreed the numbers and read-once order are internally consistent. It correctly found that the feature control shuffled columns 1-6 but retained conflict geometry columns 18-21. Codex rejects calling those columns teacher leakage: they are detector-only observables, but their survival makes the control too narrow for a global geometry claim.

## Fit/Tune-Only Endpoint Attribution

- Version/commit: `det.energy.dense_endpoint.attribution.001`, `5f8d1f49...`; artifact SHA256 `37ea25835ad2bef2e25aebcb6b0ce91e6c4d59085ee4047581c0bffb7ecfcbbf`.
- Scope: frozen full checkpoint plus fit/tune cache only; no training, new data, outer cache, or outer aggregate used for the decision.
- Tune baseline: MAE `0.30081`, pairwise `0.88718`.
- Zero all box/conflict geometry: MAE `1.01798` (`+0.71717`), pairwise `0.79368` (`-0.09350`).
- Disable pair branch: MAE `0.98129` (`+0.68048`), pairwise `0.84109` (`-0.04609`).
- Conflict-only removals were also material: zeroing conflict geometry raised MAE to `0.77644`; raw node conflict columns 18 and 21 correlate with fit target at `-0.66258/-0.65058`.
- All preregistered diagnostic gates passed. Interpretation: the original feature control preserved major geometry channels, so a fresh strong-control design is scientifically warranted. This remains researcher-adaptive diagnosis, not validation.

## Strong Geometry Control Launch

- Version/commit: `det.energy.dense_endpoint.geometry_control.001`, `c2926418...`.
- New split is deterministic inside the old 318-image inner-fit pool: fit 254, hash `468ce612...abee7`; holdout 64, hash `cde545e8...ceefe`.
- The combined fit/tune cache must be deserialized, but its final 68 old-tune records are discarded before statistics, training, evaluation, and gates. Old outer cache and new images are not read.
- Strong control jointly shuffles node columns 1-6 and 18-21 and uses the precomputed box-to-node topology-shuffled pair graph. A score+class-only arm disables all geometry and the pair branch. Hyperparameters are frozen from the absolute probe.
- Started on physical GPU2 with `41528 MiB` free, estimated peak `1024 MiB`, steady free `41038 MiB`; PID `3643909`.

## Strong Geometry Control Result

- Completed at commit `c29264184acafb17d436272cd143e079080ac2a0`; artifact SHA256 `ca5fbbef4ee19faa35a925eec6245bac456bc90525c76426e250a24f69b1171d`.
- Protocol boundary is explicit: the combined cache deserialized 68 old-tune rows but discarded them before all statistics, training, evaluation, and gates. Old outer cache, new images, and detector inference were not read/run.
- Full 64-image holdout: MAE `0.42829`, Pearson `0.97231`, pairwise `0.87847`, AUROC `0.98438`; constant MAE improvement `73.64%`; fit-to-holdout pairwise gap `0.04945`.
- Full minus strong node-geometry + pair-topology shuffle: `+0.05655` pairwise and `+0.36842` MAE gain.
- Full minus score+class-only: `+0.06548` pairwise and `+0.52797` MAE gain.
- All support, absolute-quality, constant, control, and gap gates passed. Decision: retain a researcher-adaptive train-only geometry endpoint signal and permit one separately preregistered detector-unseen absolute endpoint validation. This is not formal validation, local Delta-Q, action, or AP evidence.

## Detector-Unseen Absolute Endpoint Launch

- Version/commit: `det.energy.dense_endpoint.cleanval.001`, `b0cc5428...`; config SHA256 `82f56e7772042904c6da3dbb8559e630a351a35ccba5582421aeb1d0d8f8d05e`.
- Training uses all 454 cached NWPU train images. Architecture, teacher basis, optimization, and thresholds are frozen. Arms are full, strong geometry shuffle, score+class-only, and teacher shuffle.
- The 196-image detector validation IDs, images, and GT teacher targets are materialized once only after all four models freeze. Detector weights remain frozen and no AP/action is evaluated.
- Started on physical GPU2 with `42365 MiB` free and estimated peak `6144 MiB`; immediate free memory `42739 MiB`; PID `3782055`.

## Detector-Unseen Absolute Endpoint Result

- Completed at clean commit `b0cc54282b50005f81e1e1a11028d5ef7bbec6a7`; artifact SHA256 `024f9ed8cd89a52ac743d35b1bb6a54717772dfd51b2027e47ad30de51f80f8f`; one-time validation cache SHA256 `3d39f276...d7a11`.
- Runtime evidence shows all four models trained before `detector_validation_read_once` progressed through 196/196 images. `validation_read_count=1`, models frozen before read, detector parameters unchanged, and no actions evaluated.
- Full validation: MAE `1.76984`, Pearson `0.88461`, pairwise `0.80926`, AUROC `0.93580`; best-constant MAE gain `22.77%`.
- Pairwise gains: strong geometry shuffle `+0.04150`, score+class-only `+0.05455`, teacher shuffle `+0.30382`. MAE gains were `+0.45970/+0.32361/+1.03049`. Every absolute, support, and control gate passed.
- Train pairwise was `0.92617`; validation gap `0.11691` exceeded the locked `0.10` gate. Decision: `detector_unseen_absolute_endpoint_frozen`. Do not relax or reinterpret the gate after seeing the result.
- Narrow evidence retained: relative set-quality ranking and independent geometry/score information transfer to detector-unseen images. Absolute endpoint validation, local Delta-Q, actions, and AP remain unvalidated.
- DeepSeek agreed the frozen status is procedurally mandatory and highlighted the strong rank/control evidence. Codex rejects using the reviewer's statement that the gate measures the “wrong quantity” to change the result; that is a future protocol-design lesson only.

## Clean-Validation Shift Diagnosis

- Version/commit: `det.energy.dense_endpoint.shift_audit.001`, `b831dcba...`; artifact SHA256 `62fc164491e325926873771a1941da0fd8f2a94cdb231b48fa12033d66096ce1`.
- Cache-only; no new inference or training; original frozen status unchanged.
- Validation residual mean is `+1.70739`; oracle mean-centering reduces MAE from `1.76984` to `1.06457` (`39.85%`) while pairwise/AUROC remain unchanged.
- Coverage/GT shifts down `1.1740` train standard deviations (KS `0.50088`); total-error/prediction shifts up `1.3835` (KS `0.37492`); duplicate/edge is stable (`0.0233` std, KS `0.08943`).
- Only `11.73%` of validation images exceed the train 95th-percentile 27-D node/pair Mahalanobis distance, below the locked 25% feature-OOD threshold.
- Diagnosis: `calibration_shift_without_feature_ood`. The model preserves rank but misses a conditional quality-level shift; this motivates shift-invariant local differences, not validation-fitted intercept correction.

## Dense Local Delta-Q Support

- V1 commit `19116060...`, artifact SHA256 `f4277ec117102582c96af5662a6bbd532d37fc0347db8fb77942e15ea571b43c`. All density gates passed, but identity permutation max error `9.54e-7` exceeded `1e-7`; status frozen.
- Root cause was order-dependent float reduction. The teacher now canonical-sorts prediction and GT sets before reductions; a bitwise permutation-invariance regression test passes. The gate was not relaxed.
- V2 commit `84bd1f7e...`, config SHA256 `86b2d70495c8174b22fa325ae76ac0410414ce47da643b81bc8376c4cd26ffc6`; artifact SHA256 `10804a48fae1b3d81d7ebac1b51660524f2008f57a333a4da15ed185fe4d3010`.
- Identity permutation is exactly `Delta-Q=0`. Across 1557 non-identity train-only perturbations: nonzero `94.22%`, positive `27.17%`, negative `67.05%`, 1434 rounded unique values, median absolute Delta-Q `0.02663`, range `[-6.5105, 1.0147]`.
- All gates passed. Decision: a future train-only shift-invariant local `Q_theta(S') - Q_theta(S)` learner is warranted. This does not restore the frozen absolute endpoint and does not authorize detector actions; perturbations were applied to native kept sets without rerunning NMS.

## Dense Local Delta-Q Learner Result

- Version/commit: `det.energy.dense_local_delta_learner.001`, `58908126...`; train-only researcher-adaptive split `48/16` inside the locked 64-image source pool.
- Artifact: `runs/nwpu_dense_local_delta_learner_s42_48_16/eval_metrics.json`, SHA256 `a7570b9e89e733174272cd06956e6c2646b3fc73274c46e8b04ed71d5f969f30`; pair cache SHA256 `16758625844bebb617b6ed2552adccb2d58a722966b277677b422ad2ff063c41`.
- Full tune metrics: MAE `0.08585` versus zero predictor `0.10997` (`21.94%` relative gain), sign AUROC `0.69011`, exact identity error `0`, and pooled within-image pairwise accuracy `0.57469`.
- Full-control differences passed the original point-estimate gates: pairwise `+0.03302/+0.05195` and MAE `+0.01298/+0.02334` versus strong-geometry/utility shuffle. Fit-to-tune pairwise gap was `0.06933`.
- The locked absolute pairwise gate was `0.60`; `0.57469` failed it. Decision: `train_only_local_delta_learner_frozen`. The sign and MAE evidence show weak differential signal, but do not authorize a larger confirmation, detector validation, actions, or AP claims.
- DeepSeek agreed that freezing is procedurally mandatory and highlighted the Smooth-L1/ranking mismatch. Codex rejects its stronger wording that the result is merely underpowered: only 16 tune images were available, but uncertainty and control attribution must be measured rather than assumed.

## Post-Hoc Family And Margin Audit

- Version/commit: `det.energy.dense_local_delta_family_audit.001`, `923db6a7...`; no new training, detector inference, teacher call, NMS, or validation read.
- Artifact: `runs/nwpu_dense_local_delta_learner_s42_48_16/family_audit_metrics.json`, SHA256 `4c0a150e97135d44c3eca5b6caeaf74c28a298b6ed0ccd1314f0f5747a399b53`.
- The original metric name was misleading: it pools eligible pairs across images. Replayed pooled accuracy is exactly `0.57469`; a true per-image-equal aggregation is `0.57147`. Therefore unequal image pair counts do not explain the failure.
- Margin-stratified image-equal accuracy is `0.6429` for `(1e-6,0.01]`, `0.4894` for `(0.01,0.05]`, and `0.6246` for `>0.05`. Near-zero differences are not the aggregate bottleneck; the medium band is.
- Image-paired full-minus-strong-geometry gain is `+0.02497`, bootstrap 95% CI `[-0.01264, 0.06651]`; full-minus-utility gain is `+0.04612`, CI `[-0.00610, 0.10028]`. Neither overall interval excludes zero on 16 images.
- Family behavior is highly heterogeneous. `score_down|translate_up` is `0.28472`, while `score_down|score_up` is `0.88889` and `drop|scale_up` is `0.79861`. Some apparently strong cells are equally strong under controls; they cannot justify deleting difficult families or claiming learned geometry.
- Decision: the post-hoc audit does not reopen the learner. It refutes image-weighting and near-zero-margin explanations, and it weakens the geometry-attribution story. No larger learner or loss retuning is launched from this reused split.

## Fit-Only Perturbation-Family Prior

- Version/commit: `det.energy.dense_local_delta_family_prior.001`, `d81b27a6...`; no neural training, detector/teacher inference, NMS, or validation read.
- Artifact: `runs/nwpu_dense_local_delta_learner_s42_48_16/family_prior_metrics.json`, SHA256 `2d578fd55f64f28f5fb2d363dc4d8df7be8997aa9487ae12ca9caa8e016481c6`.
- A nine-entry lookup table was fitted using only the 48 fit images: each perturbation family maps to its mean fit Delta-Q. It was evaluated once on the already reused 16-image tune split.
- Family prior versus neural endpoint: pairwise `0.67241` versus `0.57469`; MAE `0.07722` versus `0.08585`; sign AUROC `0.70841` versus `0.69008`. The family prior also improves MAE over zero by `29.78%`, compared with `21.94%` for the neural endpoint.
- The largest family mean is `drop=-0.47800`; score changes are near zero (`-0.00051/-0.00162`), while scale/translation means range from about `-0.0182` to `-0.0455`. Much of the apparent Delta-Q predictability is therefore explained by perturbation identity rather than image content.
- Neural and family-prior predictions correlate only `0.24049`; this is descriptive disagreement, not an explained-variance estimate.
- Decision: no content-conditioned local Delta-Q gain is established. Freeze capacity growth, loss retuning, and family-ID augmentation on the reused split. A future experiment, if pursued, must fit the family prior on fit only, learn only residual `Delta-Q - mean_family`, and evaluate on newly locked train-only images with zero-residual and shuffle controls.
- DeepSeek agreed that the exact learner should remain frozen and that the family prior is the decisive baseline. Codex rejects three reviewer errors: global-mean pairwise is `0.5`, not `0`; prediction correlation is not variance explained; a constant negative prediction alone has AUROC `0.5`, not `0.69`.

## Residual Protocol Split Gate

- Draft: `docs/dense_local_delta_residual_preregistration.md`; one-shot split builder commit `df400296...`.
- Fixed source: the 254-image geometry-control fit pool, excluding the old 64-image local-learner source. Proposed capacities were 96 fit, 32 tune, 126 untouched reserve; seed `52042`; rare-first greedy multilabel class+density assignment.
- The builder was run once on the remote annotation and failed before writing a manifest. Tune class-image support was class 3=`1` and class 8=`1`, below the locked minimum `3`. Fit class 8 support was `3`. All four density bins were represented.
- Rejected deterministic hashes: fit `098cf446...3113`, tune `4053f4a6...f1b7`, reserve `cb73c0ae...1069`. These identify the failed split only; they are not an authorized manifest.
- No cache, detector inference, teacher call, training, GPU job, old inner-tune/outer read, or detector-validation read occurred. The seed, capacities, and support gate were not changed and no alternative split was generated.
- Decision: the proposed residual confirmation is blocked at its preregistered data-support gate. Combined with the stronger family-prior baseline, this ends the current synthetic post-NMS local Delta-Q learner branch.
