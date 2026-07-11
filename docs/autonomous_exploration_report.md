# Autonomous Research Exploration Report

Status: active

Window: 2026-07-12 03:06 to 11:06 Asia/Shanghai.

The two-hour heartbeat automation is active as `manifold-autonomous-8h-20260712`.

Cycle 0 implemented D1 proposal-set context under the locked C3 whole-image Delta-U cache. The model is permutation-equivariant over detector-observable proposals and uses set mean/max context while preserving the C3b endpoint, action table, balanced loss, optimizer, split, and controls. Focused verification currently passes 15 tests. Remote execution is pending commit and GPU2 reserve checks.

D1 completed and failed its preregistered detector/control gates. The policy was non-degenerate (9/32 actions) but full metrics were exactly identity, and AP75 did not beat feature shuffle. D1 is frozen without train-size expansion. The next queue item is D2 explicit detector-boundary/NMS topology evidence, subject to DeepSeek review that it is a genuinely new observable signal rather than capacity tuning.

DeepSeek agreed with the freeze, but its cheapest static boundary proposal overlaps D1's existing conflict statistics. Codex therefore refined D2 to action-conditioned NMS topology: post-action higher-score same-class overlap, survival margin, and topology change for each action, with a dedicated topology-shuffle control. This changes observable information rather than merely model capacity.

D2 is now implemented and preregistered. It reuses the locked detector-only C3 candidate cache and whole-image Delta-U endpoint; no GT candidate filtering or larger offline probe was introduced. The only added information is action-conditioned native-NMS topology. Its dedicated control destroys proposal/action alignment while preserving the topology-feature values. Local focused/regression verification passes 46 tests; remote execution remains gated on review, a clean synchronized commit, and the GPU2 reserve calculation.

D2 pairwise-proxy smoke completed with native parity and non-degenerate policies, but all three arms were exactly equal to identity on detector metrics. A post-run Terra audit then identified that the implementation was not an exact native-NMS topology test and that the shuffle control was restored to true topology at evaluation. The result is therefore retained as a negative result for the pairwise top-1 proxy only, not as evidence against native topology. D2b is a bounded correctness repair: class-expanded native kept-set observables and persistent train/eval shuffle, with no new images, actions, GT filtering, loss tuning, or offline capacity expansion.

D2b is implemented with a torchvision-equivalent class-expanded postprocessing trace and an exact-zero identity contract. It enriches the same locked C3 records only after detector/cache alignment checks, leaving Delta-U untouched. The new topology control is shuffled in both train and validation. Focused tests currently pass 27 checks; the next step is the clean remote smoke under the unchanged GPU2 reserve gate.

Pre-launch review caught a subtle identity drift from applying a mathematically zero box transform. D2b now reuses the baseline trace exactly for candidate 0, and its launcher rejects stale artifacts whose HEAD, clean state, source hash, config hash, or scope do not match. The native smoke remains pending commit and remote verification.

See `docs/autonomous_exploration_ledger.md` for the authoritative queue, commands, gates, and artifacts.
