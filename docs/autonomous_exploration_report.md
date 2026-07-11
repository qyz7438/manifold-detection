# Autonomous Research Exploration Report

Status: active

Window: 2026-07-12 03:06 to 11:06 Asia/Shanghai.

The two-hour heartbeat automation is active as `manifold-autonomous-8h-20260712`.

Cycle 0 implemented D1 proposal-set context under the locked C3 whole-image Delta-U cache. The model is permutation-equivariant over detector-observable proposals and uses set mean/max context while preserving the C3b endpoint, action table, balanced loss, optimizer, split, and controls. Focused verification currently passes 15 tests. Remote execution is pending commit and GPU2 reserve checks.

D1 completed and failed its preregistered detector/control gates. The policy was non-degenerate (9/32 actions) but full metrics were exactly identity, and AP75 did not beat feature shuffle. D1 is frozen without train-size expansion. The next queue item is D2 explicit detector-boundary/NMS topology evidence, subject to DeepSeek review that it is a genuinely new observable signal rather than capacity tuning.

DeepSeek agreed with the freeze, but its cheapest static boundary proposal overlaps D1's existing conflict statistics. Codex therefore refined D2 to action-conditioned NMS topology: post-action higher-score same-class overlap, survival margin, and topology change for each action, with a dedicated topology-shuffle control. This changes observable information rather than merely model capacity.

See `docs/autonomous_exploration_ledger.md` for the authoritative queue, commands, gates, and artifacts.
