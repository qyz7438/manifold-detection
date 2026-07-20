# Manifold Persistent Profile Registry

Version: 2026-07-21

## Project Boundary

Repository: `E:/CLIproject/.worktrees/manifold-research-state-refactor`

This project owns the unknown-endpoint, energy-guided ROI transport research
line. It is distinct from the historical RLimage workspace and from
manifoldRL. The main profile owns scientific claims, leakage boundaries, and
the boundary between action-local transport, historical training mechanisms,
and architecture baselines.

## Profile Group

| Profile | Model / effort | Stable responsibility | Owned handoff artifacts | Session reference |
|---|---|---|---|---|
| `manifold-research-main` | GPT-5.6 Sol / XHigh | Scientific decisions, causal protocols, leakage, integration, and final evidence-bound conclusions | Research status/decision documents, final reports, Vault state/evidence deltas | Current Codex desktop task; host task ID is not exposed |
| `manifold-experiment-ops` | GPT-5.6 Terra / High | Bounded implementation support, tests, Git/remote sync, GPU2 admission, process checks, and raw artifact verification | Verified command/test/sync deltas and immutable artifact hashes | Create or resume one persistent ops task on the first independent operational handoff |
| `manifold-protocol-review` | GPT-5.6 Terra / High plus Kimi Code | Milestone-scoped implementation/protocol traceability review and the prescribed one-shot Terra fallback | Read-only findings with exact files and one acceptance question | Prior Kimi implementation review timed out and received its permitted Terra fallback; open no replacement without a new material milestone |

Exactly one profile owns scientific conclusions. Support profiles do not edit
overlapping source surfaces or select claims.

## Routing

- Method, protocol, leakage, result-to-claim, and next-mechanism decisions:
  `manifold-research-main`.
- Deterministic tests, registry/doc generation, Git/remote synchronization,
  GPU2 checks, logs, and artifact collection: `manifold-experiment-ops`.
- Material implementation/protocol milestones only:
  `manifold-protocol-review`, with one Kimi attempt and one same-scope Terra
  fallback after an explicit availability/quota failure.
- DeepSeek v4 Pro is a bounded advisory scientific-evidence reviewer only when
  explicitly required by a frozen protocol or when a distinct unresolved
  evidence question remains. Codex decides which points to accept.
- Ordinary continuation and status checks reuse the current owner; they do not
  create temporary Sol tasks.

## Cache Baseline

No rollout JSONL or exposed per-task cache audit is available for the migrated
Codex desktop task, so the initial rolling cache baseline is `unknown`, not
zero. The observed target remains at least 98%. Until a deterministic audit is
available, handoffs contain only the chain ID, prior decision, material delta,
exact paths, one question, and an acceptance check.

## Current Handoff Contract

| Unit | Consumes | Emits | Must not change | Acceptance check |
|---|---|---|---|---|
| Main | Handoff, reviewed artifacts, registry/test results | Integrated status, claim boundary, and user decision | Sealed cohorts and frozen mechanisms | Reviewed report, generated-doc agreement, full tests |
| Ops | Exact commit/command/path and GPU policy | Raw hashes, test/sync status, remote fingerprint | Scientific interpretation and unrelated processes | GPU2 free memory greater than 8192 MiB before any authorized GPU invocation |
| Review | Exact report/manifest/metrics delta | Concise findings and missing-evidence risks | Source files and final claims | One completed review or one recorded failure plus allowed fallback |

## Current Chain

Chain `det.energy.re_roi_closure.001` closes
`det.energy.re_roi_counterfactual_evidence.001` and the blocked
`det.energy.oracle_utility_boxhead.001` preflight. No experiment is currently
authorized. Outer heldout and detector validation remain sealed.
