# Manifold Execution Lane Initialization Report

Date: 2026-07-22 (Asia/Shanghai)

## Verdict

The Manifold project now has project-local, persistent Kimi and Qwen execution
lanes selected by the exact pair of canonical absolute Git root and chain ID.
Neither lane reuses the read-only protocol-review session. Kimi has completed a
traceable implementation task; Qwen has a frozen task and a guarded one-shot
night-window dispatch.

## Bound Identity

- Canonical Git root:
  `E:/CLIproject/.worktrees/manifold-research-state-refactor`
- Chain ID: `det.energy.re_roi_closure.001`
- Workflow DB:
  `docs/agent_profiles/orchestrator/workflow.sqlite3` (local runtime state)
- Token ledger:
  `docs/agent_profiles/orchestrator/token-ledger` (local runtime state)
- Kimi Profile/session: `manifold-kimi-executor` /
  `session_aa114e05-51fb-46e2-9f5d-483e07a50c19`
- Qwen Profile/session: `manifold-qwen-executor` /
  `a8a962e4-0a35-426a-9eef-0bff9e9b00d2`
- Kimi isolated worktree/branch:
  `E:/CLIproject/.worktrees/manifold-kimi-executor` /
  `codex/manifold-kimi-executor-v1`
- Qwen isolated worktree/branch:
  `E:/CLIproject/.worktrees/manifold-qwen-executor` /
  `codex/manifold-qwen-executor-v1`

The workflow snapshot contains both `kimi-execution` and `qwen-execution`
lanes with distinct Profiles, sessions, and worktrees. Status, log inspection,
and bounded repairs resume the recorded session. Idle Profiles have no running
model process.

## Kimi Traceable Delivery

The first Kimi packet owned exactly:

1. `tools/validate_project_execution_profile.py`
2. `tests/test_validate_project_execution_profile.py`
3. `docs/agent_profiles/reports/MANIFOLD_KIMI_EXECUTOR_BOOTSTRAP_V1.md`

Executor commits were `7d662229`, `9516c15`, and the report-only security
amendment `471ec213`. Main integrated their content as `a02c5f1`, `8d1a9a8`,
and `ef2b561` after independent verification. The workflow task
`manifold-kimi-executor-bootstrap-v1` is `completed-reusable`; the separate
security amendment is also traceable.

Main verification results:

- focused pytest: `45 passed`;
- compileall: exit 0;
- Kimi real-binding validation: exit 0, `ok=true`;
- Qwen real-binding validation: exit 0, `ok=true`;
- manifest and workflow DB SHA-256 values unchanged by both validator runs;
- changed-file boundary: exactly the three packet-owned files;
- Kimi delivery worktree: clean.

The workflow DB schema has no worktree column. Role, kind, Profile, and session
agreement are checked against SQLite; isolated and mutually distinct worktrees
are checked from the tracked routing manifest. This limitation is explicit and
must not be reported as DB-level worktree agreement.

## Token Ledger

The Kimi ledger entry records provider `kimi`, model `kimi-code/k3`, the
persistent session, 37 model calls, 4,360,104 input tokens, 4,101,888 cached
input tokens, zero cache-creation tokens, and 43,220 output tokens. The measured
input cache ratio is 94.08%, below the 98% target. It is retained as the honest
cold-start baseline rather than being inferred or rewritten.

## Security Disposition

During the first Kimi bootstrap, an environment inspection printed a
credential-bearing variable into the local Kimi wire log. No credential value
is present in the three Git delivery files. The Kimi report was amended to
`FAIL_SECURITY_PENDING_MAIN` and the original local wire log is retained as an
audit record.

Main changed the Kimi launcher so provider credential variables are removed
from the child process environment and reran the orchestration test suite:
`14 tests passed`. The code delivery is accepted after that containment fix,
but rotation of the exposed credential remains an external required action.
The report's security verdict is intentionally not rewritten to PASS.

## Qwen Night-Window Dispatch

Automation `manifold-qwen-executor` is active for 22:00 Asia/Shanghai. It is
bound to the Manifold project, exact root/chain pair, Qwen Profile, preallocated
session, isolated worktree, and frozen packet
`packets/manifold-qwen-executor-bootstrap-v1.txt`. Qwen owns implementation,
the full acceptance suite, self-review, two commits, and a complete report.
Main performs later independent verification and integration.

The 17:09 schedule-gate check correctly rejected a model call and reported the
next opening as 22:00. After the frozen packet reaches waiting-for-main-review
or failed, the automation is instructed to pause itself so idle state does not
produce recurring model calls.

## Scientific Boundary

No experiment, training, inference, detector evaluation, GPU action, remote
host mutation, outer-heldout read, sealed-outcome read, dataset access, or
frozen research route was started or restored while initializing these lanes.
The remote experiment provenance boundary remains unchanged.
