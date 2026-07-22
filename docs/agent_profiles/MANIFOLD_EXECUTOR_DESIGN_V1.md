# Manifold Execution Profile Design V1

Date: 2026-07-22 (Asia/Shanghai)

## Decision

Add two independent persistent writable execution lanes to the existing
Manifold profile group:

- `manifold-kimi-executor`: all-day `kimi-code/k3`, Thinking Max, context
  1,048,576, 1,800-second turns.
- `manifold-qwen-executor`: guarded `qwen3.8-max-preview`, new turns only in
  Asia/Shanghai `22:00-06:00`.

The existing `manifold-protocol-review` remains read-only. Execution sessions
must never reuse its review context.

## Project Identity

- Canonical absolute Git root:
  `E:/CLIproject/.worktrees/manifold-research-state-refactor`
- Chain ID: `det.energy.re_roi_closure.001`
- Workflow DB:
  `E:/CLIproject/.worktrees/manifold-research-state-refactor/docs/agent_profiles/orchestrator/workflow.sqlite3`
- Token ledger root:
  `E:/CLIproject/.worktrees/manifold-research-state-refactor/docs/agent_profiles/orchestrator/token-ledger`

Profile selection is the pair `(canonical absolute Git root, chain_id)`.
Window title, current directory name, repository nickname, or another
project's session is never sufficient. Any identity mismatch fails closed.

## Ownership

| Unit | Owns | Consumes | Emits | Must not change | Verification |
|---|---|---|---|---|---|
| Main | profile registry, integration, final decisions | executor commits and reports | accepted integration and workflow transition | sealed evidence or frozen research routes | independent tests, hashes, boundary audit |
| Kimi executor | one frozen packet in its isolated worktree | exact owned/forbidden paths and acceptance commands | implementation commit, self-tests, report commit | review session, research code, registries, DB/ledger | packet suite plus main re-run |
| Qwen executor | one frozen packet in its isolated worktree | same complete delivery contract | implementation commit, self-tests, report commit | out-of-window calls, research code, DB/ledger | schedule gate, packet suite, main re-run |

## Lifecycle

Profiles and session identifiers may remain idle without a running process or
token consumption. A model starts only after `workflow_state.py start` returns
`start` or `resume`. Status, log inspection, and short follow-ups resume the
recorded Profile/session and never create a replacement Agent.

## Acceptance

1. Profile TOMLs parse and bind the exact project identity.
2. Workflow snapshot contains distinct `kimi-execution` and
   `qwen-execution` lanes with distinct sessions and worktrees.
3. Kimi completes one traceable implementation task with commit, tests, and
   full report; main independently verifies and integrates it.
4. Qwen has an equally complete frozen task and an active 22:00 scheduler;
   outside the window both workflow and wrapper gates reject dispatch.
5. No experiment, GPU process, remote mutation, heldout read, or frozen-route
   revival occurs during bootstrap.
