# Manifold Executor Delivery Contract V1

## Scope

`manifold-kimi-executor` and `manifold-qwen-executor` are complete bounded-task
owners, not patch suggestion lanes. For every accepted packet the executor
must implement, run the complete acceptance suite, self-review, audit the exact
file boundary, commit, and write a full report. Main Codex independently
verifies and integrates.

The read-only `manifold-protocol-review` remains separate. No execution lane
may reuse its session or convert a review request into a write task.

## Identity Gate

Dispatch requires an exact match for all of:

- canonical Git root
  `E:/CLIproject/.worktrees/manifold-research-state-refactor`;
- chain ID `det.energy.re_roi_closure.001`;
- project workflow DB and token ledger recorded in
  `profile-routing-v1.json`;
- lane role, kind, Profile, persistent session, and isolated worktree;
- packet-owned and forbidden paths.

Window title, current directory name, or repository nickname alone cannot
select a Profile. A foreign project identity, mismatched session, missing
ownership field, active conflicting writer, or dirty executor worktree blocks
dispatch.

## Packet Contract

Every packet states:

1. chain ID, role, kind, Profile, work key, and prior decision;
2. material delta and exactly one bounded implementation objective;
3. canonical Git root, workflow DB, token ledger, session, and write root;
4. exact writable files and all forbidden surfaces;
5. implementation requirements and test scenarios;
6. complete acceptance commands and failure rules;
7. commit message and full report schema.

Patch-only, suggestion-only, or untested output is failure. If a repair needs
a forbidden path, the executor stops and reports FAIL.

## Lifecycle

An idle Profile has no running model process and consumes no tokens. A model
starts only after the project workflow gate returns `start` or `resume`.
Status, log, and short follow-up requests resume the recorded Profile/session;
they never create a replacement Agent. One role has at most one active task,
one session has at most one active turn, and the chain has at most two active
support tasks.

Kimi uses `kimi-code/k3`, Thinking Max, context 1,048,576, timeout 1,800
seconds, and the official OAuth route. Qwen uses
`qwen3.8-max-preview` through the guarded local proxy. New Qwen requests are
allowed only in Asia/Shanghai `22:00-06:00`; an in-flight request may finish
after 06:00 but receives no new turn until the next window.

## Forbidden Scientific And Runtime Surfaces

Executor bootstrap tasks must not:

- launch training, inference, evaluation, qualification, or GPU work;
- access or mutate a remote host;
- read outer heldout, detector validation, sealed outcomes, real datasets,
  checkpoints, or run artifacts;
- edit research registries, current-status conclusions, research methods,
  experiment scripts, or frozen routes;
- edit workflow DBs, token ledgers, Profile registries, AGENTS, automation,
  another worktree, or another executor's files;
- integrate or merge their own commit.

## Main Acceptance

Main verifies exact changed files, commit ancestry, report completeness,
commands and exit codes, focused and relevant tests, hashes, packet immutability,
worktree cleanliness, and absence of forbidden access. Only accepted bytes are
integrated. Workflow transition and token recording occur after verification.
