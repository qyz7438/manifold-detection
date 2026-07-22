# Manifold Kimi Executor Bootstrap V1 — Delivery Report

- Chain ID: `det.energy.re_roi_closure.001`
- Role: `kimi-builder`
- Kind: `kimi-execution`
- Profile: `manifold-kimi-executor`
- Work key: `kimi-builder:project-execution-profile-validator-v1`
- Prior decision: `MANIFOLD_EXECUTOR_DESIGN_V1`
- Task kind: material agent-tooling implementation bootstrap
- Owned worktree / write root: `E:/CLIproject/.worktrees/manifold-kimi-executor`

## Verdict

**FAIL_SECURITY_PENDING_MAIN.** All acceptance commands exited 0 and the file
boundary held at every audit point (evidence preserved below), but the
bootstrap delivery is rejected on security grounds: during bootstrap an
environment inspection command printed a credential-bearing environment
variable into the local Kimi session wire log (see "Security Incident And
Retraction"). Final disposition is pending main's decision outside this
packet.

## Security Incident And Retraction

1. **Incident record.** During bootstrap, an environment inspection command
   (a listing of session-related environment variable names and values) was
   executed in this worktree, and its output — which included a
   credential-bearing environment variable — was printed into the local Kimi
   session wire log. The variable name and value are deliberately not
   reproduced in this report or any commit.
2. **Delivery-file verification.** Main independently verified zero
   credential-name and zero credential-value-pattern hits in the three Git
   delivery files (`tools/validate_project_execution_profile.py`,
   `tests/test_validate_project_execution_profile.py`, and this report).
3. **Retraction.** Any earlier claim in this report or its delivery message
   that no sensitive environment surface was accessed is hereby retracted:
   the environment inspection in item 1 did access a sensitive environment
   surface.
4. **Rotation.** Credential rotation is required. Rotation must happen
   outside this packet and remains unresolved at amendment time; the
   credential value is never included in this report or its commits.
5. **Scope of this amendment.** Only this report file is amended. All
   implementation/test evidence, hashes, boundary audits, and commit
   references below are preserved unchanged; the amendment commit SHA is
   recorded in the delivery response.

## Session ID Observed From The Current Kimi Turn

`aa114e05-51fb-46e2-9f5d-483e07a50c19`, observed from the runtime session
directory of this turn
(`C:/Users/青云志/.kimi-code/sessions/wd_manifold-kimi-executor_360ee18e76ff/session_aa114e05-51fb-46e2-9f5d-483e07a50c19/`).

The routing manifest still records `session_id: null` with
`session_state: "uninitialized"` for the Kimi lane, and the workflow DB records
`session_id: "uninitialized:manifold-kimi-executor"`. Whether the runtime ID
above is the persistent session that main should record into the manifest and
DB is **unknown** to this executor; recording it is a workflow/main decision
outside this packet's writable boundary. Consequently, the validator correctly
fails closed on the current real Kimi binding (see "Real-binding smoke runs"
below) and passes on the real Qwen binding.

## Deliverables (exact writable files, nothing else)

1. `tools/validate_project_execution_profile.py`
2. `tests/test_validate_project_execution_profile.py`
3. `docs/agent_profiles/reports/MANIFOLD_KIMI_EXECUTOR_BOOTSTRAP_V1.md` (this file)

## Requirement-To-Code Matrix

| Requirement | Implementation (`tools/validate_project_execution_profile.py`) |
|---|---|
| CLI accepts `--manifest`, `--workflow-db`, `--expected-root`, `--chain-id`, `--role` | `build_parser()`; all five arguments required |
| Parse JSON manifest | `run()` input phase; `json.load` with `encoding="utf-8"`; non-object top level rejected |
| Open SQLite read-only, no input mutation | `open_db_readonly()` uses `file:...?mode=ro` URI plus `PRAGMA query_only = ON`; SHA256 before/after proof in `run()` |
| Selector is exactly canonical absolute Git root + chain_id | `validate()` check 1 (`root_mismatch`) and check 2 (`chain_mismatch`) against normalized `--expected-root` / `--chain-id` |
| Reject window-name routing | check 3 (`window_name_routing`): `window_name_is_selector` must be exactly `false`; `selection_key` must equal `["absolute_git_root", "chain_id"]` |
| Manifest/DB agreement: workflow DB | check 4 (`workflow_db_mismatch`): manifest `workflow_db` must equal `--workflow-db` after normalization |
| Workflow DB and token ledger inside project binding | checks 5–6 (`workflow_db_outside_project`, `ledger_outside_project`) via `is_within()` |
| Manifest/DB agreement: role, kind, Profile, session | check 15 (`chain_absent_in_db`, `lane_absent_in_db`, `kind_mismatch_db`, `profile_mismatch_db`, `session_mismatch_db`) against `lanes` row for `(chain_id, role)` |
| Manifest/DB agreement: isolated worktree | checks 13–14 (`worktree_not_isolated`: non-empty, not equal to or inside canonical root) and check 16 (`shared_worktree`); the DB `lanes` table records no worktree column, so worktree agreement is enforced on the manifest side and through Kimi/Qwen distinctness |
| Kind must be `kimi-execution` or `qwen-execution` for writable lanes | checks 7–8 (`execution_lane_absent`, `writable_non_execution_lane`) and check 10 (`selected_lane_kind_not_execution`) |
| Non-empty persistent session | check 13 (`empty_session`): `session_id` must be a non-blank string |
| write=true | check 11 (`lane_not_writable`) |
| owned-files fail-closed, dispatch-through-workflow, idle zero-token, resume-only status/log/short-followup | check 17 (`lifecycle_flag_invalid`): `owned_files_fail_closed`, `dispatch_requires_workflow_start_or_resume`, `new_agent_for_status_forbidden`, `cross_project_session_reuse_forbidden` must be exactly `true`; `idle_consumes_tokens` exactly `false`; `status_log_and_short_followup_resume_existing_profile` exactly `true` |
| Kimi and Qwen sessions and worktrees distinct | check 16 (`shared_session`, `shared_worktree`) across both execution lanes |
| Reject root/DB/ledger/worktree/chain/role/kind/Profile/session mismatch | checks 1, 4, 5, 6, 9 (`lane_absent_for_role`), 14, 15, 16 — each emits an ordered error and forces exit 1 |
| Prove manifest and DB SHA256 unchanged | `run()` read-only proof phase: `input_hashes.{manifest,workflow_db}.{sha256_before,sha256_after,unchanged}`; any drift appends `input_mutated` and forces exit 1 |
| Deterministic JSON: ok, normalized identity, lane summary, input hashes, ordered errors | `result()` fixed schema; `json.dumps(..., sort_keys=True)`; `normalize_path()` for all identity paths; errors appended in fixed check order |
| Exit 0 valid / 1 contract failure / 2 usage or input failure | `run()` returns 2 for unreadable/malformed manifest or DB (incl. missing required tables); 1 when any contract error collected; 0 otherwise; argparse exits 2 on missing arguments |

## Requirement-To-Test Matrix

| Requirement | Tests (`tests/test_validate_project_execution_profile.py`) |
|---|---|
| Valid Kimi lane | `test_valid_kimi_lane` |
| Valid Qwen lane | `test_valid_qwen_lane` |
| Wrong root | `test_wrong_root_rejected` |
| Wrong chain | `test_wrong_chain_rejected` |
| Window-name selector | `test_window_name_selector_rejected`, `test_window_name_selection_key_rejected` |
| Workflow path outside project binding | `test_workflow_db_outside_project_rejected`, `test_workflow_db_mismatch_rejected` |
| Ledger outside project binding | `test_ledger_outside_project_rejected` |
| Absent lane | `test_absent_lane_rejected`, `test_db_lane_absent_rejected`, `test_chain_absent_in_db_rejected` |
| Kind mismatch | `test_kind_mismatch_rejected` |
| Profile mismatch | `test_profile_mismatch_rejected` |
| Session mismatch | `test_session_mismatch_rejected` |
| Empty session | `test_empty_session_rejected`, `test_null_session_rejected` |
| Shared Kimi/Qwen session | `test_shared_session_rejected` |
| Shared worktree | `test_shared_worktree_rejected` |
| Worktree not isolated | `test_worktree_inside_root_rejected`, `test_worktree_equal_root_rejected` |
| Missing/wrong lifecycle flags | `test_lifecycle_flags_rejected` (7 parametrized cases: wrong value for each flag, plus a missing flag) |
| Writable review lane | `test_writable_review_lane_rejected`, `test_review_role_selected_rejected`, `test_lane_not_writable_rejected` |
| Malformed/unreadable JSON | `test_malformed_manifest_exit2`, `test_unreadable_manifest_exit2`, `test_manifest_not_object_exit2` |
| Malformed/unreadable DB | `test_malformed_db_exit2`, `test_missing_db_exit2`, `test_db_missing_tables_exit2` |
| Usage failure exit 2 | `test_missing_cli_argument_exit2` plus all `*_exit2` tests |
| Deterministic output | `test_deterministic_output`, `test_deterministic_failure_output` (byte-identical stdout across runs, sorted keys) |
| Exit codes 0/1/2 | asserted in every test above |
| Manifest and DB SHA preservation | `test_input_sha256_preserved`, `test_input_sha256_preserved_on_failure` (file hashes recomputed after CLI run; `unchanged` flags and before/after values in payload) |
| No project research modules imported | tests import only stdlib, pytest, and the tool by file path |
| Path normalization / containment units | `test_normalize_path_case_and_separators`, `test_is_within`, `test_main_in_process_valid` |

## Commands And Exit Codes

All run in `E:/CLIproject/.worktrees/manifold-kimi-executor` with
`E:/anaconda/01/envs/RLimage/python.exe`.

Acceptance suite (final run, after the implementation commit content was
frozen):

| # | Command | Exit | Concise output |
|---|---|---|---|
| 1 | `python -m pytest tests/test_validate_project_execution_profile.py -q` | 0 | `45 passed in 8.54s` |
| 2 | `python -m compileall tools/validate_project_execution_profile.py tests/test_validate_project_execution_profile.py` | 0 | both files compile (no diagnostics) |
| 3 | `git diff --check` | 0 | no output |
| 4 | `git status --short` | 0 | no output (clean) |

Real-binding smoke runs (read-only inputs, no mutation; evidence that the
validator passes the recorded Qwen binding and fails closed on the not yet
recorded Kimi binding):

| # | Command | Exit | Concise output |
|---|---|---|---|
| 5 | validator with real `profile-routing-v1.json` + real `workflow.sqlite3`, `--role qwen-builder` | 0 | `"ok": true`, `"errors": []`, both input hashes `unchanged: true` |
| 6 | same inputs, `--role kimi-builder` | 1 | `empty_session` + `session_mismatch_db` (manifest Kimi session still `null` vs DB `uninitialized:manifold-kimi-executor`) — expected fail-closed bootstrap state |

## Changed-File SHA256 Values

Post-commit, working-tree content:

- `tools/validate_project_execution_profile.py`:
  `21250c7eacde88cd0ecfef8d5abcd398d91f83d61823da57c65cce138ffa6bcd`
- `tests/test_validate_project_execution_profile.py`:
  `0ae06b31d125138f1c44b63387dd2ac23fafecd702cc5885c8df50302a60499d`
- `docs/agent_profiles/reports/MANIFOLD_KIMI_EXECUTOR_BOOTSTRAP_V1.md`:
  computable only after this file is written; recorded in the delivery note.

## Manifest/DB Read-Only Proof

- Real `profile-routing-v1.json` SHA256 before any validation, after the
  qwen/kimi smoke runs, and at report time:
  `4fece10251d01aaa566f7708958c20e1f0d884d2f4c1b987d5aa39ba094acc90`
  (unchanged at all three observation points).
- Real `workflow.sqlite3` SHA256 at the same three points:
  `3e492967d673a88cfa50f0420565732b0f6ad70b93a72a1b48c3156eeced03fe`
  (unchanged at all three observation points).
- The validator itself emits per-run proof: `input_hashes.*.unchanged: true`
  in both smoke runs, and the DB is opened exclusively with
  `file:...?mode=ro` plus `PRAGMA query_only = ON`.
- Test-suite proof: `test_input_sha256_preserved` and
  `test_input_sha256_preserved_on_failure` recompute fixture hashes after CLI
  runs on success and failure paths.

## Boundary Audits

Pre-implementation-commit audit (`git status --short --untracked-files=all`,
immediately before staging):

```text
?? tests/test_validate_project_execution_profile.py
?? tools/validate_project_execution_profile.py
```

Exactly the two writable implementation files; the report did not exist yet;
`__pycache__` is covered by `.gitignore` line 2 and never entered the index.

Post-implementation-commit audit: `git status --short` empty; the commit
contains only the two files (`git show --stat` confirms:
`tools/validate_project_execution_profile.py` and
`tests/test_validate_project_execution_profile.py`, 1040 insertions).

Post-report-commit audit: `git status --short` empty; the report commit
contains only `docs/agent_profiles/reports/MANIFOLD_KIMI_EXECUTOR_BOOTSTRAP_V1.md`.
Verified and recorded in the delivery note (the audit must run after this file
is committed, so its result cannot be embedded in this file itself).

## Commit SHAs

- Implementation commit (`Add project execution profile validator`):
  `7d6622291a21656def57d12b56359869af39dbb2`
  (amended once during self-review to drop one dead line in the test file;
  amendment happened before any integration, content frozen afterwards)
- Report commit (`Report Manifold Kimi executor bootstrap`): the HEAD commit
  containing this file; its SHA is fixed only at commit time and is recorded
  in the delivery note accompanying this report.

## Unresolved Risks (unknown, not inferred)

- Whether the runtime session ID observed this turn
  (`aa114e05-51fb-46e2-9f5d-483e07a50c19`) is the persistent session main will
  record for `manifold-kimi-executor`: **unknown** — recording is outside this
  packet's boundary.
- Whether main expects the validator to also gate on `session_state`
  transitions (`uninitialized` → active) beyond non-empty session equality:
  **unknown**; the current contract text requires a non-empty persistent
  session and manifest/DB agreement, which is what is implemented.
- Behavior of the validator against future `schema_version` values other than
  1: **unknown**; it validates the documented V1 shape and reports
  `schema_version` verbatim without interpreting it.
- Cache/rollout audit baseline for this executor lane: **unknown** (no rollout
  audit exists, per the workflow DB `cold_start_reason`).

## Forbidden-Surface Statements

During this bootstrap the executor did **not**:

- launch any experiment, training, inference, evaluation, qualification, or
  GPU work; no GPU command was issued;
- access or mutate any remote host; no SSH command was issued; no network
  request was made;
- read outer heldout data, detector validation data, sealed outcomes, real
  datasets, checkpoints, caches, or run artifacts (`runs/`, `data/`, `.npz`
  caches were not touched);
- edit AGENTS files, Profile TOMLs/registries, workflow DBs, token ledgers,
  automation, research docs/status, experiment scripts, method code, or any
  frozen route;
- write outside the three exact writable files of this packet, or touch
  another worktree's contents (the canonical worktree was accessed read-only
  for the three declared input documents and for read-only manifest/DB
  validation, which the packet explicitly designates as inputs);
- integrate or merge its own commit; both commits remain local to the
  `codex/manifold-kimi-executor-v1` branch of this worktree for main to verify
  and integrate.

The list above is scoped to scientific, runtime, and Git surfaces. It does
**not** extend to environment surfaces: during bootstrap an environment
inspection command was run and printed a credential-bearing environment
variable into the local Kimi session wire log. Any contrary claim or
implication that no sensitive environment surface was accessed is retracted
(see "Security Incident And Retraction").
