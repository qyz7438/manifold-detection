# Repository Research-State Refactor Ledger

- Refactor plan: `docs/superpowers/plans/2026-07-13-manifold-repository-research-state-refactor.md`
- Date: 2026-07-13
- Base commit (audited): `7e1f2404699df997883678274491e88d3ec253c0`
- Plan-only commit (execution start): `2533d1325dbd5c5fba397aa5613eb30930918dea`
- Branch: `codex/repository-research-state-refactor` (worktree `E:/CLIproject/.worktrees/manifold-research-state-refactor`)
- Test baseline: 549 passed / 549 collected (`PYTHONUTF8=1 python -m pytest tests -q`), maintained collection sha256 `d406a78167a16b85c65c706e37b5fffee15fdd94c8c6f875d1cce7342c45a1a3`
- User-file exclusion: `docs/energy_transport_vs_fpn_sm_analysis.md` is an untracked user file in the main repository; it is never staged, moved, or treated as a clean-worktree blocker.
- Remote path: `ps@122.51.19.136:/home/ps/lzz/manifold-detection-energy-transport` (HEAD `7e1f240`, python 3.10.20, torch 2.1.0+cu121, torchvision 0.16.0+cu121; read-only SSH verified)
- Staffing note: per user instruction on 2026-07-13, Kimi executes all roles (main integration, builders, reviewers) in place of the GPT/Codex lineup named in the plan.

**No experiment is authorized by this refactor.**

## Baseline reconciliation notes

- `core.autocrlf=true` (system gitconfig `E:/Git/etc/gitconfig`) converted LF blobs to CRLF at worktree creation, breaking 31 hash-locked config tests (549 collected, 518 passed, 31 failed).
- Fix applied once: `git -c core.autocrlf=false archive HEAD | tar -x -C .`, then `git add -u` to refresh index stat info. Worktree files are now byte-identical to blobs and all 549 tests pass.
- Any future `git checkout` / `git switch` / `git worktree` operation in this worktree may re-apply CRLF; re-run the same fix if hash-locked tests fail again.

## Commit log

Convention: the commit hash of entry N is filled in by the ledger update of entry N+1, because a ledger cannot contain the hash of the commit that introduces it.

### 2026-07-13 — docs: lock repository refactor baseline

- commit hash: `0065de4`
- task ID: T0
- agent/model and write owner: Kimi (main orchestrator; write owner of ledger and shared files)
- files changed: `docs/research/refactor_ledger.md`
- RED command/result: n/a (baseline locking). Pre-existing environment failure observed and reconciled: 31 hash-locked config tests failed under CRLF checkout.
- GREEN command/result: `PYTHONUTF8=1 python -m pytest tests -q` -> 549 passed
- full-suite result: 549 passed in ~31s
- reviewer and findings accepted/rejected: n/a (T0 has no reviewer gate)
- scientific artifacts checked: none touched; protected history inventory (36 entries, docs/reports/** + docs/decisions/** + docs/autonomous_exploration_report.md) sha256 `384405371a051a5f3356af2b860a61bc859a435a56b29cf4b0ea263295588368` recorded in `.agent_reports/refactor/protected_history_inventory.txt`
- rollback command: `git revert <this-commit>` (docs-only)
- remaining risks: autocrlf CRLF regression on future checkout operations (mitigation documented above)

### 2026-07-13 — test: lock maintained repository state

- commit hash: `57b198c`
- task ID: T1
- agent/model and write owner: Kimi coder subagent (validator + contract test); Kimi main orchestrator owns `pyproject.toml` and the commit
- files changed: `pyproject.toml`, `scripts/dev/validate_repository_state.py`, `tests/contracts/test_repository_state.py` (`.gitignore` reviewed, no change needed)
- RED command/result: `pytest tests/contracts/test_repository_state.py -q` -> 1 passed, 25 errors (validator module not yet implemented)
- GREEN command/result: same command -> 26 passed; `python scripts/dev/validate_repository_state.py --json` -> `{"checked": 911, "violations": []}`, exit 0
- full-suite result: 575 passed (549 baseline + 26 new)
- reviewer and findings accepted/rejected: main-orchestrator review; accepted (patterns precise, no generic `*.json`, exact-name secrets, read-only CLI)
- scientific artifacts checked: none touched; validator confirms zero tracked runtime artifacts (dataset/checkpoint/cache/log/runs//.agent_reports//secret patterns)
- rollback command: `git revert <this-commit>`
- remaining risks: new artifact classes not covered by current patterns (e.g. future `*.parquet`) would need a pattern update with a corresponding contract test

### 2026-07-13 — feat: add research status registry

- commit hash: `987a134`
- task ID: T2 (W1 parallel wave)
- agent/model and write owner: Kimi coder subagent (contracts + registry + tests); Kimi main orchestrator reviewed and committed
- files changed: `spectral_detection_posttrain/experiments/contracts.py`, `spectral_detection_posttrain/experiments/research_status.py`, `spectral_detection_posttrain/configs/registry/research_lines.json`, `tests/experiments/test_research_status.py`
- RED command/result: `pytest tests/experiments/test_research_status.py -q` -> ModuleNotFoundError: `spectral_detection_posttrain.experiments.contracts`
- GREEN command/result: same command -> 33 passed
- full-suite result: 712 passed (575 baseline + T2 33 + concurrent W1 tests)
- reviewer and findings accepted/rejected: main-orchestrator review of scientific wording. Accepted: (1) strict transition-matrix reading (unlisted transitions rejected even with a decision document; superseding-decision process applies to the listed families only); (2) top-level `schema_version: "research_lines.v1"` envelope; (3) `docs/energy_transport_vs_fpn_sm_analysis.md` used only as `uncommitted_context` for c1_e1 and fpn_sm_afm, never as evidence. Mechanically verified: 10 registry statuses match plan Section 3.1 exactly; zero active/queued/validated; all 15 evidence pointers are tracked files.
- scientific artifacts checked: none touched; registry is metadata only. **No experiment is authorized by this refactor** — registry contains zero authorized lines.
- rollback command: `git revert <this-commit>`
- remaining risks: transition matrix is the strict reading; if a future decision requires an unlisted transition (e.g. active -> historical), the matrix needs a reviewed amendment commit.

### 2026-07-13 — feat: add immutable artifact manifests

- commit hash: `fdf4374`
- task ID: T4 (W1 parallel wave). Note: committed before T3, deviating from the Section 8 example order (T2 -> T3 -> T4); dependencies are unaffected because T3 depends only on T2 contracts, and this reorder is recorded here as the actual order.
- agent/model and write owner: Kimi coder subagent (artifacts module + tests + fixtures); Kimi main orchestrator reviewed and committed
- files changed: `spectral_detection_posttrain/experiments/artifacts.py`, `tests/experiments/test_artifact_manifest.py`, `tests/fixtures/artifacts/completed_manifest.json`, `tests/fixtures/artifacts/invalid_dirty_formal_manifest.json`
- RED command/result: `pytest tests/experiments/test_artifact_manifest.py -q` -> ModuleNotFoundError: `spectral_detection_posttrain.experiments.artifacts`
- GREEN command/result: same command -> 94 passed
- full-suite result: 712 passed (main-verified)
- reviewer and findings accepted/rejected: main-orchestrator review. Accepted judgment calls: (1) substring secret patterns (stricter than word-boundary); (2) `fail_manifest` stores `"<type>: <message>"` in `unavailable_reason` with reject-on-secret-hit; (3) reviewed manifests require `runtime_manifest_sha256` whenever completion != unavailable; (4) runtime manifests keep observation fields null until promotion; (5) `ObservationRecord` extended with expected hash / manifest_id / supersedes for testable binding; (6) fixtures generated through the serializer for byte-identical round-trip. Self-contained `EvaluationScope` in `artifacts.py` duplicates the contracts v1 type by design (disjoint write surfaces); unification happens at contracts v2 (Task 5).
- scientific artifacts checked: none touched; module defines provenance contracts only. Secret rejection, dirty-formal rejection, and reviewed append-only semantics are test-enforced.
- rollback command: `git revert <this-commit>`
- remaining risks: duplicate `EvaluationScope` definitions in `contracts.py` and `artifacts.py` until Task 5 reconciles exports; secret substring matching may need tuning if future legitimate fields contain the scanned substrings.

### 2026-07-13 — chore: inventory research scripts and entrypoints

- commit hash: pending (filled by next ledger entry)
- task ID: T9 (W1 parallel wave). Note: committed before T3, deviating from the Section 8 example order; no files were moved in this task, and the inventory is an input to T3/T7/T10 adjudication.
- agent/model and write owner: Kimi coder subagent (generator + inventory + tests); Kimi main orchestrator reviewed and committed
- files changed: `scripts/dev/inventory_scripts.py`, `spectral_detection_posttrain/configs/registry/script_inventory.json`, `tests/contracts/test_script_inventory.py`
- RED command/result: `pytest tests/contracts/test_script_inventory.py -q` -> 2 failed, 8 errors (missing inventory/APIs)
- GREEN command/result: same command -> 10 passed; generation run twice, sha256 identical: `4228098d6e0edf4fd52a308a68e93502c893291c516e887e89aed12c268a0cf1`; `--check` reports up to date
- full-suite result: 712 passed (main-verified)
- reviewer and findings accepted/rejected: main-orchestrator review. Accepted: 228 entries with exact git-tracked coverage (206 `scripts/` + 8 root `*.py` + 13 root launchers + generator self-inclusion); only 3 `reviewed` entries (`scripts/dev/*`, `scripts/round28_train_eval.py`); 225 `proposed` + `review_required` — adjudication deferred to the tasks that need them. Finding recorded for T3/T8: `det.dpo.smoke.001.json` references stale script paths (`scripts/round2129_nwpu_posttrain_smoke.py` untracked; `scripts/run_round2218_short_dpo_sweep.py` lives at `legacy/scripts/`).
- scientific artifacts checked: none touched; no script was moved, deleted, or executed.
- rollback command: `git revert <this-commit>`
- remaining risks: 225 proposed classifications await bounded adjudication before their archive wave (Task 16); scientific status must come from registries, not these heuristic labels.
