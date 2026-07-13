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

- commit hash: `fc20975`
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

### 2026-07-13 — feat: register versioned research experiments

- commit hash: `f620072`
- task ID: T3
- agent/model and write owner: Kimi coder subagent (registry + experiments.json + tests); Kimi main orchestrator reviewed and committed
- files changed: `spectral_detection_posttrain/experiments/registry.py`, `spectral_detection_posttrain/configs/registry/experiments.json`, `tests/experiments/test_experiment_registry.py`
- RED command/result: `pytest tests/experiments/test_experiment_registry.py -q` -> ModuleNotFoundError: `spectral_detection_posttrain.experiments.registry`
- GREEN command/result: same command -> 35 passed
- full-suite result: 747 passed (712 baseline + 35 new)
- reviewer and findings accepted/rejected: main-orchestrator review. Accepted judgment calls: (1) c2 and c1 scope `limited_unknown` on conflicting/absent scope evidence (non-formal, honest); (2) dense executable records point decision_document at the dense preregistration/spec docs rather than the line-level freeze doc; (3) uniform `gpu_policy: remote_gpu2_guarded`; (4) closed `EVIDENCE_KINDS` vocabulary (6 values); (5) top-level `{"schema_version": "experiments.v1", "records": [...]}` envelope. Mechanically verified: 21 records (13 executable + 8 historical), zero authorized, zero records on active/queued lines, all config/entrypoint/evidence/decision paths exist, historical records carry explicit null execution fields, runnables limited to diagnostic_only/frozen_reproduction_only per the frozen-line mapping. `det.energy.set_policy.m1.001` registered as historical_artifact per plan despite having a version config (committed evidence only, no `runs/` pointer).
- scientific artifacts checked: none touched; every required backfill ID (union of the 13 executable and 8 historical IDs) resolves. **No experiment is authorized by this refactor** — `can_dispatch` is False for the entire initial registry.
- rollback command: `git revert <this-commit>`
- remaining risks: scope determinations marked `limited_unknown` must never be read as formal; handler names are validated as a static set but implemented only in Task 7.

### 2026-07-13 — fix: record explicit evaluation scope

- commit hash: `dbff3a4`
- task ID: T5 (W2 parallel wave) + main serial v2 integration
- agent/model and write owner: Kimi coder subagent (contracts v2 + schema/metadata/round28 + tests); Kimi main orchestrator performed the serial v2 export integration (`experiments/__init__.py`) and the 4-test reconciliation
- files changed: `spectral_detection_posttrain/experiments/contracts.py` (v1->v2), `schema.py`, `metadata.py`, `scripts/round28_train_eval.py`, `tests/experiments/test_evaluation_scope.py`, `tests/test_round28_experiment_hygiene.py`, plus main-owned integration edits: `spectral_detection_posttrain/experiments/__init__.py` (explicit shared exports; contracts `EvaluationScope` is the canonical shared type), `tests/experiments/test_research_status.py` and `tests/experiments/test_experiment_registry.py` (helper scopes given a positive explicit limit to satisfy the contracts v2 smoke/limited rule)
- RED command/result: `pytest tests/experiments/test_evaluation_scope.py tests/test_round28_experiment_hygiene.py -q` -> ImportError on `CONTRACTS_VERSION` (scope contract absent)
- GREEN command/result: same command -> 31 passed; after main reconciliation the previously failing 4 committed tests (kind vocabulary loop + 3 registry helper-scope tests) pass: 99 passed across the four touched modules
- full-suite result: 817 passed (main-verified)
- reviewer and findings accepted/rejected: main-orchestrator review. Accepted judgment calls: (1) formal=True + missing scope normalizes to `limited_unknown` with `formal: false` + warning instead of raising (raising would break committed `test_experiment_schema.py`); (2) formal marker kept top-level (`evaluation_scope_formal`) so `config["evaluation_scope"]` stays a strict 4-key dict under re-validation; (3) `image_count` = post-limit evaluated val images; (4) the 4-test reconciliation edits minimal helper scopes rather than weakening the v2 rule. Builder-reported stop condition (4 committed tests red) resolved by main as documented above.
- scientific artifacts checked: one temporary synthetic 6-image CPU run produced `config.json` / `metadata.json` / `eval_metrics.json`; all three carry the scope-provenance block (`limited_unknown`, non-formal, explicit limits, normalization warning); existing metric keys/values unchanged (additive key only). Run artifacts deleted; no experiment authorized.
- rollback command: `git revert <this-commit>`
- remaining risks: `artifacts.py` still carries its own self-contained `EvaluationScope` (deliberate duplication; canonical shared export is the contracts v2 type — dedup deferred to a future cleanup pass); old configs without scope are forever non-formal.

### 2026-07-13 — docs: align repository with research registries

- commit hash: `92855f0`
- task ID: T8 (W2 parallel wave) + main serial README/AGENTS integration
- agent/model and write owner: Kimi coder subagent (deterministic generator + four generated docs + drift tests, returned proposals only); Kimi main orchestrator applied the README.md / AGENTS.md edits, reviewed, and committed
- files changed: `scripts/dev/generate_research_docs.py`, `docs/research/README.md`, `docs/research/current_status.md`, `docs/research/experiment_index.md`, `docs/research/artifact_index.md`, `tests/experiments/test_generated_research_docs.py`, plus main-owned `README.md`, `AGENTS.md`, `docs/research/refactor_ledger.md`
- RED command/result: `pytest tests/experiments/test_generated_research_docs.py -q` -> collection error (generator script and generated docs absent)
- GREEN command/result: same command -> 38 passed; generation run twice, sha256 identical across runs (`docs/research/README.md` `f7ad8d27…`, `current_status.md` `dbaad429…`, `experiment_index.md` `2b24d611…`, `artifact_index.md` `47bb9d8f…`); `--check` reports up to date
- full-suite result: 817 passed (main-verified after README/AGENTS edits)
- reviewer and findings accepted/rejected: main-orchestrator review of builder proposals J1-J7. Accepted: (J1) generated links use repo-relative link text with file-relative hrefs; (J2) optional AGENTS hunk adopted — the `Current Active Experiment Matrix (June 2026)` section replaced by `Research Status And GPU Policy` (registry authority, no authorized experiment, user GPU rules preserved: physical GPU2 only, `memory.free > 8192 MiB` post-peak, never interfere with other processes, no waiting once the gate passes); (J3) archive-path references now point to the tracked `legacy/` directory (`scripts/legacy/` and `docs/reports/archive/` are gitignore-only paths that do not exist in this worktree); (J4) all four generated files carry a unified header with generation-input sha256 block — accepted with the consequence that any registry change drifts all four files, so Task 10 must regenerate after every backfill family; (J5) (a) AGENTS.md line-7 framing prose left unchanged (registry and user wording govern scientific status), (b)(c) adopted (README remote block now `cd`s into `manifold-detection-energy-transport` without the stale PYTHONPATH export; test-suite note updated: full suite green in the maintained environment, scikit-learn optional legacy), (d) skipped; (J7) empty `configs/registry/artifacts/` renders an explicit empty state instead of fabricated rows. Mechanically verified: `current_status.md` line 10 is exactly `No experiment is currently authorized.`; Authorization Summary reads Active 0 / Queued 0 / Authorized 0; layout blocks no longer list nonexistent `defense/`, `segmentation/`, `pixel_classification/` paths.
- scientific artifacts checked: none touched; generator reads the two registry JSONs and the (empty) artifacts directory only. **No experiment is authorized by this refactor** — confirmed again by the generated Authorization Summary.
- rollback command: `git revert <this-commit>`
- remaining risks: registry edits now require a generator re-run in the same commit (drift test enforces); the stale-script-path finding for `det.dpo.smoke.001.json` (recorded in the T9 entry) is still open and scheduled for Task 18 adjudication.

### 2026-07-13 — docs: record reproducible execution environments

- commit hash: `c3652e6`
- task ID: T11 (W2 parallel wave)
- agent/model and write owner: Kimi coder subagent (read-only collector + snapshots + maintained prose + tests); Kimi main orchestrator reviewed and committed
- files changed: `scripts/dev/snapshot_environment.py`, `docs/research/environment.md`, `spectral_detection_posttrain/configs/registry/environments/local_windows_py310.json`, `spectral_detection_posttrain/configs/registry/environments/remote_gpu2_py310_cu121.json`, `tests/contracts/test_environment_snapshot.py`, `requirements.txt` (comments only)
- RED command/result: `pytest tests/contracts/test_environment_snapshot.py -q` -> collection error (collector and snapshots absent)
- GREEN command/result: same command -> 11 passed
- full-suite result: 817 passed (main-verified; T8 commit `92855f0` suite run)
- reviewer and findings accepted/rejected: main-orchestrator review. Verified mechanically: (1) remote snapshot redacts user-profile paths as `/home/<user>` and records host identity via `host_alias`; (2) GPU2 facts match the user's hard rules — RTX 4090, 48628 MiB free of 49140 MiB at capture, launch gate `memory.free > 8192 MiB` post-peak; (3) local snapshot records RTX 3070 Laptop, torch 2.1.0+cu121, scikit-learn 1.7.2 present; remote records scikit-learn absent; (4) `requirements.txt` diff is comments-only — no pin added, removed, or changed (torch/torchvision deliberately not repinned to a CUDA wheel URL); (5) `captured_at_utc` is format-validated but never equality-compared, so re-capture cannot break the suite.
- scientific artifacts checked: none touched; collector is read-only (interpreter version, installed distributions, `nvidia-smi` query). No remote writes, no package installs, no GPU work.
- rollback command: `git revert <this-commit>`
- remaining risks: snapshots are point-in-time observations; remote environment drift (driver, env packages) requires an explicit re-capture commit rather than silent regeneration.

### 2026-07-13 — docs: backfill critical experiment provenance

- commit hash: `b0a1d73`
- task ID: T10 family 1/6 (nwpu-mobile) + main integration fixes
- agent/model and write owner: Kimi coder subagent (generator + 15 manifests + 31 tests, "Luna" role); Kimi explore subagent ("Terra" provenance review of all 15); Kimi main orchestrator applied the review adjustments, regenerated docs/inventory, and owns the per-family commits
- files changed: `spectral_detection_posttrain/configs/registry/artifacts/nwpu_mob_strong_cosine_s42_bs8_36ep.json` (family 1 only; the other 14 manifests are held outside the worktree for their own family commits), `scripts/dev/backfill_artifact_manifests.py`, `tests/experiments/test_backfilled_artifacts.py`, regenerated `docs/research/{README,current_status,experiment_index,artifact_index}.md`, regenerated `spectral_detection_posttrain/configs/registry/script_inventory.json` (adds the 3 dev scripts from T8/T10/T11 — the T8/T11 drift is fixed here), `docs/research/environment.md` (path-alias section), `docs/research/refactor_ledger.md`
- RED command/result: `pytest tests/experiments/test_backfilled_artifacts.py -q` -> 21 failed, 1 passed, 9 errors (generator and manifests absent)
- GREEN command/result: same command with all 15 manifests present -> 31 passed; `test_artifact_manifest.py` + backfill -> 125 passed; backfill generator two runs byte-identical and `--check` clean; docs generator two runs byte-identical and `--check` clean
- full-suite result: 828 passed + 1 skipped; 18 failed + 9 errors, ALL inside `test_backfilled_artifacts.py` and ALL transitional: 14 parametrized contract tests for the held-out family manifests, 2 generator-vs-directory consistency tests, 2 required-set tests, 9 fixture-setup errors for the same missing 14. Zero failures anywhere else in the suite. This transitional red is the designed per-family commit state (plan Task 10 Step 5); it must shrink family by family and family 6 must be fully green.
- reviewer and findings accepted/rejected: Terra reviewed all 15 manifests read-only: every metric and gate float-exact against the bound evidence JSONs; all 15 primary result JSON sha256 verified; all 13 repo-relative refs re-hashed; redaction grep clean; claim ceilings pass (nothing reads as validated/promoted; M1 correctly shows a failed gate). Terra judgment-call verdicts 4a-4e all ACCEPT, including the `runtime_manifest_sha256` substitution (contract `_validate_reviewed` requires the field unconditionally for completed reviewed manifests; Terra recommends a future contract-docstring amendment, no data change). Adjustments applied by main before this commit: (1) strong-run git_commit provenance disclosure added to `missing_evidence`; (2) shift_audit null `image_count` disclosure added; (3) shared invocation clause softened to "where recorded" (7 `historical_artifact` records have null `legacy_entrypoint`); (4) parity pointer disclosure (c0 doc predates the runs; parity results live in the strong review doc); (5) `environment.md` gained the `remote:manifold` / `remote:rlimage` / `derived:` alias section. Main independently spot-checked the flagship manifest end-to-end over read-only SSH: remote sha256+size of `eval_metrics.json` / `config.json` / `checkpoint_best.pth` match the manifest; all 12 headline metrics are float-exact against the remote `eval_metrics.json`; run commit `dfaf494b8` is an ancestor of remote HEAD with subject "exp: add warmup cosine strong baseline".
- scientific artifacts checked: remote cross-check (read-only SSH, 2026-07-13): workspace `7e1f240…` clean; all 15 result JSONs, strong `config.json`, NWPU annotation JSON, and both verifiable checkpoints match their recorded hashes/sizes. No remote writes, no installs, no GPU use. **No experiment is authorized by this refactor** — manifests are provenance records, not run requests.
- rollback command: `git revert <this-commit>` (held-out manifests live outside the worktree and are unaffected)
- remaining risks: 14 manifests await their family commits from the hold directory; the transitional suite red (above) persists until family 6; `runtime_manifest_sha256` binding a result JSON (contract-required) needs a future docstring amendment to stop describing only the idealized path.

### 2026-07-13 — docs: backfill critical experiment provenance

- commit hash: `02c6ddb`
- task ID: T10 family 2/6 (zero-parity)
- agent/model and write owner: same pipeline as family 1 (coder builder, Terra review, main commit); no new builder output in this commit
- files changed: `spectral_detection_posttrain/configs/registry/artifacts/native_zero_parity_baseline.json`, `.../native_zero_parity_fullft.json`, regenerated `docs/research/{README,current_status,experiment_index,artifact_index}.md`, `docs/research/refactor_ledger.md`
- RED command/result: covered by the family-1 entry (required-artifact tests failed before any manifest existed)
- GREEN command/result: `pytest tests/experiments/test_artifact_manifest.py tests/experiments/test_generated_research_docs.py tests/contracts/test_script_inventory.py -q` -> 141 passed, 1 skipped; docs generator `--check` clean after regeneration
- full-suite result: 830 passed + 1 skipped; 16 failed + 9 errors, ALL transitional inside `test_backfilled_artifacts.py` (12 held-out manifests now; shrank by exactly 2 from family 1 as designed)
- reviewer and findings accepted/rejected: Terra verdicts for both parity manifests: ACCEPT. Verified: full_val/196 scope matches the strict-parity evidence (196 images, 0 mismatched); all 12 metrics float-exact vs the bound result JSONs (baseline sha `d4db4d9f…`, fullft sha `5324392a…`); `da54b19` equals the evidence `config.git_state.commit`; fullft ap75 `0.281522` matches the c0 doc native AP75. Disclosures carried: RLimage initial checkpoint unverifiable (outside permitted read scope, sha copied from evidence); c0 doc predates the runs (parity results reported in the strong review doc; metrics bind the hashed JSONs).
- scientific artifacts checked: remote cross-check already recorded in the family-1 entry covers both parity result JSONs; fullft initial checkpoint `950bc89a…` verified remotely (size 76205690). **No experiment is authorized by this refactor.**
- rollback command: `git revert <this-commit>`
- remaining risks: 12 manifests remain in the hold directory; transitional suite red persists until family 6.

### 2026-07-13 — docs: backfill critical experiment provenance

- commit hash: `a260d97`
- task ID: T10 family 3/6 (set-policy)
- agent/model and write owner: same pipeline as family 1 (coder builder, Terra review, main commit); no new builder output in this commit
- files changed: `spectral_detection_posttrain/configs/registry/artifacts/det.energy.set_policy.m1.001.json`, regenerated `docs/research/{README,current_status,experiment_index,artifact_index}.md`, `docs/research/refactor_ledger.md`
- RED command/result: covered by the family-1 entry
- GREEN command/result: `pytest tests/experiments/test_artifact_manifest.py tests/experiments/test_generated_research_docs.py tests/contracts/test_script_inventory.py -q` -> 141 passed, 1 skipped; docs generator `--check` clean after regeneration
- full-suite result: 831 passed + 1 skipped; 15 failed + 9 errors, ALL transitional inside `test_backfilled_artifacts.py` (11 held-out manifests now; shrank by exactly 1 from family 2 as designed)
- reviewer and findings accepted/rejected: Terra verdict for `det.energy.set_policy.m1.001`: ACCEPT (fully verified end-to-end). Verified: full_val/196 scope matches report section 5 and the run config `validation_images=196`; all 8 metrics and all 7 gates float-exact vs the local run JSON `runs/nwpu_m1_set_policy_s42_fulltrain_fullval/eval_metrics.json` (sha `9b089b66…` matches the report's recorded result SHA); `bf857497` equals the report + evidence commit; the manifest correctly shows `all_passed: false` with gate G2 failed — no promotion wording anywhere.
- scientific artifacts checked: remote cross-check recorded in the family-1 entry covers the M1 result JSON; local run directory hash independently matches. **No experiment is authorized by this refactor.**
- rollback command: `git revert <this-commit>`
- remaining risks: 11 manifests remain in the hold directory; transitional suite red persists until family 6.

### 2026-07-13 — docs: backfill critical experiment provenance

- commit hash: `0311589`
- task ID: T10 family 4/6 (native-actions)
- agent/model and write owner: same pipeline as family 1 (coder builder, Terra review, main commit); no new builder output in this commit
- files changed: `spectral_detection_posttrain/configs/registry/artifacts/det.energy.global_delta_u.c3b.balanced.001.json`, `.../det.energy.native_topology.d2b.001.json`, `.../det.energy.post_nms_suppress.e1.001.json`, regenerated `docs/research/{README,current_status,experiment_index,artifact_index}.md`, `docs/research/refactor_ledger.md`
- RED command/result: covered by the family-1 entry
- GREEN command/result: `pytest tests/experiments/test_artifact_manifest.py tests/experiments/test_generated_research_docs.py tests/contracts/test_script_inventory.py -q` -> 141 passed, 1 skipped; docs generator `--check` clean after regeneration
- full-suite result: 834 passed + 1 skipped; 12 failed + 9 errors, ALL transitional inside `test_backfilled_artifacts.py` (8 held-out manifests now; shrank by exactly 3 from family 3 as designed)
- reviewer and findings accepted/rejected: Terra verdict for all three native-actions manifests: ACCEPT. Spot-checks against `docs/autonomous_exploration_ledger.md`: d2b (commit `6a3634d`, artifact sha, identity/full APs, gates, 32-image smoke — all match); c3b (ledger "AP75 fell 0.008688" = manifest `-0.008687760351983542`); all three carry smoke scope with `limit_train/limit_val=32/32` and non-formal wording — no smoke-vs-full-val conflation.
- scientific artifacts checked: remote cross-check recorded in the family-1 entry covers all three result JSONs. **No experiment is authorized by this refactor.**
- rollback command: `git revert <this-commit>`
- remaining risks: 8 manifests remain in the hold directory; transitional suite red persists until family 6.

### 2026-07-13 — docs: backfill critical experiment provenance

- commit hash: `60bdbd1`
- task ID: T10 family 5/6 (dense-endpoint)
- agent/model and write owner: same pipeline as family 1 (coder builder, Terra review, main commit); no new builder output in this commit
- files changed: `spectral_detection_posttrain/configs/registry/artifacts/det.energy.dense_endpoint.{absolute,geometry_control,cleanval,shift_audit}.001.json`, regenerated `docs/research/{README,current_status,experiment_index,artifact_index}.md`, `docs/research/refactor_ledger.md`
- RED command/result: covered by the family-1 entry
- GREEN command/result: `pytest tests/experiments/test_artifact_manifest.py tests/experiments/test_generated_research_docs.py tests/contracts/test_script_inventory.py -q` -> 141 passed, 1 skipped; docs generator `--check` clean after regeneration
- full-suite result: 838 passed + 1 skipped; 8 failed + 9 errors, ALL transitional inside `test_backfilled_artifacts.py` (4 held-out manifests now; shrank by exactly 4 from family 4 as designed)
- reviewer and findings accepted/rejected: Terra verdict for all four dense-endpoint manifests: ACCEPT (one optional nit, resolved by main in the family-1 generator adjustments: shift_audit null `image_count` now carries an explicit disclosure). Spot-checks against `docs/autonomous_exploration_ledger.md`: absolute (`52c82685`, artifact `d0e4af…`, nested train-only protocol); cleanval (`b0cc5428`, val MAE `1.76984` / pairwise `0.80926` / AUROC `0.93580`, gap-gate failure recorded); scopes stay train_only / detector_unseen / cache_only — nothing reads as full-val.
- scientific artifacts checked: remote cross-check recorded in the family-1 entry covers all four result JSONs. **No experiment is authorized by this refactor.**
- rollback command: `git revert <this-commit>`
- remaining risks: 4 manifests remain in the hold directory; transitional suite red persists until family 6.

### 2026-07-13 — docs: backfill critical experiment provenance

- commit hash: `a0ed29e`
- task ID: T10 family 6/6 (dense-local-delta) — final family; closes Task 10
- agent/model and write owner: same pipeline as family 1 (coder builder, Terra review, main commit); no new builder output in this commit
- files changed: `spectral_detection_posttrain/configs/registry/artifacts/det.energy.dense_local_delta_{stats.002,learner.001,family_audit.001,family_prior.001}.json`, regenerated `docs/research/{README,current_status,experiment_index,artifact_index}.md`, `docs/research/refactor_ledger.md`
- RED command/result: covered by the family-1 entry
- GREEN command/result: backfill generator `--check` reports up to date with all 15 manifests present; `pytest tests/experiments/test_backfilled_artifacts.py tests/experiments/test_artifact_manifest.py -q` -> 125 passed
- full-suite result: **855 passed + 1 skipped, zero failures/errors** — the transitional red is fully cleared (855 = 817 baseline + 31 backfill + 8 lifecycle tests from the concurrently developed, not-yet-committed Task 6 files; skip is the generated-docs empty-state check, inapplicable now that reviewed manifests exist)
- reviewer and findings accepted/rejected: Terra verdicts for all four dense-local-delta manifests: ACCEPT. Spot-checks against `docs/autonomous_exploration_ledger.md`: learner (`58908126`, 0.08585 vs 0.10997, pairwise `0.57469` below the 0.60 gate — failure recorded, not hidden); family_prior (`d81b27a6`, 0.67241/0.07722); bundle-commit provenance for family_audit (`923db6a7…`) and family_prior (`d81b27a6…`) mechanically verified via `git bundle list-heads`, both ancestors of the observed revision. Scopes stay train_only / limited(48/16) / cache_only — non-formal, no claim inflation.
- scientific artifacts checked: remote cross-check recorded in the family-1 entry covers all four result JSONs and both evidence bundles. Hold directory is empty; all 15 manifests are now committed across the six family commits. **No experiment is authorized by this refactor.**
- rollback command: `git revert <this-commit>`
- remaining risks: none transitional left; standing risks from earlier entries apply (contract-docstring amendment for `runtime_manifest_sha256`; `det.dpo.smoke.001.json` stale script paths await Task 18 adjudication).

### 2026-07-13 — feat: track canonical experiment lifecycle

- commit hash: pending (filled by next ledger entry)
- task ID: T6 (developed in parallel with the T10 per-family commits; committed after them so its gate is the fully green post-T10 suite)
- agent/model and write owner: Kimi coder subagent (lifecycle module + hooks + tests); Kimi main orchestrator reviewed the diff and committed
- files changed: `spectral_detection_posttrain/experiments/canonical_runner.py` (additive: lifecycle import, `ExperimentContext.manifest_path` default field, precondition check + manifest start inside `prepare_experiment_from_config`; metadata collection reordered before `mkdir` so a formal dirty tree is rejected before any run-directory mutation; signatures unchanged), `spectral_detection_posttrain/experiments/lifecycle.py` (new), `tests/experiments/test_experiment_lifecycle.py` (new, 8 tests), `tests/test_canonical_runner_e2e.py` (additive manifest assertions)
- RED command/result: `pytest tests/experiments/test_experiment_lifecycle.py tests/test_canonical_runner_e2e.py -q` -> ImportError on `fail_experiment`; E2E 1 failed (`manifest.json` not written by prepare)
- GREEN command/result: `pytest tests/experiments/test_experiment_lifecycle.py tests/test_canonical_runner.py tests/test_canonical_runner_e2e.py -q` -> 12 passed; touched-module gate (scope + artifact manifest + schema + metadata + canonical + lifecycle) -> 135 passed
- full-suite result: 855 passed + 1 skipped (the post-T10 fully green suite, which includes these files)
- reviewer and findings accepted/rejected: main-orchestrator review of the diff. Verified directly: `check_formal_run_preconditions` keys on `evaluation_scope_formal AND git_dirty` only, so non-formal callers (including `limited_unknown` scope-less configs) are unaffected; the `mkdir` reorder is the only behavioral change and matches plan Step 1's "rejected before run directory mutation". Accepted judgment calls: (1) formal = explicit-scope marker, not the bare `formal` flag (keying on `formal` alone would break committed canonical-runner tests on any dirty worktree); (2) explicit missing-input-hash rejection (`_require_input_hashes`); (3) `LifecycleError` subclasses `ManifestValidationError` so both raise surfaces work; (4) sanitization reuses artifacts' `_SECRET_PATTERNS` (private import, documented) so stored failure reasons always pass the manifest scrubber — whole-string replacement on any pattern hit; (5) no `experiments/__init__.py` export changes (not an owned file; callers import from `lifecycle`/`canonical_runner` directly — a future export pass may add them). Backward compatibility asserted by tests: callers that never finalize leave a `started` manifest; no tracked reviewed manifest is ever written by this path.
- scientific artifacts checked: none touched; runtime `manifest.json` is written only under the ignored run directory. **No experiment is authorized by this refactor.**
- rollback command: `git revert <this-commit>`
- remaining risks: existing production callers are non-formal until they adopt `finalize_experiment`; checkpoint logical_path uses the basename to keep absolute temp paths out of the secret scrubber (documented in-code).
