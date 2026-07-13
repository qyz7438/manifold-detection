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

- commit hash: `c405ea1`
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

### 2026-07-13 — docs: define energy transport package boundaries

- commit hash: `3f69594`
- task ID: T12 (developed in parallel with T7; disjoint file surfaces)
- agent/model and write owner: Kimi coder subagent (ADR + AST boundary test + decision index row); Kimi main orchestrator reviewed research semantics and committed
- files changed: `docs/decisions/adr-energy-transport-package-boundaries.md` (new), `tests/contracts/test_energy_transport_import_boundaries.py` (new, 11 tests), `docs/decisions/index.md` (one index row appended — plan-mandated modification; the T0 protected-history inventory records the pre-change blob, and the diff is exactly one line)
- RED command/result: boundary test absent at HEAD; the ADR/test pair is documentation-plus-contract, so the meaningful RED is the classifier exercised out-of-band by the builder: 12 hypothetical new violations (trainer/experiment/dataset/scripts/runs/legacy imports, native->policy, action->policy, endpoint->core, cross-family, third-party) all flagged; 9 allowed cases all clean; stale-allowlist detection verified
- GREEN command/result: `pytest tests/contracts/test_energy_transport_import_boundaries.py -q` -> 11 passed (main-verified)
- full-suite result: 866 passed + 1 skipped = 855 baseline + 11 new, zero failures (run with `--ignore=tests/experiments/test_dispatcher.py` — that untracked file belongs to the in-flight T7 builder and errors at collection until T7 lands; no relation to this task)
- reviewer and findings accepted/rejected: main-orchestrator review per plan Step 4 (boundaries must preserve research semantics). Verified: the ADR reproduces the plan's 5-subpackage mapping and 5 dependency rules verbatim and moves no implementation code; only 2 allowlist violations exist (`adaptive_consensus.py` and `set_search.py` importing `spectral_detection_posttrain.core.matching` directly), both assigned to Task 14 for removal; the ratchet (stale allowlist entries fail) forces ADR+allowlist cleanup when Task 14 fixes them. Accepted interpretation judgments: (1) the "not" clause is the enforced prohibition and `core.*` access is restricted to action/native — this is what produces the 2 entries; (2) diagnostics = leaf layer over the other four subpackages; (3) endpoint = self-contained (dense_set_energy teacher) + substrate; (4) side-effect rule enforced at import time only, function bodies and `__main__` blocks exempt; (5) imports of future `energy_transport.<subpkg>.*` shim paths are classified by target subpackage so Tasks 13/14 do not trip the rules.
- scientific artifacts checked: none touched; no module was moved or edited. **No experiment is authorized by this refactor.**
- rollback command: `git revert <this-commit>`
- remaining risks: 2 allowlist entries must be cleared by Task 14 policy migration (stale-entry test will fail until allowlist and ADR table are emptied together); Task 14 Step 3 adds the stricter import-safety gate (no reads/mkdir/CUDA at import) not covered here.

### 2026-07-13 — feat: add explicit experiment dispatcher

- commit hash: `b6d3628`
- task ID: T7
- agent/model and write owner: Kimi coder subagent (dispatcher + handlers + CLI + 65 tests); Kimi main orchestrator reviewed the guards, regenerated the script inventory, and committed
- files changed: `spectral_detection_posttrain/experiments/dispatcher.py`, `spectral_detection_posttrain/experiments/handlers/{__init__,standard_detection,native_contract,dense_endpoint}.py`, `scripts/run_experiment.py`, `tests/experiments/test_dispatcher.py`, regenerated `spectral_detection_posttrain/configs/registry/script_inventory.json` (adds `scripts/run_experiment.py`), `docs/research/refactor_ledger.md`
- RED command/result: `pytest tests/experiments/test_dispatcher.py -q` -> ModuleNotFoundError: `spectral_detection_posttrain.experiments.dispatcher` (collection error)
- GREEN command/result: same command -> 65 passed
- full-suite result: 931 passed + 1 skipped (855 baseline + 65 dispatcher + 11 T12 boundary tests integrated mid-task)
- reviewer and findings accepted/rejected: main-orchestrator review. Guards verified directly: `run` on a historical artifact exits 1 with a clear refusal; frozen dry-run without `--allow-frozen-reproduction` exits 1; no run directory is created by dry-run; `status` reports active 0 / queued 0 / authorized 0 and prints "No experiment is currently authorized."; a static source-scan test forbids `importlib`/`__import__`/`eval`/`exec` and dataset/torch/canonical_runner imports in dispatcher+handlers. Accepted judgment calls: (1) `ExperimentCapability` + exceptions live in `handlers/__init__.py` with concrete handlers imported after base definitions (cycle-free DAG; registry types TYPE_CHECKING-only); (2) "exact command" renders `python <legacy_entrypoint>` only, because frozen entrypoints hardcode their locked config with an internal sha256 check — fabricating flags would be dishonest; (3) the frozen flag gates dry-run too, per plan section 3.2; (4) `DispatchRefusedError` subclasses `DispatchError` so both guard surfaces hold; (5) no `experiments/__init__.py` export changes (main-owned surface; tests import the submodule directly — a future export pass may add them). The four required verification dry-runs all render `authorized: no` (C1 `det.energy.native_listwise.c1.001`, c3b balanced, dense absolute, family prior).
- scientific artifacts checked: none touched; dry-run only hashes the locked config JSON. **No experiment is authorized by this refactor** — `run` is gated on `registry.can_dispatch`, currently false for all 21 records.
- rollback command: `git revert <this-commit>`
- remaining risks: `run` capability is unexercised by design (zero authorized records); when a record is ever authorized, its handler's `run` path needs a fresh review gate before first use.

### 2026-07-13 — refactor: isolate energy transport action core

- commit hash: `a12684f`
- task ID: T13 phase 1 of 2 (action core; native phase follows as a separate commit per plan serialization)
- agent/model and write owner: Kimi coder subagent (parity tests + fixture + byte-identical moves + shims); Kimi main orchestrator reviewed the diff summary and committed
- files changed: moved byte-identical into `spectral_detection_posttrain/methods/energy_transport/action/`: `actions.py`, `contracts.py`, `operators.py`, `preferences.py`, `geometric_constraints.py` (sizes verified identical pre/post move); new `action/__init__.py` (imports only from own submodules); the 5 flat modules rewritten as pure forwarding shims (docstring + imports + `__all__`, zero function bodies); `energy_transport/__init__.py` import-source lines only (143-name public `__all__` unchanged); new `tests/compatibility/test_energy_transport_flat_shims.py` (72 tests); new `tests/fixtures/checkpoints/action_local_transport_head_seed42.pt` (5,490-byte synthetic CPU state_dict, seed 42) + `energy_transport_checkpoint_manifest.json` (sha256-locked, provenance explicitly SYNTHETIC T13 parity fixture, not historical weights; 4 non-serializable action modules listed in `skipped` with reasons); `.gitignore` (main-owned: appended `!tests/fixtures/checkpoints/` + `!tests/fixtures/checkpoints/**` negations — the pre-existing `checkpoints/` and `*.pt` rules otherwise excluded the plan-mandated fixture files); `scripts/dev/validate_repository_state.py` (main-owned: exact-path `ALLOWED_FIXTURE_PATHS` allowlist so the T1 runtime-artifact contract admits only the named sha256-pinned fixture file — deliberately an exact path, not a directory prefix); `docs/research/refactor_ledger.md`
- RED command/result: `pytest tests/compatibility/test_energy_transport_flat_shims.py -q` before the subpackage existed -> 72 failed, `ModuleNotFoundError: spectral_detection_posttrain.methods.energy_transport.action`
- GREEN command/result: same command -> 72 passed; T12 contract `tests/contracts/test_energy_transport_import_boundaries.py` still green (shim imports of `energy_transport.action.*` classify action->action, allowed)
- full-suite result: 1003 passed + 1 skipped = 931+1 baseline + 72 new
- reviewer and findings accepted/rejected: main-orchestrator review of builder report. Accepted judgment calls: (1) moved files kept strictly byte-identical, so `preferences.py`/`geometric_constraints.py` still import through the flat shims (safe: neither has intra-package deps; internalizing those imports would violate the no-body-edit constraint); (2) shims re-export the full module public surface (incl. `SAMPLE_*` constants and non-facade names such as `bbox_aware_direct_loss` used by direct-importing tests), pinned by an `__all__` drift test; (3) shim purity enforced by an AST test (no function bodies); (4) fixture manifest enforced by `test_manifest_covers_every_action_module` so no action module can be silently dropped. Consumer sweep: no direct importers of the 5 moved modules in `scripts/` or `experiments/handlers/`; in-package consumers (`candidate_energy`, `benefit_energy`, `high_water_mark`, `native_topology`, `global_top1`) import via flat shims and were verified importable post-move.
- scientific artifacts checked: none touched; the fixture is synthetic and labeled as such. **No experiment is authorized by this refactor.**
- rollback command: `git revert <this-commit>`
- remaining risks: native phase (T13 steps 4-6) still pending; `action/` internal imports through flat shims could be internalized in a later cleanup pass (would require body edits, out of scope now).

### 2026-07-13 — refactor: isolate native detection transport

- commit hash: `500169e`
- task ID: T13 phase 2 of 2 (native detection transport; completes the Task 13 mapping for action+native)
- agent/model and write owner: Kimi coder subagent (parity tests + fixtures + byte-identical moves + shims, resumed agent from phase 1); Kimi main orchestrator extended the validator fixture allowlist, verified fixture sha256 against the manifest independently, and committed
- files changed: moved byte-identical into `spectral_detection_posttrain/methods/energy_transport/native/`: `native_contract.py`, `native_topology.py`, `candidate_energy.py`, `benefit_energy.py` (sizes verified identical pre/post move); new `native/__init__.py` (own submodules only); the 4 flat modules rewritten as pure forwarding shims; `energy_transport/__init__.py` 3 import-source lines only (143-name public `__all__` unchanged); extended `tests/compatibility/test_energy_transport_flat_shims.py` (120 tests total, +48 net new); 3 new synthetic seeded fixtures under `tests/fixtures/checkpoints/` (`spatial_candidate_energy_head_seed43.pt` 30,698 B, `context_only_candidate_energy_head_seed44.pt` 5,148 B, `action_benefit_energy_head_seed45.pt` 7,972 B — all sha256-verified by main against the manifest); manifest updated (4 fixtures + 6 skipped with reasons, provenance still explicitly SYNTHETIC); `scripts/dev/validate_repository_state.py` (`ALLOWED_FIXTURE_PATHS` +3 exact paths, comment relaxed from "a few KB" to "tens of KB" — the 30.7 KB spatial fixture is the module's architecture floor with hardcoded conv channels); `docs/research/refactor_ledger.md`
- RED command/result: `pytest tests/compatibility/test_energy_transport_flat_shims.py -q` before the subpackage existed -> 51 failed, `ModuleNotFoundError: spectral_detection_posttrain.methods.energy_transport.native`
- GREEN command/result: same command -> 120 passed; T12 boundary test and all pre-existing native contract/topology/candidate/benefit tests green unchanged
- full-suite result: 1051 passed + 1 skipped = 1003+1 phase-1 baseline + 48 net new (run AFTER staging so the T1 validator sees the fixtures via `git ls-files`)
- reviewer and findings accepted/rejected: main-orchestrator review. Behavioral coverage per plan verified in the diff report: zero-action parity helpers + contract gates, class expansion, BoxCoder decode, small-box filtering, class-wise NMS, normalized-rank ordering, top-K cap, seeded forward parity for all 3 heads. Accepted judgment calls: (1) `native_topology` shim additionally imports `apply_box_delta` (excluded from `__all__`, commented) because the read-only existing test `tests/test_energy_transport_native_topology.py:137` patches that attribute path — the identity-candidate path never calls it; (2) `native_contract` stays out of the facade `__all__` as before, 7 known direct consumers keep importing the flat shim; (3) spatial fixture is 30.7 KB (architecture floor) — accepted with exact-path allowlist entry rather than weakening coverage; (4) moved files keep their original absolute imports of the flat action shims per the byte-identical rule; identity tests prove they resolve to the canonical `action.*` objects.
- scientific artifacts checked: none touched; fixtures synthetic and labeled. **No experiment is authorized by this refactor.**
- rollback command: `git revert <this-commit>`
- remaining risks: policy/endpoint/diagnostics migrations (Task 14) still pending, incl. clearing the 2 T12 allowlist entries; the `apply_box_delta` patch-compat import should move to the canonical `native.*` path when the owning test is next edited (out of scope: read-only existing test).

### 2026-07-13 — refactor: isolate energy transport policies

- commit hash: `9b437c9`
- task ID: T14 phase 1 of 3 (policy; endpoint and diagnostics follow as separate commits per plan serialization)
- agent/model and write owner: Kimi coder subagent (parity tests + fixtures + moves + shims + allowlist cleanup; first attempt timed out with no persisted changes, resumed and completed); Kimi main orchestrator verified byte-identity of all moves, normalized the manifest to LF, extended the validator fixture allowlist, and committed
- files changed: moved into `spectral_detection_posttrain/methods/energy_transport/policy/`: `search.py`, `set_policy.py`, `global_top1.py`, `listwise_noop.py`, `joint_delta_u.py`, `post_nms_suppress.py` (6 byte-identical, main-verified against HEAD blobs) plus `set_search.py` and `adaptive_consensus.py` with exactly one sanctioned import-line edit each (diff-verified by main: only the `core.matching` import line changed); new `policy/__init__.py`; the 8 flat modules rewritten as pure forwarding shims; `energy_transport/__init__.py` 7 import-source lines only (143-name public `__all__` unchanged); `native/native_contract.py` gained 2 sanctioned re-exports (`box_iou`, `match_predictions_to_gt` from `core.matching`) with a comment citing Task 14/ADR; `tests/contracts/test_energy_transport_import_boundaries.py` allowlist emptied (typed empty dict, ratchet comment); `docs/decisions/adr-energy-transport-package-boundaries.md` violation table converted to historical with resolutions, stale sentence corrected; extended `tests/compatibility/test_energy_transport_flat_shims.py` (241 tests total); 8 new synthetic seeded fixtures under `tests/fixtures/checkpoints/` (seeds 51-58, all sha256-verified by main against the manifest); manifest normalized to LF by main (builder's Windows heredoc had regressed the worktree file to pure CRLF; the T13 git blob was always LF); `scripts/dev/validate_repository_state.py` (`ALLOWED_FIXTURE_PATHS` +8 exact paths); `docs/research/refactor_ledger.md`
- RED command/result: `pytest tests/compatibility/test_energy_transport_flat_shims.py -q` before the subpackage existed -> 123 failed, `ModuleNotFoundError: spectral_detection_posttrain.methods.energy_transport.policy`
- GREEN command/result: same command + focused policy/boundary set (335 tests) passed; `pytest tests/contracts/test_energy_transport_import_boundaries.py -q` -> 11 passed with `ALLOWLIST: {}` and zero violations found
- full-suite result: 1172 passed + 1 skipped = 1051+1 baseline + 121 net new
- reviewer and findings accepted/rejected: main-orchestrator review with independent diff/hash verification. Accepted judgment calls: (1) routing solution — 2 re-exports placed in `native/native_contract.py` (native_topology was unsuitable: it already imports torchvision's `box_iou`); `core/matching` has no back-imports, no cycle; (2) `global_top1.py`/`post_nms_suppress.py` keep their flat shim imports per the byte-identical rule; (3) `SetEvaluator` public type alias included in the shim surface as cheap insurance; (4) one test-only input fix (`set_policy_loss` parity input `move_targets` row 0->1) because the loss validator rejects identity action targets — no production code touched; (5) ADR edits slightly beyond "violation table only" (correcting the now-false "no native module uses this grant" sentence and one tense) accepted as required for accuracy; (6) builder's `joint_delta_u.py` size-discrepancy flag investigated by main: HEAD blob and moved file are byte-identical at 26,234 B, the earlier figure was a stale measurement from the timed-out first attempt. Builder's claim that the manifest CRLF was "pre-existing from T13" was FALSE — the T13 git blob is LF; the CRLF was introduced by the builder's own generation heredoc this session. Corrected by main before staging.
- scientific artifacts checked: none touched; fixtures synthetic and labeled. **No experiment is authorized by this refactor.**
- rollback command: `git revert <this-commit>`
- remaining risks: endpoint and diagnostics migrations (Task 14 phases 2-3) pending, incl. the stricter import-safety gate for diagnostics; facade eager-import reduction is Task 14 Step 4; the `native_contract` re-export route should be revisited if `core.matching` ever grows policy-inappropriate surface.


### 2026-07-13 — refactor: isolate dense set endpoints

- commit hash: `8ae3f33`
- task ID: T14 phase 2 of 3 (endpoint; diagnostics follows as the final migration commit)
- agent/model and write owner: Kimi coder subagent (parity tests + fixture + byte-identical moves + shims); Kimi main orchestrator verified move byte-identity and fixture sha256, extended the validator fixture allowlist, and committed
- files changed: moved byte-identical into `spectral_detection_posttrain/methods/energy_transport/endpoint/`: `dense_set_energy.py` (9,894 B) and `dense_endpoint.py` (6,815 B) — main-verified identical to HEAD blobs, zero body edits (`dense_endpoint.py` keeps its flat sibling import, resolved via shim per the byte-identical rule); new `endpoint/__init__.py` (9 names, standalone-endpoint layering docstring); the 2 flat modules rewritten as pure forwarding shims; `energy_transport/__init__.py` 2 import-source lines only (143-name public `__all__` unchanged); extended `tests/compatibility/test_energy_transport_flat_shims.py` (260 tests total); 1 new synthetic seeded fixture `dense_set_energy_endpoint_seed59.pt` (sha256-verified by main; manifest now 13 fixtures + 11 skipped, written LF-only this time — no CRLF regression); `scripts/dev/validate_repository_state.py` (`ALLOWED_FIXTURE_PATHS` +1 exact path); `docs/research/refactor_ledger.md`
- RED command/result: `pytest tests/compatibility/test_energy_transport_flat_shims.py -q` before the subpackage existed -> 21 failed, `ModuleNotFoundError: spectral_detection_posttrain.methods.energy_transport.endpoint`
- GREEN command/result: focused set (parity + boundary + dense endpoint/set-energy/teacher audit + dispatcher + backfilled-artifact tests, 378 tests) passed; T12 boundary test green unchanged (endpoint mapping and path already in the ADR from T12 — no ADR edit needed)
- full-suite result: 1191 passed + 1 skipped = 1172+1 baseline + 19 net new
- reviewer and findings accepted/rejected: main-orchestrator review with independent diff/hash verification. Plan-required behaviors covered per builder diff: teacher component values, permutation invariance via canonical sorting (pred and GT shuffles), state-dict key equality, strict old-path loading, `robust_scalar_summary`, `reduced_teacher_values`, `RobustTeacherStats.fit/.quality`, `build_sparse_pair_features`, seeded endpoint forward parity. Accepted judgment calls: (1) no ADR edit this phase — boundary cross-check passes as-is; (2) only one fixture — `DenseSetEnergyEndpoint` is the sole nn.Module across the two modules, `dense_set_energy` recorded in manifest `skipped` with reason.
- scientific artifacts checked: none touched; fixture synthetic and labeled. **No experiment is authorized by this refactor.**
- rollback command: `git revert <this-commit>`
- remaining risks: diagnostics migration (Task 14 phase 3) pending, incl. the stricter import-safety gate (no reads/mkdir/CUDA at import) and the plan Step 4 facade eager-import reduction; Kimi full-package review of all three Task 14 diffs follows phase 3.


### 2026-07-13 — refactor: isolate transport diagnostics

- commit hash: `df2bc4c`
- task ID: T14 phase 3 of 3 (diagnostics; completes the Task 14 package migrations; plan Step 4 facade eager-import reduction and Step 5 full verification follow)
- agent/model and write owner: Kimi coder subagent (parity tests + import-safety gate + moves + shims; first attempt timed out after persisting the migration, resumed and closed out verification); Kimi main orchestrator verified every move diff against HEAD blobs and the manifest sha256, and committed
- files changed: moved into `spectral_detection_posttrain/methods/energy_transport/diagnostics/`: 9 modules — `cone_projection.py`, `linear_identifiability.py`, `step_strata.py`, `top_focused_audit.py`, `decomposed_actionability.py` byte-identical (main-verified), plus 4 with ONLY sanctioned import-line edits (main diff-verified): `structure_metrics.py` (1 line: sibling import -> `diagnostics.cone_projection`), `high_water_mark.py` (2 lines: -> `action.actions` / `action.contracts`), `spatial_counterfactual.py` (1 line: -> `diagnostics.linear_identifiability`), `joint_probe_validation.py` (the relative-import trap: `from .joint_delta_u` -> absolute `policy.joint_delta_u`); new `diagnostics/__init__.py` (86 public names, leaf-layer docstring); the 9 flat modules rewritten as pure forwarding shims; `energy_transport/__init__.py` 4 import-source lines only (143-name public `__all__` unchanged); `tests/contracts/test_energy_transport_import_boundaries.py` extended with the plan-mandated import-safety gate: static dotted-name module-level call walker (forbids import-time reads/writes/`torch.cuda.*`, function bodies and `__main__` exempt, scans flat + diagnostics paths) plus a runtime subprocess probe (imports all 9 modules + 9 shims in an empty tmp cwd, asserts no CUDA init and no new files); `tests/compatibility/test_energy_transport_flat_shims.py` extended (86-name diagnostics parity + section 3e deterministic-output tests); manifest appended with 9 `skipped` entries (no nn.Module anywhere in diagnostics — AST-confirmed, so zero new fixtures and no validator allowlist change); `docs/research/refactor_ledger.md`
- RED command/result: `pytest tests/compatibility/test_energy_transport_flat_shims.py tests/contracts/test_energy_transport_import_boundaries.py -q --tb=no` before the subpackage existed -> 186 failed (`ModuleNotFoundError: ...energy_transport.diagnostics`); the new static import-safety gate passed pre-move by design (zero module-level calls in all 9 modules today)
- GREEN command/result: focused set (parity + boundary + 8 diagnostics test files + 4 nwpu script tests, 522 tests) passed; boundary contract 12 tests green with the extended gate
- full-suite result: 1376 passed + 1 skipped = 1191+1 baseline + 185 net new
- reviewer and findings accepted/rejected: main-orchestrator review with independent diff/hash verification of all 9 moves (5 byte-identical, 4 import-line-only — exactly matching the builder's declared edit list). Accepted judgment calls: (1) cone_projection shim surface = 11 names incl. non-facade `normalize_l2`/`project_to_tangent`/`EnergyFn` because scripts import them from the flat module; (2) parity call coverage kept at 2-3 deterministic calls per module with identity/signature parametrization covering all 86 names; (3) `load` short-name in the static gate is deliberately strict (no module-level `*.load()` at all); (4) subprocess probe timeout 180s for cold torch import. Side note: the 26,598 B figure flagged as a stale measurement in T14a turns out to be structure_metrics' HEAD blob size — the builder had confused the two files; both are consistent with HEAD.
- scientific artifacts checked: none touched; no new fixtures. **No experiment is authorized by this refactor.**
- rollback command: `git revert <this-commit>`
- remaining risks: Task 14 Step 4 (facade eager-import reduction with optional-deps-absent import smoke) and Step 5 (full verification incl. native zero-action parity command against the locked fixture) still pending; then Kimi full-package review of the complete Task 14 diff (all three migration commits) before Task 15.


### 2026-07-13 — refactor: make energy_transport facade lazy with PEP 562

- commit hash: `636e31b`
- task ID: T14 Steps 4-5 (facade eager-import reduction + full verification; completes Task 14)
- agent/model and write owner: Kimi coder subagent (lazy facade + smoke tests); Kimi main orchestrator verified `__all__` byte-identity and mapping completeness, and committed
- files changed: `spectral_detection_posttrain/methods/energy_transport/__init__.py` rewritten from eager imports to PEP 562 lazy loading — docstring + stdlib `importlib` only at import time, static `_SYMBOL_MODULES` mapping (143 names -> 21 defining subpackage modules), `__getattr__` with globals() caching, `__dir__` returning sorted `__all__`; `__all__` block byte-identical to HEAD (main-verified via AST: same 143 names, same order, no duplicates; mapping covers all 143 exactly once); new `tests/compatibility/test_energy_transport_facade_lazy.py` (6 tests: `__all__` uniqueness + `dir()` consistency, AST assertion of zero eager package imports, mapping resolution + object identity, `from ... import` form + caching, AttributeError on unknown names, subprocess smoke with `sys.modules['sklearn']=None` proving facade insulation and zero subpackage modules loaded at import); `docs/research/refactor_ledger.md`
- RED command/result: lazy-facade tests absent at HEAD; the meaningful RED is the eager-import AST assertion, which fails against the pre-change facade (175/185-line diff)
- GREEN command/result: `pytest tests/compatibility/test_energy_transport_facade_lazy.py tests/compatibility/test_energy_transport_flat_shims.py -q` -> all passed; the pre-existing `test_facade_reexports_match_new_subpackage` (all 143 names via getattr) passes unchanged, proving both import forms keep working
- full-suite result: 1382 passed + 1 skipped = 1376+1 baseline + 6 new
- reviewer and findings accepted/rejected: main-orchestrator review. Optional-dependency finding accepted: no facade-reachable module imports sklearn or any uninstalled optional dep (only `torchvision.ops`, an installed hard dependency already in the ADR base-allowed set); sklearn is confined to legacy raw-iFFT verifier modules unreachable from the facade, so the smoke proves insulation with sklearn blocked. Plan Step 5 native zero-action parity: canonical command `python scripts/verify_nwpu_native_c1_contract.py` is CPU-only but its locked config references three sha256-locked sources under `runs/` that do not exist in this worktree (`runs/` is untracked local state) — NOT run; substituted with the equivalent CPU coverage `pytest tests/test_action_zero_parity.py tests/test_nwpu_native_c1_contract_script.py` (9 passed), which exercises the same locked config + `build_contract_report` path (strict zero-action parity artifact, exact-C1 candidate contract, budget-1 policy gates). Accepted judgment calls: mapping produced by a generator with bidirectional assertions rather than hand-written; subpackage `__init__.py` files stay eager per instruction; no existing test modified. Task 14 complete-package review: all four commits (9b437c9 policy, 8ae3f33 endpoint, df2bc4c diagnostics, this one) reviewed per-phase with independent diff/hash verification; the 5-subpackage ADR mapping is now fully realized on disk, the allowlist is empty with the ratchet active, and the import-safety gate covers diagnostics at both static and runtime levels.
- scientific artifacts checked: none touched. **No experiment is authorized by this refactor.**
- rollback command: `git revert <this-commit>`
- remaining risks: lazy facade changes import timing — code that relied on `import energy_transport` transitively importing a specific subpackage module (rather than a public name) would now need an explicit import; the parity suite covers the known consumers and is green. `from energy_transport import <flat_shim_submodule>` still works via normal submodule import fallback (verified by the AttributeError-on-unknown test only covering non-module names).


### 2026-07-13 — refactor: add structured detection trainer protocols

- commit hash: `6f52c73`
- task ID: T15 (trainer protocols; depends on T5-T7 and the T13/T14 package migrations, all landed)
- agent/model and write owner: Kimi coder subagent (protocols + 2 adapters + 62 tests); Kimi main orchestrator reviewed the adapter composition and committed
- files changed: new `spectral_detection_posttrain/trainers/detection/protocols.py` (frozen `TrainRequest`/`TrainResult` exactly per plan shape, `@runtime_checkable DetectionTrainer` Protocol, `filter_scalar_metrics` helper, `ArtifactRef` REUSED from `experiments.artifacts` and re-exported — identity asserted by contract test); new `standard.py` (`StandardDetectionTrainer` composing maintained package functions only: `prepare_experiment_from_config` -> penn_fudan loaders -> `build_experiment_model` -> `train_baseline.train_one_epoch` -> `finalize_experiment`/`fail_experiment`; checkpoint refs taken from the completed manifest's own hashed `outputs`, so `TrainResult.checkpoints` and the manifest agree byte-for-byte; `scripts/round28_train_eval.py` untouched and existing CLIs operational); new `action_transport.py` (`ActionTransportTrainer` composing package-level action primitives — `extract_proposal_action_batch`, `supervised_action_transport_loss`, `action_batch_to_predictions`, `ActionLocalTransportHead`, `evaluate_detection_predictions` — with NO import from `scripts/`, creating zero new violations for Task 17's trainers->not-scripts boundary; docstring scopes the loop as minimal, full CLI keeps HWM-teacher/parity features); new `tests/contracts/test_trainer_protocols.py` (59 tests: frozen shapes, metric scalar-union enforcement, ArtifactRef-only checkpoints, forbidden research-status token scan over metrics + nested diagnostics, protocol conformance, adapter signatures, AST scans proving no subprocess/os.system and no hardcoded write paths); new `tests/contracts/test_trainer_protocols_integration.py` (3 fake-data CPU tests: real `build_detector(pretrained=False)` offline + deterministic injected fake loader, prepare->train->eval->checkpoint ArtifactRef sha256/size->scalar metrics->manifest `completion == "completed"`->reload into fresh head with tensor-exact weights->finite supervised loss; a third test proves an exploding loader flips the manifest to `failed`); `docs/research/refactor_ledger.md`
- RED command/result: `pytest tests/contracts/test_trainer_protocols.py -q` before the modules existed -> collection error `ModuleNotFoundError: spectral_detection_posttrain.trainers.detection.action_transport`
- GREEN command/result: contract file 59 passed; integration file 3 passed in 10.35s (real extract->loss->eval pipeline on CPU, zero downloads)
- full-suite result: 1444 passed + 1 skipped = 1382+1 baseline + 62 new (suite wall time rises from ~44s to ~71s, driven by the real-detector integration test)
- reviewer and findings accepted/rejected: main-orchestrator review. Accepted judgment calls: (1) "trainers never decide research status" operationalized as a token scan in `TrainResult.__post_init__` against `{validated, authorized, promoted, promotion, approved, reviewed}` over metric keys/values and recursive diagnostics — lifecycle words (`completed`, `started`, `runtime`, `failed`) pass by design; watch for false positives if future diagnostics legitimately carry those tokens; (2) `standard.py` composes canonical-runner functions directly rather than wrapping `canonical_runner.main(argv)` (which only prepares run dirs), gaining runs_root control and explicit manifest completion the `train_baseline.py` CLI lacks; (3) constructor seams (`loader_builder`/`model_builder`) defaulting to the maintained functions — the seam is what makes the honest fake-data exercise possible without inventing a synthetic-dataset config surface; (4) `finalize_experiment` reuses the lifecycle module's `_output_ref` for checkpoint refs; (5) no AGENTS.md change needed — new code fits the documented canonical layout. Integration test scoping accepted as honest: it exercises the real pipeline on injected fake data rather than claiming full-dataset coverage.
- scientific artifacts checked: none touched; fake data is in-memory/tmp and synthetic. **No experiment is authorized by this refactor.**
- rollback command: `git revert <this-commit>`
- remaining risks: adapters cover the minimal honest loop; HWM-teacher and strict-parity training features remain CLI-only by design — a future task may promote them behind the same protocol. The forbidden-status token scan may need refinement if legitimate diagnostics ever trip it.


### 2026-07-13 — chore: archive scratch research scripts

- commit hash: pending (filled by next ledger entry)
- task ID: T16 Wave A of 3 (scratch archival; analysis and frozen-launcher waves follow)
- agent/model and write owner: Kimi coder subagent (manifest contract test + manifest + byte-identical moves); Kimi main orchestrator adjudicated the wave membership, regenerated the script inventory and research docs, and committed
- adjudication: wave membership = exactly the 34 inventory entries with `research_status="scratch"` (builder's zero-reference evidence and the inventory classification agree independently; main spot-checked 4 stems for zero inbound refs via `git grep` and 3 files for blob identity — all clean). `_analyze_*`/`_read_round282.py`/`_scan_*` inclusions accepted under the plan Step 3 catch-all (no maintained imports, no documented reproduction command). No `.gitignore` change: nothing ignores `scripts/archive/`, and the archive must stay tracked (manifest blob-identity checks depend on it).
- files changed: 34 byte-identical moves `scripts/_*.py`/`_*.txt` -> `scripts/archive/scratch/` (git rename-detected; every destination `git hash-object` equals the recorded `original_blob_sha`); new `scripts/archive/manifest.json` (schema `script-archive-manifest.v1`, 12 fields per entry: original path/blob SHA/destination/wave/kind/research_status/historical_invocation/wrapper/replacement/runnable_state/required_ignored_artifacts/known_failure_modes; `historical_invocation: "none documented"` and wrapper/replacement `"none"` for all Wave A entries — no invocation was invented); new `tests/compatibility/test_script_archive_manifest.py` (6 tests: canonical JSON, field set/types/enums, blob identity via `git hash-object` and object-DB existence, Wave-A old-path absence, wrapper-wave blob-distinctness for future B/C, no maintained file imports/references a moved module); regenerated `spectral_detection_posttrain/configs/registry/script_inventory.json` (34 scratch paths now under `scripts/archive/scratch/`) and `docs/research/{current_status,experiment_index,artifact_index}.md`; `docs/research/refactor_ledger.md`
- RED command/result: `pytest tests/compatibility/test_script_archive_manifest.py -q` before the manifest existed -> 6 setup errors (`scripts/archive/manifest.json has not been created`)
- GREEN command/result: same command -> 6 passed; manifest+inventory focused set 16 passed; `backfill_artifact_manifests.py --check` up to date
- full-suite result: 1450 passed + 1 skipped = 1444+1 baseline + 6 new (run with moves in the working tree, proving no test imports any moved script)
- reviewer and findings accepted/rejected: main-orchestrator adjudication and review. Accepted judgment calls: (1) manifest schema wave-generic (per-entry `archive_wave`, wrapper-semantics branch) so Waves B/C extend without rewrites; (2) `required_ignored_artifacts` sourced from hardcoded path literals in each file, not guessed; (3) flat layout under `scripts/archive/scratch/` (no name collisions); (4) one CRLF stumble in the manifest (Windows text mode) fixed by the builder as LF bytes before handoff.
- scientific artifacts checked: none touched; moved files are analysis scratch code, byte-identical. **No experiment is authorized by this refactor.**
- rollback command: `git revert <this-commit>`
- remaining risks: Wave B (maintained analysis CLIs -> `scripts/analysis/` with thin old-path wrappers) and Wave C (frozen launchers -> `scripts/archive/historical/` with frozen-status wrappers) pending; each requires its own adjudication of the remaining `proposed` inventory entries. T9 found `det.dpo.smoke.001.json` references a nonexistent script path and `scripts/run_round2218_short_dpo_sweep.py` lives in `legacy/scripts/` — deferred to T18 adjudication.
