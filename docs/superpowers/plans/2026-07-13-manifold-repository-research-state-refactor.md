# Manifold Detection Research-State Repository Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `manifold-detection` into an auditable research repository whose scientific status, experiment execution, artifact provenance, compatibility boundaries, and documentation all agree, without changing detector algorithms or rewriting historical evidence.

**Architecture:** Introduce three independent sources of truth: a research-status registry for claim authorization, an experiment registry for executable definitions, and immutable artifact manifests for completed runs. Add a thin explicit dispatcher over existing runners, then migrate package and script organization incrementally behind compatibility shims. Every phase is contract-first, behavior-preserving, reversible, and independently mergeable.

**Tech Stack:** Python 3.10, PyTorch 2.1 / TorchVision 0.16, JSON/YAML configs, pytest, Git/GitHub, PowerShell and Bash launchers, remote Linux GPU2 execution, SHA-256 provenance.

---

## 0. Plan Authority And Current Baseline

This document is the master execution plan. No agent may infer a different architecture from an older report or start a migration task not listed here. Amendments require a dedicated plan-only commit and a new review before implementation continues.

Authoritative starting state:

- Repository: `E:/CLIproject/manifold`.
- Base branch: `codex/energy-guided-roi-transport`.
- Audited base commit: `7e1f2404699df997883678274491e88d3ec253c0`.
- Remote workspace: `ps@122.51.19.136:/home/ps/lzz/manifold-detection-energy-transport`.
- GitHub remote: `qyz7438/manifold-detection`.
- Current maintained test baseline: `549 passed` with `PYTHONUTF8=1`.
- Current feature branch is approximately 120 commits ahead of `origin/main`; the refactor must not be based directly on `main`.
- Local untracked file `docs/energy_transport_vs_fpn_sm_analysis.md` is user-owned. It must never be staged, moved, rewritten, normalized, archived, or used as a clean-worktree blocker without explicit user approval.
- No research experiment is currently authorized to expand. Action C1-E1, dense absolute endpoint, and the first local Delta-Q learner are frozen; the residual protocol is blocked.

### 0.1 Non-Negotiable Scientific Invariants

1. Refactoring must not change model arithmetic, random-number consumption, proposal order, BoxCoder behavior, score thresholds, NMS order, matching, top-K, metric definitions, or default hyperparameters.
2. A successful import, script exit code, smoke metric, geometry metric, offline AUC, or training-loss reduction must never automatically change a scientific status.
3. `smoke`, `limited`, `train_only`, `cache_only`, `detector_unseen`, and `full_val` are distinct evaluation scopes and must be stored explicitly.
4. Frozen experiments may be reproduced only with their original config and hashes. They may not be tuned, widened to new seeds, or relabeled as active.
5. Existing version configs, split manifests, artifact hashes, native parity contracts, and result JSON semantics are immutable historical contracts.
6. Physical checkpoints, caches, datasets, logs, and large arrays remain outside Git. Git tracks only schemas, manifests, hashes, commands, summaries, and decisions.
7. No migration commit may combine code movement with behavior changes. Algorithm changes require a separate future research plan.
8. No agent may write to `/home/ps/lzz/RLimage`; remote work is restricted to the manifold workspace.

### 0.2 Explicit Non-Goals

- Do not design a new manifold loss, action head, endpoint, FPN adapter, or training objective.
- Do not rerun or reinterpret failed experiments during the refactor.
- Do not merge the 120-commit research branch into `main` as part of the first refactor PR.
- Do not delete legacy scripts, compatibility imports, reports, or invalidated evidence.
- Do not build one universal runner that dynamically executes arbitrary Python files.
- Do not install new packages on the remote server.
- Do not normalize every old document or mass-reformat the repository.

## 1. Multi-Agent Execution Contract

### 1.1 Command Structure

- **Main agent: GPT-5.6 Sol XHigh.** Owns architecture decisions, task ordering, integration, conflict resolution, scientific-status wording, final tests, commits, and user-facing reports.
- **Default builder: GPT-5.6 Terra High.** Owns bounded multi-file implementation with an exact write set and explicit tests.
- **Routine worker/verifier: GPT-5.6 Luna High.** Used only for deterministic inventories, generated JSON/docs, link checks, test execution, and isolated changes with a complete contract.
- **Required independent reviewer: Kimi K2.7 through the local OAuth route.** Reviews each substantive phase read-only. If Kimi reports quota, OAuth, rate-limit, or availability failure once, record it and use a fresh read-only Terra High reviewer with the same prompt. Do not retry Kimi or substitute DeepSeek automatically.
- **DeepSeek v4 Pro:** optional only for a distinct unresolved scientific-evidence question or explicit user request; never used as the implementation owner.

The current repository-local `AGENTS.md` statement that a historical VOC matrix must finish before any GPU2 launch is stale state, not a live user authorization rule. The controlling launch rule for this refactor is the user's later explicit instruction: use physical GPU2 only, require `memory.free > 8192 MiB` before launch, do not wait merely because another process exists, and never stop or interfere with another process. Task 8 must remove the stale matrix statement while preserving this rule.

### 1.2 One-Writer Rule

Each task has exactly one write owner. Parallel agents must have disjoint write surfaces. Shared files such as `README.md`, `AGENTS.md`, `pyproject.toml`, registry indexes, and package `__init__.py` are integration-owned by the main agent and may not be edited concurrently.

Every delegated brief must contain:

```text
Task ID and base commit
Owned files/directories
Read-only inputs
Forbidden files/directories
Required failing test and expected failure
Required implementation behavior
Exact verification commands
Expected output/artifacts
Commit message
Stop conditions
```

### 1.3 Review Sequence Per Task

1. Main agent confirms dependency gates and assigns one builder.
2. Builder writes the failing test and returns the RED evidence before implementation.
3. Builder implements the smallest passing change and runs task-local tests.
4. Luna runs deterministic verification without editing.
5. Kimi reviews the phase diff read-only; Terra fallback applies only on one explicit Kimi failure.
6. Main agent accepts/rejects findings, runs integration tests, and creates the commit.
7. No next dependent task begins until the commit hash and gate result are recorded in the refactor ledger.

### 1.4 Parallel Waves

| Wave | Parallel lanes | Allowed overlap |
|---|---|---|
| W0 | Baseline lock only | Serial; no parallel writes |
| W1 | Research-status contracts; artifact contracts; script inventory | Disjoint files only |
| W2 | Experiment registry; documentation generator; environment snapshot | Disjoint files; main integrates indexes |
| W3 | Dispatcher handlers; eval-scope hygiene; artifact backfill | Disjoint modules and tests |
| W4 | Energy package subpackage migrations | One subpackage per Terra builder; flat shims integration-owned |
| W5 | Script archive waves; docs migration; CI | Archive manifest, docs, and CI may run in parallel |
| W6 | Final parity, remote verification, PR preparation | Verification only; serial integration |

## 2. Target Repository Architecture

```text
spectral_detection_posttrain/
  experiments/
    contracts.py               # enums and immutable dataclasses
    research_status.py         # research-line registry loader and transition validation
    registry.py                # experiment-definition registry loader
    artifacts.py               # artifact-manifest creation and validation
    lifecycle.py               # formal/smoke/frozen/validated guards
    dispatcher.py              # explicit capability-to-handler dispatch
    handlers/
      standard_detection.py
      native_contract.py
      dense_endpoint.py
    canonical_runner.py        # compatibility facade over the new components
  configs/
    registry/
      research_lines.json
      experiments.json
      script_inventory.json
      artifacts/
        *.json
  methods/
    energy_transport/
      action/
      native/
      policy/
      endpoint/
      diagnostics/
      __init__.py              # stable compatibility facade, no eager experimental imports
  trainers/
    detection/
      protocols.py
      standard.py
      action_transport.py
  eval/
    detection.py

scripts/
  run_experiment.py            # thin dispatcher CLI
  dev/
    inventory_scripts.py
    validate_repository_state.py
    snapshot_environment.py
    generate_research_docs.py
  analysis/                    # maintained offline analysis CLIs
  archive/
    manifest.json
    scratch/
    historical/

docs/
  research/
    README.md
    current_status.md          # generated; never hand-edited
    experiment_index.md        # generated; never hand-edited
    artifact_index.md          # generated; never hand-edited
    refactor_ledger.md
  reports/
    generated/
  decisions/

tests/
  experiments/
  contracts/
  compatibility/
  fixtures/

.github/workflows/ci.yml
pyproject.toml
```

Existing `spectral_detection_posttrain/configs/versions/*.json` and `configs/splits/*` remain in place. The registry references them; it does not duplicate or rename them during the first refactor.

## 3. Three Sources Of Truth

### 3.1 Research Status Registry

`spectral_detection_posttrain/configs/registry/research_lines.json` is the only machine-readable authority for whether a research line is authorized, frozen, blocked, diagnostic, baseline-only, historical, invalidated, or validated.

Allowed statuses:

```text
active       authorized to implement/run the exact next gate
queued       approved, but blocked by an explicit predecessor gate
paused       temporarily blocked by external state; hypothesis not adjudicated
frozen       tested formulation failed; no tuning, seed expansion, or claim promotion
blocked      protocol cannot execute under its locked contract
diagnostic   retained only for analysis; cannot support detector-gain claims
baseline     comparison architecture/method only
historical   reproducibility asset, not a current claim
invalidated  prior evidence contaminated by bug, weak baseline, or evaluation drift
validated    fixed claim passed its required clean/full gates
```

Initial registry must state:

- `energy_transport.native_actions.c1_e1`: `frozen`.
- `energy_transport.dense_absolute_endpoint`: `frozen`.
- `energy_transport.local_delta_q`: `frozen`.
- `energy_transport.residual_content_protocol`: `blocked`.
- `manifold.prototype_attraction`: `diagnostic`.
- `manifold.intrinsic_dimension`: `diagnostic`.
- `manifold.dual_energy`: `diagnostic`.
- `signals.fft_reward`: `invalidated` for causal reward, retained as diagnostic signal.
- `rlvr_grpo_dpo`: `historical` for the AP mainline, with reusable constraints.
- `architecture.fpn_sm_afm`: `baseline`.
- No line is initially `active` or `queued`.

Scientific status is changed only by a reviewed decision commit. Result parsers may propose a status transition but may not write it.

The status-to-execution mapping is fixed and is not independently configurable:

| Research status | Allowed experiment runnable modes |
|---|---|
| `active` | `authorized`, `diagnostic_only`, `frozen_reproduction_only` |
| `queued` | `disabled`, `diagnostic_only` |
| `paused` | `disabled`, `diagnostic_only` |
| `frozen` | `disabled`, `diagnostic_only`, `frozen_reproduction_only` |
| `blocked` | `disabled`, `diagnostic_only` |
| `diagnostic` | `disabled`, `diagnostic_only` |
| `baseline` | `disabled`, `diagnostic_only`, `frozen_reproduction_only` |
| `historical` | `disabled`, `frozen_reproduction_only` |
| `invalidated` | `disabled`, `diagnostic_only` |
| `validated` | `disabled`, `frozen_reproduction_only` |

Only `active` may authorize a new run. `queued` never means runnable. Transitions use one committed matrix: `active -> queued|paused|frozen|blocked|diagnostic|validated`, `queued -> active|paused|frozen|blocked`, `paused -> active|queued|frozen|blocked`, and any transition out of `frozen|blocked|invalidated|validated` requires a new decision document that explicitly supersedes the prior evidence. `historical`, `baseline`, and `diagnostic` may be reclassified only by the same superseding-decision process. No status or runnable mode may be promoted by a result parser.

### 3.2 Experiment Registry

`spectral_detection_posttrain/configs/registry/experiments.json` defines how a versioned experiment can be inspected or reproduced. It must never imply scientific authorization.

Required record fields:

```json
{
  "id": "det.energy.global_delta_u.c3b.balanced.001",
  "research_line": "energy_transport.native_actions.c1_e1",
  "capability": "detection.energy_transport.c3b",
  "config_path": "spectral_detection_posttrain/configs/versions/det.energy.global_delta_u.c3b.balanced.001.json",
  "legacy_entrypoint": "scripts/train_nwpu_c3b_balanced.py",
  "handler": "native_contract",
  "runnable": "frozen_reproduction_only",
  "evaluation_scope": "smoke",
  "required_inputs": ["checkpoint", "annotation", "split_manifest"],
  "expected_outputs": ["eval_metrics", "launcher_log"],
  "gpu_policy": "remote_gpu2_guarded",
  "decision_document": "docs/autonomous_exploration_report.md"
}
```

`record_kind` forms a discriminated union:

- `executable_definition` requires every execution field in the example above: `capability`, `config_path`, `legacy_entrypoint`, `handler`, `required_inputs`, `expected_outputs`, `gpu_policy`, and `decision_document`. Path-existence and handler-map tests apply.
- `historical_artifact` exists only to give archived/backfilled evidence a non-orphan identity. It requires `research_line`, `runnable="disabled"`, `evidence_kind`, `source_evidence_pointer`, and `decision_document`; `capability`, `config_path`, `legacy_entrypoint`, `handler`, `required_inputs`, `expected_outputs`, and `gpu_policy` must be explicit JSON `null`. It can never be dispatched or authorize execution, and executable-path tests do not apply.

Every `ArtifactManifest.experiment_id` must resolve to exactly one of these two record kinds. Unknown or mixed fields are rejected rather than coerced.

Frozen reproduction requires `--allow-frozen-reproduction`, exact config hash, clean Git, and matching input hashes. The dispatcher must reject any override that changes training epochs, seed, split, action space, gate, or evaluation scope.

### 3.3 Artifact Manifest

Artifact provenance has two deliberately different layers:

1. A **runtime manifest** at `runs/<run_id>/manifest.json` is ignored by Git and may move only through `started -> completed|failed|invalid`. It is written by the runner and is never itself treated as reviewed scientific evidence.
2. A **reviewed manifest** under `spectral_detection_posttrain/configs/registry/artifacts/` is a tracked, immutable snapshot. It references the runtime manifest's SHA-256, records the observation host/workspace revision/time, and is created only after review. Reviewed manifests are append-only: correcting one requires a new manifest ID with `supersedes`, never editing the old file.

Required conceptual schema:

```python
@dataclass(frozen=True)
class ArtifactRef:
    semantic_kind: str
    logical_path: str
    sha256: str
    size_bytes: int | None

@dataclass(frozen=True)
class EvaluationScope:
    kind: Literal[
        "smoke", "limited", "train_only", "cache_only",
        "detector_unseen", "full_val", "synthetic", "oracle",
        "limited_unknown"
    ]
    image_count: int | None
    limit_train: int | None
    limit_val: int | None

@dataclass(frozen=True)
class ArtifactManifest:
    schema_version: str
    manifest_kind: Literal["runtime", "reviewed"]
    manifest_id: str
    experiment_id: str
    run_id: str
    completion: Literal["started", "completed", "failed", "invalid", "unavailable"]
    invocation: tuple[str, ...]
    resolved_config: ArtifactRef
    inputs: tuple[ArtifactRef, ...]
    outputs: tuple[ArtifactRef, ...]
    git_commit: str
    git_dirty: bool
    environment: dict[str, str]
    evaluation_scope: EvaluationScope
    metrics_summary: dict[str, float | int | str | bool | None]
    gates: dict[str, bool]
    runtime_manifest_sha256: str | None
    observed_at_utc: str | None
    observed_host_alias: str | None
    observed_workspace_revision: str | None
    source_evidence_pointer: str | None
    missing_evidence: tuple[str, ...]
    unavailable_reason: str | None
    supersedes: str | None
```

`limited_unknown` is non-formal and can never become `validated`. A completed formal detection manifest additionally requires named refs for the dataset/annotation identity, split manifest, initial checkpoint or weight source, metric protocol/version, and the resolved score-threshold/NMS/decode configuration. Missing any one rejects formal completion.

Reviewed manifests additionally require `runtime_manifest_sha256` when a runtime manifest exists, `observed_at_utc`, `observed_host_alias`, `observed_workspace_revision`, `source_evidence_pointer`, and either verified refs or a specific `missing_evidence`/`unavailable_reason`. A reviewed manifest may not have `completion="started"`; runtime-only observation fields are null until promotion. Manifests must not contain tokens, passwords, private keys, embedded checkpoints, dataset contents, or user profile paths. Remote paths use logical aliases such as `remote:manifold/runs/<run_id>/eval_metrics.json`.

## 4. Implementation Tasks

### Task 0: Create The Isolated Refactor Worktree And Research Lock

**Agent:** Main Sol XHigh; Luna verifies. No builder delegation.

**Files:**
- Create: `docs/research/refactor_ledger.md`
- Create: `.agent_reports/refactor/baseline_state.json` locally only; never stage
- Do not modify: `docs/energy_transport_vs_fpn_sm_analysis.md`

- [ ] **Step 1: Verify the audited base and preserve the user file boundary**

Run:

```powershell
cd E:\CLIproject\manifold
git rev-parse HEAD
git rev-parse codex/energy-guided-roi-transport
git merge-base --is-ancestor 7e1f2404699df997883678274491e88d3ec253c0 HEAD
if ($LASTEXITCODE -ne 0) { throw 'audited code base is not an ancestor of HEAD' }
git status --short --branch
$allowed = @(
  '?? docs/energy_transport_vs_fpn_sm_analysis.md',
  '?? docs/superpowers/plans/2026-07-13-manifold-repository-research-state-refactor.md'
)
$actual = @(git status --porcelain=v1 --untracked-files=all)
$delta = @(Compare-Object -ReferenceObject $allowed -DifferenceObject $actual)
if ($delta.Count -ne 0) { $delta | Format-Table | Out-String | Write-Error; throw 'unexpected worktree state' }
git worktree list --porcelain
git show-ref --verify --quiet refs/heads/codex/repository-research-state-refactor; if ($LASTEXITCODE -eq 0) { throw 'target branch already exists' }
```

Expected before the plan-only commit: HEAD and `codex/energy-guided-roi-transport` are `7e1f2404699df997883678274491e88d3ec253c0`; the only worktree entries are the user-owned `?? docs/energy_transport_vs_fpn_sm_analysis.md` and this plan. The target branch and worktree path do not exist.

Stop if the exact porcelain assertion fails, the source ref differs, the audited code base is not an ancestor, the target branch/worktree exists, or another path is unexplained. No staged or tracked modification is allowed; the two explicitly listed untracked files are the only exception.

- [ ] **Step 2: Create and verify the plan-only execution revision**

Stage only this plan, commit it, and prove that the delta from the audited code base contains no implementation change:

```powershell
git add docs/superpowers/plans/2026-07-13-manifold-repository-research-state-refactor.md
git diff --cached --name-only
git commit -m "docs: add repository research-state refactor plan"
$executionStart = git rev-parse HEAD
git diff --name-only 7e1f2404699df997883678274491e88d3ec253c0..$executionStart
```

Expected: the staged and committed path list contains exactly this plan. The user-owned document remains untracked. Record `$executionStart`; it is the execution revision, while `7e1f240...` remains the audited code revision.

- [ ] **Step 3: Create a worktree from the plan-only execution revision**

Run:

```powershell
git worktree add E:\CLIproject\.worktrees\manifold-research-state-refactor `
  -b codex/repository-research-state-refactor `
  $executionStart
```

Expected: a clean worktree on `codex/repository-research-state-refactor` containing the plan and no implementation delta from the audited code revision. Do not copy the user-owned untracked document into it.

- [ ] **Step 4: Capture immutable local and remote facts**

Record in `.agent_reports/refactor/baseline_state.json`:

```json
{
  "base_commit": "7e1f2404699df997883678274491e88d3ec253c0",
  "source_branch": "codex/energy-guided-roi-transport",
  "main_commit": "1dfc57f",
  "maintained_tests": 549,
  "remote_workspace": "/home/ps/lzz/manifold-detection-energy-transport",
  "remote_gpu": 2,
  "user_untracked_exclusions": ["docs/energy_transport_vs_fpn_sm_analysis.md"]
}
```

Insert the actual `$executionStart` value as `execution_start` when writing the file; do not retain a symbolic or example value. Also capture `git remote -v`, local Python/Torch/TorchVision versions, remote versions, and current remote HEAD. Redact credentials.

Create a protected-history inventory containing tracked path plus Git blob SHA for every file under `docs/reports/` and every committed decision/evidence document referenced by the initial research registry. CI later requires these blobs to remain unchanged; any contextual banner must live in a new generated index, never inside a historical report.

- [ ] **Step 5: Lock collection and run baseline tests in the new worktree**

Run:

```powershell
$env:PYTHONUTF8='1'
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests --collect-only -q | Out-File -Encoding utf8 .agent_reports/refactor/maintained_collection.txt
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests -q
```

Expected in the audited environment: `549 passed`. Record the collection-file SHA-256, interpreter path, and any explicitly non-collected legacy suites. There are no implicit allowed failures: if this environment no longer produces the locked maintained set, stop and reconcile the environment/collection contract without installing remote packages or changing research behavior.

- [ ] **Step 6: Write the initial ledger entry and commit**

`docs/research/refactor_ledger.md` must record date, base commit, branch, test count, user-file exclusion, remote path, and the statement `No experiment is authorized by this refactor`.

Run:

```powershell
git add docs/research/refactor_ledger.md
git commit -m "docs: lock repository refactor baseline"
```

### Task 1: Add Test Discovery And Repository-State Contracts

**Agent:** Terra High builder owns the validator and test. Main alone creates shared `pyproject.toml` from the accepted test-discovery contract. Luna verifies.

**Files:**
- Create: `pyproject.toml`
- Create: `scripts/dev/validate_repository_state.py`
- Create: `tests/contracts/test_repository_state.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing tests for maintained test discovery and forbidden tracked artifacts**

Create assertions that:

```python
def test_pytest_collects_only_maintained_tests():
    config = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'testpaths = ["tests"]' in config

def test_repository_state_rejects_runtime_artifacts():
    violations = find_tracked_runtime_artifacts(ROOT)
    assert violations == []

def test_user_owned_untracked_document_is_never_an_auto_stage_target():
    assert "docs/energy_transport_vs_fpn_sm_analysis.md" in EXPLICIT_EXCLUSIONS
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/contracts/test_repository_state.py -q
```

Expected: failure because `pyproject.toml` and the validator do not exist.

- [ ] **Step 3: Add minimal test configuration**

`pyproject.toml` must contain only currently enforced settings:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-ra"
```

Do not add formatter, linter, packaging, or dependency settings in this task.

- [ ] **Step 4: Implement repository-state validation**

The validator must detect tracked files matching dataset, checkpoint, cache, log, `runs/`, `.agent_reports/`, and secret patterns. It must support `--json` and return exit code 1 on violations. It must not delete or modify files.

- [ ] **Step 5: Verify GREEN and full baseline**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/contracts/test_repository_state.py -q
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests -q
E:\anaconda\01\envs\RLimage\python.exe scripts/dev/validate_repository_state.py --json
```

Expected: contract tests pass; full maintained suite passes; validator reports zero tracked runtime artifacts.

- [ ] **Step 6: Commit**

```powershell
git add pyproject.toml .gitignore scripts/dev/validate_repository_state.py tests/contracts/test_repository_state.py
git commit -m "test: lock maintained repository state"
```

### Task 2: Implement Research-Status Contracts And Initial Registry

**Agent:** Terra High builder. Kimi reviews scientific wording read-only. Main resolves status disputes.

**Files:**
- Create: `spectral_detection_posttrain/experiments/contracts.py`
- Create: `spectral_detection_posttrain/experiments/research_status.py`
- Create: `spectral_detection_posttrain/configs/registry/research_lines.json`
- Create: `tests/experiments/test_research_status.py`
- Read-only: `docs/autonomous_exploration_report.md`, `docs/reports/manifold_research_line_retirement_report_2026-07-10.md`

- [ ] **Step 1: Write failing enum and registry tests**

Tests must assert the exact status vocabulary, unique IDs, nonempty hypothesis, `claim_ceiling`, evidence pointers, decision document, the complete transition matrix, and the fixed status-to-runnable mapping. They must explicitly assert that no initial line is `active`, `queued`, or `validated`, and that no non-active line can authorize a new run.

```python
def test_initial_registry_authorizes_no_experiment_expansion():
    registry = load_research_status_registry(REGISTRY_PATH)
    assert not [line for line in registry.lines if line.status in {"active", "queued"}]

def test_frozen_line_requires_decision_evidence():
    with pytest.raises(ValueError, match="decision_document"):
        ResearchLine(id="x", status=ResearchStatus.FROZEN, hypothesis="h")
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/experiments/test_research_status.py -q
```

Expected: import failure for the new contracts.

- [ ] **Step 3: Implement strict immutable contracts**

Use string enums and frozen dataclasses. This task freezes contracts v1: `ResearchStatus`, `RunnableMode`, `ExperimentDefinition`, and `EvaluationScope`, including `limited_unknown`. Reject unknown fields, duplicate IDs, missing evidence, invalid status/runnable combinations, invalid relative paths, and automatic status changes. Provide `load_research_status_registry(path)` and `validate_status_transition(old, new, decision_document)`.

- [ ] **Step 4: Populate the initial registry**

The initial statuses must match Section 3.1 exactly. Each line links committed evidence only. The untracked comparison document may be listed as `uncommitted_context` but cannot be a required evidence pointer.

- [ ] **Step 5: Verify registry and full tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/experiments/test_research_status.py -q
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests -q
```

Expected: all pass; registry contains zero authorized lines.

- [ ] **Step 6: Independent review and commit**

Kimi must check only whether statuses and claim ceilings match committed evidence. Main records accepted/rejected review points in the ledger, then commits:

```powershell
git add spectral_detection_posttrain/experiments/contracts.py `
  spectral_detection_posttrain/experiments/research_status.py `
  spectral_detection_posttrain/configs/registry/research_lines.json `
  tests/experiments/test_research_status.py docs/research/refactor_ledger.md
git commit -m "feat: add research status registry"
```

### Task 3: Implement Experiment-Definition Registry

**Agent:** Terra High builder. Luna checks all referenced paths and IDs.

**Files:**
- Create: `spectral_detection_posttrain/experiments/registry.py`
- Create: `spectral_detection_posttrain/configs/registry/experiments.json`
- Create: `tests/experiments/test_experiment_registry.py`
- Read-only: `spectral_detection_posttrain/experiments/contracts.py` (contracts v1 from Task 2)

- [ ] **Step 1: Write failing schema tests**

Cover unique experiment IDs, existing config/entrypoint paths, known research-line IDs, explicit evaluation scope, explicit capability, immutable frozen reproduction, and rejection of dynamic module strings.

```python
def test_frozen_experiment_cannot_accept_overrides():
    definition = registry.require("det.energy.global_delta_u.c3b.balanced.001")
    with pytest.raises(ValueError, match="frozen reproduction"):
        definition.resolve_overrides({"seed": 2024})
```

- [ ] **Step 2: Verify RED**

Run the single new test module and expect missing registry APIs.

- [ ] **Step 3: Implement `ExperimentDefinition` and loader**

Implement the two mutually exclusive record shapes from Section 3.2. Capability values for executable records are enums, not import paths. `runnable` is one of `disabled`, `diagnostic_only`, `frozen_reproduction_only`, or `authorized`; initial registry must contain no `authorized` record. Enforce the Section 3.1 status mapping through the research-line reference, not through a second mutable authorization flag. Tests must reject historical records with non-null execution fields and executable records missing any required path/handler field.

- [ ] **Step 4: Register the first complete migration set**

Register, at minimum:

```text
det.energy.native_listwise.c1.001
det.energy.native_budget1.c2.001
det.energy.global_delta_u.c3.001
det.energy.global_delta_u.c3b.balanced.001
det.energy.set_context.d1.001
det.energy.native_topology.d2b.001
det.energy.fine_action.d3.001
det.energy.adaptive_consensus.d4.001
det.energy.post_nms_suppress.e1.001
det.energy.dense_endpoint.absolute.001
det.energy.dense_endpoint.cleanval.001
det.energy.dense_local_delta_learner.001
det.energy.dense_local_delta_family_prior.001
```

Every `executable_definition` references the current version config and exact legacy entrypoint. Every `historical_artifact` instead uses the explicit null execution fields and evidence/provenance fields defined in Section 3.2. Do not mark D1/D2 as active or queued.

Also register a disabled `historical_artifact` record for every Task 10 evidence family that has no versioned executable definition, including the strong baseline and both native zero-parity artifacts. Tests must prove that every required backfill ID resolves and that historical-artifact records cannot be dispatched.

- [ ] **Step 5: Verify path completeness and commit**

Run registry tests, a Luna path inventory, and the full suite. Commit:

```powershell
git add spectral_detection_posttrain/experiments/registry.py `
  spectral_detection_posttrain/configs/registry/experiments.json `
  tests/experiments/test_experiment_registry.py
git commit -m "feat: register versioned research experiments"
```

### Task 4: Implement Artifact Manifest Contracts

**Agent:** Terra High builder. Write surface is disjoint from Task 3. Task 4 defines artifact types in `artifacts.py`; main integrates shared exports only after both commits.

**Files:**
- Create: `spectral_detection_posttrain/experiments/artifacts.py`
- Create: `tests/experiments/test_artifact_manifest.py`
- Create: `tests/fixtures/artifacts/completed_manifest.json`
- Create: `tests/fixtures/artifacts/invalid_dirty_formal_manifest.json`

- [ ] **Step 1: Write failing manifest tests**

Tests must cover SHA-256 format, logical paths, input/output kinds, evaluation scope, runtime started/completed/failed transitions, reviewed-manifest append-only behavior, runtime-to-reviewed hash binding, secret rejection, dirty formal-run rejection, missing `limit_*` rejection, missing formal detection refs, `limited_unknown` non-formal behavior, and stable JSON serialization.

- [ ] **Step 2: Verify RED**

Run the artifact test module and expect missing APIs.

- [ ] **Step 3: Implement strict manifest APIs**

Required functions:

```python
load_artifact_manifest(path: Path) -> ArtifactManifest
write_artifact_manifest(path: Path, manifest: ArtifactManifest) -> None
start_manifest(context: ExperimentContext, scope: EvaluationScope) -> ArtifactManifest
complete_manifest(manifest: ArtifactManifest, outputs: Sequence[ArtifactRef], metrics: dict, gates: dict) -> ArtifactManifest
fail_manifest(manifest: ArtifactManifest, failure_type: str, message: str) -> ArtifactManifest
promote_reviewed_manifest(runtime_path: Path, observation: ObservationRecord) -> ArtifactManifest
verify_artifact_ref(ref: ArtifactRef, resolver: ArtifactResolver) -> bool
```

Serialization must be sorted, UTF-8, deterministic, and end with one newline. Runtime writes use atomic replace inside `runs/`; reviewed writes use create-exclusive semantics and reject overwrite. Failure messages must not include environment secrets. Formal completion requires named dataset/annotation, split, initial-weight, metric-protocol, and postprocess-config refs.

- [ ] **Step 4: Verify GREEN and commit**

Run artifact tests and the full suite. Commit:

```powershell
git add spectral_detection_posttrain/experiments/artifacts.py `
  tests/experiments/test_artifact_manifest.py tests/fixtures/artifacts
git commit -m "feat: add immutable artifact manifests"
```

### Task 5: Make Evaluation Scope And Limits Unambiguous

**Agent:** Terra High builder. This is a behavior-observability change, not a metric change.

**Files:**
- Modify: `spectral_detection_posttrain/experiments/contracts.py`
- Modify: `spectral_detection_posttrain/experiments/schema.py`
- Modify: `scripts/round28_train_eval.py`
- Modify: `spectral_detection_posttrain/experiments/metadata.py`
- Create: `tests/experiments/test_evaluation_scope.py`
- Modify: `tests/test_round28_experiment_hygiene.py`

- [ ] **Step 1: Write failing scope-provenance tests**

Tests must prove that `limit_train`, `limit_val`, image counts, and `evaluation_scope.kind` appear in resolved config, metadata, and final metrics. Formal `full_val` must reject non-null limits. `smoke` must require a positive explicit limit.

```python
def test_full_val_rejects_limit_val():
    with pytest.raises(ValueError, match="full_val"):
        EvaluationScope(kind="full_val", image_count=196, limit_val=32)
```

- [ ] **Step 2: Verify RED using focused tests**

Expected: current `round28` metadata omits the complete scope contract.

- [ ] **Step 3: Add normalized scope without changing evaluation**

Extend config normalization and metadata writing only. Existing CLI flags keep their names and behavior. Existing metrics stay byte-compatible except for additive provenance keys under `evaluation_scope`.

This is the only task after Task 2 allowed to change `experiments/contracts.py`; it creates contracts v2. Tasks 3 and 4 must already be committed and are read-only inputs. Main performs the v2 export integration serially before Task 6.

- [ ] **Step 4: Add backward compatibility**

Older configs without `evaluation_scope` may run only in `non-formal` mode and must be normalized to `limited_unknown` with an explicit warning in metadata. They cannot produce `validated` manifests.

- [ ] **Step 5: Verify**

Run scope tests, round28 hygiene tests, canonical runner tests, and full maintained tests. Inspect one temporary CPU run's `config.json`, `metadata.json`, and `eval_metrics.json`.

- [ ] **Step 6: Commit**

```powershell
git add spectral_detection_posttrain/experiments/contracts.py `
  spectral_detection_posttrain/experiments/schema.py `
  spectral_detection_posttrain/experiments/metadata.py `
  scripts/round28_train_eval.py tests/experiments/test_evaluation_scope.py `
  tests/test_round28_experiment_hygiene.py
git commit -m "fix: record explicit evaluation scope"
```

### Task 6: Add Artifact Lifecycle To The Canonical Runner

**Agent:** Terra High builder. Kimi reviews manifest lifecycle and failure behavior.

**Files:**
- Modify: `spectral_detection_posttrain/experiments/canonical_runner.py`
- Create: `spectral_detection_posttrain/experiments/lifecycle.py`
- Create: `tests/experiments/test_experiment_lifecycle.py`
- Modify: `tests/test_canonical_runner_e2e.py`

- [ ] **Step 1: Write failing lifecycle tests**

Tests must assert:

```text
prepare -> manifest completion=started
successful finalize -> completed with output hashes
exception -> failed with sanitized failure type
formal dirty tree -> rejected before run directory mutation
missing input hash -> rejected
```

- [ ] **Step 2: Verify RED**

Run lifecycle and canonical-runner E2E tests. Expect missing manifest behavior.

- [ ] **Step 3: Implement additive lifecycle hooks**

`prepare_experiment` writes `config.yaml`, `metadata.json`, and a runtime `manifest.json` under the ignored run directory. Add explicit `finalize_experiment(context, outputs, metrics, gates)` and `fail_experiment(context, error)`. Do not infer completion from file existence. This task never writes a tracked reviewed manifest; promotion happens only through the reviewed backfill/promotion path after verification.

- [ ] **Step 4: Preserve old callers**

Existing callers that never finalize leave a `started` manifest and otherwise behave exactly as before. They are non-formal until migrated.

- [ ] **Step 5: Verify and commit**

Run canonical tests, metadata/schema tests, and full suite. Commit:

```powershell
git add spectral_detection_posttrain/experiments/canonical_runner.py `
  spectral_detection_posttrain/experiments/lifecycle.py `
  tests/experiments/test_experiment_lifecycle.py tests/test_canonical_runner_e2e.py
git commit -m "feat: track canonical experiment lifecycle"
```

### Task 7: Build The Explicit Experiment Dispatcher

**Agent:** Terra High builder. Main owns capability registration approval.

**Files:**
- Create: `spectral_detection_posttrain/experiments/dispatcher.py`
- Create: `spectral_detection_posttrain/experiments/handlers/__init__.py`
- Create: `spectral_detection_posttrain/experiments/handlers/standard_detection.py`
- Create: `spectral_detection_posttrain/experiments/handlers/native_contract.py`
- Create: `spectral_detection_posttrain/experiments/handlers/dense_endpoint.py`
- Create: `scripts/run_experiment.py`
- Create: `tests/experiments/test_dispatcher.py`

- [ ] **Step 1: Write failing command and guard tests**

Required CLI:

```text
python scripts/run_experiment.py list
python scripts/run_experiment.py status
python scripts/run_experiment.py validate --experiment <id>
python scripts/run_experiment.py dry-run --experiment <id> --run-name <name>
python scripts/run_experiment.py run --experiment <id> --run-name <name>
```

Tests must reject unknown IDs, unknown capabilities, dynamic import strings, frozen experiments without the explicit reproduction flag, and any override on frozen records.

- [ ] **Step 2: Verify RED**

Run dispatcher tests and expect missing CLI/module failures.

- [ ] **Step 3: Implement a static handler map**

```python
HANDLERS: dict[ExperimentCapability, ExperimentHandler] = {
    ExperimentCapability.STANDARD_DETECTION: StandardDetectionHandler(),
    ExperimentCapability.NATIVE_CONTRACT: NativeContractHandler(),
    ExperimentCapability.DENSE_ENDPOINT: DenseEndpointHandler(),
}
```

Handlers initially expose `validate` and `dry_run`. `run` remains disabled unless the experiment registry permits it. Frozen handlers may reproduce exact commands but cannot alter configs.

- [ ] **Step 4: Make dry-run deterministic**

Dry-run prints resolved experiment ID, research status, config hash, required inputs, evaluation scope, GPU policy, expected outputs, and exact command. It must not create a run directory or import dataset code.

- [ ] **Step 5: Verify and commit**

Run dispatcher tests, `list`, `status`, and dry-runs for C1, C3b, dense absolute, and family prior. Confirm all are non-authorized. Commit the listed files.

Commit message:

```text
feat: add explicit experiment dispatcher
```

### Task 8: Generate Current Research Documentation From Registries

**Agent:** Luna High builder for generator, generated files, and deterministic tests. Main is the sole writer for `README.md` and `AGENTS.md` and owns prose templates and scientific wording.

**Files:**
- Create: `scripts/dev/generate_research_docs.py`
- Create: `docs/research/README.md`
- Create: `docs/research/current_status.md`
- Create: `docs/research/experiment_index.md`
- Create: `docs/research/artifact_index.md`
- Create: `tests/experiments/test_generated_research_docs.py`
- Modify: `README.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Write failing generation and drift tests**

Tests regenerate into a temporary directory and compare byte-for-byte with tracked generated files. Assert that current status contains zero active/queued lines and names all frozen/blocked main lines.

- [ ] **Step 2: Verify RED**

Run the generator test and expect missing files.

- [ ] **Step 3: Implement deterministic generation**

Sort by stable IDs, use UTF-8, one trailing newline, relative links, and visible generated-file headers. Metrics come only from artifact manifests; status comes only from the research registry.

- [ ] **Step 4: Correct top-level documentation**

Luna returns a proposed diff but does not edit the two shared files. Main applies and verifies the `README.md` and `AGENTS.md` changes serially after generated output is green.

`README.md` must:

- distinguish maintained implementation from authorized research;
- state that no experiment is currently authorized;
- point to `docs/research/current_status.md`;
- correct the remote workspace and remove the `/home/ps/lzz/RLimage` PYTHONPATH example;
- describe `round28` as the standard detector runner, not the universal energy runner;
- stop referencing nonexistent active directories.

`AGENTS.md` must preserve the user's current GPU2-only, `memory.free > 8192 MiB`, no-interference rule while removing stale “matrix in progress” and wait-for-VOC statements. It must say registries control current status.

- [ ] **Step 5: Protect historical reports**

Do not rewrite old report conclusions or dates. Historical report blobs locked in Task 0 remain byte-identical; add context only through generated indexes. Do not stage the user-owned untracked comparison document.

- [ ] **Step 6: Verify and commit**

Run generator drift tests, link checks introduced later in Task 16, full tests, and `git status`. Commit only the listed files.

Commit message:

```text
docs: align repository with research registries
```

### Task 9: Build A Complete Script Inventory Before Moving Anything

**Agent:** Luna High builder. Main reviews ambiguous classifications.

**Files:**
- Create: `scripts/dev/inventory_scripts.py`
- Create: `spectral_detection_posttrain/configs/registry/script_inventory.json`
- Create: `tests/contracts/test_script_inventory.py`
- Do not move files in this task

- [ ] **Step 1: Write failing completeness tests**

The inventory must include every tracked file under `scripts/` and every root `run_*.bat`, `run_*.sh`, `run_*.py`, analysis script, and trainer. Required fields:

```json
{
  "path": "scripts/train_nwpu_c3b_balanced.py",
  "kind": "trainer",
  "research_status": "frozen_reproduction",
  "capability": "detection.energy_transport.c3b",
  "replacement": "scripts/run_experiment.py",
  "archive_wave": 0,
  "known_inputs": [],
  "known_outputs": [],
  "notes": ""
}
```

Add `classification_state` as `proposed` or `reviewed`. Complete path coverage is required in this task, but a proposed ambiguous classification is allowed until the archive wave that would move it.

- [ ] **Step 2: Verify RED**

Expect missing inventory and APIs.

- [ ] **Step 3: Implement deterministic discovery**

The script proposes classifications by path/name but writes `review_required=true` for ambiguous files. It must never move, delete, or execute a discovered script.

- [ ] **Step 4: Adjudicate the first governance wave**

Main reviews every entry needed by Tasks 3, 7, 10, and the first archive wave. Remaining ambiguous entries stay `classification_state="proposed"`, may not be moved, and are adjudicated in bounded batches immediately before their archive wave. Scientific status comes from registries, not filename heuristics.

- [ ] **Step 5: Verify coverage and commit**

Run inventory generation twice and compare hashes; run completeness tests and full tests. Commit inventory, generator, and tests.

Commit message:

```text
chore: inventory research scripts and entrypoints
```

### Task 10: Backfill Critical Artifact Manifests

**Agent:** Luna High prepares deterministic manifests; Terra High reviews provenance; main approves scientific scope.

**Files:**
- Create files under: `spectral_detection_posttrain/configs/registry/artifacts/`
- Create: `scripts/dev/backfill_artifact_manifests.py`
- Create: `tests/experiments/test_backfilled_artifacts.py`
- Regenerate: `docs/research/artifact_index.md` as a main-owned integration step
- Read-only: `.agent_reports/c1_native/*`, remote `runs/*`, committed reports

- [ ] **Step 1: Write failing required-artifact tests**

Require manifests for:

```text
nwpu_mob_strong_cosine_s42_bs8_36ep
native_zero_parity_baseline
native_zero_parity_fullft
det.energy.set_policy.m1.001
det.energy.global_delta_u.c3b.balanced.001
det.energy.native_topology.d2b.001
det.energy.post_nms_suppress.e1.001
det.energy.dense_endpoint.absolute.001
det.energy.dense_endpoint.geometry_control.001
det.energy.dense_endpoint.cleanval.001
det.energy.dense_endpoint.shift_audit.001
det.energy.dense_local_delta_stats.002
det.energy.dense_local_delta_learner.001
det.energy.dense_local_delta_family_audit.001
det.energy.dense_local_delta_family_prior.001
```

- [ ] **Step 2: Verify RED**

Expect missing manifest failures.

- [ ] **Step 3: Backfill without inventing evidence**

Read existing result JSON and reports. Copy only observed values and hashes. Unknown values are `null` with `missing_evidence` entries. Never compute a new status or silently shorten a hash. Each reviewed manifest records whether its evidence is full-val, smoke, train-only, cache-only, or post-hoc; its runtime-manifest hash when available; observation host alias, workspace revision and UTC time; original evidence pointer; and unavailable reason when applicable.

- [ ] **Step 4: Cross-check remote artifacts**

Use read-only SSH. Verify current SHA-256 and sizes and compare the observed remote revision with the evidence's expected revision. If an artifact is absent, differs, or has no trustworthy selection lineage, mark the reviewed manifest `invalid` or `unavailable`; do not overwrite remote or local files. An unavailable family does not block unrelated governance/package work and is delivered as its own artifact-family commit.

- [ ] **Step 5: Verify and commit**

Process and commit one artifact family at a time. Run manifest validation, artifact index generation, and full tests for each family. Terra reviews every full-val scope and scientific claim ceiling; main regenerates and stages `docs/research/artifact_index.md` after each accepted family.

Commit message:

```text
docs: backfill critical experiment provenance
```

### Task 11: Record Reproducible Environment Contracts

**Agent:** Luna High collects versions; Terra High implements validation. No remote package installation.

**Files:**
- Create: `scripts/dev/snapshot_environment.py`
- Create: `docs/research/environment.md`
- Create: `spectral_detection_posttrain/configs/registry/environments/local_windows_py310.json`
- Create: `spectral_detection_posttrain/configs/registry/environments/remote_gpu2_py310_cu121.json`
- Create: `tests/contracts/test_environment_snapshot.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Write failing environment-schema tests**

Require Python, OS, CUDA availability, GPU identity, Torch, TorchVision, NumPy, Pillow, pytest, optional dependency states, and executable path. Reject secrets and user tokens.

- [ ] **Step 2: Verify RED**

Run the new tests and expect missing snapshots.

- [ ] **Step 3: Implement read-only snapshot collection**

The script prints or writes JSON only when given `--output`. It must not install packages, mutate conda, initialize CUDA workloads, or query GPUs other than the explicitly requested index.

- [ ] **Step 4: Capture local and remote environments**

Run locally and through read-only SSH in the manifold workspace. Record that remote training is restricted to physical GPU2. Use semantic notes for optional legacy dependencies instead of adding them to the maintained install by default.

- [ ] **Step 5: Clarify requirements without pretending to lock CUDA wheels**

Separate maintained runtime requirements from optional legacy analysis requirements using comments or companion documentation. Do not pin an unverified CUDA wheel URL. `requirements.txt` must remain installable for the maintained CPU test path.

- [ ] **Step 6: Verify and commit**

Run environment tests and full tests. Commit the snapshots, docs, script, and requirements clarification.

Commit message:

```text
docs: record reproducible execution environments
```

### Task 12: Define Energy-Transport Subpackage Boundaries

**Agent:** Main Sol locks the mapping; Terra High builders later migrate one subpackage each. This task creates boundaries and tests but moves no implementation.

**Files:**
- Create: `docs/decisions/adr-energy-transport-package-boundaries.md`
- Create: `tests/contracts/test_energy_transport_import_boundaries.py`
- Modify: `docs/decisions/index.md`

- [ ] **Step 1: Write the ADR with exact mapping**

Target mapping:

| Target package | Current modules |
|---|---|
| `energy_transport/action/` | `actions.py`, `contracts.py`, `operators.py`, `preferences.py`, `geometric_constraints.py` |
| `energy_transport/native/` | `native_contract.py`, `native_topology.py`, `candidate_energy.py`, `benefit_energy.py` |
| `energy_transport/policy/` | `search.py`, `set_search.py`, `set_policy.py`, `global_top1.py`, `listwise_noop.py`, `joint_delta_u.py`, `adaptive_consensus.py`, `post_nms_suppress.py` |
| `energy_transport/endpoint/` | `dense_set_energy.py`, `dense_endpoint.py` |
| `energy_transport/diagnostics/` | `structure_metrics.py`, `cone_projection.py`, `high_water_mark.py`, `linear_identifiability.py`, `spatial_counterfactual.py`, `step_strata.py`, `top_focused_audit.py`, `decomposed_actionability.py`, `joint_probe_validation.py` |

Each current flat module remains a compatibility shim for at least one complete refactor release. `__init__.py` retains the public names but must eventually use lazy or explicit imports that do not import every experimental module eagerly.

- [ ] **Step 2: Write failing dependency-rule tests**

Rules:

```text
action may depend on torch and core types, not trainer/experiment/dataset
native may depend on action and eval primitives, not trainers
policy may depend on action/native, not scripts or runs
endpoint may depend on pure feature/teacher modules, not runners
diagnostics may depend on stable method APIs, never write artifacts implicitly
```

- [ ] **Step 3: Verify current violations are reported, not auto-fixed**

The test initially reads an allowlist of known violations. The ADR lists every allowlist item and its removal task. New violations fail immediately.

- [ ] **Step 4: Independent review and commit**

Kimi reviews whether boundaries preserve research semantics. Commit ADR, test, and decision index only.

Commit message:

```text
docs: define energy transport package boundaries
```

### Task 13: Migrate Energy Action And Native Modules Behind Shims

**Agent:** Two Terra High builders in sequence, not parallel, because native depends on action. Luna verifies imports after each commit.

**Files:**
- Create: `spectral_detection_posttrain/methods/energy_transport/action/*.py`
- Create: `spectral_detection_posttrain/methods/energy_transport/native/*.py`
- Replace current flat source modules with forwarding shims
- Create: `tests/compatibility/test_energy_transport_flat_shims.py`
- Create: `tests/fixtures/checkpoints/energy_transport_checkpoint_manifest.json`
- Modify: `spectral_detection_posttrain/methods/energy_transport/__init__.py`

- [ ] **Step 1: Add failing old/new API parity tests for the action package**

For every public symbol, assert old and new imports reference the same object or have identical signatures and outputs. Include `ActionLocalTransportHead`, action dataclasses, score/box application, energy, threshold preservation, and rescue budget. The checkpoint fixture manifest points to small CPU-safe, hash-locked historical state dicts outside Git when weights are nontrivial; tests strict-load every serializable action module through both old and new import paths and compare keys, tensors, and deterministic outputs.

- [ ] **Step 2: Move action modules with `git mv` and add shims**

Do not edit function bodies. Flat shims contain imports and `__all__` only. Run focused tests and confirm no source diff other than module paths and shim content.

- [ ] **Step 3: Commit action migration**

```text
refactor: isolate energy transport action core
```

- [ ] **Step 4: Add failing parity tests for native modules**

Cover strict zero-action native parity helpers, class expansion, BoxCoder decode, small-box filtering, class-wise NMS, ordering, and top-K. Strict-load any serializable native module from the checkpoint fixture and compare old/new outputs on the locked CPU input.

- [ ] **Step 5: Move native modules and add shims**

Again, no function-body change. Run existing native contract tests plus new shim tests.

- [ ] **Step 6: Run the full suite and commit**

```text
refactor: isolate native detection transport
```

Stop and revert the individual migration commit if any numerical, ordering, import, or serialization parity fails.

### Task 14: Migrate Policy, Endpoint, And Diagnostic Modules In Separate Commits

**Agent:** One Terra High builder per subpackage, serial integration. Kimi reviews the complete package diff after all three commits.

**Files:**
- Create: `energy_transport/policy/*.py`
- Create: `energy_transport/endpoint/*.py`
- Create: `energy_transport/diagnostics/*.py`
- Replace corresponding flat modules with shims
- Extend: `tests/compatibility/test_energy_transport_flat_shims.py`
- Modify: `energy_transport/__init__.py`

- [ ] **Step 1: Migrate policy modules**

Write API and deterministic-output parity tests first. Preserve all candidate ordering, tie handling, no-op semantics, and random seeds. Strict-load every serializable policy module through old and new paths using the hash-locked CPU fixture; compare state-dict keys/values and outputs. Commit as:

```text
refactor: isolate energy transport policies
```

- [ ] **Step 2: Migrate endpoint modules**

Write tests for permutation invariance, canonical sorting, state-dict key equality, teacher component values, and strict old-checkpoint loading through old/new paths. Commit as:

```text
refactor: isolate dense set endpoints
```

- [ ] **Step 3: Migrate diagnostics modules**

Write import/output parity tests. Strict-load any serializable diagnostic module covered by the checkpoint manifest. Diagnostics must stay side-effect free: importing them cannot read files, create directories, initialize CUDA, or write outputs. Commit as:

```text
refactor: isolate transport diagnostics
```

- [ ] **Step 4: Reduce eager imports**

Refactor the top-level facade only after all shims pass. Preserve the documented public `__all__`. Add an import smoke that succeeds when optional legacy dependencies are absent.

- [ ] **Step 5: Full verification**

Run all 549 baseline tests plus all new tests, then run the native zero-action parity command against the locked fixture. No GPU training is permitted.

### Task 15: Extract Trainer Protocols Without Changing Training

**Agent:** Terra High builder. This begins only after Tasks 5-7 and package migrations pass.

**Files:**
- Create: `spectral_detection_posttrain/trainers/detection/protocols.py`
- Create: `spectral_detection_posttrain/trainers/detection/standard.py`
- Create: `spectral_detection_posttrain/trainers/detection/action_transport.py`
- Keep: `action_local_transport.py`, `roi_state.py`, and legacy trainer shims
- Create: `tests/contracts/test_trainer_protocols.py`

- [ ] **Step 1: Write failing protocol tests**

Define:

```python
class DetectionTrainer(Protocol):
    def train(self, request: TrainRequest) -> TrainResult: ...

@dataclass(frozen=True)
class TrainResult:
    checkpoints: tuple[ArtifactRef, ...]
    metrics: dict[str, float | int | str | bool | None]
    diagnostics: dict[str, object]
```

Tests require that trainers return data and never decide research status.

- [ ] **Step 2: Verify RED**

Run the protocol test and expect missing types.

- [ ] **Step 3: Add adapters around existing code**

Do not move the 656-line `round28_train_eval.py` in this task. `standard.py` and `action_transport.py` are adapters that call existing maintained functions and return structured results. Existing CLIs remain operational.

- [ ] **Step 4: Add a fake-data CPU integration test**

Exercise prepare, train adapter, eval adapter, checkpoint reload, metrics, and manifest completion on a tiny deterministic fake dataset. Do not download weights or data.

- [ ] **Step 5: Verify and commit**

Run trainer protocol, canonical runner, action trainer, and full tests. Commit:

```text
refactor: add structured detection trainer protocols
```

### Task 16: Archive Scripts In Controlled Waves

**Agent:** Luna prepares move list; Terra performs one wave at a time; main owns compatibility wrappers.

**Files:**
- Create: `scripts/archive/manifest.json`
- Create: `scripts/archive/scratch/`
- Create: `scripts/archive/historical/`
- Create: `scripts/analysis/`
- Create: `scripts/run/`
- Modify: `.gitignore`
- Create: `tests/compatibility/test_script_archive_manifest.py`

- [ ] **Step 1: Write failing archive-manifest tests**

Every moved script needs `original_entrypoint_path`, original Git blob SHA, destination, kind, research status, exact historical invocation, `current_reproduction_wrapper`, replacement command, runnable state, required ignored artifacts, and known failure modes. Every replacement wrapper must be tested, and tests must distinguish the original blob identity from the wrapper currently occupying the old path.

- [ ] **Step 2: Verify RED**

Expect missing archive manifest.

- [ ] **Step 3: Wave A moves scratch-only files**

Move only private `_check_*`, `_counterfactual_*`, captured output text, and scripts proven to have no maintained imports or documented reproduction command. Preserve them under `scripts/archive/scratch/` using `git mv`.

- [ ] **Step 4: Wave B moves maintained analysis tools**

Move selected `analyze_*`, `audit_*`, and pure `validate_*` CLIs to `scripts/analysis/`. Leave thin old-path wrappers until all committed reports and tests use the new path.

- [ ] **Step 5: Wave C moves historical trainers and launchers**

Only after dispatcher dry-run parity exists, move frozen C1-E1 and dense launchers to `scripts/archive/historical/<line>/`. Old paths become thin wrappers printing the frozen status and delegating exact reproduction only.

- [ ] **Step 6: Do not move general root scripts automatically**

Root batch/shell scripts, `mfvpt/`, `MPLSeg/`, and `legacy/` require separate user-approved waves. Inventory them, but leave them in place in this plan.

- [ ] **Step 7: Verify each wave and commit separately**

Before each wave, main adjudicates only that wave's `classification_state="proposed"` entries; no proposed entry may move. For each bounded wave, run archive manifest tests, all tests importing moved scripts, generated docs, and the full suite. Each wave is independently reviewable and revertible; do not combine all 205 scripts in one PR. Commit messages:

```text
chore: archive scratch research scripts
refactor: organize maintained analysis scripts
chore: archive frozen experiment launchers
```

### Task 17: Add Compatibility And Import-Boundary Coverage

**Agent:** Terra High builder; Luna enumerates imports.

**Files:**
- Create: `tests/compatibility/test_legacy_imports.py`
- Create: `tests/contracts/test_dependency_boundaries.py`
- Create: `scripts/dev/check_import_boundaries.py`
- Read-only: `spectral_detection_posttrain/models/`, `rlvr/`, `train/`, `spectral/`, `matching/`, `legacy/`

- [ ] **Step 1: Write tests for every compatibility namespace**

Import every tracked shim and assert its intended public symbols resolve. Optional legacy dependencies must produce an explicit skipped capability or a targeted error, not break unrelated imports.

- [ ] **Step 2: Write dependency-boundary tests**

Parse imports with Python `ast`, not regex. Enforce:

```text
core must not import methods, trainers, experiments, scripts, or legacy
methods must not import trainers, experiments, scripts, or runs
signals must not import trainers or execute training
trainers may import core/methods/datasets/eval, not scripts
experiments may orchestrate all canonical layers, not legacy scripts directly
new canonical code must not import compatibility namespaces
```

Known violations are stored in a shrinking explicit allowlist with owner and removal task. New violations fail CI.

- [ ] **Step 3: Verify RED then GREEN**

First show current violations. Add only the audited existing allowlist; do not broaden rules to make tests pass.

- [ ] **Step 4: Run import smoke in a fresh Python process**

Verify that importing action contracts, dense endpoint, canonical runner, and registry performs no dataset read, CUDA initialization, file write, or remote access.

- [ ] **Step 5: Commit**

```text
test: enforce canonical import boundaries
```

### Task 18: Add Documentation Integrity Checks

**Agent:** Luna High builder for checker and tests. Main is the sole writer for current shared documentation and reviews scientific wording and archive labels.

**Files:**
- Create: `scripts/dev/check_docs.py`
- Create: `tests/contracts/test_documentation_integrity.py`
- Modify: `docs/reports/index.md`
- Modify: `docs/architecture.md`
- Main-only possible modifications: `README.md`, `AGENTS.md`

- [ ] **Step 1: Write failing documentation tests**

Check UTF-8 decoding, relative links, referenced tracked paths, generated-doc drift, duplicate experiment IDs, forbidden current-status phrases, explicit evaluation-scope labels on generated metric tables, and every current `active`/`authorized`/`in progress` statement against the research registry. Static current docs may link to generated status but may not independently declare a different status.

- [ ] **Step 2: Verify RED**

Current stale references to nonexistent active directories and archives should fail with actionable paths.

- [ ] **Step 3: Implement checker and fix only current documentation**

Luna returns findings and a proposed patch but edits only the checker/tests. Main updates README, AGENTS, architecture, and report index serially. Historical report blobs from Task 0 remain byte-unchanged; contextual banners live only in new/generated indexes.

- [ ] **Step 4: Handle encoding conservatively**

Do not mass-normalize 145 documents. Check all; repair only current source-of-truth files in explicit commits. Garbled historical files are labeled `encoding_legacy` in the archive index.

- [ ] **Step 5: Verify and commit**

Run doc tests, generator drift tests, and full tests. Commit:

```text
docs: validate current research documentation
```

### Task 19: Add CPU CI Without Pretending To Validate Research Runs

**Agent:** Terra High builder owns `.github/workflows/ci.yml` and `tests/contracts/test_ci_contract.py`. Main alone modifies shared `pyproject.toml` and `docs/research/environment.md` after the builder returns the required deltas. Luna verifies workflow syntax.

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `tests/contracts/test_ci_contract.py`
- Modify: `pyproject.toml`
- Modify: `docs/research/environment.md`

- [ ] **Step 1: Write failing CI-contract tests**

Require jobs for repository state, generated-doc drift, import boundaries, maintained tests, and registry/manifest validation. Forbid GPU jobs, dataset downloads, remote SSH, secret-dependent tests, and research-result promotion.

- [ ] **Step 2: Verify RED**

Expect missing workflow.

- [ ] **Step 3: Implement Linux CPU CI**

Use Python 3.10. Install only maintained test requirements. Cache package downloads, not datasets or checkpoints. Run:

```text
python scripts/dev/validate_repository_state.py
python scripts/dev/check_import_boundaries.py
python scripts/dev/check_docs.py
python scripts/dev/generate_research_docs.py --check
python -m pytest tests -q
```

- [ ] **Step 4: Add explicit legacy-optional reporting**

Do not silently collect ignored `legacy/tests` or root analysis scripts. Document them as a separate non-gating local suite with optional dependencies.

- [ ] **Step 5: Verify and commit**

Validate YAML locally without adding a new dependency if unavailable; otherwise rely on GitHub's first Draft PR run. Commit:

```text
ci: add maintained CPU repository checks
```

### Task 20: Add The Remote GPU2 Runbook And Guarded Launcher Contract

**Agent:** Terra High builder; Luna performs read-only remote checks. No training job is launched.

**Files:**
- Create: `docs/research/remote_gpu2_runbook.md`
- Create: `scripts/run/guard_gpu2.py`
- Create: `tests/contracts/test_gpu2_guard.py`
- Modify: generated/current documentation links only

- [ ] **Step 1: Write failing guard tests with mocked `nvidia-smi`**

Cover physical index 2, malformed output, unavailable GPU, the strict free-memory threshold, command environment, dirty-tree rejection, and no process-kill behavior.

- [ ] **Step 2: Verify RED**

Expect missing guard.

- [ ] **Step 3: Implement a dry-run-first guard**

The guard may query GPU2 and print a command. It must never kill, stop, reprioritize, or signal another process. It requires:

```text
CUDA_VISIBLE_DEVICES=2
current_free_mib > 8192
clean Git
matching registry/config/input hashes
authorized or exact frozen-reproduction status
```

The user rule that other processes need not block launch applies only after the memory inequality passes. Never use GPU0/1/3.

A launcher identity must come from the reviewed experiment definition; CLI overrides are forbidden. `estimated_peak_mib` may be recorded from the registry for audit but is not a second launch gate and may not weaken or strengthen the user's strict `current_free_mib > 8192` rule. A successful launch records the newly created PID, command hash, run ID, and manifest path, but the guard exposes no kill, stop, or reprioritize operation. Cleanup or cancellation is outside this guard and may act only through a separately verified owned-PID procedure.

- [ ] **Step 4: Write the runbook**

Document local/remote path mapping, conda executable, `PYTHONPATH` rooted at the manifold workspace, log paths, manifest lifecycle, failure handling, and the rule against installing packages.

- [ ] **Step 5: Read-only remote verification**

Run only `status`, `dry-run`, GPU query, Git status, and version commands. Do not launch training.

- [ ] **Step 6: Commit**

```text
docs: add guarded GPU2 execution contract
```

### Task 21: Final Integration, Remote Verification, And PR Preparation

**Agent:** Main Sol XHigh. Luna runs test/report commands. Kimi performs final read-only diff review; Terra fallback on one explicit Kimi failure.

**Files:**
- Modify: `docs/research/refactor_ledger.md`
- Create: `docs/research/refactor_completion_report.md`
- Do not modify implementation unless a separate failing-test fix commit is created

- [ ] **Step 1: Run local completion suite**

```powershell
$env:PYTHONUTF8='1'
E:\anaconda\01\envs\RLimage\python.exe scripts/dev/validate_repository_state.py
E:\anaconda\01\envs\RLimage\python.exe scripts/dev/check_import_boundaries.py
E:\anaconda\01\envs\RLimage\python.exe scripts/dev/check_docs.py
E:\anaconda\01\envs\RLimage\python.exe scripts/dev/generate_research_docs.py --check
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests -q
git diff --check
git status --short --branch
```

Expected: all checks pass; test count is at least 549 plus new tests; worktree clean after the completion-report commit.

- [ ] **Step 2: Run compatibility and dispatcher proofs**

Prove old/new imports, frozen dry-runs, exact config resolution, generated status, and artifact verification. Capture outputs in the completion report, not as untracked binary fixtures.

- [ ] **Step 3: Run remote CPU verification**

Fast-forward the dedicated refactor branch into a clean remote refactor worktree or use a verified bundle. Run maintained CPU tests and registry/manifest checks. Do not change the primary experiment worktree or launch GPU training.

- [ ] **Step 4: Compare scientific artifacts**

Re-hash all backfilled remote artifacts. Confirm no result JSON, checkpoint, cache, split manifest, or version config changed during refactor.

- [ ] **Step 5: Independent review**

Kimi reviews the complete diff for behavior drift, status drift, missing compatibility, and research-evidence corruption. Main lists every accepted and rejected finding in the completion report.

- [ ] **Step 6: Commit the completion report**

```text
docs: complete repository research-state refactor
```

- [ ] **Step 7: Push and open a Draft PR**

The first PR targets `codex/energy-guided-roi-transport`, not `main`. Include phase commits, test counts, parity evidence, artifact hashes, migration map, residual risks, and explicit statement that no new research result is claimed.

- [ ] **Step 8: Merge strategy after review**

Merge the refactor PR only after CI and required reviews pass. Plan a separate consolidation PR from the research branch to `main`; do not squash away experiment lineage unless the user explicitly chooses that history strategy.

## 5. Phase Gates And Dependency Graph

```text
Task 0 research lock
  -> Task 1 repository contract
  -> [Task 2 status registry || Task 4 artifact contracts || Task 9 script inventory]
  -> Task 3 experiment registry
  -> [Task 5 eval scope || Task 8 generated docs || Task 10 artifact backfill]
  -> Task 6 lifecycle
  -> Task 7 dispatcher
  -> Task 11 environment lock
  -> Task 12 package-boundary ADR
  -> Task 13 action/native migration
  -> Task 14 policy/endpoint/diagnostic migration
  -> Task 15 trainer protocols
  -> [Task 16 script archive || Task 17 import boundaries || Task 18 docs integrity]
  -> Task 19 CI
  -> Task 20 remote runbook
  -> Task 21 final integration and Draft PR
```

### Gate A: Baseline Locked

Required before Task 1:

- exact base commit and branch;
- clean tracked worktree;
- user-owned untracked file excluded;
- local 549-test baseline green;
- local/remote environment and remote HEAD captured;
- no active experiment or remote manifold process assumed.

### Gate B: Governance Layer Complete

Required before dispatcher or file moves:

- status registry validates and authorizes zero experiments;
- experiment registry references every first-wave version/config/runner;
- artifact schema rejects dirty or scope-ambiguous formal runs;
- script inventory covers 100% of tracked entrypoints, and every entry selected for the next move wave is reviewed;
- generated status exactly matches registries.

### Gate C: Execution Layer Complete

Required before package migration:

- eval scope appears in config, metadata, metrics, and manifests;
- canonical lifecycle records started/completed/failed;
- dispatcher uses a static handler map;
- dry-run has zero filesystem/run side effects;
- frozen reproduction rejects overrides.

### Gate D: Package Migration Complete

Required before archiving scripts:

- old/new import parity for every moved symbol;
- native ordering and zero-action parity unchanged;
- dense endpoint state-dict keys and outputs unchanged;
- optional legacy dependencies do not break canonical imports;
- full maintained suite green after every migration commit.

### Gate E: Repository Hygiene Complete

Required before CI/PR:

- script archive manifest complete;
- current docs pass UTF-8/link/drift checks;
- critical artifact manifests validate against remote hashes;
- environment snapshots committed;
- no current documentation references RLimage as the manifold runtime;
- no generated document is manually edited.

## 6. Scientific Evidence Protection Matrix

| Protected evidence | Immutable property | Verification |
|---|---|---|
| Strong NWPU baseline | final and best metrics, checkpoint hash, full-val scope | Backfilled manifest plus remote SHA |
| Native zero-action parity | 196 images, zero mismatches, exact zero actions | Existing parity JSON and contract tests |
| M1 full-val set policy | identity/learned/control metrics and failed gates | Manifest, config SHA, no result rewrite |
| C1-E1 smoke actions | smoke scope, exact action candidates, frozen outcomes | Registry status and artifact hashes |
| Dense absolute endpoint | nested split, controls, train/validation gap | Split/config/result hashes |
| Detector-unseen validation | one-time validation read and calibration-shift diagnosis | Lifecycle evidence and cleanval/shift artifacts |
| Local Delta-Q learner | 48/16 split, pairwise 0.57469, failed 0.60 gate | Learner/family manifests |
| Family-prior audit | fit-only LUT metrics and post-hoc scope | Manifest records reused tune boundary |
| Residual protocol | blocked before cache/training | Status registry and failed support evidence |
| FPN-SM/AFM | baseline-only, mixed evidence | Research status; never merged into transport claim |

If a refactor test requires rewriting an immutable result file, the test design is wrong. Add an external manifest or compatibility adapter instead.

## 7. Risk Register, Detection, And Rollback

| Risk | Detection | Prevention | Rollback/stop rule |
|---|---|---|---|
| Smoke reported as full-val | scope contract and generated-table tests | mandatory scope and `limit_*` fields | mark run invalid; revert promotion commit |
| Proposal/NMS ordering changes during moves | native parity fixture and exact sequence tests | no body edits in move commits | revert the single migration commit immediately |
| Scientific status changes from parser output | registry write-protection tests | human-reviewed decision commits only | reject diff; restore registry from prior commit |
| Artifact hash/path drift | manifest verifier against remote | immutable logical refs and SHA-256 | mark unavailable/invalid; never overwrite evidence |
| Dirty-tree formal run | lifecycle guard | fail before run directory mutation | delete only newly created empty temp run after path verification |
| User untracked document staged | repository-state exclusion test | explicit path exclusion and isolated worktree | unstage only; never delete or checkout the file |
| Legacy import breaks | compatibility import matrix | shims retained for one release | revert individual namespace move |
| Optional dependency breaks canonical import | fresh-process import smoke | lazy/explicit imports | remove eager export; keep optional feature isolated |
| Randomness changes | fixed seed/output fixtures | no mixed behavior/path commits | revert; compare RNG call path before retry |
| Checkpoint no longer loads | state-dict key and fixture tests | module shims and unchanged class APIs | revert endpoint/policy move |
| Documentation silently changes old conclusions | byte/hash checks for historical reports | generated indexes, no mass rewrite | revert docs commit; add external banner/index |
| Dispatcher runs arbitrary code | static capability tests | explicit enum-to-handler map | block merge; no plugin discovery |
| GPU2 interference | guard unit tests and dry-run | no automatic retry/kill; user-authoritative `memory.free > 8192 MiB`; physical GPU2 only | guard never signals a process; cancellation requires a separate owned-PID procedure and never signals others |
| Large PR becomes unreviewable | commit/phase size checks | phase PRs and atomic commits | split before Ready review |
| Main history loses experiment lineage | branch comparison and PR review | first target feature branch; no squash by default | close PR and redesign integration strategy |

### 7.1 Global Stop Conditions

Stop the entire refactor and report to the user if any of these repeats after one focused fix attempt:

1. Exact native parity fails.
2. A critical artifact hash differs from the recorded remote artifact.
3. The maintained baseline suite cannot be restored without changing research behavior.
4. Old checkpoint compatibility requires an algorithmic change.
5. Registry statuses cannot be reconciled with committed decision documents.
6. A migration would require deleting or rewriting historical evidence.
7. The remote worktree contains unexplained tracked changes.
8. Required credentials, data, or artifacts are unavailable and no read-only verification path exists.

Do not mark the refactor blocked merely because it is large. Block only on a repeated hard condition that prevents safe progress.

## 8. Commit And Review Ledger

Expected commit sequence:

```text
docs: lock repository refactor baseline
test: lock maintained repository state
feat: add research status registry
feat: register versioned research experiments
feat: add immutable artifact manifests
fix: record explicit evaluation scope
feat: track canonical experiment lifecycle
feat: add explicit experiment dispatcher
docs: align repository with research registries
chore: inventory research scripts and entrypoints
docs: backfill critical experiment provenance
docs: record reproducible execution environments
docs: define energy transport package boundaries
refactor: isolate energy transport action core
refactor: isolate native detection transport
refactor: isolate energy transport policies
refactor: isolate dense set endpoints
refactor: isolate transport diagnostics
refactor: add structured detection trainer protocols
chore: archive scratch research scripts
refactor: organize maintained analysis scripts
chore: archive frozen experiment launchers
test: enforce canonical import boundaries
docs: validate current research documentation
ci: add maintained CPU repository checks
docs: add guarded GPU2 execution contract
docs: complete repository research-state refactor
```

Each commit entry in `docs/research/refactor_ledger.md` records:

```text
commit hash
task ID
agent/model and write owner
files changed
RED command/result
GREEN command/result
full-suite result
reviewer and findings accepted/rejected
scientific artifacts checked
rollback command
remaining risks
```

## 9. Definition Of Done

The refactor is complete only when all statements below are proven by current evidence:

### Research Truth

- [ ] The machine registry and generated current-status document both show zero authorized experiments.
- [ ] Frozen, blocked, diagnostic, baseline, historical, invalidated, and validated meanings are documented and tested.
- [ ] No result parser can change research status.
- [ ] Every current quantitative claim links a tracked artifact manifest or is labeled unverifiable/historical.

### Execution

- [ ] `scripts/run_experiment.py list/status/validate/dry-run` works deterministically.
- [ ] Frozen reproduction rejects all scientific overrides.
- [ ] Formal runs reject dirty Git, ambiguous scope, missing hashes, and unregistered capabilities.
- [ ] Canonical lifecycle writes started/completed/failed manifests without changing old result semantics.

### Package Structure

- [ ] Energy action, native, policy, endpoint, and diagnostic subpackages obey dependency rules.
- [ ] All old flat imports and documented compatibility namespaces still work.
- [ ] Importing canonical APIs performs no dataset read, file write, CUDA initialization, or remote access.
- [ ] No new canonical module imports `legacy`, `scripts`, or compatibility shims.

### Scripts And Documentation

- [ ] Every tracked script/root entrypoint appears exactly once in the reviewed inventory.
- [ ] Every moved script has an archive manifest and replacement or explicit no-replacement reason.
- [ ] README, AGENTS, architecture, report index, and generated status agree.
- [ ] Current docs pass UTF-8 and link checks; historical docs were not mass-rewritten.
- [ ] The user-owned untracked document remains untouched.

### Reproducibility And Verification

- [ ] Local and remote environment snapshots exist and contain no secrets.
- [ ] Critical artifact manifests match remote hashes.
- [ ] Native zero-action parity is unchanged.
- [ ] All 549 baseline tests and every newly added test pass locally.
- [ ] Maintained CPU tests and registry checks pass in the clean remote refactor worktree.
- [ ] GitHub CI passes without GPU, data download, or remote credentials.
- [ ] `git diff --check` passes and the refactor worktree is clean.

### Delivery

- [ ] Kimi or the documented Terra fallback has reviewed every substantive phase and the final diff.
- [ ] The completion report lists accepted/rejected reviewer findings and residual risks.
- [ ] The branch is pushed and a Draft PR targets `codex/energy-guided-roi-transport`.
- [ ] No new research result, AP claim, or experiment authorization is introduced by the refactor.

## 10. Execution Handoff

Recommended mode: **Subagent-Driven Development**. The main agent dispatches a fresh bounded builder for each task, requires RED/GREEN evidence, runs Luna verification, obtains the required Kimi review at phase gates, and integrates one atomic commit at a time.

Inline execution is permitted only for Task 0 and main-agent integration edits. Do not execute this plan as one long autonomous patch or assign the whole repository to one worker.

Unless the user explicitly overrides it before execution, the refactor worktree branch targets the current research branch exactly as specified. All other technical decisions are locked by this plan.

## 11. Plan Review Record

The plan was assembled from three disjoint read-only lanes: Luna High repository inventory, Terra High architecture design, and Terra High risk/verification audit. Main Sol XHigh reconciled their outputs against the latest committed evidence and rejected any suggestion that D1/D2 or another experiment was currently active or queued.

The required Kimi K2.7 review was attempted once through the local OAuth CLI. It produced no review because the CLI resolved the Chinese Windows user profile to a corrupted path and failed with `PermissionError` while creating `.kimi`. Per the routing contract, Kimi was not retried and DeepSeek was not substituted. A fresh read-only Terra High fallback reviewed the same files and questions.

Accepted fallback findings incorporated into this revision:

- plan-only execution revision plus audited-code-base ancestry/worktree guards;
- exact status transition and status-to-runnable matrices;
- `limited_unknown` scope taxonomy;
- separate mutable runtime and append-only reviewed manifests;
- non-runnable historical-artifact registry records;
- formal dataset/split/weight/metric/postprocess provenance;
- all-subpackage checkpoint/import/output parity;
- main-only ownership for shared files;
- artifact-index regeneration after backfill;
- original-script blob identity distinct from current wrappers;
- observation revision/time and unavailable evidence handling;
- protected historical-report blob inventory;
- locked pytest collection and bounded script/artifact waves.

Rejected fallback finding:

- Requiring GPU2 to be process-empty was rejected because it came from a stale repository-local matrix statement and conflicts with the user's later explicit rule. The locked rule is physical GPU2 only, `memory.free > 8192 MiB`, no wait solely because another process exists, and no interference with any process.
