---
status: accepted
contact: Kimi (refactor builder, Task 12)
date: 2026-07-13
deciders: Main Sol (refactor plan owner)
consulted: Kimi (boundary review, plan Task 12 Step 4)
informed: Terra High builders (Tasks 13/14 migration owners)
---

# Energy-Transport Package Boundaries

## Context and Problem Statement

`spectral_detection_posttrain/methods/energy_transport/` currently holds 28 flat
modules mixing four concerns: action primitives, native detector-coupled energy
heads, selection policies, dense set endpoints, and diagnostics. Refactor plan
Task 12 (`docs/superpowers/plans/2026-07-13-manifold-repository-research-state-refactor.md`)
locks the target subpackage layout so that Tasks 13/14 can migrate one
subpackage at a time behind compatibility shims. This ADR records the exact
mapping, the dependency rules, the shim policy, and the currently known rule
violations. **This task moves no implementation code** — it defines boundaries
and the contract test that enforces them.

## Decision Drivers

- The mapping is fixed by the refactor plan; this ADR must reproduce it faithfully.
- Dependency rules must be checkable by static (AST) analysis, without importing
  the modules under test.
- Current violations must be reported, never auto-fixed; any new violation must
  fail immediately.
- The boundary test must stay green across the Tasks 13/14 migration (flat
  modules become forwarding shims) and must ratchet: allowlist entries that no
  longer violate are stale and must be removed.

## Considered Options

- **Option A: Five subpackages per the plan mapping, flat modules kept as shims.**
  Layering `action <- native <- policy`, `endpoint` standalone, `diagnostics` as
  the leaf layer. Enforced by `tests/contracts/test_energy_transport_import_boundaries.py`.
- **Option B: Keep the flat layout and document conventions only.**
  Rejected: 28 flat modules with eager `__init__.py` imports give no
  machine-checkable structure, and the plan requires migratable boundaries.
- **Option C: Move modules immediately (no shim phase).**
  Rejected: every existing caller imports the flat paths; the plan mandates a
  shim phase with parity tests (Tasks 13/14) before any import-path break.

## Decision Outcome

Chosen option: "Five subpackages per the plan mapping, flat modules kept as
shims", because it is the plan-mandated layout and the only option that makes
the boundaries testable before any code moves.

### Target mapping (plan Task 12, verbatim)

| Target package | Current modules |
|---|---|
| `energy_transport/action/` | `actions.py`, `contracts.py`, `operators.py`, `preferences.py`, `geometric_constraints.py` |
| `energy_transport/native/` | `native_contract.py`, `native_topology.py`, `candidate_energy.py`, `benefit_energy.py` |
| `energy_transport/policy/` | `search.py`, `set_search.py`, `set_policy.py`, `global_top1.py`, `listwise_noop.py`, `joint_delta_u.py`, `adaptive_consensus.py`, `post_nms_suppress.py` |
| `energy_transport/endpoint/` | `dense_set_energy.py`, `dense_endpoint.py` |
| `energy_transport/diagnostics/` | `structure_metrics.py`, `cone_projection.py`, `high_water_mark.py`, `linear_identifiability.py`, `spatial_counterfactual.py`, `step_strata.py`, `top_focused_audit.py`, `decomposed_actionability.py`, `joint_probe_validation.py` |

### Dependency rules (plan Task 12, verbatim)

```text
action may depend on torch and core types, not trainer/experiment/dataset
native may depend on action and eval primitives, not trainers
policy may depend on action/native, not scripts or runs
endpoint may depend on pure feature/teacher modules, not runners
diagnostics may depend on stable method APIs, never write artifacts implicitly
```

### Interpretation of the rules (resolved ambiguities)

- The "not ..." clause of each rule is the enforced prohibition; the "may depend
  on ..." clause defines the allowed dependency set. The explicit grant of
  "core types" to `action` only means other subpackages must not import
  `spectral_detection_posttrain.core.*` directly (see the allowlist below).
- `torch`, `torchvision`, `numpy`, and the Python standard library are the base
  substrate allowed in every subpackage (`torch` is named in rule 1;
  `torchvision.ops` box/NMS utilities are detection eval primitives).
- For `native`, "eval primitives" includes `spectral_detection_posttrain.core`
  matching/box utilities. Since Task 14 phase 1, `native_contract.py` uses this
  grant to re-export `box_iou` and `match_predictions_to_gt` as the sanctioned
  route by which `policy` modules reach those primitives without importing
  `core` directly.
- Imports within the same target subpackage are always allowed.
- For `policy`, the allowed set is exactly `action` + `native` (+ substrate);
  "not scripts or runs" also forbids `scripts/`, `runs/`, and `legacy/` imports.
- For `endpoint`, "pure feature/teacher modules" means the endpoint subpackage
  itself (`dense_set_energy` is the teacher module) plus substrate; no other
  `energy_transport` subpackage and no `core` imports.
- For `diagnostics`, "stable method APIs" means the other four
  `energy_transport` subpackages (the maintained public surface re-exported by
  `__init__.py`); diagnostics is the leaf layer.
- "never write artifacts implicitly" is enforced as: no import-time
  (module-level) calls to artifact-writing APIs (`open` in write/append/create
  mode, `write_text`, `write_bytes`, `save`, `savefig`, `dump`, `to_csv`,
  `to_json`, `to_pickle`, `savez`, `savez_compressed`, `mkdir`, `makedirs`).
  Function bodies are exempt (a caller invoking a write is explicit);
  `if __name__ == "__main__":` blocks are exempt (explicit script entry).
  Task 14 Step 3 will add the stricter import-safety gate (no file reads, no
  directory creation, no CUDA initialization at import time); this ADR covers
  the write side only.
- `trainer/experiment/dataset` in rule 1 maps to
  `spectral_detection_posttrain.trainers`, `spectral_detection_posttrain.experiments`,
  and `spectral_detection_posttrain.datasets`; these prefixes, plus `scripts`,
  `runs`, and `legacy`, are forbidden for every subpackage.
- Any in-package import of a future subpackage directory (`energy_transport.action.*`
  etc.) is classified by that directory's subpackage, so the contract test
  stays green while Tasks 13/14 turn flat modules into forwarding shims.
- Any other internal (`spectral_detection_posttrain.*`) or third-party import
  not covered above is a violation by default; adding one requires an ADR
  amendment plus an allowlist entry.

### Shim policy

Each current flat module remains a compatibility shim for at least one complete
refactor release. `__init__.py` retains the public names but must eventually use
lazy or explicit imports that do not import every experimental module eagerly.
Concretely: Tasks 13/14 replace flat modules with forwarding shims (imports and
`__all__` only), keep the public `__all__` of the package facade unchanged, and
only then reduce the facade's eager imports (Task 14 Step 4). This contract
test classifies shim imports by their target subpackage, so the migration does
not trip the boundary rules.

### Known violations (allowlist)

The contract test reads an allowlist of known current violations. Every entry
is listed here with its removal task; the test cross-checks that this list and
the ADR agree, and fails if an allowlist entry stops being a violation (stale
entries must be deleted, not kept).

**Current status: the allowlist is empty.** The two violations recorded at
Task 12 were both resolved by Task 14 phase 1 (policy migration), which routed
the `policy` -> `core` imports through `native_contract.py` re-exports and then
deleted the allowlist entries:

| Module | Historical violation | Resolution |
|---|---|---|
| `adaptive_consensus.py` | `policy` module imported `spectral_detection_posttrain.core.matching.box_iou` directly | Task 14 phase 1 (policy migration): imports `box_iou` from `energy_transport.native.native_contract` instead; allowlist entry deleted |
| `set_search.py` | `policy` module imported `spectral_detection_posttrain.core.matching` directly | Task 14 phase 1 (policy migration): imports `match_predictions_to_gt` from `energy_transport.native.native_contract` instead; allowlist entry deleted |

No other violations exist: all `action`, `native`, `policy`, `endpoint`, and
`diagnostics` modules respect their rules, and no diagnostics module writes
artifacts at import time.

### Consequences

- Good, because the Tasks 13/14 builders get a machine-checked boundary that
  fails on any new cross-layer dependency instead of relying on review alone.
- Good, because the allowlist ratchet (stale entries fail) forces cleanup when
  Task 14 fixes the two policy violations.
- Bad, because the rule interpretation (e.g. `core` access restricted to
  `action`/`native`) is stricter than pre-refactor practice; two policy modules
  carried documented debt until Task 14 phase 1 resolved it.
- Neutral, because `__init__.py` keeps importing eagerly for now; the lazy
  facade is deferred to Task 14 Step 4.

## Validation

Compliance is validated by
`tests/contracts/test_energy_transport_import_boundaries.py`, an AST-based
static analysis over the flat modules (no imports of the code under test). It
checks: (1) the mapping covers exactly the modules on disk; (2) dependency-rule
violations equal the allowlist — new violations fail immediately, stale
allowlist entries fail; (3) diagnostics modules have no import-time
artifact-writing calls; (4) this ADR contains the five verbatim dependency
rules and every allowlist entry, so documentation and test cannot drift apart.
Full-suite gate at the time of writing: 855 passed + 1 skipped at HEAD
`c405ea1`; this task adds tests only.

## More Information

- Refactor plan: `docs/superpowers/plans/2026-07-13-manifold-repository-research-state-refactor.md` (Tasks 12-14)
- Refactor ledger: `docs/research/refactor_ledger.md`
- Removal tasks for allowlist entries: Task 13 (`refactor: isolate energy transport action core`, `refactor: isolate native detection transport`) and Task 14 (`refactor: isolate energy transport policies`, `refactor: isolate dense set endpoints`, `refactor: isolate transport diagnostics`)
- Revisit this ADR when Task 14 completes (allowlist should be empty) or when a
  new dependency need arises that the current rules reject.
