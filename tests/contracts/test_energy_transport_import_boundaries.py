"""Static import-boundary contract for the energy_transport package.

Refactor plan Task 12: this test pins the target subpackage mapping and the
five dependency rules for ``spectral_detection_posttrain/methods/energy_transport/``
while the package is still flat. It uses AST analysis only — the modules under
test are never imported — so it cannot be defeated by import-time side effects.

Known current violations are read from ALLOWLIST and reported, never
auto-fixed. Any violation not in the allowlist fails immediately; any allowlist
entry that stops being a violation also fails (stale entries must be removed).
The ADR cross-check keeps
``docs/decisions/adr-energy-transport-package-boundaries.md`` in lockstep with
the rules and the allowlist.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = "spectral_detection_posttrain.methods.energy_transport"
PACKAGE_DIR = ROOT / "spectral_detection_posttrain" / "methods" / "energy_transport"
ADR_PATH = ROOT / "docs" / "decisions" / "adr-energy-transport-package-boundaries.md"

# The five dependency rules, verbatim from the refactor plan (Task 12).
DEPENDENCY_RULES = (
    "action may depend on torch and core types, not trainer/experiment/dataset",
    "native may depend on action and eval primitives, not trainers",
    "policy may depend on action/native, not scripts or runs",
    "endpoint may depend on pure feature/teacher modules, not runners",
    "diagnostics may depend on stable method APIs, never write artifacts implicitly",
)

# Target mapping, verbatim from the refactor plan (Task 12).
SUBPACKAGE_MODULES = {
    "action": (
        "actions",
        "contracts",
        "operators",
        "preferences",
        "geometric_constraints",
    ),
    "native": (
        "native_contract",
        "native_topology",
        "candidate_energy",
        "benefit_energy",
    ),
    "policy": (
        "search",
        "set_search",
        "set_policy",
        "global_top1",
        "listwise_noop",
        "joint_delta_u",
        "adaptive_consensus",
        "post_nms_suppress",
    ),
    "endpoint": (
        "dense_set_energy",
        "dense_endpoint",
    ),
    "diagnostics": (
        "structure_metrics",
        "cone_projection",
        "high_water_mark",
        "linear_identifiability",
        "spatial_counterfactual",
        "step_strata",
        "top_focused_audit",
        "decomposed_actionability",
        "joint_probe_validation",
    ),
}

MODULE_TO_SUBPACKAGE = {
    module: subpackage
    for subpackage, modules in SUBPACKAGE_MODULES.items()
    for module in modules
}
SUBPACKAGE_NAMES = frozenset(SUBPACKAGE_MODULES)

# Allowed cross-subpackage imports inside energy_transport (see ADR rule
# interpretation: layering action <- native <- policy, endpoint standalone,
# diagnostics as the leaf layer over the stable method APIs).
ALLOWED_CROSS_IMPORTS = {
    "action": {"action"},
    "native": {"native", "action"},
    "policy": {"policy", "action", "native"},
    "endpoint": {"endpoint"},
    "diagnostics": {"diagnostics", "action", "native", "policy", "endpoint"},
}

# Subpackages allowed to import spectral_detection_posttrain.core.* directly
# (rule 1 grants action "core types"; rule 2 counts core matching/box
# utilities as native eval primitives).
CORE_TYPES_ALLOWED = {"action", "native"}

# Base substrate allowed in every subpackage.
BASE_ALLOWED_IMPORTS = {"torch", "torchvision", "numpy"}

# Forbidden for every subpackage (trainer/experiment/dataset in rule 1,
# scripts/runs in rule 3, runners in rule 4, plus the legacy tree).
FORBIDDEN_EXTERNAL_PREFIXES = (
    "spectral_detection_posttrain.trainers",
    "spectral_detection_posttrain.experiments",
    "spectral_detection_posttrain.datasets",
    "scripts",
    "runs",
    "legacy",
)

# Import-time artifact-writing APIs forbidden in diagnostics modules
# (rule 5: "never write artifacts implicitly").
ARTIFACT_WRITE_CALLS = {
    "write_text",
    "write_bytes",
    "save",
    "savefig",
    "dump",
    "to_csv",
    "to_json",
    "to_pickle",
    "savez",
    "savez_compressed",
    "mkdir",
    "makedirs",
}

# Known current violations: (flat module stem, imported target) -> removal task.
# Every entry must also appear in the ADR. Entries are removed by the task
# listed; the test fails if an entry stops being a violation before removal.
# Task 14 phase 1 (policy migration) resolved the two policy -> core entries
# by routing box_iou and match_predictions_to_gt through
# energy_transport.native.native_contract; the allowlist is intentionally
# empty and the ratchet keeps it that way.
ALLOWLIST: dict[tuple[str, str], str] = {}


def _resolve_import_from(node: ast.ImportFrom) -> str:
    """Resolve an ImportFrom node to an absolute dotted module path."""
    if node.level:
        # Relative imports inside this package are always level-1.
        return f"{PACKAGE_ROOT}.{node.module}" if node.module else PACKAGE_ROOT
    return node.module or ""


def _iter_import_targets(path: Path):
    """Yield every dotted import target declared in a module (any depth)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            module = _resolve_import_from(node)
            if module == PACKAGE_ROOT:
                # `from <pkg> import actions` names a submodule.
                for alias in node.names:
                    yield f"{module}.{alias.name}"
            else:
                yield module


def _classify_violation(module_stem: str, target: str) -> str | None:
    """Return the rule broken by `module_stem` importing `target`, or None."""
    subpackage = MODULE_TO_SUBPACKAGE[module_stem]
    root = target.split(".")[0]
    if root == "__future__":
        return None

    # In-package imports (flat modules today, subpackage dirs after Tasks 13/14).
    if target == PACKAGE_ROOT or target.startswith(PACKAGE_ROOT + "."):
        remainder = target[len(PACKAGE_ROOT) + 1 :].split(".")[0]
        if not remainder:
            return None
        target_subpackage = MODULE_TO_SUBPACKAGE.get(remainder, remainder)
        if target_subpackage in SUBPACKAGE_NAMES and (
            target_subpackage not in ALLOWED_CROSS_IMPORTS[subpackage]
        ):
            return (
                f"cross-subpackage import {subpackage} -> {target_subpackage} "
                f"is not allowed"
            )
        return None

    if root in BASE_ALLOWED_IMPORTS or root in sys.stdlib_module_names:
        return None

    if any(
        target == prefix or target.startswith(prefix + ".")
        for prefix in FORBIDDEN_EXTERNAL_PREFIXES
    ):
        return "forbidden external import (trainer/experiment/dataset/scripts/runs/legacy)"
    if target.startswith("spectral_detection_posttrain."):
        if target.startswith("spectral_detection_posttrain.core"):
            if subpackage not in CORE_TYPES_ALLOWED:
                return f"direct core import is not allowed for {subpackage}"
            return None
        return f"unclassified internal import is not allowed for {subpackage}"

    return f"unclassified third-party import is not allowed for {subpackage}"


def find_boundary_violations() -> dict[tuple[str, str], str]:
    """Map (module stem, import target) -> broken rule for every violation."""
    violations: dict[tuple[str, str], str] = {}
    for module_stem in MODULE_TO_SUBPACKAGE:
        path = PACKAGE_DIR / f"{module_stem}.py"
        for target in _iter_import_targets(path):
            broken_rule = _classify_violation(module_stem, target)
            if broken_rule is not None:
                violations[(module_stem, target)] = broken_rule
    return violations


def _module_level_calls(path: Path):
    """Yield (lineno, call name) for calls executed at import time.

    Function/class bodies are exempt (a caller invoking a write is explicit),
    and `if __name__ == "__main__":` blocks are exempt (script entry point).
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if (
            isinstance(statement, ast.If)
            and isinstance(statement.test, ast.Compare)
            and isinstance(statement.test.left, ast.Name)
            and statement.test.left.id == "__name__"
        ):
            continue
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name in ARTIFACT_WRITE_CALLS:
                yield node.lineno, name
            elif name == "open":
                mode = None
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    mode = node.args[1].value
                for keyword in node.keywords:
                    if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                        mode = keyword.value.value
                if isinstance(mode, str) and any(flag in mode for flag in "wax+"):
                    yield node.lineno, "open(write mode)"


def find_diagnostics_side_effects() -> list[str]:
    """Report import-time artifact-writing calls in diagnostics modules."""
    findings = []
    for module_stem in SUBPACKAGE_MODULES["diagnostics"]:
        path = PACKAGE_DIR / f"{module_stem}.py"
        for lineno, name in _module_level_calls(path):
            findings.append(f"{module_stem}.py:{lineno}: import-time call to {name}()")
    return findings


# ---------------------------------------------------------------------------
# Mapping drift guard
# ---------------------------------------------------------------------------


def test_mapping_covers_exactly_the_flat_modules_on_disk():
    on_disk = {
        path.stem
        for path in PACKAGE_DIR.glob("*.py")
        if path.name != "__init__.py"
    }
    mapped = set(MODULE_TO_SUBPACKAGE)
    assert mapped == on_disk, (
        f"mapping drift: unmapped modules {sorted(on_disk - mapped)}, "
        f"missing files {sorted(mapped - on_disk)}"
    )


@pytest.mark.parametrize(
    "subpackage,modules",
    sorted(SUBPACKAGE_MODULES.items()),
    ids=[name for name in sorted(SUBPACKAGE_MODULES)],
)
def test_mapping_matches_plan_table(subpackage, modules):
    # Every mapped module exists on disk and maps to exactly one subpackage.
    for module in modules:
        assert (PACKAGE_DIR / f"{module}.py").exists(), module
        assert MODULE_TO_SUBPACKAGE[module] == subpackage
    assert len(set(modules)) == len(modules), f"duplicate in {subpackage} mapping"


# ---------------------------------------------------------------------------
# Dependency rules
# ---------------------------------------------------------------------------


def test_no_new_dependency_rule_violations():
    violations = find_boundary_violations()
    new_violations = {
        key: rule for key, rule in violations.items() if key not in ALLOWLIST
    }
    assert not new_violations, (
        "new energy_transport import-boundary violations (reported, not "
        "auto-fixed; amend the ADR and allowlist to accept): "
        + "; ".join(
            f"{module}.py imports {target} ({rule})"
            for (module, target), rule in sorted(new_violations.items())
        )
    )


def test_allowlist_entries_are_current_violations():
    violations = find_boundary_violations()
    stale = set(ALLOWLIST) - set(violations)
    assert not stale, (
        "allowlist entries no longer violate the rules; remove them from the "
        "allowlist and the ADR: "
        + "; ".join(f"{module}.py -> {target}" for module, target in sorted(stale))
    )


def test_known_violations_are_reported_not_fixed():
    # Documentation test: the allowlist records each known violation with its
    # removal task. The violations themselves must still be present (this task
    # moves no implementation code).
    violations = find_boundary_violations()
    for (module, target), removal_task in sorted(ALLOWLIST.items()):
        assert (module, target) in violations
        assert removal_task, f"{module}.py -> {target} has no removal task"


def test_diagnostics_modules_write_no_artifacts_at_import_time():
    findings = find_diagnostics_side_effects()
    assert not findings, (
        "diagnostics must never write artifacts implicitly: " + "; ".join(findings)
    )


# ---------------------------------------------------------------------------
# ADR cross-check
# ---------------------------------------------------------------------------


def test_adr_documents_rules_mapping_and_allowlist():
    assert ADR_PATH.exists(), "boundary ADR has not been written"
    adr = ADR_PATH.read_text(encoding="utf-8")
    for rule in DEPENDENCY_RULES:
        assert rule in adr, f"ADR is missing dependency rule: {rule}"
    for subpackage, modules in SUBPACKAGE_MODULES.items():
        assert f"energy_transport/{subpackage}/" in adr
        for module in modules:
            assert f"`{module}.py`" in adr, f"ADR is missing module {module}.py"
    for (module, target), removal_task in sorted(ALLOWLIST.items()):
        assert f"`{module}.py`" in adr, f"ADR is missing allowlist module {module}.py"
        assert target in adr, f"ADR is missing allowlist target {target}"
        assert removal_task in adr, f"ADR is missing removal task {removal_task}"
