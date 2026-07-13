"""Standalone AST-based import-boundary checker for canonical layers.

Refactor plan Task 17.  Enforces the six boundary constraints verbatim:

* core must not import methods, trainers, experiments, scripts, or legacy
* methods must not import trainers, experiments, scripts, or runs
* signals must not import trainers or execute training
* trainers may import core/methods/datasets/eval, not scripts
* experiments may orchestrate all canonical layers, not legacy scripts directly
* new canonical code must not import compatibility namespaces

The checker never imports the package; it parses ``.py`` files with ``ast`` and
matches import targets against the rule table.  Known violations are listed in
``ALLOWLIST`` with an ``owner`` and a ``removal_task``.  The checker exits 0
only when every current violation is in the allowlist and every allowlist entry
is still a current violation.

Usage::

    python scripts/dev/check_import_boundaries.py
    python scripts/dev/check_import_boundaries.py --json
"""

from __future__ import annotations

import ast
import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPO_PKG = "spectral_detection_posttrain"

# Canonical layers that participate in the boundary contract.
CANONICAL_LAYERS = (
    "core",
    "methods",
    "signals",
    "trainers",
    "experiments",
    "datasets",
    "eval",
    "analysis",
    "utils",
    "visualization",
    "configs",
)

# Compatibility shim namespaces (forwarding re-exports).  New canonical code
# must not depend on these; the canonical target should be imported directly.
COMPATIBILITY_NAMESPACES = ("spectral", "models", "rlvr", "train", "matching")

# Import targets forbidden for a given importer layer.  Targets are matched as
# top-level ``scripts`` / ``legacy`` / ``runs`` or as
# ``spectral_detection_posttrain.<layer>[.*]``.
FORBIDDEN_TARGETS: dict[str, tuple[str, ...]] = {
    "core": ("methods", "trainers", "experiments", "scripts", "legacy"),
    "methods": ("trainers", "experiments", "scripts", "runs"),
    "signals": ("trainers",),
    "trainers": ("scripts",),
    "experiments": ("legacy",),
}

# Module-level call names that count as "executing training" for the signals
# layer.  Function bodies and ``if __name__ == "__main__":`` blocks are exempt.
TRAINING_EXECUTION_CALLS = {
    "train",
    "fit",
    "train_detector",
    "run_training",
    "train_model",
}

# Known current violations: (relative path, imported target) -> metadata.
# Every entry must name an owner and a concrete removal task.
ALLOWLIST: dict[tuple[str, str], dict[str, str]] = {
    # core/models/build_detector.py imports method-specific detector options at
    # module and function level.  Inverting this dependency is the core builder
    # refactor tracked separately; until then the imports are lazy where possible.
    (
        "spectral_detection_posttrain/core/models/build_detector.py",
        "spectral_detection_posttrain.experiments.schema",
    ): {
        "owner": "core.models",
        "removal_task": (
            "Move experiment-schema resolution out of core builder; "
            "core/models/build_detector.py should receive a validated config "
            "from callers and stop importing experiments.schema"
        ),
    },
    (
        "spectral_detection_posttrain/core/models/build_detector.py",
        "spectral_detection_posttrain.methods.detection.pah",
    ): {
        "owner": "core.models",
        "removal_task": (
            "Register ResidualPrototypeHead via a methods-side detector "
            "builder hook so core/models/build_detector.py can drop "
            "methods.detection.pah"
        ),
    },
    (
        "spectral_detection_posttrain/core/models/build_detector.py",
        "spectral_detection_posttrain.methods.afm.micro_afm",
    ): {
        "owner": "core.models",
        "removal_task": (
            "Register MultiScaleAFM / build_afm_block via a methods-side "
            "detector builder hook so core builder no longer imports "
            "methods.afm.micro_afm"
        ),
    },
    (
        "spectral_detection_posttrain/core/models/build_detector.py",
        "spectral_detection_posttrain.methods.manifold",
    ): {
        "owner": "core.models",
        "removal_task": (
            "Register FPN spectral/real/attention manifold wrappers through a "
            "methods-side builder registry; remove methods.manifold import "
            "from core builder"
        ),
    },
    (
        "spectral_detection_posttrain/core/models/build_detector.py",
        "spectral_detection_posttrain.methods.detection.pbg",
    ): {
        "owner": "core.models",
        "removal_task": (
            "Register PhaseBoundaryGate / LearnedSpectralGate via a "
            "methods-side detector builder hook; remove methods.detection.pbg "
            "import from core builder"
        ),
    },
    (
        "spectral_detection_posttrain/core/models/build_detector.py",
        "spectral_detection_posttrain.methods.detection.tam",
    ): {
        "owner": "core.models",
        "removal_task": (
            "Register TaskAlignedManifold via a methods-side detector builder "
            "hook; remove methods.detection.tam import from core builder"
        ),
    },
    # core/models/spectral_quality_head.py is a legacy re-export shim living
    # inside the canonical core tree.  It should be migrated into core or deleted.
    (
        "spectral_detection_posttrain/core/models/spectral_quality_head.py",
        "legacy.core.models.spectral_quality_head",
    ): {
        "owner": "core.models",
        "removal_task": (
            "Migrate legacy/core/models/spectral_quality_head.py content into "
            "a canonical core location and delete the core/models/ "
            "spectral_quality_head.py shim"
        ),
    },
    # experiments/nni_quality_trial.py is a legacy re-export shim.  The
    # experiment should be ported into canonical experiments or retired.
    (
        "spectral_detection_posttrain/experiments/nni_quality_trial.py",
        "legacy.experiments.nni_quality_trial",
    ): {
        "owner": "experiments",
        "removal_task": (
            "Port legacy.experiments.nni_quality_trial into canonical "
            "experiments or archive the experiment; remove the re-export shim "
            "experiments/nni_quality_trial.py"
        ),
    },
}


class Violation:
    """One boundary rule break: relative path, import target, rule description."""

    def __init__(self, path: str, target: str, rule: str, line: int | None = None):
        self.path = path
        self.target = target
        self.rule = rule
        self.line = line

    def key(self) -> tuple[str, str]:
        return (self.path, self.target)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "target": self.target,
            "rule": self.rule,
            "line": self.line,
        }


def _iter_import_targets(path: Path):
    """Yield every dotted import target declared in a module (any depth)."""
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise SyntaxError(f"{path}: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                # Resolve level-1/2 relative imports against the package path.
                pkg_parts = path.relative_to(ROOT).with_suffix("").parts[:-1]
                module = ".".join(pkg_parts[: len(pkg_parts) - node.level + 1])
                if node.module:
                    module = f"{module}.{node.module}" if module else node.module
            yield module, node.lineno


def _iter_module_level_calls(path: Path):
    """Yield (lineno, call name) for calls executed at import time."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
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
            if isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                if name:
                    yield node.lineno, name


def _layer_of(path: Path) -> str | None:
    """Return the canonical/shim layer directory stem, or None for root files."""
    rel = path.relative_to(ROOT / REPO_PKG)
    if len(rel.parts) < 2:
        return None
    return rel.parts[0]


def _matches(target: str, prefix: str) -> bool:
    return target == prefix or target.startswith(prefix + ".")


def _is_forbidden_for_layer(layer: str, target: str) -> str | None:
    """Return the rule description if ``target`` is forbidden for ``layer``."""
    forbidden = FORBIDDEN_TARGETS.get(layer)
    if forbidden:
        for prefix in forbidden:
            if prefix in ("scripts", "legacy", "runs"):
                if target == prefix or target.startswith(prefix + "."):
                    return f"{layer} must not import {prefix}"
            elif _matches(target, f"{REPO_PKG}.{prefix}"):
                return f"{layer} must not import {prefix}"
    return None


def _is_compatibility_namespace(target: str) -> bool:
    for ns in COMPATIBILITY_NAMESPACES:
        if _matches(target, f"{REPO_PKG}.{ns}"):
            return True
    return False


def find_violations() -> list[Violation]:
    """Scan every canonical-layer .py file and return all boundary violations."""
    violations: list[Violation] = []
    for layer in CANONICAL_LAYERS:
        layer_dir = ROOT / REPO_PKG / layer
        if not layer_dir.exists():
            continue
        for path in sorted(layer_dir.rglob("*.py")):
            if path.name == "__init__.py":
                continue
            rel = path.relative_to(ROOT).as_posix()
            for target, lineno in _iter_import_targets(path):
                if target == "__future__":
                    continue
                # Rule 6: no canonical layer may import a compatibility shim.
                if _is_compatibility_namespace(target):
                    violations.append(
                        Violation(
                            rel,
                            target,
                            "canonical code must not import compatibility namespaces",
                            lineno,
                        )
                    )
                    continue
                # Layer-specific rules.
                reason = _is_forbidden_for_layer(layer, target)
                if reason:
                    violations.append(Violation(rel, target, reason, lineno))

            # Rule 3 extra: signals must not execute training at import time.
            if layer == "signals":
                for lineno, name in _iter_module_level_calls(path):
                    if name in TRAINING_EXECUTION_CALLS:
                        violations.append(
                            Violation(
                                rel,
                                f"<module-level call: {name}()>",
                                "signals must not execute training at import time",
                                lineno,
                            )
                        )
    return violations


def _partition(violations: list[Violation]) -> tuple[list[Violation], list[tuple[str, str]]]:
    """Return (unallowlisted violations, stale allowlist keys)."""
    current_keys = {v.key() for v in violations}
    allowlist_keys = set(ALLOWLIST)
    unallowlisted = [v for v in violations if v.key() not in allowlist_keys]
    stale = sorted(allowlist_keys - current_keys)
    return unallowlisted, stale


def check() -> dict[str, Any]:
    """Run the boundary check and return a serializable result object."""
    violations = find_violations()
    unallowlisted, stale = _partition(violations)
    return {
        "violations": [v.to_dict() for v in violations],
        "unallowlisted": [v.to_dict() for v in unallowlisted],
        "stale_allowlist": [
            {"path": p, "target": t, "meta": ALLOWLIST[(p, t)]} for p, t in stale
        ],
        "allowlist_size": len(ALLOWLIST),
        "clean": not unallowlisted and not stale,
    }


def _format_human(result: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"Canonical import-boundary check: {len(result['violations'])} total violation(s)")
    if result["unallowlisted"]:
        lines.append("")
        lines.append("NEW / UNALLOWLISTED VIOLATIONS (must be fixed or allowlisted with owner + task):")
        for v in result["unallowlisted"]:
            lines.append(f"  {v['path']}:{v['line']} imports {v['target']} -> {v['rule']}")
    if result["stale_allowlist"]:
        lines.append("")
        lines.append("STALE ALLOWLIST ENTRIES (no longer violations; remove them):")
        for item in result["stale_allowlist"]:
            lines.append(f"  {item['path']} -> {item['target']}")
    if result["clean"]:
        lines.append("Result: clean (all violations are current and allowlisted)")
    else:
        lines.append("Result: FAIL")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Check canonical import boundaries.")
    parser.add_argument("--json", action="store_true", help="emit JSON output")
    args = parser.parse_args(argv)

    result = check()

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(_format_human(result))

    return 0 if result["clean"] else 1


if __name__ == "__main__":
    sys.exit(main())
