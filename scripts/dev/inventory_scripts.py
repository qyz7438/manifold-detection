"""Deterministic inventory of tracked research scripts and entrypoints.

Refactor Task 9: build ``spectral_detection_posttrain/configs/registry/
script_inventory.json`` covering every tracked file under ``scripts/`` plus
every tracked root-level ``*.py`` / ``*.bat`` / ``*.sh`` launcher, analysis
script, or trainer.  The generator NEVER moves, deletes, or executes a
discovered script; discovery is pure ``git ls-files`` plus path/name
heuristics.

Coverage rule (must stay in sync with
``tests/contracts/test_script_inventory.py``):

* every path reported by ``git ls-files -- scripts/``;
* every path reported by ``git ls-files -- '*.py' | '*.bat' | '*.sh'`` that
  has no ``/`` (root-level launchers such as ``run_smoke.bat`` and root
  analysis/trainer scripts such as ``analyze_*.py``);
* self-inclusion: ``scripts/dev/inventory_scripts.py`` itself, whenever it
  exists on disk.  It is committed together with the inventory, so after the
  commit ``git ls-files`` yields the identical set; this rule only bridges
  the pre-commit window.

Classification conventions:

* ``kind`` and ``research_status`` are proposed from the ordered RULES table
  below (first match wins).  Scientific status authority belongs to the
  research registries, not these heuristics, so every heuristic match keeps
  ``classification_state="proposed"`` and ``review_required=true``.
* ``classification_state="reviewed"`` is used ONLY where the classification
  is unambiguous from the path alone: ``scripts/legacy/**`` (historical),
  ``scripts/dev/**`` (maintained dev tooling), and
  ``scripts/round28_train_eval.py`` (maintained canonical detection runner).
* ``capability`` convention: a dotted capability string where confidently
  inferable from the filename (see CAPABILITY_RULES), otherwise the empty
  string ``""``.  The string ``"unknown"`` is never used for capability.
* ``replacement``: ``scripts/run_experiment.py`` for trainer entries the
  registry/dispatcher is expected to supersede (all ``train_*`` scripts
  other than the canonical runner), else ``""``.
* ``archive_wave`` is 0 for every entry; waves are adjudicated in Task 16.
* ``known_inputs``/``known_outputs`` stay empty unless trivially obvious
  from the filename (evaluators output evaluation metrics); empty lists are
  the honest default.

CLI:

* ``python scripts/dev/inventory_scripts.py --write``  regenerate the JSON
* ``python scripts/dev/inventory_scripts.py --check``  exit 1 on drift
* ``python scripts/dev/inventory_scripts.py --json``   print stats
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = (
    ROOT
    / "spectral_detection_posttrain"
    / "configs"
    / "registry"
    / "script_inventory.json"
)
GENERATOR_REPO_PATH = "scripts/dev/inventory_scripts.py"
REPLACEMENT_RUNNER = "scripts/run_experiment.py"

RULE_TABLE_VERSION = 1

# Ordered heuristic rule table.  First match wins.  Match keys:
#   exact             full repo-relative path
#   path_prefix       repo-relative path prefix
#   basename_prefix   str or tuple tested with str.startswith on the basename
#   basename_contains str or tuple tested with substring containment
#   extension         str or tuple of file suffixes
# Result keys default to: kind="unknown", research_status="unknown",
# classification_state="proposed", review_required=True, and empty strings
# or lists for the remaining fields.
RULES = [
    {
        "id": "legacy-tree",
        "match": {"path_prefix": "scripts/legacy/"},
        "kind": "unknown",
        "research_status": "historical",
        "classification_state": "reviewed",
        "review_required": False,
        "notes": "archived under scripts/legacy/ (historical)",
    },
    {
        "id": "dev-tooling",
        "match": {"path_prefix": "scripts/dev/"},
        "kind": "utility",
        "research_status": "maintained",
        "classification_state": "reviewed",
        "review_required": False,
        "notes": "maintained dev tooling",
    },
    {
        "id": "canonical-runner",
        "match": {"exact": "scripts/round28_train_eval.py"},
        "kind": "trainer",
        "research_status": "maintained",
        "capability": "detection.runner.round28",
        "classification_state": "reviewed",
        "review_required": False,
        "notes": "round28 canonical detection runner",
    },
    {
        "id": "scratch-oneoff",
        "match": {"basename_prefix": "_"},
        "kind": "diagnostic",
        "research_status": "scratch",
        "notes": "underscore-prefixed one-off script (proposed scratch)",
    },
    {
        "id": "shell-launcher",
        "match": {"extension": (".sh", ".bat")},
        "kind": "launcher",
    },
    {
        "id": "run-launcher",
        "match": {"basename_prefix": "run_"},
        "kind": "launcher",
    },
    {
        "id": "embedded-run-launcher",
        "match": {"basename_contains": "_run_"},
        "kind": "launcher",
    },
    {
        "id": "pipeline-launcher",
        "match": {"basename_contains": "pipeline"},
        "kind": "launcher",
    },
    {
        "id": "trainer",
        "match": {"basename_prefix": "train_"},
        "kind": "trainer",
        "replacement": REPLACEMENT_RUNNER,
    },
    {
        "id": "evaluator",
        "match": {"basename_prefix": ("eval_", "re_evaluate_")},
        "kind": "evaluator",
        "known_outputs": ["evaluation metrics"],
    },
    {
        "id": "analysis",
        "match": {"basename_prefix": ("analyze_", "summarize_", "scan_", "aggregate_")},
        "kind": "analysis",
    },
    {
        "id": "analysis-token",
        "match": {"basename_contains": ("_analysis", "_summarize")},
        "kind": "analysis",
    },
    {
        "id": "diagnostic",
        "match": {
            "basename_prefix": (
                "audit_",
                "check_",
                "diagnose_",
                "explore_",
                "probe_",
                "profile_",
                "test_",
                "trace_",
                "validate_",
                "verify_",
            )
        },
        "kind": "diagnostic",
    },
    {
        "id": "diagnostic-token",
        "match": {"basename_contains": "diagnostic"},
        "kind": "diagnostic",
    },
    {
        "id": "generator",
        "match": {"basename_prefix": ("build_", "write_")},
        "kind": "generator",
    },
    {
        "id": "utility",
        "match": {"basename_prefix": ("download_", "notify_")},
        "kind": "utility",
    },
]

# Capability strings where confidently inferable from the filename.
# ("exact", path, capability) or ("regex", pattern_with_group_1, template).
CAPABILITY_RULES = [
    (
        "exact",
        "scripts/train_nwpu_dense_endpoint_absolute.py",
        "detection.energy_transport.dense_endpoint",
    ),
    (
        "exact",
        "scripts/train_dense_endpoint_geometry_control.py",
        "detection.energy_transport.dense_endpoint.geometry_control",
    ),
    (
        "exact",
        "scripts/train_dense_local_delta_learner.py",
        "detection.energy_transport.dense_local_delta",
    ),
    (
        "regex",
        r"^scripts/(?:train|eval)_nwpu_([a-z]\d[a-z]?)_",
        "detection.energy_transport.{0}",
    ),
]

CODE_EXTENSIONS = (".py", ".sh", ".bat")

CONVENTIONS = {
    "archive_wave_policy": "0 for all entries in Task 9; archive waves are adjudicated in Task 16",
    "capability_not_inferable": "",
    "classification_states": ["proposed", "reviewed"],
    "coverage": "git ls-files -- scripts/ plus tracked root-level *.py/*.bat/*.sh, plus generator self-inclusion while untracked",
    "generated_by": "scripts/dev/inventory_scripts.py --write",
    "replacement_default": "",
    "status_authority": "research registries; filename heuristics are proposals only and keep review_required=true",
}


def git_ls_files(*pathspecs: str, root: Path = ROOT) -> list[str]:
    result = subprocess.run(
        ["git", "-c", "core.quotepath=false", "ls-files", "--", *pathspecs],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def discover_inventory_paths(root: Path = ROOT) -> list[str]:
    """Sorted repo-relative paths the inventory must cover."""
    paths = set(git_ls_files("scripts/", root=root))
    for pattern in ("*.py", "*.bat", "*.sh"):
        paths.update(p for p in git_ls_files(pattern, root=root) if "/" not in p)
    if (root / GENERATOR_REPO_PATH).exists():
        paths.add(GENERATOR_REPO_PATH)
    return sorted(paths)


def _matches(rule: dict, path: str, basename: str) -> bool:
    spec = rule["match"]
    if "exact" in spec and path == spec["exact"]:
        return True
    if "path_prefix" in spec and path.startswith(spec["path_prefix"]):
        return True
    if "basename_prefix" in spec and basename.startswith(spec["basename_prefix"]):
        return True
    if "basename_contains" in spec:
        needles = spec["basename_contains"]
        if isinstance(needles, str):
            needles = (needles,)
        if any(needle in basename for needle in needles):
            return True
    if "extension" in spec and basename.endswith(spec["extension"]):
        return True
    return False


def infer_capability(path: str) -> str:
    for mode, pattern, value in CAPABILITY_RULES:
        if mode == "exact":
            if path == pattern:
                return value
        else:
            match = re.match(pattern, path)
            if match:
                return value.format(*match.groups())
    return ""


def classify_path(path: str) -> dict:
    basename = path.rsplit("/", 1)[-1]
    entry = {
        "path": path,
        "kind": "unknown",
        "research_status": "unknown",
        "capability": "",
        "replacement": "",
        "archive_wave": 0,
        "known_inputs": [],
        "known_outputs": [],
        "notes": "",
        "classification_state": "proposed",
        "review_required": True,
    }
    for rule in RULES:
        if not _matches(rule, path, basename):
            continue
        for key in (
            "kind",
            "research_status",
            "capability",
            "replacement",
            "known_inputs",
            "known_outputs",
            "notes",
            "classification_state",
            "review_required",
        ):
            if key in rule:
                entry[key] = rule[key]
        break
    if not basename.endswith(CODE_EXTENSIONS) and entry["kind"] != "unknown":
        # Captured artifacts (e.g. *.txt outputs) are not classifiable code.
        entry["kind"] = "unknown"
    if not entry["capability"]:
        entry["capability"] = infer_capability(path)
    return entry


def build_inventory(root: Path = ROOT) -> dict:
    entries = [classify_path(path) for path in discover_inventory_paths(root)]
    return {
        "conventions": CONVENTIONS,
        "entries": entries,
        "rule_table_version": RULE_TABLE_VERSION,
    }


def serialize(inventory: dict) -> str:
    return json.dumps(inventory, indent=2, sort_keys=True) + "\n"


def check_drift(root: Path = ROOT, inventory_path: Path = INVENTORY_PATH) -> tuple[bool, str]:
    generated = serialize(build_inventory(root))
    if not inventory_path.exists():
        return False, f"inventory file is missing: {inventory_path}"
    on_disk = inventory_path.read_text(encoding="utf-8")
    if on_disk != generated:
        return False, (
            "inventory drift: regenerate with "
            "`python scripts/dev/inventory_scripts.py --write`"
        )
    return True, "inventory is up to date"


def collect_stats(inventory: dict) -> dict:
    by_kind: dict[str, int] = {}
    by_research_status: dict[str, int] = {}
    by_classification_state: dict[str, int] = {}
    review_required = 0
    for entry in inventory["entries"]:
        by_kind[entry["kind"]] = by_kind.get(entry["kind"], 0) + 1
        by_research_status[entry["research_status"]] = (
            by_research_status.get(entry["research_status"], 0) + 1
        )
        by_classification_state[entry["classification_state"]] = (
            by_classification_state.get(entry["classification_state"], 0) + 1
        )
        if entry["review_required"]:
            review_required += 1
    return {
        "by_classification_state": by_classification_state,
        "by_kind": by_kind,
        "by_research_status": by_research_status,
        "review_required": review_required,
        "total": len(inventory["entries"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="regenerate the inventory file")
    group.add_argument("--check", action="store_true", help="exit 1 when the file drifts")
    group.add_argument("--json", action="store_true", help="print inventory statistics")
    args = parser.parse_args(argv)

    if args.write:
        content = serialize(build_inventory())
        INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        INVENTORY_PATH.write_text(content, encoding="utf-8", newline="\n")
        print(f"wrote {INVENTORY_PATH} ({len(content.encode('utf-8'))} bytes)")
        return 0
    if args.check:
        ok, message = check_drift()
        print(message)
        return 0 if ok else 1
    stats = collect_stats(build_inventory())
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
