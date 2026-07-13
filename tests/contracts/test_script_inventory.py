"""Contract tests for the repository script inventory (refactor Task 9).

The inventory at
``spectral_detection_posttrain/configs/registry/script_inventory.json`` must
cover every tracked file under ``scripts/`` plus every tracked root-level
launcher/analysis/trainer file (``*.py``, ``*.bat``, ``*.sh``).  The expected
path set is computed independently here from ``git ls-files`` so that the
test does not simply mirror the generator's discovery code.

Both the generator and this test apply one documented self-inclusion rule:
``scripts/dev/inventory_scripts.py`` is part of the expected set whenever it
exists on disk, even before it is tracked (it is committed together with the
inventory, so post-commit ``git ls-files`` yields the identical set).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GENERATOR_PATH = ROOT / "scripts" / "dev" / "inventory_scripts.py"
INVENTORY_PATH = (
    ROOT
    / "spectral_detection_posttrain"
    / "configs"
    / "registry"
    / "script_inventory.json"
)
GENERATOR_REPO_PATH = "scripts/dev/inventory_scripts.py"

REQUIRED_FIELDS = {
    "path": str,
    "kind": str,
    "research_status": str,
    "capability": str,
    "replacement": str,
    "archive_wave": int,
    "known_inputs": list,
    "known_outputs": list,
    "notes": str,
    "classification_state": str,
    "review_required": bool,
}
KINDS = {
    "trainer",
    "evaluator",
    "analysis",
    "launcher",
    "diagnostic",
    "generator",
    "utility",
    "unknown",
}
RESEARCH_STATUSES = {"maintained", "frozen_reproduction", "historical", "scratch", "unknown"}
CLASSIFICATION_STATES = {"proposed", "reviewed"}
# Path-only unambiguous classifications that may carry classification_state
# "reviewed"; everything else must stay proposed + review_required.
REVIEWED_PREFIXES = ("scripts/dev/", "scripts/legacy/")
REVIEWED_EXACT = {"scripts/round28_train_eval.py"}


def load_generator():
    assert GENERATOR_PATH.exists(), "script inventory generator has not been implemented"
    spec = importlib.util.spec_from_file_location("inventory_scripts", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_ls_files(*pathspecs: str) -> list[str]:
    result = subprocess.run(
        ["git", "-c", "core.quotepath=false", "ls-files", "--", *pathspecs],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def expected_inventory_paths() -> set[str]:
    """Independent computation of the required coverage set."""
    paths = set(git_ls_files("scripts/"))
    for pattern in ("*.py", "*.bat", "*.sh"):
        paths.update(p for p in git_ls_files(pattern) if "/" not in p)
    if (ROOT / GENERATOR_REPO_PATH).exists():
        paths.add(GENERATOR_REPO_PATH)
    return paths


@pytest.fixture(scope="module")
def generator():
    return load_generator()


@pytest.fixture(scope="module")
def inventory():
    assert INVENTORY_PATH.exists(), "script_inventory.json has not been generated"
    return json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def entries(inventory) -> list[dict]:
    return inventory["entries"]


def test_inventory_file_exists():
    assert INVENTORY_PATH.exists(), "script_inventory.json has not been generated"


def test_generator_exists():
    assert GENERATOR_PATH.exists(), "script inventory generator has not been implemented"


def test_coverage_is_exactly_complete(entries):
    inventory_paths = {entry["path"] for entry in entries}
    expected = expected_inventory_paths()
    missing = expected - inventory_paths
    extras = inventory_paths - expected
    assert not missing, f"inventory is missing tracked scripts: {sorted(missing)}"
    assert not extras, f"inventory contains paths outside the tracked set: {sorted(extras)}"
    assert len(entries) == len(inventory_paths), "duplicate paths in inventory"


def test_every_entry_matches_schema(entries):
    tracked = set(git_ls_files())
    assert entries, "inventory has no entries"
    for entry in entries:
        assert set(entry) == set(REQUIRED_FIELDS), (
            f"{entry.get('path')}: field set mismatch "
            f"(missing={set(REQUIRED_FIELDS) - set(entry)}, extra={set(entry) - set(REQUIRED_FIELDS)})"
        )
        for field, field_type in REQUIRED_FIELDS.items():
            value = entry[field]
            if field_type is bool:
                assert isinstance(value, bool), f"{entry['path']}.{field} must be bool"
            else:
                assert isinstance(value, field_type) and not isinstance(value, bool), (
                    f"{entry['path']}.{field} must be {field_type.__name__}"
                )
        assert entry["kind"] in KINDS, f"{entry['path']}: bad kind {entry['kind']}"
        assert entry["research_status"] in RESEARCH_STATUSES, (
            f"{entry['path']}: bad research_status {entry['research_status']}"
        )
        assert entry["classification_state"] in CLASSIFICATION_STATES, (
            f"{entry['path']}: bad classification_state"
        )
        assert entry["archive_wave"] == 0, f"{entry['path']}: archive_wave must be 0"
        assert all(isinstance(x, str) for x in entry["known_inputs"])
        assert all(isinstance(x, str) for x in entry["known_outputs"])
        assert entry["path"] in tracked or entry["path"] == GENERATOR_REPO_PATH, (
            f"{entry['path']}: not a tracked repository file"
        )


def test_review_semantics_are_consistent(entries):
    for entry in entries:
        path = entry["path"]
        if entry["classification_state"] == "reviewed":
            assert entry["review_required"] is False, f"{path}: reviewed must not require review"
            assert path.startswith(REVIEWED_PREFIXES) or path in REVIEWED_EXACT, (
                f"{path}: reviewed is only allowed for path-unambiguous entries"
            )
        else:
            assert entry["review_required"] is True, f"{path}: proposed entries need review"


def test_generation_is_deterministic(generator):
    first = generator.serialize(generator.build_inventory())
    second = generator.serialize(generator.build_inventory())
    assert first == second, "generator output is not deterministic"
    first_hash = hashlib.sha256(first.encode("utf-8")).hexdigest()
    second_hash = hashlib.sha256(second.encode("utf-8")).hexdigest()
    assert first_hash == second_hash


def test_shipped_inventory_has_no_drift(generator):
    ok, message = generator.check_drift()
    assert ok, message


def test_serialized_form_is_canonical(entries):
    raw = INVENTORY_PATH.read_text(encoding="utf-8")
    assert raw.endswith("\n") and not raw.endswith("\n\n"), "exactly one trailing newline"
    parsed = json.loads(raw)
    assert raw == json.dumps(parsed, indent=2, sort_keys=True) + "\n", (
        "inventory file is not in canonical sorted-key form"
    )
    paths = [entry["path"] for entry in entries]
    assert paths == sorted(paths), "entries must be sorted by path"


def test_round28_canonical_runner_spot_check(entries):
    by_path = {entry["path"]: entry for entry in entries}
    assert "scripts/round28_train_eval.py" in by_path
    entry = by_path["scripts/round28_train_eval.py"]
    assert entry["kind"] == "trainer"
    assert entry["research_status"] == "maintained"
    assert entry["classification_state"] == "reviewed"
    assert entry["review_required"] is False


def test_legacy_scripts_are_historical_when_present(entries):
    legacy_tracked = set(git_ls_files("scripts/legacy/"))
    by_path = {entry["path"]: entry for entry in entries}
    for path in sorted(legacy_tracked):
        entry = by_path[path]
        assert entry["research_status"] == "historical", path
        assert entry["classification_state"] == "reviewed", path
    # Empty-safe: with no tracked scripts/legacy/ files this loop is a no-op.
    legacy_entries = [e for e in entries if e["path"].startswith("scripts/legacy/")]
    assert {e["path"] for e in legacy_entries} == legacy_tracked
