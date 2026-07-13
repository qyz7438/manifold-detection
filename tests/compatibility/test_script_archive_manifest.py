"""Contract tests for the script archive manifest (refactor Task 16).

``scripts/archive/manifest.json`` records every script moved out of its
historical entrypoint path during the controlled archival waves. Each moved
script must carry enough evidence to reproduce or retire the original
invocation: original path, original git blob identity, destination, archive
wave, kind, research status, exact historical invocation (or an explicit
"none documented"), current reproduction wrapper (or "none"), replacement
command (or "none"), runnable state, required ignored artifacts, and known
failure modes.

Moves are byte-identical: the archived file's current ``git hash-object``
must equal the recorded original blob SHA, and the recorded blob must exist
in the git object database. Scratch-wave (A) entries must have no wrapper at
the old path; future waves that leave wrappers must keep the original blob
identity distinct from the wrapper now occupying the old path. Maintained
code (``tests/`` and ``spectral_detection_posttrain/``) must not import or
reference scratch-wave (A) modules by their original path; wave-B/C modules are
intentionally still reachable through their wrappers at the old paths.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "scripts" / "archive" / "manifest.json"
SCHEMA_VERSION = "script-archive-manifest.v1"

REQUIRED_FIELDS: dict[str, type] = {
    "original_entrypoint_path": str,
    "original_blob_sha": str,
    "destination": str,
    "archive_wave": str,
    "kind": str,
    "research_status": str,
    "historical_invocation": str,
    "current_reproduction_wrapper": str,
    "replacement_command": str,
    "runnable_state": str,
    "required_ignored_artifacts": list,
    "known_failure_modes": list,
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
ARCHIVE_WAVES = {"A", "B", "C"}
RUNNABLE_STATES = {"runnable_as_archived", "runnable_via_wrapper", "not_runnable"}
BLOB_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
NONE_MARKERS = {"none", "none documented"}

SCANNED_TREES = (ROOT / "tests", ROOT / "spectral_detection_posttrain")

#: Where each archive wave must place moved files.
DESTINATION_PREFIXES = {
    "A": "scripts/archive/",
    "B": "scripts/analysis/",
    "C": "scripts/archive/historical/",
}


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


@pytest.fixture(scope="module")
def manifest() -> dict:
    assert MANIFEST_PATH.exists(), "scripts/archive/manifest.json has not been created"
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def entries(manifest) -> list[dict]:
    return manifest["entries"]


@pytest.fixture(scope="module")
def maintained_sources() -> dict[Path, str]:
    """Text of every maintained Python source, read once for all checks."""
    sources: dict[Path, str] = {}
    for tree_root in SCANNED_TREES:
        for path in sorted(tree_root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            sources[path] = path.read_text(encoding="utf-8")
    return sources


def _imported_names(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


# ---------------------------------------------------------------------------
# Manifest shape
# ---------------------------------------------------------------------------


def test_manifest_is_canonical_json(manifest) -> None:
    raw = MANIFEST_PATH.read_text(encoding="utf-8")
    assert raw.endswith("\n") and not raw.endswith("\n\n"), "exactly one trailing newline"
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert isinstance(manifest["entries"], list) and manifest["entries"], "manifest has no entries"
    assert raw == json.dumps(manifest, indent=2, sort_keys=True) + "\n", (
        "manifest is not in canonical sorted-key form"
    )


def test_entries_are_sorted_and_unique(entries) -> None:
    originals = [entry["original_entrypoint_path"] for entry in entries]
    destinations = [entry["destination"] for entry in entries]
    assert originals == sorted(originals), "entries must be sorted by original_entrypoint_path"
    assert len(set(originals)) == len(originals), "duplicate original paths"
    assert len(set(destinations)) == len(destinations), "duplicate destinations"


def test_every_entry_matches_schema(entries) -> None:
    for entry in entries:
        path = entry.get("original_entrypoint_path", "<unknown>")
        assert set(entry) == set(REQUIRED_FIELDS), (
            f"{path}: field set mismatch "
            f"(missing={set(REQUIRED_FIELDS) - set(entry)}, extra={set(entry) - set(REQUIRED_FIELDS)})"
        )
        for field, field_type in REQUIRED_FIELDS.items():
            value = entry[field]
            assert isinstance(value, field_type) and not isinstance(value, bool), (
                f"{path}.{field} must be {field_type.__name__}"
            )
        assert entry["kind"] in KINDS, f"{path}: bad kind {entry['kind']}"
        assert entry["research_status"] in RESEARCH_STATUSES, (
            f"{path}: bad research_status {entry['research_status']}"
        )
        assert entry["archive_wave"] in ARCHIVE_WAVES, f"{path}: bad archive_wave"
        assert entry["runnable_state"] in RUNNABLE_STATES, (
            f"{path}: bad runnable_state {entry['runnable_state']}"
        )
        assert entry["historical_invocation"].strip(), f"{path}: empty historical_invocation"
        assert entry["current_reproduction_wrapper"].strip(), f"{path}: empty wrapper field"
        assert entry["replacement_command"].strip(), f"{path}: empty replacement_command"
        assert all(isinstance(x, str) for x in entry["required_ignored_artifacts"])
        assert all(isinstance(x, str) for x in entry["known_failure_modes"])


# ---------------------------------------------------------------------------
# Blob identity: moves are byte-identical
# ---------------------------------------------------------------------------


def test_original_blobs_exist_and_moves_are_byte_identical(entries) -> None:
    for entry in entries:
        original = entry["original_entrypoint_path"]
        sha = entry["original_blob_sha"]
        assert BLOB_SHA_RE.match(sha), f"{original}: original_blob_sha must be 40 lowercase hex"
        assert _git("cat-file", "-t", sha) == "blob", (
            f"{original}: recorded blob {sha} is not in the git object database"
        )
        destination = ROOT / entry["destination"]
        assert destination.is_file(), f"{original}: destination missing at {entry['destination']}"
        expected_prefix = DESTINATION_PREFIXES[entry["archive_wave"]]
        assert entry["destination"].startswith(expected_prefix), (
            f"{original}: wave-{entry['archive_wave']} destination must stay "
            f"under {expected_prefix}"
        )
        current_sha = _git("hash-object", str(destination))
        assert current_sha == sha, (
            f"{original}: archived file was modified during the move "
            f"(recorded {sha}, current {current_sha})"
        )


def test_wrapper_semantics_match_the_old_path(entries) -> None:
    """Wave A leaves no wrapper; wrapper waves must not fake the original blob."""
    for entry in entries:
        original = ROOT / entry["original_entrypoint_path"]
        wrapper = entry["current_reproduction_wrapper"]
        if entry["archive_wave"] == "A":
            assert wrapper == "none", (
                f"{entry['original_entrypoint_path']}: scratch-wave moves leave no wrapper"
            )
        else:
            assert wrapper == entry["original_entrypoint_path"], (
                f"{entry['original_entrypoint_path']}: wave-{entry['archive_wave']} wrapper "
                "must occupy the original entrypoint path"
            )
        if wrapper in NONE_MARKERS:
            assert not original.exists(), (
                f"{entry['original_entrypoint_path']}: no wrapper declared but the old path exists"
            )
        else:
            assert original.is_file(), (
                f"{entry['original_entrypoint_path']}: wrapper declared but the old path is missing"
            )
            wrapper_sha = _git("hash-object", str(original))
            assert wrapper_sha != entry["original_blob_sha"], (
                f"{entry['original_entrypoint_path']}: wrapper must be distinguishable "
                "from the archived original blob"
            )


def test_wave_b_wrappers_delegate_to_destination(entries) -> None:
    """Wave-B wrappers must point at the moved destination they delegate to."""
    for entry in entries:
        if entry["archive_wave"] != "B":
            continue
        original = ROOT / entry["original_entrypoint_path"]
        wrapper_source = original.read_text(encoding="utf-8")
        assert entry["destination"] in wrapper_source, (
            f"{entry['original_entrypoint_path']}: wrapper does not reference "
            f"{entry['destination']}"
        )
        assert entry["destination"] in entry["replacement_command"], (
            f"{entry['original_entrypoint_path']}: replacement_command must use "
            "the destination path"
        )
        assert entry["runnable_state"] == "runnable_via_wrapper", (
            f"{entry['original_entrypoint_path']}: wave-B entries must be "
            "runnable_via_wrapper"
        )


def test_wave_c_wrappers_delegate_and_announce_frozen_status(entries) -> None:
    """Wave-C wrappers keep frozen reproduction working from the old path."""
    for entry in entries:
        if entry["archive_wave"] != "C":
            continue
        original = ROOT / entry["original_entrypoint_path"]
        wrapper_source = original.read_text(encoding="utf-8")
        assert entry["destination"] in wrapper_source, (
            f"{entry['original_entrypoint_path']}: wrapper does not reference "
            f"{entry['destination']}"
        )
        assert "frozen" in wrapper_source.lower(), (
            f"{entry['original_entrypoint_path']}: wrapper must announce the frozen status"
        )
        replacement = entry["replacement_command"]
        assert (
            "scripts/run_experiment.py dry-run" in replacement
            or entry["destination"] in replacement
        ), (
            f"{entry['original_entrypoint_path']}: replacement_command must use the "
            "dispatcher dry-run or the archived path"
        )
        assert entry["runnable_state"] == "runnable_via_wrapper", (
            f"{entry['original_entrypoint_path']}: wave-C entries must be "
            "runnable_via_wrapper"
        )
        assert entry["research_status"] == "frozen_reproduction", (
            f"{entry['original_entrypoint_path']}: wave-C entries must be "
            "frozen_reproduction"
        )


# ---------------------------------------------------------------------------
# Maintained code must not reference scratch-wave (A) moved modules
# ---------------------------------------------------------------------------


def test_moved_modules_are_not_imported_by_maintained_code(entries, maintained_sources) -> None:
    """Only scratch-wave (A) moves cut the old path; wave-B wrappers keep it live."""
    by_stem = {
        Path(entry["original_entrypoint_path"]).stem: entry
        for entry in entries
        if entry["archive_wave"] == "A"
    }
    offenders: list[str] = []
    for source_path, text in maintained_sources.items():
        imported = _imported_names(text)
        for stem, entry in by_stem.items():
            if stem in imported or any(name.endswith(f".{stem}") for name in imported):
                offenders.append(f"{source_path.relative_to(ROOT)} imports {stem}")
            if entry["original_entrypoint_path"] in text:
                offenders.append(
                    f"{source_path.relative_to(ROOT)} references {entry['original_entrypoint_path']}"
                )
    assert not offenders, "moved scripts are still referenced by maintained code:\n" + "\n".join(
        sorted(offenders)
    )
