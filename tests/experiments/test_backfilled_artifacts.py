"""Required-artifact tests for the reviewed artifact manifests.

Seventeen critical experiment artifacts must carry a reviewed manifest under
``spectral_detection_posttrain/configs/registry/artifacts/``. The manifests
are emitted by the deterministic generator
``scripts/dev/backfill_artifact_manifests.py`` and must never be hand-edited.

These tests:

* require one reviewed manifest per required experiment ID and validate every
  manifest through the strict Task 4 contract
  (``spectral_detection_posttrain/experiments/artifacts.py``);
* check that each manifest's ``experiment_id`` resolves in the experiment
  registry and that its evaluation scope matches the registry record;
* enforce evidence honesty: tracked source-evidence pointers, the documented
  runtime-manifest substitution for historical runs, no absolute user paths,
  complete formal-detection refs for completed formal scopes, explicit
  reasons for invalid/unavailable manifests, and byte-verified repo-relative
  artifact refs;
* enforce generator determinism: two ``--write`` runs are byte-identical and
  ``--check`` reports the shipped manifests up to date.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from spectral_detection_posttrain.experiments.artifacts import (
    FORMAL_DETECTION_REF_KINDS,
    FORMAL_SCOPE_KINDS,
    ArtifactManifest,
    load_artifact_manifest,
)

ROOT = Path(__file__).resolve().parents[2]
GENERATOR_PATH = ROOT / "scripts" / "dev" / "backfill_artifact_manifests.py"
REGISTRY_DIR = ROOT / "spectral_detection_posttrain" / "configs" / "registry"
ARTIFACTS_DIR = REGISTRY_DIR / "artifacts"
EXPERIMENTS_PATH = REGISTRY_DIR / "experiments.json"

REQUIRED_EXPERIMENT_IDS = (
    "nwpu_mob_strong_cosine_s42_bs8_36ep",
    "native_zero_parity_baseline",
    "native_zero_parity_fullft",
    "det.energy.set_policy.m1.001",
    "det.energy.global_delta_u.c3b.balanced.001",
    "det.energy.native_topology.d2b.001",
    "det.energy.post_nms_suppress.e1.001",
    "det.energy.dense_endpoint.absolute.001",
    "det.energy.dense_endpoint.geometry_control.001",
    "det.energy.dense_endpoint.cleanval.001",
    "det.energy.dense_endpoint.shift_audit.001",
    "det.energy.dense_local_delta_stats.002",
    "det.energy.dense_local_delta_learner.001",
    "det.energy.dense_local_delta_family_audit.001",
    "det.energy.dense_local_delta_family_prior.001",
    "det.energy.re_roi_counterfactual_evidence.001",
    "det.energy.oracle_utility_boxhead.001",
)

REVIEWED_COMPLETION_STATES = ("completed", "failed", "invalid", "unavailable")


def load_generator():
    assert GENERATOR_PATH.exists(), "backfill generator has not been implemented"
    spec = importlib.util.spec_from_file_location("backfill_artifact_manifests", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest_path(experiment_id: str) -> Path:
    return ARTIFACTS_DIR / f"{experiment_id}.json"


def load_experiment_records() -> dict[str, dict]:
    payload = json.loads(EXPERIMENTS_PATH.read_text(encoding="utf-8"))
    return {record["id"]: record for record in payload["records"]}


@pytest.fixture(scope="module")
def records() -> dict[str, dict]:
    return load_experiment_records()


@pytest.fixture(scope="module")
def manifests() -> dict[str, ArtifactManifest]:
    loaded = {}
    for experiment_id in REQUIRED_EXPERIMENT_IDS:
        path = manifest_path(experiment_id)
        assert path.is_file(), f"missing reviewed manifest for {experiment_id}: {path}"
        loaded[experiment_id] = load_artifact_manifest(path)
    return loaded


# ---------------------------------------------------------------------------
# Required manifests and contract validation
# ---------------------------------------------------------------------------


def test_all_required_manifests_exist() -> None:
    missing = [eid for eid in REQUIRED_EXPERIMENT_IDS if not manifest_path(eid).is_file()]
    assert not missing, f"missing reviewed manifests: {missing}"


def test_artifacts_directory_contains_exactly_the_required_manifests() -> None:
    if not ARTIFACTS_DIR.is_dir():
        pytest.fail(f"artifacts directory does not exist: {ARTIFACTS_DIR}")
    shipped = sorted(path.stem for path in ARTIFACTS_DIR.glob("*.json"))
    assert shipped == sorted(REQUIRED_EXPERIMENT_IDS)


@pytest.mark.parametrize("experiment_id", REQUIRED_EXPERIMENT_IDS)
def test_manifest_validates_through_contract(experiment_id: str) -> None:
    path = manifest_path(experiment_id)
    assert path.is_file(), f"missing reviewed manifest for {experiment_id}: {path}"
    manifest = load_artifact_manifest(path)
    assert manifest.schema_version == "artifact-manifest.v1"
    assert manifest.manifest_kind == "reviewed"
    assert manifest.experiment_id == experiment_id
    assert manifest.completion in REVIEWED_COMPLETION_STATES
    assert manifest.completion != "started"
    assert manifest.manifest_id
    assert manifest.run_id
    assert manifest.observed_at_utc
    assert manifest.observed_host_alias
    assert manifest.observed_workspace_revision
    assert manifest.source_evidence_pointer
    assert manifest.supersedes is None


def test_experiment_ids_resolve_in_registry(manifests: dict[str, ArtifactManifest], records: dict) -> None:
    for experiment_id, manifest in manifests.items():
        assert experiment_id in records, f"{experiment_id} does not resolve in experiments.json"
        assert manifest.experiment_id in records


def test_scope_matches_registry_record(manifests: dict[str, ArtifactManifest], records: dict) -> None:
    for experiment_id, manifest in manifests.items():
        record_scope = records[experiment_id]["evaluation_scope"]
        scope = manifest.evaluation_scope
        assert scope.kind == record_scope["kind"], (
            f"{experiment_id}: scope kind {scope.kind!r} != registry {record_scope['kind']!r}"
        )
        assert scope.image_count == record_scope["image_count"]
        assert scope.limit_train == record_scope["limit_train"]
        assert scope.limit_val == record_scope["limit_val"]


# ---------------------------------------------------------------------------
# Evidence honesty
# ---------------------------------------------------------------------------


def test_source_evidence_pointer_is_tracked_file(manifests: dict[str, ArtifactManifest]) -> None:
    for experiment_id, manifest in manifests.items():
        pointer = manifest.source_evidence_pointer
        assert pointer is not None
        assert "\\" not in pointer, f"{experiment_id}: evidence pointer must be POSIX"
        assert not pointer.startswith("/"), f"{experiment_id}: evidence pointer must be relative"
        target = ROOT / pointer
        assert target.is_file(), f"{experiment_id}: evidence pointer does not exist: {pointer}"


def test_runtime_manifest_substitution_is_documented(manifests: dict[str, ArtifactManifest]) -> None:
    """Historical runs predate runtime manifests; the binding hash substitutes
    the primary result JSON and every manifest must say so explicitly."""
    for experiment_id, manifest in manifests.items():
        if manifest.completion == "unavailable":
            continue
        assert manifest.runtime_manifest_sha256 is not None, experiment_id
        runtime_outputs = [
            ref for ref in manifest.outputs if ref.semantic_kind == "runtime_manifest"
        ]
        if runtime_outputs:
            assert manifest.runtime_manifest_sha256 in {
                ref.sha256 for ref in runtime_outputs
            }, experiment_id
            continue
        assert any("runtime manifest" in entry for entry in manifest.missing_evidence), (
            f"{experiment_id}: missing_evidence must document the runtime-manifest substitution"
        )


def test_terminal_closure_manifests_preserve_sealed_boundaries(
    manifests: dict[str, ArtifactManifest],
) -> None:
    re_roi = manifests["det.energy.re_roi_counterfactual_evidence.001"]
    assert re_roi.metrics_summary["outer_heldout_read"] is False
    assert re_roi.metrics_summary["detector_validation_read"] is False
    assert re_roi.gates == {
        "support": True,
        "identity": True,
        "re_roi_gain": False,
        "bundle_integrity": False,
        "static_baseline": True,
        "calibration": False,
        "generalization": False,
        "all_passed": False,
    }

    awr = manifests["det.energy.oracle_utility_boxhead.001"]
    assert awr.metrics_summary["training_started"] is False
    assert awr.metrics_summary["downstream_arms_started"] is False
    assert awr.metrics_summary["positive_candidates"] == 93
    assert awr.metrics_summary["minimum_positive_candidates"] == 500
    assert awr.gates["support"] is False


def test_no_absolute_user_paths_in_raw_manifest_text() -> None:
    for experiment_id in REQUIRED_EXPERIMENT_IDS:
        path = manifest_path(experiment_id)
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        assert "/home/" not in text, f"{experiment_id}: absolute home path leaked"
        assert "C:\\Users" not in text and "C:/Users" not in text, experiment_id
        assert "E:\\" not in text and "E:/" not in text, f"{experiment_id}: absolute drive path leaked"
        assert "\\" not in text, f"{experiment_id}: backslash path separator leaked"
        assert "\r" not in text, f"{experiment_id}: CR character present (must be LF)"


def test_formal_completed_manifests_have_required_refs(manifests: dict[str, ArtifactManifest]) -> None:
    for experiment_id, manifest in manifests.items():
        if manifest.completion != "completed" or manifest.evaluation_scope.kind not in FORMAL_SCOPE_KINDS:
            continue
        present = {ref.semantic_kind for ref in (*manifest.inputs, *manifest.outputs, manifest.resolved_config)}
        missing = [kind for kind in FORMAL_DETECTION_REF_KINDS if kind not in present]
        assert not missing, f"{experiment_id}: formal refs missing: {missing}"


def test_invalid_and_unavailable_manifests_state_reasons(manifests: dict[str, ArtifactManifest]) -> None:
    for experiment_id, manifest in manifests.items():
        if manifest.completion == "unavailable":
            assert manifest.unavailable_reason, experiment_id
        if manifest.completion in ("invalid", "unavailable"):
            assert manifest.missing_evidence or manifest.unavailable_reason, experiment_id


def test_unavailable_reason_absent_for_completed(manifests: dict[str, ArtifactManifest]) -> None:
    for experiment_id, manifest in manifests.items():
        if manifest.completion == "completed":
            assert manifest.unavailable_reason is None, experiment_id


def _ref_has_alias(logical_path: str) -> bool:
    # Mirrors artifacts._ALIAS_RE: alias prefixes are lowercase-letter led.
    head, sep, _body = logical_path.partition(":")
    return bool(sep) and head.isascii() and head.replace("+", "").replace(".", "").replace("-", "").replace(
        "_", ""
    ).isalnum() and head[0].islower() and len(head) >= 2


def test_repo_relative_refs_verify_against_worktree(manifests: dict[str, ArtifactManifest]) -> None:
    """Refs without a logical alias must point at tracked repo files whose
    bytes match the recorded sha256 (and size when declared)."""
    for experiment_id, manifest in manifests.items():
        refs = (*manifest.inputs, *manifest.outputs, manifest.resolved_config)
        for ref in refs:
            if _ref_has_alias(ref.logical_path):
                continue
            target = ROOT / ref.logical_path
            assert target.is_file(), f"{experiment_id}: repo-relative ref missing: {ref.logical_path}"
            data = target.read_bytes()
            assert hashlib.sha256(data).hexdigest() == ref.sha256, (
                f"{experiment_id}: sha256 mismatch for {ref.logical_path}"
            )
            if ref.size_bytes is not None:
                assert len(data) == ref.size_bytes, f"{experiment_id}: size mismatch for {ref.logical_path}"


def test_completed_manifests_bind_a_primary_result_output(manifests: dict[str, ArtifactManifest]) -> None:
    for experiment_id, manifest in manifests.items():
        if manifest.completion != "completed":
            continue
        assert manifest.outputs, f"{experiment_id}: completed manifest has no verified outputs"
        output_hashes = {ref.sha256 for ref in manifest.outputs}
        assert manifest.runtime_manifest_sha256 in output_hashes, (
            f"{experiment_id}: runtime_manifest_sha256 must bind one of the verified output refs"
        )


# ---------------------------------------------------------------------------
# Generator determinism
# ---------------------------------------------------------------------------


def _run_generator(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(GENERATOR_PATH), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_generator_check_reports_up_to_date() -> None:
    if not GENERATOR_PATH.exists():
        pytest.fail(f"backfill generator has not been implemented: {GENERATOR_PATH}")
    result = _run_generator("--check")
    assert result.returncode == 0, f"--check failed: {result.stdout}{result.stderr}"
    assert "up to date" in result.stdout


def test_generator_write_twice_is_byte_identical(tmp_path: Path) -> None:
    if not GENERATOR_PATH.exists():
        pytest.fail(f"backfill generator has not been implemented: {GENERATOR_PATH}")
    first = tmp_path / "first"
    second = tmp_path / "second"
    for out_dir in (first, second):
        result = _run_generator("--write", "--out-dir", str(out_dir))
        assert result.returncode == 0, f"--write failed: {result.stdout}{result.stderr}"
    first_files = sorted(first.glob("*.json"))
    second_files = sorted(second.glob("*.json"))
    assert [path.name for path in first_files] == [path.name for path in second_files]
    assert len(first_files) == len(REQUIRED_EXPERIMENT_IDS)
    for a, b in zip(first_files, second_files):
        assert a.read_bytes() == b.read_bytes(), f"generator output not deterministic: {a.name}"


def test_generator_output_matches_shipped_manifests(tmp_path: Path) -> None:
    if not GENERATOR_PATH.exists():
        pytest.fail(f"backfill generator has not been implemented: {GENERATOR_PATH}")
    result = _run_generator("--write", "--out-dir", str(tmp_path))
    assert result.returncode == 0, f"--write failed: {result.stdout}{result.stderr}"
    for experiment_id in REQUIRED_EXPERIMENT_IDS:
        generated = tmp_path / f"{experiment_id}.json"
        assert generated.is_file(), f"generator did not emit {experiment_id}"
        shipped = manifest_path(experiment_id)
        assert shipped.is_file(), f"shipped manifest missing: {experiment_id}"
        assert generated.read_bytes() == shipped.read_bytes(), (
            f"shipped manifest {experiment_id} differs from generator output; "
            "regenerate with `python scripts/dev/backfill_artifact_manifests.py --write`"
        )


def test_generator_emits_valid_utf8_lf_single_trailing_newline(tmp_path: Path) -> None:
    if not GENERATOR_PATH.exists():
        pytest.fail(f"backfill generator has not been implemented: {GENERATOR_PATH}")
    result = _run_generator("--write", "--out-dir", str(tmp_path))
    assert result.returncode == 0, f"--write failed: {result.stdout}{result.stderr}"
    for path in sorted(tmp_path.glob("*.json")):
        data = path.read_bytes()
        assert b"\r" not in data, f"{path.name}: CR present"
        assert not data.startswith(b"\xef\xbb\xbf"), f"{path.name}: BOM present"
        assert data.endswith(b"\n") and not data.endswith(b"\n\n"), f"{path.name}: must end with one newline"
        data.decode("utf-8")
