"""Tests for strict artifact manifest contracts (Task 4).

Covers SHA-256 format, logical paths, input/output kinds, evaluation scope,
runtime lifecycle transitions, reviewed append-only semantics,
runtime-to-reviewed hash binding, secret rejection, dirty formal rejection,
missing limit_* rejection, missing formal detection refs, limited_unknown
non-formal behavior, and stable JSON serialization.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from spectral_detection_posttrain.experiments.artifacts import (
    FORMAL_DETECTION_REF_KINDS,
    ArtifactManifest,
    ArtifactRef,
    EvaluationScope,
    ExperimentContext,
    ManifestValidationError,
    ObservationRecord,
    complete_manifest,
    fail_manifest,
    load_artifact_manifest,
    promote_reviewed_manifest,
    serialize_artifact_manifest,
    start_manifest,
    verify_artifact_ref,
    write_artifact_manifest,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "artifacts"

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64


def make_ref(
    kind: str = "eval_metrics",
    path: str = "runs/run-1/eval_metrics.json",
    sha: str = SHA_A,
    size: int | None = 128,
) -> ArtifactRef:
    return ArtifactRef(semantic_kind=kind, logical_path=path, sha256=sha, size_bytes=size)


def make_scope(kind: str = "smoke", **overrides) -> EvaluationScope:
    values = {"kind": kind, "image_count": 32, "limit_train": 16, "limit_val": 16}
    values.update(overrides)
    return EvaluationScope(**values)


def make_context(**overrides) -> ExperimentContext:
    values = {
        "experiment_id": "det.rlvr.smoke.001",
        "run_id": "run-1",
        "invocation": ("python", "scripts/round28_train_eval.py", "--dataset", "penn_fudan"),
        "resolved_config": make_ref("resolved_config", "runs/run-1/config.json", SHA_B, 512),
        "inputs": (make_ref("training_config", "configs/smoke.yaml", SHA_C, 256),),
        "git_commit": "57b198cfdc8586900785cc534345dfdfeb7681b2",
        "git_dirty": False,
        "environment": {"python": "3.10", "torch": "2.1.0"},
    }
    values.update(overrides)
    return ExperimentContext(**values)


def make_started(**overrides) -> ArtifactManifest:
    scope = overrides.pop("scope", make_scope())
    manifest = start_manifest(make_context(**overrides), scope)
    assert manifest.completion == "started"
    return manifest


def make_completed(**overrides) -> ArtifactManifest:
    outputs = overrides.pop(
        "outputs",
        (make_ref("eval_metrics", "runs/run-1/eval_metrics.json", SHA_D, 2048),),
    )
    metrics = overrides.pop("metrics", {"AP50": 0.5, "num_predictions": 12, "note": "smoke", "ok": True, "extra": None})
    gates = overrides.pop("gates", {"clean_eval": True})
    manifest = make_started(**overrides)
    return complete_manifest(manifest, outputs=outputs, metrics=metrics, gates=gates)


def make_observation(**overrides) -> ObservationRecord:
    values = {
        "observed_at_utc": "2026-07-13T00:00:00Z",
        "observed_host_alias": "gpu2-box",
        "observed_workspace_revision": "57b198c",
        "source_evidence_pointer": "docs/reports/smoke_review.md",
    }
    values.update(overrides)
    return ObservationRecord(**values)


def make_reviewed(**overrides) -> ArtifactManifest:
    """Build a reviewed manifest without touching the filesystem."""
    observation_overrides = {}
    for key in (
        "observed_at_utc",
        "observed_host_alias",
        "observed_workspace_revision",
        "source_evidence_pointer",
        "verified_refs",
        "missing_evidence",
        "unavailable_reason",
        "expected_runtime_manifest_sha256",
        "manifest_id",
        "supersedes",
    ):
        if key in overrides:
            observation_overrides[key] = overrides.pop(key)
    runtime = overrides.pop("runtime", None)
    if runtime is None:
        runtime = make_completed(**overrides)
    observation = make_observation(**observation_overrides)
    return ArtifactManifest(
        schema_version=runtime.schema_version,
        manifest_kind="reviewed",
        manifest_id=observation.manifest_id or f"{runtime.manifest_id}.reviewed",
        experiment_id=runtime.experiment_id,
        run_id=runtime.run_id,
        completion=runtime.completion,
        invocation=runtime.invocation,
        resolved_config=runtime.resolved_config,
        inputs=runtime.inputs,
        outputs=runtime.outputs + tuple(observation.verified_refs),
        git_commit=runtime.git_commit,
        git_dirty=runtime.git_dirty,
        environment=runtime.environment,
        evaluation_scope=runtime.evaluation_scope,
        metrics_summary=runtime.metrics_summary,
        gates=runtime.gates,
        runtime_manifest_sha256=SHA_E,
        observed_at_utc=observation.observed_at_utc,
        observed_host_alias=observation.observed_host_alias,
        observed_workspace_revision=observation.observed_workspace_revision,
        source_evidence_pointer=observation.source_evidence_pointer,
        missing_evidence=tuple(observation.missing_evidence),
        unavailable_reason=observation.unavailable_reason,
        supersedes=observation.supersedes,
    )


# ---------------------------------------------------------------------------
# SHA-256 format
# ---------------------------------------------------------------------------


def test_artifact_ref_sha256_format_enforced():
    for bad in ("abc", "A" * 64, "g" * 64, "a" * 63, "a" * 65, ""):
        with pytest.raises(ManifestValidationError, match="sha256"):
            make_ref(sha=bad)


def test_artifact_ref_accepts_valid_sha256():
    ref = make_ref(sha="0123456789abcdef" * 4)
    assert ref.sha256 == "0123456789abcdef" * 4


def test_runtime_manifest_sha256_format_enforced():
    reviewed = make_reviewed()
    with pytest.raises(ManifestValidationError, match="sha256"):
        ArtifactManifest(**{**reviewed.__dict__, "runtime_manifest_sha256": "not-a-sha"})


# ---------------------------------------------------------------------------
# Logical paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_path",
    [
        "../escape.json",
        "runs/../../etc/passwd",
        "/absolute/path.json",
        "C:/Users/alice/file.json",
        "C:\\Users\\alice\\file.json",
        "runs\\backslash.json",
        "runs//double.json",
        "runs/./dot.json",
        "remote:manifold/../escape.json",
        "remote:C:/drive.json",
        "",
    ],
)
def test_logical_path_rejects_unsafe_values(bad_path):
    with pytest.raises(ManifestValidationError, match="logical_path"):
        make_ref(path=bad_path)


@pytest.mark.parametrize(
    "good_path",
    [
        "runs/run-1/eval_metrics.json",
        "configs/registry/artifacts/reviewed.json",
        "remote:manifold/runs/run-1/eval_metrics.json",
    ],
)
def test_logical_path_accepts_relative_posix_and_remote_alias(good_path):
    ref = make_ref(path=good_path)
    assert ref.logical_path == good_path


# ---------------------------------------------------------------------------
# Input/output kinds
# ---------------------------------------------------------------------------


def test_input_output_semantic_kinds_preserved():
    inputs = (
        make_ref("training_config", "configs/smoke.yaml", SHA_A),
        make_ref("dataset_slice", "runs/run-1/split.json", SHA_B),
    )
    outputs = (make_ref("eval_metrics", "runs/run-1/eval_metrics.json", SHA_C),)
    manifest = complete_manifest(make_started(inputs=inputs), outputs=outputs, metrics={}, gates={})
    assert [ref.semantic_kind for ref in manifest.inputs] == ["training_config", "dataset_slice"]
    assert [ref.semantic_kind for ref in manifest.outputs] == ["eval_metrics"]


def test_artifact_ref_requires_non_empty_semantic_kind():
    with pytest.raises(ManifestValidationError, match="semantic_kind"):
        make_ref(kind="")


def test_artifact_ref_rejects_negative_size():
    with pytest.raises(ManifestValidationError, match="size_bytes"):
        make_ref(size=-1)


def test_artifact_ref_rejects_bool_size():
    with pytest.raises(ManifestValidationError, match="size_bytes"):
        make_ref(size=True)


# ---------------------------------------------------------------------------
# Evaluation scope
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind",
    [
        "smoke",
        "limited",
        "train_only",
        "cache_only",
        "detector_unseen",
        "full_val",
        "synthetic",
        "oracle",
        "limited_unknown",
    ],
)
def test_evaluation_scope_accepts_all_documented_kinds(kind):
    overrides = {"image_count": None, "limit_train": None, "limit_val": None}
    if kind in ("smoke", "limited"):
        overrides["limit_val"] = 8
    scope = make_scope(kind, **overrides)
    assert scope.kind == kind


def test_evaluation_scope_rejects_unknown_kind():
    with pytest.raises(ManifestValidationError, match="kind"):
        make_scope("full-val")


@pytest.mark.parametrize("kind", ["smoke", "limited"])
def test_smoke_and_limited_require_positive_limit(kind):
    with pytest.raises(ManifestValidationError, match="limit"):
        make_scope(kind, limit_train=None, limit_val=None)
    with pytest.raises(ManifestValidationError, match="limit"):
        make_scope(kind, limit_train=0, limit_val=0)
    with pytest.raises(ManifestValidationError, match="limit"):
        make_scope(kind, limit_train=-1, limit_val=None)
    # At least one explicit positive limit is sufficient.
    assert make_scope(kind, limit_train=None, limit_val=4).limit_val == 4
    assert make_scope(kind, limit_train=4, limit_val=None).limit_train == 4


def test_scope_rejects_bool_limits():
    with pytest.raises(ManifestValidationError, match="limit_train"):
        make_scope("smoke", limit_train=True)


# ---------------------------------------------------------------------------
# Runtime lifecycle
# ---------------------------------------------------------------------------


def test_start_manifest_creates_started_runtime():
    manifest = make_started()
    assert manifest.manifest_kind == "runtime"
    assert manifest.completion == "started"
    assert manifest.outputs == ()
    assert manifest.runtime_manifest_sha256 is None
    assert manifest.observed_at_utc is None
    assert manifest.source_evidence_pointer is None


def test_complete_manifest_from_started():
    manifest = make_completed()
    assert manifest.completion == "completed"
    assert manifest.metrics_summary["AP50"] == 0.5
    assert manifest.gates == {"clean_eval": True}


def test_complete_manifest_rejects_non_started():
    completed = make_completed()
    with pytest.raises(ManifestValidationError, match="started"):
        complete_manifest(completed, outputs=(), metrics={}, gates={})
    failed = fail_manifest(make_started(), "oom", "out of memory")
    with pytest.raises(ManifestValidationError, match="started"):
        complete_manifest(failed, outputs=(), metrics={}, gates={})


def test_fail_manifest_from_started():
    failed = fail_manifest(make_started(), "data_error", "split file missing")
    assert failed.completion == "failed"
    assert failed.unavailable_reason is not None
    assert "data_error" in failed.unavailable_reason
    assert "split file missing" in failed.unavailable_reason


def test_fail_manifest_rejects_non_started():
    completed = make_completed()
    with pytest.raises(ManifestValidationError, match="started"):
        fail_manifest(completed, "late_failure", "too late")


def test_runtime_manifest_may_not_be_unavailable():
    with pytest.raises(ManifestValidationError, match="unavailable"):
        ArtifactManifest(**{**make_started().__dict__, "completion": "unavailable"})


def test_reviewed_manifest_may_not_be_started():
    reviewed = make_reviewed()
    with pytest.raises(ManifestValidationError, match="started"):
        ArtifactManifest(**{**reviewed.__dict__, "completion": "started"})


def test_complete_manifest_rejects_reviewed_manifest():
    reviewed = make_reviewed()
    with pytest.raises(ManifestValidationError, match="runtime"):
        complete_manifest(reviewed, outputs=(), metrics={}, gates={})


# ---------------------------------------------------------------------------
# Reviewed append-only + write semantics
# ---------------------------------------------------------------------------


def test_reviewed_write_is_create_exclusive(tmp_path):
    reviewed = make_reviewed()
    target = tmp_path / "registry" / "reviewed.json"
    write_artifact_manifest(target, reviewed)
    with pytest.raises(FileExistsError):
        write_artifact_manifest(target, reviewed)


def test_runtime_write_requires_runs_path(tmp_path):
    manifest = make_completed()
    with pytest.raises(ManifestValidationError, match="runs"):
        write_artifact_manifest(tmp_path / "manifest.json", manifest)
    with pytest.raises(ManifestValidationError, match="runs"):
        write_artifact_manifest(tmp_path / "not_runs_dir" / "manifest.json", manifest)


def test_runtime_write_overwrites_with_atomic_replace(tmp_path):
    started = make_started()
    target = tmp_path / "runs" / "run-1" / "manifest.json"
    write_artifact_manifest(target, started)
    completed = make_completed()
    write_artifact_manifest(target, completed)
    loaded = load_artifact_manifest(target)
    assert loaded == completed
    # No temp files left behind by atomic replace.
    leftovers = [p.name for p in target.parent.iterdir() if p.name != "manifest.json"]
    assert leftovers == []


# ---------------------------------------------------------------------------
# Runtime-to-reviewed hash binding
# ---------------------------------------------------------------------------


def test_promote_reviewed_manifest_binds_runtime_hash(tmp_path):
    runtime = make_completed()
    target = tmp_path / "runs" / "run-1" / "manifest.json"
    write_artifact_manifest(target, runtime)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    observation = make_observation(expected_runtime_manifest_sha256=digest)
    reviewed = promote_reviewed_manifest(target, observation)
    assert reviewed.manifest_kind == "reviewed"
    assert reviewed.runtime_manifest_sha256 == digest
    assert reviewed.observed_at_utc == "2026-07-13T00:00:00Z"
    assert reviewed.observed_host_alias == "gpu2-box"
    assert reviewed.source_evidence_pointer == "docs/reports/smoke_review.md"
    assert reviewed.completion == "completed"
    assert reviewed.experiment_id == runtime.experiment_id


def test_promote_reviewed_manifest_rejects_hash_mismatch(tmp_path):
    runtime = make_completed()
    target = tmp_path / "runs" / "run-1" / "manifest.json"
    write_artifact_manifest(target, runtime)
    observation = make_observation(expected_runtime_manifest_sha256="f" * 64)
    with pytest.raises(ManifestValidationError, match="sha256"):
        promote_reviewed_manifest(target, observation)


def test_promote_reviewed_manifest_rejects_started_runtime(tmp_path):
    started = make_started()
    target = tmp_path / "runs" / "run-1" / "manifest.json"
    write_artifact_manifest(target, started)
    with pytest.raises(ManifestValidationError, match="started"):
        promote_reviewed_manifest(target, make_observation())


def test_promote_reviewed_manifest_rejects_invalid_runtime_file(tmp_path):
    target = tmp_path / "runs" / "run-1" / "manifest.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"not": "a manifest"}\n', encoding="utf-8")
    with pytest.raises(ManifestValidationError):
        promote_reviewed_manifest(target, make_observation())


# ---------------------------------------------------------------------------
# Secret rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_entry",
    [
        {"api_key": "sk-0123456789abcdef"},
        {"PASSWORD": "hunter2"},
        {"auth_token": "tok_123"},
        {"client_secret": "shh"},
        {"normal": "-----BEGIN RSA PRIVATE KEY-----\nabc"},
    ],
)
def test_environment_secret_patterns_rejected(env_entry):
    with pytest.raises(ManifestValidationError, match="(?i)secret|token|password|api_key|private key"):
        make_context(environment=env_entry)


@pytest.mark.parametrize(
    "env_entry",
    [
        {"profile": "C:\\Users\\alice\\file"},
        {"profile": "C:/Users/alice/file"},
        {"profile": "/home/alice/file"},
    ],
)
def test_environment_user_profile_paths_rejected(env_entry):
    with pytest.raises(ManifestValidationError, match="(?i)user|profile|home|users"):
        make_context(environment=env_entry)


def test_environment_values_must_be_strings():
    with pytest.raises(ManifestValidationError, match="environment"):
        make_context(environment={"workers": 4})


def test_fail_manifest_scrubs_secret_message():
    started = make_started()
    with pytest.raises(ManifestValidationError, match="(?i)secret|token|password|api_key"):
        fail_manifest(started, "auth", "failed with token tok_123 and password hunter2")


def test_fail_manifest_scrubs_user_path_message():
    started = make_started()
    with pytest.raises(ManifestValidationError, match="(?i)user|profile|home|users"):
        fail_manifest(started, "io", "cannot read /home/alice/data.bin")


def test_secret_patterns_rejected_in_invocation():
    with pytest.raises(ManifestValidationError, match="(?i)secret|token|password|api_key"):
        make_context(invocation=("python", "run.py", "--api_key", "sk-abc"))


# ---------------------------------------------------------------------------
# Dirty formal rejection
# ---------------------------------------------------------------------------


def test_dirty_formal_completed_manifest_rejected():
    scope = make_scope("full_val", image_count=4952, limit_train=None, limit_val=None)
    with pytest.raises(ManifestValidationError, match="git_dirty"):
        make_completed(git_dirty=True, scope=scope)


def test_dirty_formal_detector_unseen_rejected():
    scope = make_scope("detector_unseen", image_count=100, limit_train=None, limit_val=None)
    with pytest.raises(ManifestValidationError, match="git_dirty"):
        make_completed(git_dirty=True, scope=scope)


def test_dirty_non_formal_completed_manifest_allowed():
    manifest = make_completed(git_dirty=True)
    assert manifest.git_dirty is True
    assert manifest.completion == "completed"


def test_invalid_dirty_formal_fixture_rejected():
    with pytest.raises(ManifestValidationError, match="git_dirty|secret|api_key"):
        load_artifact_manifest(FIXTURES / "invalid_dirty_formal_manifest.json")


# ---------------------------------------------------------------------------
# Missing formal detection refs
# ---------------------------------------------------------------------------


def formal_inputs():
    return (
        make_ref("dataset_annotation", "data/nwpu/annotations.json", SHA_A),
        make_ref("split_manifest", "runs/run-1/split.json", SHA_B),
        make_ref("initial_checkpoint", "remote:manifold/checkpoints/base.pth", SHA_C),
        make_ref("metric_protocol", "configs/metrics/coco_eval.json", SHA_D),
    )


def test_formal_completion_requires_named_refs():
    scope = make_scope("full_val", image_count=4952, limit_train=None, limit_val=None)
    # Missing postprocess_config -> rejected, and the missing kind is listed.
    with pytest.raises(ManifestValidationError, match="postprocess_config"):
        make_completed(scope=scope, inputs=formal_inputs())


def test_formal_completion_lists_all_missing_refs():
    scope = make_scope("full_val", image_count=4952, limit_train=None, limit_val=None)
    with pytest.raises(ManifestValidationError) as excinfo:
        make_completed(scope=scope, inputs=())
    message = str(excinfo.value)
    for kind in FORMAL_DETECTION_REF_KINDS:
        assert kind in message


def test_formal_completion_with_all_refs_accepted():
    scope = make_scope("full_val", image_count=4952, limit_train=None, limit_val=None)
    inputs = formal_inputs() + (
        make_ref("postprocess_config", "runs/run-1/postprocess.json", SHA_E),
    )
    manifest = make_completed(scope=scope, inputs=inputs)
    assert manifest.completion == "completed"


def test_started_formal_manifest_does_not_require_refs():
    scope = make_scope("full_val", image_count=4952, limit_train=None, limit_val=None)
    manifest = make_started(scope=scope, inputs=())
    assert manifest.completion == "started"


def test_non_formal_completion_does_not_require_refs():
    manifest = make_completed(inputs=())
    assert manifest.completion == "completed"


# ---------------------------------------------------------------------------
# limited_unknown non-formal behavior
# ---------------------------------------------------------------------------


def test_limited_unknown_runtime_may_complete_as_non_formal():
    scope = make_scope("limited_unknown", image_count=None, limit_train=None, limit_val=None)
    manifest = make_completed(scope=scope, git_dirty=True, inputs=())
    assert manifest.completion == "completed"


def test_limited_unknown_reviewed_requires_missing_evidence_or_reason():
    scope = make_scope("limited_unknown", image_count=None, limit_train=None, limit_val=None)
    runtime = make_completed(scope=scope)
    with pytest.raises(ManifestValidationError, match="limited_unknown"):
        make_reviewed(runtime=runtime)
    reviewed = make_reviewed(runtime=runtime, missing_evidence=("no split manifest",))
    assert reviewed.missing_evidence == ("no split manifest",)
    reviewed2 = make_reviewed(runtime=runtime, unavailable_reason="scope limits unknown")
    assert reviewed2.unavailable_reason == "scope limits unknown"


# ---------------------------------------------------------------------------
# Stable serialization
# ---------------------------------------------------------------------------


def test_serialize_twice_identical_bytes():
    manifest = make_completed()
    first = serialize_artifact_manifest(manifest).encode("utf-8")
    second = serialize_artifact_manifest(manifest).encode("utf-8")
    assert first == second


def test_serialize_ends_with_exactly_one_newline():
    text = serialize_artifact_manifest(make_completed())
    assert text.endswith("\n")
    assert not text.endswith("\n\n")
    assert "\r" not in text


def test_serialize_keys_sorted():
    text = serialize_artifact_manifest(make_completed())
    payload = json.loads(text)
    raw_keys = list(json.loads(text, object_pairs_hook=dict).keys())
    assert raw_keys == sorted(payload.keys())


def test_load_serialize_roundtrip_equality():
    manifest = make_completed()
    text = serialize_artifact_manifest(manifest)
    assert load_artifact_manifest_from_text(text) == manifest


def load_artifact_manifest_from_text(text: str) -> ArtifactManifest:
    """Helper: round-trip through a temp file under a runs/ path."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "runs" / "rt" / "manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(text, encoding="utf-8")
        return load_artifact_manifest(path)


def test_completed_fixture_roundtrips_byte_identically():
    path = FIXTURES / "completed_manifest.json"
    manifest = load_artifact_manifest(path)
    assert manifest.completion == "completed"
    assert manifest.manifest_kind == "runtime"
    assert manifest.evaluation_scope.kind == "smoke"
    assert manifest.git_dirty is False
    reserialized = serialize_artifact_manifest(manifest).encode("utf-8")
    assert reserialized == path.read_bytes()


def test_reviewed_manifest_roundtrip():
    reviewed = make_reviewed()
    text = serialize_artifact_manifest(reviewed)
    assert load_artifact_manifest_from_text(text) == reviewed


# ---------------------------------------------------------------------------
# Strict loading
# ---------------------------------------------------------------------------


def load_payload(payload: dict) -> ArtifactManifest:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "runs" / "rt" / "manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return load_artifact_manifest(path)


def base_payload() -> dict:
    return json.loads(serialize_artifact_manifest(make_completed()))


def test_load_rejects_unknown_fields():
    payload = base_payload()
    payload["unexpected_field"] = 1
    with pytest.raises(ManifestValidationError, match="unknown"):
        load_payload(payload)


def test_load_rejects_missing_fields():
    payload = base_payload()
    del payload["git_commit"]
    with pytest.raises(ManifestValidationError, match="missing"):
        load_payload(payload)


def test_load_rejects_wrong_types():
    payload = base_payload()
    payload["git_dirty"] = "false"
    with pytest.raises(ManifestValidationError, match="git_dirty"):
        load_payload(payload)


def test_load_rejects_bool_for_int_field():
    payload = base_payload()
    payload["evaluation_scope"]["image_count"] = True
    with pytest.raises(ManifestValidationError, match="image_count"):
        load_payload(payload)


def test_load_rejects_unknown_completion_enum():
    payload = base_payload()
    payload["completion"] = "done"
    with pytest.raises(ManifestValidationError, match="completion"):
        load_payload(payload)


def test_load_rejects_unknown_manifest_kind():
    payload = base_payload()
    payload["manifest_kind"] = "run_time"
    with pytest.raises(ManifestValidationError, match="manifest_kind"):
        load_payload(payload)


def test_load_rejects_unknown_scope_kind():
    payload = base_payload()
    payload["evaluation_scope"]["kind"] = "partial"
    with pytest.raises(ManifestValidationError, match="kind"):
        load_payload(payload)


def test_load_rejects_non_dict_payload():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "runs" / "rt" / "manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ManifestValidationError, match="mapping"):
            load_artifact_manifest(path)


def test_load_rejects_nested_metric_values():
    payload = base_payload()
    payload["metrics_summary"]["nested"] = {"a": 1}
    with pytest.raises(ManifestValidationError, match="metrics_summary"):
        load_payload(payload)


def test_load_rejects_non_bool_gate():
    payload = base_payload()
    payload["gates"]["clean_eval"] = "yes"
    with pytest.raises(ManifestValidationError, match="gates"):
        load_payload(payload)


# ---------------------------------------------------------------------------
# verify_artifact_ref
# ---------------------------------------------------------------------------


class MappingResolver:
    def __init__(self, mapping: dict[str, bytes]):
        self._mapping = mapping

    def resolve(self, logical_path: str):
        if logical_path not in self._mapping:
            raise KeyError(logical_path)
        return self._mapping[logical_path]


def test_verify_artifact_ref_matches_bytes():
    data = b"metrics payload"
    ref = make_ref(sha=hashlib.sha256(data).hexdigest(), size=len(data))
    resolver = MappingResolver({ref.logical_path: data})
    assert verify_artifact_ref(ref, resolver) is True


def test_verify_artifact_ref_hash_mismatch():
    ref = make_ref(sha=SHA_A, size=3)
    resolver = MappingResolver({ref.logical_path: b"abc"})
    assert verify_artifact_ref(ref, resolver) is False


def test_verify_artifact_ref_size_mismatch():
    data = b"abc"
    ref = make_ref(sha=hashlib.sha256(data).hexdigest(), size=99)
    resolver = MappingResolver({ref.logical_path: data})
    assert verify_artifact_ref(ref, resolver) is False


def test_verify_artifact_ref_missing_artifact():
    ref = make_ref()
    assert verify_artifact_ref(ref, MappingResolver({})) is False


def test_verify_artifact_ref_callable_resolver():
    data = b"payload"
    ref = make_ref(sha=hashlib.sha256(data).hexdigest(), size=None)
    assert verify_artifact_ref(ref, lambda path: data) is True
    assert verify_artifact_ref(ref, lambda path: b"other") is False


def test_verify_artifact_ref_path_resolver(tmp_path):
    data = b"file payload"
    target = tmp_path / "artifact.bin"
    target.write_bytes(data)
    ref = make_ref(sha=hashlib.sha256(data).hexdigest(), size=len(data))
    assert verify_artifact_ref(ref, lambda path: target) is True
