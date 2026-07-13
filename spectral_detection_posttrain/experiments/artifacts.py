"""Strict artifact manifest contracts (plan section 3.3).

Two deliberately different layers:

1. A **runtime manifest** at ``runs/<run_id>/manifest.json`` is ignored by Git
   and may move only through ``started -> completed|failed|invalid``. It is
   written by the runner with atomic replace and is never itself reviewed
   scientific evidence.
2. A **reviewed manifest** is a tracked, immutable snapshot created by
   :func:`promote_reviewed_manifest`. It binds the runtime manifest's SHA-256,
   records the observation host/workspace revision/time, and is append-only:
   corrections require a new ``manifest_id`` with ``supersedes`` set, never
   editing the old file.

Validation rules enforced on construction, load, and lifecycle transitions:

- SHA-256 fields are 64 lowercase hex characters.
- ``logical_path`` values are relative POSIX paths (no ``..``, no absolute
  paths, no backslashes, no drive letters). Remote references use logical
  aliases such as ``remote:manifold/runs/<run_id>/eval_metrics.json``.
- ``environment`` values must be strings; every manifest is scanned for
  secret-looking content (tokens, passwords, private keys, ``api_key``
  patterns) and user profile paths (``C:\\Users\\`` / ``/home/``) and rejected
  on hit. ``fail_manifest`` messages pass through the same scrubber.
- A ``completed`` manifest with ``git_dirty=True`` is rejected when its scope
  is formal (``full_val`` or ``detector_unseen``).
- Scope kinds ``smoke`` and ``limited`` require at least one explicit positive
  ``limit_train`` / ``limit_val``.
- A completed formal detection manifest additionally requires refs (by
  ``semantic_kind``) for dataset/annotation identity, split manifest, initial
  checkpoint/weight source, metric protocol/version, and the resolved
  score-threshold/NMS/decode postprocess config; see
  :data:`FORMAL_DETECTION_REF_KINDS`.
- ``limited_unknown`` is non-formal and can never become ``validated``:
  reviewed manifests with scope kind ``limited_unknown`` must carry non-empty
  ``missing_evidence`` or an ``unavailable_reason`` documenting why the scope
  is not fully known, otherwise construction raises.
- Serialization is sorted, UTF-8, deterministic, and ends with exactly one
  newline. Loading rejects unknown fields, wrong types, and unknown enum
  values.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol, Sequence, runtime_checkable

SCHEMA_VERSION = "artifact-manifest.v1"

SCOPE_KINDS = (
    "smoke",
    "limited",
    "train_only",
    "cache_only",
    "detector_unseen",
    "full_val",
    "synthetic",
    "oracle",
    "limited_unknown",
)

COMPLETION_STATES = ("started", "completed", "failed", "invalid", "unavailable")
RUNTIME_COMPLETION_STATES = ("started", "completed", "failed", "invalid")
MANIFEST_KINDS = ("runtime", "reviewed")

FORMAL_SCOPE_KINDS = frozenset({"full_val", "detector_unseen"})

#: semantic_kind refs required for a completed formal detection manifest.
FORMAL_DETECTION_REF_KINDS = (
    "dataset_annotation",
    "split_manifest",
    "initial_checkpoint",
    "metric_protocol",
    "postprocess_config",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_ALIAS_RE = re.compile(r"^([a-z][a-z0-9+_.-]{1,}):(.*)$")

#: (pattern, description) pairs scanned over every manifest string value.
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), description)
    for pattern, description in (
        (r"BEGIN\s+[A-Z ]*PRIVATE KEY", "private key block"),
        (r"password", "password"),
        (r"token", "token"),
        (r"secret", "secret"),
        (r"api[_-]?key", "api_key"),
        (r"[A-Za-z]:[\\/]Users[\\/]", "user profile path"),
        (r"(^|/)home/", "user home path"),
    )
)

_REF_FIELDS = frozenset({"semantic_kind", "logical_path", "sha256", "size_bytes"})
_SCOPE_FIELDS = frozenset({"kind", "image_count", "limit_train", "limit_val"})
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "manifest_kind",
        "manifest_id",
        "experiment_id",
        "run_id",
        "completion",
        "invocation",
        "resolved_config",
        "inputs",
        "outputs",
        "git_commit",
        "git_dirty",
        "environment",
        "evaluation_scope",
        "metrics_summary",
        "gates",
        "runtime_manifest_sha256",
        "observed_at_utc",
        "observed_host_alias",
        "observed_workspace_revision",
        "source_evidence_pointer",
        "missing_evidence",
        "unavailable_reason",
        "supersedes",
    }
)


class ManifestValidationError(ValueError):
    """Raised when an artifact manifest violates the section 3.3 contract."""


# ---------------------------------------------------------------------------
# Field validators
# ---------------------------------------------------------------------------


def _validate_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.match(value):
        raise ManifestValidationError(
            f"invalid sha256 for {field_name}: expected 64 lowercase hex characters, got {value!r}"
        )
    return value


def _validate_logical_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestValidationError(f"logical_path must be a non-empty string, got {value!r}")
    if "\\" in value:
        raise ManifestValidationError(f"logical_path must use POSIX separators: {value!r}")
    if _DRIVE_RE.match(value):
        raise ManifestValidationError(f"logical_path must not contain a drive letter: {value!r}")
    alias_match = _ALIAS_RE.match(value)
    body = value
    if alias_match is not None:
        body = alias_match.group(2)
        if not body:
            raise ManifestValidationError(f"logical_path alias must name a path: {value!r}")
        if ":" in body:
            raise ManifestValidationError(f"logical_path must not contain ':': {value!r}")
        if _DRIVE_RE.match(body):
            raise ManifestValidationError(f"logical_path must not contain a drive letter: {value!r}")
    segments = body.split("/")
    for segment in segments:
        if segment in ("", ".", ".."):
            raise ManifestValidationError(
                f"logical_path must be a clean relative POSIX path without '.' or '..': {value!r}"
            )
    return value


def _validate_optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestValidationError(f"{field_name} must be an int or null, got {value!r}")
    if value < 0:
        raise ManifestValidationError(f"{field_name} must not be negative, got {value!r}")
    return value


def _reject_secret_text(value: str, context: str) -> None:
    for pattern, description in _SECRET_PATTERNS:
        if pattern.search(value):
            raise ManifestValidationError(
                f"secret or user-profile pattern ({description}) rejected in {context}: {value[:60]!r}"
            )


def _walk_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk_strings(key)
            yield from _walk_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_strings(item)


def _scrub_mapping_strings(payload: Mapping[str, Any], context: str) -> None:
    for text in _walk_strings(payload):
        _reject_secret_text(text, context)


def _validate_string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ManifestValidationError(f"{field_name} must be a list of strings, got {type(value).__name__}")
    result = tuple(value)
    for entry in result:
        if not isinstance(entry, str):
            raise ManifestValidationError(f"{field_name} entries must be strings, got {entry!r}")
    return result


def _validate_str_dict(value: Any, field_name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ManifestValidationError(f"{field_name} must be a mapping, got {type(value).__name__}")
    result = dict(value)
    for key, item in result.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise ManifestValidationError(f"{field_name} keys and values must be strings, got {key!r}: {item!r}")
    return result


def _validate_metrics(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestValidationError(f"metrics_summary must be a mapping, got {type(value).__name__}")
    result = dict(value)
    for key, item in result.items():
        if not isinstance(key, str):
            raise ManifestValidationError(f"metrics_summary keys must be strings, got {key!r}")
        if item is not None and not isinstance(item, (bool, int, float, str)):
            raise ManifestValidationError(
                f"metrics_summary values must be scalar (float, int, str, bool, or null), got {key!r}: {item!r}"
            )
    return result


def _validate_gates(value: Any) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        raise ManifestValidationError(f"gates must be a mapping, got {type(value).__name__}")
    result = dict(value)
    for key, item in result.items():
        if not isinstance(key, str) or not isinstance(item, bool):
            raise ManifestValidationError(f"gates must map strings to bools, got {key!r}: {item!r}")
    return result


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactRef:
    semantic_kind: str
    logical_path: str
    sha256: str
    size_bytes: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.semantic_kind, str) or not self.semantic_kind:
            raise ManifestValidationError(f"semantic_kind must be a non-empty string, got {self.semantic_kind!r}")
        _validate_logical_path(self.logical_path)
        _validate_sha256(self.sha256, "artifact ref")
        _validate_optional_int(self.size_bytes, "size_bytes")


@dataclass(frozen=True)
class EvaluationScope:
    kind: str
    image_count: int | None
    limit_train: int | None
    limit_val: int | None

    def __post_init__(self) -> None:
        if self.kind not in SCOPE_KINDS:
            raise ManifestValidationError(f"unknown evaluation scope kind: {self.kind!r}")
        _validate_optional_int(self.image_count, "image_count")
        _validate_optional_int(self.limit_train, "limit_train")
        _validate_optional_int(self.limit_val, "limit_val")
        if self.kind in ("smoke", "limited"):
            has_positive_limit = any(
                limit is not None and limit > 0 for limit in (self.limit_train, self.limit_val)
            )
            if not has_positive_limit:
                raise ManifestValidationError(
                    f"scope kind {self.kind!r} requires at least one explicit positive "
                    f"limit_train/limit_val, got limit_train={self.limit_train!r}, limit_val={self.limit_val!r}"
                )


@dataclass(frozen=True)
class ArtifactManifest:
    schema_version: str
    manifest_kind: str
    manifest_id: str
    experiment_id: str
    run_id: str
    completion: str
    invocation: tuple[str, ...]
    resolved_config: ArtifactRef
    inputs: tuple[ArtifactRef, ...]
    outputs: tuple[ArtifactRef, ...]
    git_commit: str
    git_dirty: bool
    environment: dict[str, str]
    evaluation_scope: EvaluationScope
    metrics_summary: dict[str, Any]
    gates: dict[str, bool]
    runtime_manifest_sha256: str | None
    observed_at_utc: str | None
    observed_host_alias: str | None
    observed_workspace_revision: str | None
    source_evidence_pointer: str | None
    missing_evidence: tuple[str, ...]
    unavailable_reason: str | None
    supersedes: str | None

    def __post_init__(self) -> None:
        # Basic type checks and collection coercion.
        for name in ("schema_version", "manifest_id", "experiment_id", "run_id", "git_commit"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ManifestValidationError(f"{name} must be a non-empty string, got {value!r}")
        if self.manifest_kind not in MANIFEST_KINDS:
            raise ManifestValidationError(f"unknown manifest_kind: {self.manifest_kind!r}")
        if self.completion not in COMPLETION_STATES:
            raise ManifestValidationError(f"unknown completion state: {self.completion!r}")
        if not isinstance(self.git_dirty, bool):
            raise ManifestValidationError(f"git_dirty must be a bool, got {self.git_dirty!r}")
        if not isinstance(self.resolved_config, ArtifactRef):
            raise ManifestValidationError("resolved_config must be an ArtifactRef")
        if not isinstance(self.evaluation_scope, EvaluationScope):
            raise ManifestValidationError("evaluation_scope must be an EvaluationScope")

        object.__setattr__(self, "invocation", _validate_string_tuple(self.invocation, "invocation"))
        object.__setattr__(self, "inputs", _validate_ref_tuple(self.inputs, "inputs"))
        object.__setattr__(self, "outputs", _validate_ref_tuple(self.outputs, "outputs"))
        object.__setattr__(self, "missing_evidence", _validate_string_tuple(self.missing_evidence, "missing_evidence"))
        object.__setattr__(self, "environment", _validate_str_dict(self.environment, "environment"))
        object.__setattr__(self, "metrics_summary", _validate_metrics(self.metrics_summary))
        object.__setattr__(self, "gates", _validate_gates(self.gates))

        for name in (
            "runtime_manifest_sha256",
            "observed_at_utc",
            "observed_host_alias",
            "observed_workspace_revision",
            "source_evidence_pointer",
            "unavailable_reason",
            "supersedes",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ManifestValidationError(f"{name} must be a string or null, got {value!r}")
        if self.runtime_manifest_sha256 is not None:
            _validate_sha256(self.runtime_manifest_sha256, "runtime_manifest_sha256")

        # Secret / user-profile scrub over every serialized field value.
        _scrub_mapping_strings(_manifest_to_payload(self), "manifest")

        # Lifecycle rules.
        if self.manifest_kind == "runtime":
            self._validate_runtime()
        else:
            self._validate_reviewed()

        # Dirty formal-run rejection.
        if (
            self.completion == "completed"
            and self.git_dirty
            and self.evaluation_scope.kind in FORMAL_SCOPE_KINDS
        ):
            raise ManifestValidationError(
                f"completed manifest with git_dirty=True is rejected for formal scope "
                f"{self.evaluation_scope.kind!r}"
            )

        # Formal completion requires named detection refs.
        if self.completion == "completed" and self.evaluation_scope.kind in FORMAL_SCOPE_KINDS:
            present = {ref.semantic_kind for ref in (*self.inputs, *self.outputs, self.resolved_config)}
            missing = [kind for kind in FORMAL_DETECTION_REF_KINDS if kind not in present]
            if missing:
                raise ManifestValidationError(
                    f"formal completion for scope {self.evaluation_scope.kind!r} is missing required "
                    f"artifact refs: {', '.join(missing)}"
                )

    def _validate_runtime(self) -> None:
        if self.completion not in RUNTIME_COMPLETION_STATES:
            raise ManifestValidationError(
                f"runtime manifests may not be {self.completion!r}: unavailable is reserved for "
                "reviewed-only backfill contexts"
            )
        if self.runtime_manifest_sha256 is not None:
            raise ManifestValidationError("runtime manifests must not carry runtime_manifest_sha256")
        for name in (
            "observed_at_utc",
            "observed_host_alias",
            "observed_workspace_revision",
            "source_evidence_pointer",
            "supersedes",
        ):
            if getattr(self, name) is not None:
                raise ManifestValidationError(f"{name} is null until promotion to a reviewed manifest")

    def _validate_reviewed(self) -> None:
        if self.completion == "started":
            raise ManifestValidationError("a reviewed manifest may not have completion='started'")
        for name in (
            "observed_at_utc",
            "observed_host_alias",
            "observed_workspace_revision",
            "source_evidence_pointer",
        ):
            if not getattr(self, name):
                raise ManifestValidationError(f"reviewed manifests require {name}")
        if self.completion == "unavailable":
            if not self.unavailable_reason:
                raise ManifestValidationError("unavailable reviewed manifests require unavailable_reason")
        elif self.runtime_manifest_sha256 is None:
            raise ManifestValidationError(
                "reviewed manifests require runtime_manifest_sha256 when a runtime manifest exists"
            )
        has_verified_refs = bool(self.inputs or self.outputs)
        if not (has_verified_refs or self.missing_evidence or self.unavailable_reason):
            raise ManifestValidationError(
                "reviewed manifests require verified refs or a specific missing_evidence/unavailable_reason"
            )
        # limited_unknown is non-formal and can never become validated: a reviewed
        # manifest must document why the scope is unknown.
        if self.evaluation_scope.kind == "limited_unknown" and not (
            self.missing_evidence or self.unavailable_reason
        ):
            raise ManifestValidationError(
                "reviewed manifests with scope kind 'limited_unknown' must carry non-empty "
                "missing_evidence or an unavailable_reason"
            )


@dataclass(frozen=True)
class ExperimentContext:
    """Runner-provided context used to start a runtime manifest."""

    experiment_id: str
    run_id: str
    invocation: tuple[str, ...]
    resolved_config: ArtifactRef
    inputs: tuple[ArtifactRef, ...]
    git_commit: str
    git_dirty: bool
    environment: dict[str, str]

    def __post_init__(self) -> None:
        for name in ("experiment_id", "run_id", "git_commit"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ManifestValidationError(f"{name} must be a non-empty string, got {value!r}")
        if not isinstance(self.git_dirty, bool):
            raise ManifestValidationError(f"git_dirty must be a bool, got {self.git_dirty!r}")
        if not isinstance(self.resolved_config, ArtifactRef):
            raise ManifestValidationError("resolved_config must be an ArtifactRef")
        object.__setattr__(self, "invocation", _validate_string_tuple(self.invocation, "invocation"))
        object.__setattr__(self, "inputs", _validate_ref_tuple(self.inputs, "inputs"))
        object.__setattr__(self, "environment", _validate_str_dict(self.environment, "environment"))
        _scrub_mapping_strings(
            {"invocation": list(self.invocation), "environment": dict(self.environment)},
            "experiment context",
        )


@dataclass(frozen=True)
class ObservationRecord:
    """Review-time observation used to promote a runtime manifest.

    ``expected_runtime_manifest_sha256`` optionally pins the runtime file's
    hash; a mismatch with the actual file bytes rejects promotion. When
    ``manifest_id`` is omitted the reviewed id defaults to
    ``<runtime manifest_id>.reviewed``.
    """

    observed_at_utc: str
    observed_host_alias: str
    observed_workspace_revision: str
    source_evidence_pointer: str
    verified_refs: tuple[ArtifactRef, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    unavailable_reason: str | None = None
    expected_runtime_manifest_sha256: str | None = None
    manifest_id: str | None = None
    supersedes: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "observed_at_utc",
            "observed_host_alias",
            "observed_workspace_revision",
            "source_evidence_pointer",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ManifestValidationError(f"{name} must be a non-empty string, got {value!r}")
        object.__setattr__(self, "verified_refs", _validate_ref_tuple(self.verified_refs, "verified_refs"))
        object.__setattr__(
            self, "missing_evidence", _validate_string_tuple(self.missing_evidence, "missing_evidence")
        )
        for name in ("unavailable_reason", "manifest_id", "supersedes"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ManifestValidationError(f"{name} must be a string or null, got {value!r}")
        if self.expected_runtime_manifest_sha256 is not None:
            _validate_sha256(self.expected_runtime_manifest_sha256, "expected_runtime_manifest_sha256")


def _validate_ref_tuple(value: Any, field_name: str) -> tuple[ArtifactRef, ...]:
    if not isinstance(value, (list, tuple)):
        raise ManifestValidationError(f"{field_name} must be a list of ArtifactRef, got {type(value).__name__}")
    result = tuple(value)
    for entry in result:
        if not isinstance(entry, ArtifactRef):
            raise ManifestValidationError(f"{field_name} entries must be ArtifactRef, got {entry!r}")
    return result


@runtime_checkable
class ArtifactResolver(Protocol):
    """Maps a manifest ``logical_path`` to bytes or a readable path."""

    def resolve(self, logical_path: str) -> bytes | Path: ...


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _ref_to_payload(ref: ArtifactRef) -> dict[str, Any]:
    return {
        "semantic_kind": ref.semantic_kind,
        "logical_path": ref.logical_path,
        "sha256": ref.sha256,
        "size_bytes": ref.size_bytes,
    }


def _scope_to_payload(scope: EvaluationScope) -> dict[str, Any]:
    return {
        "kind": scope.kind,
        "image_count": scope.image_count,
        "limit_train": scope.limit_train,
        "limit_val": scope.limit_val,
    }


def _manifest_to_payload(manifest: ArtifactManifest) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "manifest_kind": manifest.manifest_kind,
        "manifest_id": manifest.manifest_id,
        "experiment_id": manifest.experiment_id,
        "run_id": manifest.run_id,
        "completion": manifest.completion,
        "invocation": list(manifest.invocation),
        "resolved_config": _ref_to_payload(manifest.resolved_config),
        "inputs": [_ref_to_payload(ref) for ref in manifest.inputs],
        "outputs": [_ref_to_payload(ref) for ref in manifest.outputs],
        "git_commit": manifest.git_commit,
        "git_dirty": manifest.git_dirty,
        "environment": dict(manifest.environment),
        "evaluation_scope": _scope_to_payload(manifest.evaluation_scope),
        "metrics_summary": dict(manifest.metrics_summary),
        "gates": dict(manifest.gates),
        "runtime_manifest_sha256": manifest.runtime_manifest_sha256,
        "observed_at_utc": manifest.observed_at_utc,
        "observed_host_alias": manifest.observed_host_alias,
        "observed_workspace_revision": manifest.observed_workspace_revision,
        "source_evidence_pointer": manifest.source_evidence_pointer,
        "missing_evidence": list(manifest.missing_evidence),
        "unavailable_reason": manifest.unavailable_reason,
        "supersedes": manifest.supersedes,
    }


def serialize_artifact_manifest(manifest: ArtifactManifest) -> str:
    """Canonical JSON text: sorted keys, UTF-8, deterministic, one newline."""
    if not isinstance(manifest, ArtifactManifest):
        raise ManifestValidationError(f"expected ArtifactManifest, got {type(manifest).__name__}")
    payload = _manifest_to_payload(manifest)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    return text + "\n"


def _ref_from_payload(payload: Any, field_name: str) -> ArtifactRef:
    if not isinstance(payload, Mapping):
        raise ManifestValidationError(f"{field_name} must be a mapping, got {type(payload).__name__}")
    keys = set(payload)
    unknown = keys - _REF_FIELDS
    if unknown:
        raise ManifestValidationError(f"unknown {field_name} fields: {sorted(unknown)}")
    missing = _REF_FIELDS - keys
    if missing:
        raise ManifestValidationError(f"missing {field_name} fields: {sorted(missing)}")
    return ArtifactRef(
        semantic_kind=payload["semantic_kind"],
        logical_path=payload["logical_path"],
        sha256=payload["sha256"],
        size_bytes=payload["size_bytes"],
    )


def _scope_from_payload(payload: Any) -> EvaluationScope:
    if not isinstance(payload, Mapping):
        raise ManifestValidationError(f"evaluation_scope must be a mapping, got {type(payload).__name__}")
    keys = set(payload)
    unknown = keys - _SCOPE_FIELDS
    if unknown:
        raise ManifestValidationError(f"unknown evaluation_scope fields: {sorted(unknown)}")
    missing = _SCOPE_FIELDS - keys
    if missing:
        raise ManifestValidationError(f"missing evaluation_scope fields: {sorted(missing)}")
    if isinstance(payload["image_count"], bool) or (
        payload["image_count"] is not None and not isinstance(payload["image_count"], int)
    ):
        raise ManifestValidationError(f"image_count must be an int or null, got {payload['image_count']!r}")
    return EvaluationScope(
        kind=payload["kind"],
        image_count=payload["image_count"],
        limit_train=payload["limit_train"],
        limit_val=payload["limit_val"],
    )


def _manifest_from_payload(payload: Any) -> ArtifactManifest:
    if not isinstance(payload, Mapping):
        raise ManifestValidationError(f"manifest payload must be a mapping, got {type(payload).__name__}")
    keys = set(payload)
    unknown = keys - _MANIFEST_FIELDS
    if unknown:
        raise ManifestValidationError(f"unknown manifest fields: {sorted(unknown)}")
    missing = _MANIFEST_FIELDS - keys
    if missing:
        raise ManifestValidationError(f"missing manifest fields: {sorted(missing)}")

    git_dirty = payload["git_dirty"]
    if not isinstance(git_dirty, bool):
        raise ManifestValidationError(f"git_dirty must be a bool, got {git_dirty!r}")
    manifest_kind = payload["manifest_kind"]
    if manifest_kind not in MANIFEST_KINDS:
        raise ManifestValidationError(f"unknown manifest_kind: {manifest_kind!r}")
    completion = payload["completion"]
    if completion not in COMPLETION_STATES:
        raise ManifestValidationError(f"unknown completion state: {completion!r}")

    inputs = payload["inputs"]
    if not isinstance(inputs, list):
        raise ManifestValidationError(f"inputs must be a list, got {type(inputs).__name__}")
    outputs = payload["outputs"]
    if not isinstance(outputs, list):
        raise ManifestValidationError(f"outputs must be a list, got {type(outputs).__name__}")

    _validate_metrics(payload["metrics_summary"])
    _validate_gates(payload["gates"])
    _validate_str_dict(payload["environment"], "environment")
    _validate_string_tuple(payload["invocation"], "invocation")
    _validate_string_tuple(payload["missing_evidence"], "missing_evidence")

    return ArtifactManifest(
        schema_version=payload["schema_version"],
        manifest_kind=manifest_kind,
        manifest_id=payload["manifest_id"],
        experiment_id=payload["experiment_id"],
        run_id=payload["run_id"],
        completion=completion,
        invocation=tuple(payload["invocation"]),
        resolved_config=_ref_from_payload(payload["resolved_config"], "resolved_config"),
        inputs=tuple(_ref_from_payload(ref, "inputs") for ref in inputs),
        outputs=tuple(_ref_from_payload(ref, "outputs") for ref in outputs),
        git_commit=payload["git_commit"],
        git_dirty=git_dirty,
        environment=dict(payload["environment"]),
        evaluation_scope=_scope_from_payload(payload["evaluation_scope"]),
        metrics_summary=dict(payload["metrics_summary"]),
        gates=dict(payload["gates"]),
        runtime_manifest_sha256=payload["runtime_manifest_sha256"],
        observed_at_utc=payload["observed_at_utc"],
        observed_host_alias=payload["observed_host_alias"],
        observed_workspace_revision=payload["observed_workspace_revision"],
        source_evidence_pointer=payload["source_evidence_pointer"],
        missing_evidence=tuple(payload["missing_evidence"]),
        unavailable_reason=payload["unavailable_reason"],
        supersedes=payload["supersedes"],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_artifact_manifest(path: Path) -> ArtifactManifest:
    """Load and strictly validate a manifest from JSON."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestValidationError(f"manifest at {path} is not valid UTF-8: {exc}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ManifestValidationError(f"manifest at {path} is not valid JSON: {exc}") from exc
    return _manifest_from_payload(payload)


def write_artifact_manifest(path: Path, manifest: ArtifactManifest) -> None:
    """Write a manifest.

    Runtime manifests are written with atomic replace (temp file in the same
    directory, then ``os.replace``) and may only target paths under ``runs/``.
    Reviewed manifests use create-exclusive semantics: an existing target path
    raises :class:`FileExistsError` and is never overwritten.
    """
    if not isinstance(manifest, ArtifactManifest):
        raise ManifestValidationError(f"expected ArtifactManifest, got {type(manifest).__name__}")
    path = Path(path)
    text = serialize_artifact_manifest(manifest)

    if manifest.manifest_kind == "runtime":
        if "runs" not in path.parts[:-1]:
            raise ManifestValidationError(
                f"runtime manifests must be written under a runs/ directory, got {path}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def start_manifest(context: ExperimentContext, scope: EvaluationScope) -> ArtifactManifest:
    """Create a ``started`` runtime manifest from runner context."""
    if not isinstance(context, ExperimentContext):
        raise ManifestValidationError(f"expected ExperimentContext, got {type(context).__name__}")
    if not isinstance(scope, EvaluationScope):
        raise ManifestValidationError(f"expected EvaluationScope, got {type(scope).__name__}")
    return ArtifactManifest(
        schema_version=SCHEMA_VERSION,
        manifest_kind="runtime",
        manifest_id=f"{context.experiment_id}:{context.run_id}",
        experiment_id=context.experiment_id,
        run_id=context.run_id,
        completion="started",
        invocation=tuple(context.invocation),
        resolved_config=context.resolved_config,
        inputs=tuple(context.inputs),
        outputs=(),
        git_commit=context.git_commit,
        git_dirty=context.git_dirty,
        environment=dict(context.environment),
        evaluation_scope=scope,
        metrics_summary={},
        gates={},
        runtime_manifest_sha256=None,
        observed_at_utc=None,
        observed_host_alias=None,
        observed_workspace_revision=None,
        source_evidence_pointer=None,
        missing_evidence=(),
        unavailable_reason=None,
        supersedes=None,
    )


def _require_started_runtime(manifest: ArtifactManifest, action: str) -> None:
    if not isinstance(manifest, ArtifactManifest):
        raise ManifestValidationError(f"expected ArtifactManifest, got {type(manifest).__name__}")
    if manifest.manifest_kind != "runtime":
        raise ManifestValidationError(f"{action} applies to runtime manifests only, got {manifest.manifest_kind!r}")
    if manifest.completion != "started":
        raise ManifestValidationError(
            f"{action} requires a started runtime manifest, got completion={manifest.completion!r}"
        )


def complete_manifest(
    manifest: ArtifactManifest,
    outputs: Sequence[ArtifactRef],
    metrics: dict,
    gates: dict,
) -> ArtifactManifest:
    """Move a started runtime manifest to ``completed``."""
    _require_started_runtime(manifest, "complete_manifest")
    return replace(
        manifest,
        completion="completed",
        outputs=tuple(outputs),
        metrics_summary=dict(metrics),
        gates=dict(gates),
    )


def fail_manifest(manifest: ArtifactManifest, failure_type: str, message: str) -> ArtifactManifest:
    """Move a started runtime manifest to ``failed``.

    ``failure_type`` and ``message`` pass through the secret scrubber; the
    combined reason is stored in ``unavailable_reason``.
    """
    _require_started_runtime(manifest, "fail_manifest")
    if not isinstance(failure_type, str) or not failure_type:
        raise ManifestValidationError(f"failure_type must be a non-empty string, got {failure_type!r}")
    if not isinstance(message, str):
        raise ManifestValidationError(f"failure message must be a string, got {type(message).__name__}")
    _reject_secret_text(failure_type, "failure_type")
    _reject_secret_text(message, "failure message")
    return replace(manifest, completion="failed", unavailable_reason=f"{failure_type}: {message}")


def promote_reviewed_manifest(runtime_path: Path, observation: ObservationRecord) -> ArtifactManifest:
    """Promote a runtime manifest file into a reviewed manifest.

    The reviewed manifest binds ``runtime_manifest_sha256`` to the SHA-256 of
    the runtime file bytes. When ``observation.expected_runtime_manifest_sha256``
    is set it must match the file bytes; a mismatch raises. Started runtime
    manifests may not be promoted.
    """
    if not isinstance(observation, ObservationRecord):
        raise ManifestValidationError(f"expected ObservationRecord, got {type(observation).__name__}")
    runtime_path = Path(runtime_path)
    try:
        data = runtime_path.read_bytes()
    except OSError as exc:
        raise ManifestValidationError(f"cannot read runtime manifest at {runtime_path}: {exc}") from exc
    digest = hashlib.sha256(data).hexdigest()
    if (
        observation.expected_runtime_manifest_sha256 is not None
        and observation.expected_runtime_manifest_sha256 != digest
    ):
        raise ManifestValidationError(
            f"runtime manifest sha256 mismatch: expected "
            f"{observation.expected_runtime_manifest_sha256}, actual {digest}"
        )
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestValidationError(f"runtime manifest at {runtime_path} is not valid JSON: {exc}") from exc
    runtime = _manifest_from_payload(payload)
    if runtime.manifest_kind != "runtime":
        raise ManifestValidationError(
            f"promote_reviewed_manifest requires a runtime manifest, got {runtime.manifest_kind!r}"
        )
    if runtime.completion == "started":
        raise ManifestValidationError("cannot promote a started runtime manifest")
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
        outputs=tuple(runtime.outputs) + tuple(observation.verified_refs),
        git_commit=runtime.git_commit,
        git_dirty=runtime.git_dirty,
        environment=dict(runtime.environment),
        evaluation_scope=runtime.evaluation_scope,
        metrics_summary=dict(runtime.metrics_summary),
        gates=dict(runtime.gates),
        runtime_manifest_sha256=digest,
        observed_at_utc=observation.observed_at_utc,
        observed_host_alias=observation.observed_host_alias,
        observed_workspace_revision=observation.observed_workspace_revision,
        source_evidence_pointer=observation.source_evidence_pointer,
        missing_evidence=tuple(observation.missing_evidence) or tuple(runtime.missing_evidence),
        unavailable_reason=(
            observation.unavailable_reason
            if observation.unavailable_reason is not None
            else runtime.unavailable_reason
        ),
        supersedes=observation.supersedes,
    )


def verify_artifact_ref(ref: ArtifactRef, resolver: ArtifactResolver) -> bool:
    """Verify an artifact's SHA-256 (and size when declared) via a resolver.

    ``resolver`` may be an :class:`ArtifactResolver` (``.resolve``) or a plain
    callable mapping ``logical_path`` to ``bytes`` or a readable path. Returns
    ``False`` on resolution failure, hash mismatch, or size mismatch.
    """
    if not isinstance(ref, ArtifactRef):
        raise ManifestValidationError(f"expected ArtifactRef, got {type(ref).__name__}")
    try:
        if callable(resolver) and not hasattr(resolver, "resolve"):
            data: Any = resolver(ref.logical_path)
        else:
            data = resolver.resolve(ref.logical_path)
    except Exception:
        return False
    if isinstance(data, (str, Path)):
        try:
            data = Path(data).read_bytes()
        except OSError:
            return False
    if isinstance(data, (bytearray, memoryview)):
        data = bytes(data)
    if not isinstance(data, bytes):
        return False
    if ref.size_bytes is not None and len(data) != ref.size_bytes:
        return False
    return hashlib.sha256(data).hexdigest() == ref.sha256
