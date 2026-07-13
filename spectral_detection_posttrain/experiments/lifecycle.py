"""Artifact lifecycle hooks for the canonical runner (plan Task 6).

This module wires the strict manifest contracts of
:mod:`spectral_detection_posttrain.experiments.artifacts` into the canonical
runner's run directory:

- :func:`start_runtime_manifest` writes a ``started`` runtime manifest at
  ``<run_dir>/manifest.json`` right after ``prepare_experiment`` has written
  ``config.yaml`` and ``metadata.json``.
- :func:`finalize_experiment` explicitly moves the started manifest to
  ``completed``, hashing the declared outputs. Completion is never inferred
  from file existence: only this function (or the reviewed backfill/promotion
  path) may produce a completed manifest.
- :func:`fail_experiment` explicitly moves the started manifest to ``failed``
  with a sanitized failure type/message.
- :func:`check_formal_run_preconditions` rejects a formal run (explicit
  evaluation scope plus ``formal=True``) on a dirty git tree *before* any
  run-directory mutation.

Runtime manifests live under the ignored ``runs/`` tree and are never
themselves reviewed evidence; promotion to a tracked reviewed manifest
happens only through the reviewed backfill/promotion path.
"""

from __future__ import annotations

import platform
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from spectral_detection_posttrain.experiments.artifacts import (
    ArtifactManifest,
    ArtifactRef,
    EvaluationScope as ManifestEvaluationScope,
    ExperimentContext as ManifestExperimentContext,
    ManifestValidationError,
    complete_manifest,
    fail_manifest,
    load_artifact_manifest,
    start_manifest,
    write_artifact_manifest,
)

# ``artifacts`` is a read-only sibling contract module; reuse its secret /
# user-profile patterns so lifecycle sanitization always redacts exactly what
# the manifest scrubber would reject.
from spectral_detection_posttrain.experiments.artifacts import _SECRET_PATTERNS
from spectral_detection_posttrain.experiments.metadata import sha256_file

if TYPE_CHECKING:  # pragma: no cover
    from spectral_detection_posttrain.experiments.canonical_runner import ExperimentContext

RUNTIME_MANIFEST_NAME = "manifest.json"
RESOLVED_CONFIG_NAME = "config.yaml"

_FAILURE_TYPE_FALLBACK = "Error"
_MESSAGE_REDACTED = "<redacted>"
_MAX_FAILURE_TYPE_LENGTH = 80
_MAX_FAILURE_MESSAGE_LENGTH = 240


class LifecycleError(ManifestValidationError):
    """Raised when a canonical-runner lifecycle transition is invalid."""


class FormalRunRejectedError(LifecycleError):
    """Raised when a formal run is rejected before run-directory mutation."""


class MissingInputHashError(LifecycleError):
    """Raised when a declared input has no recorded SHA-256 hash."""


def runtime_manifest_path(context: ExperimentContext) -> Path:
    """Path of the runtime manifest inside the run directory."""
    return Path(context.run_dir) / RUNTIME_MANIFEST_NAME


def check_formal_run_preconditions(config: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
    """Reject a formal run on a dirty git tree before any mutation.

    A run counts as formal when config normalization attached an explicit
    evaluation scope under ``formal=True`` (the ``evaluation_scope_formal``
    marker). Non-formal runs (including ``limited_unknown`` configs) are
    unaffected.
    """
    if bool(config.get("evaluation_scope_formal")) and bool(metadata.get("git_dirty")):
        raise FormalRunRejectedError(
            "formal run rejected: git working tree is dirty "
            f"(commit {metadata.get('git_commit')!r}); commit or stash changes before a formal run"
        )


def start_runtime_manifest(context: ExperimentContext) -> ArtifactManifest:
    """Write the ``started`` runtime manifest for a prepared run directory."""
    resolved_config_path = Path(context.run_dir) / RESOLVED_CONFIG_NAME
    if not resolved_config_path.is_file():
        raise LifecycleError(
            f"resolved config missing at {resolved_config_path}; "
            "prepare_experiment must write config.yaml before the manifest starts"
        )
    _require_input_hashes(context)
    manifest = start_manifest(
        _manifest_context(context, resolved_config_path),
        _scope_from_metadata(context.metadata),
    )
    write_artifact_manifest(runtime_manifest_path(context), manifest)
    return manifest


def finalize_experiment(
    context: ExperimentContext,
    outputs: Mapping[str, str | Path] | None,
    metrics: Mapping[str, Any] | None,
    gates: Mapping[str, bool] | None,
) -> ArtifactManifest:
    """Move the started runtime manifest to ``completed``.

    ``outputs`` maps each output's ``semantic_kind`` to a file path; every
    file must exist and is hashed (SHA-256 plus size) into the manifest.
    The updated manifest is written back with atomic replace.
    """
    started = _load_started_manifest(context, "finalize_experiment")
    refs = tuple(_output_ref(context, kind, path) for kind, path in _iter_outputs(outputs))
    completed = complete_manifest(started, refs, dict(metrics or {}), dict(gates or {}))
    write_artifact_manifest(runtime_manifest_path(context), completed)
    return completed


def fail_experiment(context: ExperimentContext, error: object) -> ArtifactManifest:
    """Move the started runtime manifest to ``failed``.

    The exception type name and message are sanitized (secret- and
    user-profile-looking content is redacted wholesale, values truncated)
    before being stored as ``unavailable_reason``.
    """
    started = _load_started_manifest(context, "fail_experiment")
    failure_type, message = _sanitize_failure(error)
    failed = fail_manifest(started, failure_type, message)
    write_artifact_manifest(runtime_manifest_path(context), failed)
    return failed


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _require_input_hashes(context: ExperimentContext) -> None:
    metadata = context.metadata
    if context.checkpoint_path is not None and not metadata.get("checkpoint_hash"):
        raise MissingInputHashError(
            f"input checkpoint {context.checkpoint_path} has no recorded SHA-256 hash; "
            "refusing to start a runtime manifest without input hashes"
        )


def _manifest_context(context: ExperimentContext, resolved_config_path: Path) -> ManifestExperimentContext:
    metadata = context.metadata
    inputs: list[ArtifactRef] = []
    if context.checkpoint_path is not None:
        checkpoint = Path(context.checkpoint_path)
        inputs.append(
            ArtifactRef(
                semantic_kind="initial_checkpoint",
                logical_path=checkpoint.name,
                sha256=str(metadata["checkpoint_hash"]),
                size_bytes=checkpoint.stat().st_size,
            )
        )
    run_root_name = Path(context.run_dir).parent.name
    resolved_config = ArtifactRef(
        semantic_kind="resolved_config",
        logical_path=PurePosixPath(run_root_name, context.run_name, RESOLVED_CONFIG_NAME).as_posix(),
        sha256=sha256_file(resolved_config_path),
        size_bytes=resolved_config_path.stat().st_size,
    )
    experiment_id = context.config.get("experiment_id") or f"canonical.{context.phase}"
    return ManifestExperimentContext(
        experiment_id=str(experiment_id),
        run_id=str(context.run_name),
        invocation=("canonical-runner", str(context.phase), str(context.run_name)),
        resolved_config=resolved_config,
        inputs=tuple(inputs),
        git_commit=str(metadata.get("git_commit") or "unknown"),
        git_dirty=bool(metadata.get("git_dirty", False)),
        environment=_environment(metadata),
    )


def _scope_from_metadata(metadata: Mapping[str, Any]) -> ManifestEvaluationScope:
    provenance = metadata.get("evaluation_scope")
    if not isinstance(provenance, Mapping):
        return ManifestEvaluationScope(
            kind="limited_unknown", image_count=None, limit_train=None, limit_val=None
        )
    return ManifestEvaluationScope(
        kind=str(provenance.get("kind") or "limited_unknown"),
        image_count=provenance.get("image_count"),
        limit_train=provenance.get("limit_train"),
        limit_val=provenance.get("limit_val"),
    )


def _environment(metadata: Mapping[str, Any]) -> dict[str, str]:
    environment = {"python": platform.python_version()}
    torch_version = metadata.get("torch_version")
    if torch_version:
        environment["torch"] = str(torch_version)
    torchvision_version = metadata.get("torchvision_version")
    if torchvision_version:
        environment["torchvision"] = str(torchvision_version)
    return environment


def _load_started_manifest(context: ExperimentContext, action: str) -> ArtifactManifest:
    path = runtime_manifest_path(context)
    if not path.is_file():
        raise LifecycleError(
            f"{action} requires a started runtime manifest at {path}; run prepare_experiment first"
        )
    manifest = load_artifact_manifest(path)
    if manifest.manifest_kind != "runtime":
        raise LifecycleError(
            f"{action} applies to runtime manifests only, got {manifest.manifest_kind!r}"
        )
    if manifest.completion != "started":
        raise LifecycleError(
            f"{action} requires completion='started', got {manifest.completion!r}"
        )
    return manifest


def _iter_outputs(outputs: Mapping[str, str | Path] | None) -> Iterable[tuple[str, str | Path]]:
    if outputs is None:
        return ()
    if not isinstance(outputs, Mapping):
        raise LifecycleError(
            f"outputs must map semantic_kind to a file path, got {type(outputs).__name__}"
        )
    items = []
    for kind, path in outputs.items():
        if not isinstance(kind, str) or not kind:
            raise LifecycleError(f"output semantic_kind must be a non-empty string, got {kind!r}")
        items.append((kind, path))
    return tuple(items)


def _output_ref(context: ExperimentContext, semantic_kind: str, raw_path: str | Path) -> ArtifactRef:
    path = Path(raw_path)
    if not path.is_file():
        raise LifecycleError(f"output {semantic_kind!r} does not exist: {path}")
    run_dir = Path(context.run_dir)
    try:
        relative = path.resolve().relative_to(run_dir.resolve()).as_posix()
        logical_path = PurePosixPath(run_dir.parent.name, context.run_name, relative).as_posix()
    except ValueError:
        logical_path = path.name
    return ArtifactRef(
        semantic_kind=semantic_kind,
        logical_path=logical_path,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _sanitize_failure(error: object) -> tuple[str, str]:
    if isinstance(error, BaseException):
        failure_type = type(error).__name__
        message = str(error)
    elif isinstance(error, str):
        failure_type, message = _FAILURE_TYPE_FALLBACK, error
    else:
        failure_type, message = _FAILURE_TYPE_FALLBACK, str(error)
    failure_type = _sanitize_text(
        failure_type, replacement=_FAILURE_TYPE_FALLBACK, limit=_MAX_FAILURE_TYPE_LENGTH
    )
    message = _sanitize_text(message, replacement=_MESSAGE_REDACTED, limit=_MAX_FAILURE_MESSAGE_LENGTH)
    return failure_type, message


def _sanitize_text(text: str, *, replacement: str, limit: int) -> str:
    for pattern, _description in _SECRET_PATTERNS:
        if pattern.search(text):
            return replacement
    return text[:limit]
