"""Explicit experiment dispatcher (refactor Task 7).

A thin, deterministic, side-effect-free dispatch layer over the experiment
registry (``configs/registry/experiments.json``) and the research status
registry (``configs/registry/research_lines.json``).

Design rules:

- **Static handler map only.** ``HANDLERS`` maps ``ExperimentCapability``
  members to handler instances. A handler name is never treated as a dotted
  or dynamic import string; unknown capabilities and import-looking strings
  are rejected.
- **Guards.** Unknown experiment IDs, historical artifacts, frozen records
  without the explicit ``--allow-frozen-reproduction`` flag, and overrides
  rejected by the record's own ``resolve_overrides`` (any override on a
  frozen record; restricted keys on any non-authorized record) all raise
  :class:`DispatchError`.
- **``run`` is disabled.** Execution requires ``registry.can_dispatch``;
  the current registry authorizes zero experiments, so ``run`` refuses
  every record with a clear message.
- **Dry-run purity.** ``dispatch_dry_run`` prints resolved experiment ID,
  research status, config hash, required inputs, evaluation scope, GPU
  policy, expected outputs, and the exact command. It never creates a run
  directory, never calls ``prepare_experiment``, and never imports dataset
  or training code.

The CLI lives in ``scripts/run_experiment.py``; this module is the library
surface it renders from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from spectral_detection_posttrain.experiments.contracts import RunnableMode
from spectral_detection_posttrain.experiments.handlers import (
    DenseEndpointHandler,
    DispatchError,
    DispatchPlan,
    DispatchRefusedError,
    ExperimentCapability,
    ExperimentHandler,
    NativeContractHandler,
    StandardDetectionHandler,
    format_evaluation_scope,
)
from spectral_detection_posttrain.experiments.registry import (
    ExecutableDefinition,
    ExperimentRecord,
    ExperimentRegistry,
    HistoricalArtifact,
    load_experiment_registry,
)

__all__ = [
    "HANDLERS",
    "DispatchError",
    "DispatchPlan",
    "DispatchRefusedError",
    "ExperimentCapability",
    "ExperimentHandler",
    "dispatch_dry_run",
    "dispatch_run",
    "dispatch_validate",
    "load_dispatcher_registry",
    "render_plan",
    "render_record_list",
    "render_status",
    "require_executable",
    "resolve_handler",
    "validate_run_name",
]


# Static capability-to-handler map (plan Task 7). Membership in this map is
# the only way a registry handler name resolves to code; there is no plugin
# discovery and no dynamic import.
HANDLERS: dict[ExperimentCapability, ExperimentHandler] = {
    ExperimentCapability.STANDARD_DETECTION: StandardDetectionHandler(),
    ExperimentCapability.NATIVE_CONTRACT: NativeContractHandler(),
    ExperimentCapability.DENSE_ENDPOINT: DenseEndpointHandler(),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_dispatcher_registry(root: str | Path | None = None) -> ExperimentRegistry:
    """Load the experiment registry (plus its research-line status reference)."""
    base = Path(root) if root is not None else _repo_root()
    return load_experiment_registry(
        base / "spectral_detection_posttrain" / "configs" / "registry" / "experiments.json"
    )


def resolve_handler(record: ExecutableDefinition | Any) -> ExperimentHandler:
    """Resolve a record's static handler name to a handler instance.

    Dynamic import strings (anything containing ``.``, ``:``, ``/``, or
    ``\\``), unknown capability names, and capabilities missing from the
    static ``HANDLERS`` map are all rejected.
    """
    name = record if isinstance(record, str) else getattr(record, "handler", None)
    if not isinstance(name, str) or not name:
        raise DispatchError(f"handler name must be a non-empty string, got {name!r}")
    if any(marker in name for marker in (".", ":", "/", "\\")):
        raise DispatchError(
            f"dynamic import strings are not dispatchable handlers: {name!r}"
        )
    try:
        capability = ExperimentCapability(name)
    except ValueError:
        raise DispatchError(
            f"unknown handler {name!r}; static handlers: "
            f"{sorted(capability.value for capability in HANDLERS)}"
        ) from None
    handler = HANDLERS.get(capability)
    if handler is None:
        raise DispatchError(
            f"no handler registered for capability {capability.value!r}"
        )
    return handler


def require_executable(
    registry: ExperimentRegistry, experiment_id: str
) -> ExecutableDefinition:
    """Resolve an experiment ID to an executable record, guarding the union."""
    try:
        record: ExperimentRecord = registry.require(experiment_id)
    except KeyError:
        raise DispatchError(f"unknown experiment id: {experiment_id!r}") from None
    if isinstance(record, HistoricalArtifact):
        raise DispatchError(
            f"historical artifact {experiment_id!r} can never be dispatched; "
            "it exists only as archived evidence"
        )
    if not isinstance(record, ExecutableDefinition):
        raise DispatchError(
            f"experiment {experiment_id!r} is not an executable definition"
        )
    return record


_RUN_NAME_FORBIDDEN = ("/", "\\", "..")


def validate_run_name(run_name: str) -> str:
    """Validate a dispatcher run name: a single non-empty path segment."""
    if not isinstance(run_name, str) or not run_name.strip():
        raise DispatchError(f"run name must be a non-empty string, got {run_name!r}")
    if run_name in (".", "..") or any(part in run_name for part in _RUN_NAME_FORBIDDEN):
        raise DispatchError(
            f"run name must be a single path segment without '/', '\\\\', or '..': "
            f"{run_name!r}"
        )
    return run_name


def _check_frozen_guard(record: ExecutableDefinition, allow_frozen_reproduction: bool) -> None:
    if (
        record.runnable is RunnableMode.FROZEN_REPRODUCTION_ONLY
        and not allow_frozen_reproduction
    ):
        raise DispatchError(
            f"experiment {record.id!r} is frozen (runnable='frozen_reproduction_only'); "
            "re-run with --allow-frozen-reproduction to acknowledge exact-config "
            "reproduction"
        )


def _resolve_overrides(
    record: ExecutableDefinition, overrides: Mapping[str, Any] | None
) -> dict[str, Any]:
    try:
        return record.resolve_overrides(overrides or {})
    except ValueError as exc:
        raise DispatchError(str(exc)) from None


def _research_status(registry: ExperimentRegistry, record: ExecutableDefinition) -> str:
    return registry.research_lines.get(record.research_line).status.value


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def dispatch_validate(registry: ExperimentRegistry, experiment_id: str) -> str:
    """Validate one record through its handler and render the result.

    Raises :class:`DispatchError` for unknown IDs, historical artifacts, or
    records that fail handler validation.
    """
    record = require_executable(registry, experiment_id)
    handler = resolve_handler(record)
    problems = handler.validate(record, registry)
    status = _research_status(registry, record)
    header = f"experiment {record.id}"
    if problems:
        lines = [f"{header}: invalid"]
        lines.extend(f"  - {problem}" for problem in problems)
        raise DispatchError("\n".join(lines))
    frozen = record.runnable is RunnableMode.FROZEN_REPRODUCTION_ONLY
    dry_run_note = (
        "allowed (requires --allow-frozen-reproduction)" if frozen else "allowed"
    )
    run_note = (
        "allowed"
        if registry.can_dispatch(record)
        else "refused (record is not authorized)"
    )
    lines = [
        f"{header}: valid",
        f"  record kind: {record.record_kind}",
        f"  research line: {record.research_line} (status {status})",
        f"  capability: {record.capability}",
        f"  handler: {record.handler}",
        f"  runnable: {record.runnable.value}",
        f"  dry-run: {dry_run_note}",
        f"  run: {run_note}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# dry-run
# ---------------------------------------------------------------------------


def dispatch_dry_run(
    registry: ExperimentRegistry,
    experiment_id: str,
    run_name: str,
    *,
    allow_frozen_reproduction: bool = False,
    overrides: Mapping[str, Any] | None = None,
) -> DispatchPlan:
    """Build the deterministic dry-run plan for a record.

    Guard order: unknown ID / historical artifact, run-name shape, frozen
    reproduction flag, override policy, handler resolution. The plan is pure
    data; rendering happens in :func:`render_plan`. No run directory is
    created and no dataset or training code is imported.
    """
    record = require_executable(registry, experiment_id)
    run_name = validate_run_name(run_name)
    _check_frozen_guard(record, allow_frozen_reproduction)
    resolved_overrides = _resolve_overrides(record, overrides)
    handler = resolve_handler(record)
    return handler.dry_run(record, registry, run_name, resolved_overrides)


def render_plan(plan: DispatchPlan) -> str:
    """Render a dry-run plan deterministically (no timestamps, repo-relative)."""
    lines = [
        "dry-run plan (no side effects; no run directory created)",
        f"experiment: {plan.experiment_id}",
        f"record kind: {plan.record_kind}",
        f"research line: {plan.research_line}",
        f"research status: {plan.research_status}",
        f"capability: {plan.capability}",
        f"handler: {plan.handler}",
        f"runnable: {plan.runnable}",
        f"authorized: {'yes' if plan.authorized else 'no'}",
        f"config: {plan.config_path}",
        f"config sha256: {plan.config_sha256}",
        f"required inputs: {', '.join(plan.required_inputs)}",
        f"evaluation scope: {format_evaluation_scope(plan.evaluation_scope)}",
        f"gpu policy: {plan.gpu_policy} — {plan.gpu_policy_note}",
        f"expected outputs: {', '.join(plan.expected_outputs)}",
        f"decision document: {plan.decision_document}",
        f"run name: {plan.run_name}",
        f"planned run directory: {plan.planned_run_dir} (not created by dry-run)",
        f"exact command: {' '.join(plan.command)}",
    ]
    if plan.overrides:
        rendered = ", ".join(f"{key}={value}" for key, value in plan.overrides)
        lines.append(f"overrides: {rendered}")
    lines.append("notes:")
    lines.extend(f"  - {note}" for note in plan.notes)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# run (disabled)
# ---------------------------------------------------------------------------


def dispatch_run(
    registry: ExperimentRegistry,
    experiment_id: str,
    run_name: str,
    *,
    allow_frozen_reproduction: bool = False,
    overrides: Mapping[str, Any] | None = None,
) -> None:
    """Refuse execution unless the registry authorizes the record.

    The current registry authorizes zero experiments, so this always raises
    :class:`DispatchRefusedError`. Even a future authorized record raises
    until a reviewed execution task implements handler ``run``.
    """
    try:
        record: ExperimentRecord = registry.require(experiment_id)
    except KeyError:
        raise DispatchError(f"unknown experiment id: {experiment_id!r}") from None
    if isinstance(record, HistoricalArtifact):
        raise DispatchRefusedError(
            f"run refused: historical artifact {experiment_id!r} can never be "
            "dispatched; it exists only as archived evidence"
        )
    if not isinstance(record, ExecutableDefinition):
        raise DispatchRefusedError(
            f"run refused: experiment {experiment_id!r} is not an executable definition"
        )
    run_name = validate_run_name(run_name)
    _check_frozen_guard(record, allow_frozen_reproduction)
    _resolve_overrides(record, overrides)
    if not registry.can_dispatch(record):
        status = _research_status(registry, record)
        raise DispatchRefusedError(
            f"run refused: experiment {record.id!r} is not authorized to run "
            f"(runnable={record.runnable.value!r}, research line "
            f"{record.research_line!r} status {status!r}); the current registry "
            "authorizes zero experiments — execution requires a reviewed status "
            "transition committed to research_lines.json"
        )
    handler = resolve_handler(record)
    handler.run(record, registry, run_name, overrides)


# ---------------------------------------------------------------------------
# list / status rendering
# ---------------------------------------------------------------------------


def render_record_list(registry: ExperimentRegistry) -> str:
    """Render every registry record deterministically, grouped by record kind."""
    executable = sorted(
        (record for record in registry if isinstance(record, ExecutableDefinition)),
        key=lambda record: record.id,
    )
    historical = sorted(
        (record for record in registry if isinstance(record, HistoricalArtifact)),
        key=lambda record: record.id,
    )
    total = len(executable) + len(historical)
    lines = [
        f"experiment registry: {total} records "
        f"({len(executable)} executable_definition, {len(historical)} historical_artifact)",
        "",
        f"executable definitions ({len(executable)}):",
    ]
    for record in executable:
        lines.append(
            f"  {record.id} | line={record.research_line} | "
            f"runnable={record.runnable.value} | handler={record.handler} | "
            f"scope={record.evaluation_scope.kind}"
        )
    lines.extend(["", f"historical artifacts ({len(historical)}):"])
    for record in historical:
        lines.append(
            f"  {record.id} | line={record.research_line} | "
            f"evidence={record.evidence_kind} | scope={record.evaluation_scope.kind}"
        )
    return "\n".join(lines)


def render_status(registry: ExperimentRegistry) -> str:
    """Render the authorization status deterministically."""
    lines_registry = registry.research_lines
    status_counts: dict[str, int] = {}
    for line in lines_registry.lines:
        status_counts[line.status.value] = status_counts.get(line.status.value, 0) + 1
    counts_text = ", ".join(
        f"{status} {status_counts[status]}" for status in sorted(status_counts)
    )
    active = sum(
        1 for line in lines_registry.lines if line.status.value == "active"
    )
    queued = sum(
        1 for line in lines_registry.lines if line.status.value == "queued"
    )
    authorized = sum(1 for record in registry if registry.can_dispatch(record))
    out = [
        f"research status registry: {len(lines_registry.lines)} lines ({counts_text})",
        f"active lines: {active}",
        f"queued lines: {queued}",
        f"authorized experiments: {authorized}",
    ]
    if authorized == 0:
        out.append("No experiment is currently authorized.")
    out.extend(["", "lines:"])
    for line in sorted(lines_registry.lines, key=lambda entry: entry.id):
        out.append(f"  {line.id}: {line.status.value}")
    return "\n".join(out)
