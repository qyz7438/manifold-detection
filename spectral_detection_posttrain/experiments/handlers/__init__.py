"""Static handler protocol for the explicit experiment dispatcher (Task 7).

A handler is the executable side of one registry ``handler`` name. The map
from capability to handler is static: capabilities are an enum whose values
are exactly the registry's static ``HANDLERS`` names, and dispatch never
treats a handler name as a dotted/dynamic import string.

Handlers expose ``validate`` and ``dry_run``. ``run`` stays disabled: the
dispatcher only invokes it when the experiment registry authorizes the
record, and the current registry authorizes zero experiments.

Dry-run is deterministic and side-effect free: it reads the locked config
file only to hash it, never creates a run directory, never calls
``prepare_experiment``, and never imports dataset or training code.
"""

from __future__ import annotations

import hashlib
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Mapping

from spectral_detection_posttrain.experiments.contracts import EvaluationScope

if TYPE_CHECKING:  # pragma: no cover
    from spectral_detection_posttrain.experiments.registry import (
        ExecutableDefinition,
        ExperimentRegistry,
    )

__all__ = [
    "GPU_POLICY_NOTES",
    "RUNNABLE_NOTES",
    "DenseEndpointHandler",
    "DispatchError",
    "DispatchPlan",
    "DispatchRefusedError",
    "ExperimentCapability",
    "ExperimentHandler",
    "NativeContractHandler",
    "StandardDetectionHandler",
    "format_evaluation_scope",
]


class DispatchError(ValueError):
    """Raised when a dispatch request violates a registry guard."""


class DispatchRefusedError(DispatchError):
    """Raised when ``run`` is requested for a non-authorized record."""


class ExperimentCapability(str, Enum):
    """The static dispatch capabilities.

    Values are exactly the handler names validated by the experiment
    registry (``registry.HANDLERS``). There is no dynamic import path:
    a capability resolves to a handler only through the static
    ``dispatcher.HANDLERS`` map.
    """

    STANDARD_DETECTION = "standard_detection"
    NATIVE_CONTRACT = "native_contract"
    DENSE_ENDPOINT = "dense_endpoint"


# Human-readable, deterministic notes for known GPU policies. Unknown
# policies render without a note rather than failing.
GPU_POLICY_NOTES: dict[str, str] = {
    "remote_gpu2_guarded": (
        "physical GPU2 only; require memory.free > 8192 MiB post-peak; "
        "never signal or interfere with other processes"
    ),
}

# Deterministic per-runnable-mode notes rendered into every dry-run plan.
RUNNABLE_NOTES: dict[str, str] = {
    "frozen_reproduction_only": (
        "frozen reproduction: requires --allow-frozen-reproduction, the exact "
        "registered config hash, a clean git tree, and matching input hashes; "
        "no overrides are accepted"
    ),
    "diagnostic_only": (
        "diagnostic-only record: inspection and analysis use; it cannot "
        "support detector-gain claims"
    ),
    "disabled": "record is disabled and cannot be dispatched",
    "authorized": "record is authorized to run under an active research line",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def format_evaluation_scope(scope: EvaluationScope) -> str:
    """Render an evaluation scope deterministically; ``None`` fields omitted."""
    details = []
    if scope.image_count is not None:
        details.append(f"image_count={scope.image_count}")
    if scope.limit_train is not None:
        details.append(f"limit_train={scope.limit_train}")
    if scope.limit_val is not None:
        details.append(f"limit_val={scope.limit_val}")
    if details:
        return f"{scope.kind} ({', '.join(details)})"
    return str(scope.kind)


@dataclass(frozen=True)
class DispatchPlan:
    """The deterministic, side-effect-free result of a dry-run.

    Rendering is handled by ``dispatcher.render_plan``; this dataclass holds
    only plain data so two dry-runs of the same record always render
    byte-identical text.
    """

    experiment_id: str
    record_kind: str
    research_line: str
    research_status: str
    capability: str
    handler: str
    runnable: str
    authorized: bool
    config_path: str
    config_sha256: str
    required_inputs: tuple[str, ...]
    expected_outputs: tuple[str, ...]
    evaluation_scope: EvaluationScope
    gpu_policy: str
    gpu_policy_note: str
    decision_document: str
    run_name: str
    planned_run_dir: str
    command: tuple[str, ...]
    notes: tuple[str, ...] = ()
    overrides: tuple[tuple[str, str], ...] = field(default=())


class ExperimentHandler(ABC):
    """Base class for the static dispatch handlers.

    Subclasses set ``capability`` and ``family_note`` and may add
    family-specific checks in ``validate``. The shared ``dry_run``
    implementation builds the full :class:`DispatchPlan` from registry data;
    it performs no filesystem writes and imports nothing beyond the standard
    library and the experiment contracts.
    """

    capability: ClassVar[ExperimentCapability]
    family_note: ClassVar[str] = ""

    def validate(
        self, record: ExecutableDefinition, registry: ExperimentRegistry | None = None
    ) -> tuple[str, ...]:
        """Return handler-level problems for a record (empty tuple = valid)."""
        problems: list[str] = []
        if getattr(record, "handler", None) != self.capability.value:
            problems.append(
                f"record handler {getattr(record, 'handler', None)!r} does not match "
                f"handler capability {self.capability.value!r}"
            )
        return tuple(problems)

    def dry_run(
        self,
        record: ExecutableDefinition,
        registry: ExperimentRegistry,
        run_name: str,
        overrides: Mapping[str, Any] | None = None,
    ) -> DispatchPlan:
        """Build the deterministic dry-run plan; no side effects."""
        line = registry.research_lines.get(record.research_line)
        config_file = _repo_root() / record.config_path
        config_sha256 = hashlib.sha256(config_file.read_bytes()).hexdigest()
        resolved_overrides = tuple(
            sorted((str(key), str(value)) for key, value in (overrides or {}).items())
        )
        notes = [self.family_note, RUNNABLE_NOTES[record.runnable.value]]
        if not registry.can_dispatch(record):
            notes.append(
                "run is refused for this record (the current registry authorizes "
                "zero experiments)"
            )
        return DispatchPlan(
            experiment_id=record.id,
            record_kind=record.record_kind,
            research_line=record.research_line,
            research_status=line.status.value,
            capability=record.capability,
            handler=record.handler,
            runnable=record.runnable.value,
            authorized=registry.can_dispatch(record),
            config_path=record.config_path,
            config_sha256=config_sha256,
            required_inputs=tuple(record.required_inputs),
            expected_outputs=tuple(record.expected_outputs),
            evaluation_scope=record.evaluation_scope,
            gpu_policy=record.gpu_policy,
            gpu_policy_note=GPU_POLICY_NOTES.get(
                record.gpu_policy, "no dispatcher note registered for this policy"
            ),
            decision_document=record.decision_document,
            run_name=run_name,
            planned_run_dir=f"runs/{run_name}",
            command=("python", record.legacy_entrypoint),
            notes=tuple(note for note in notes if note),
            overrides=resolved_overrides,
        )

    def run(
        self,
        record: ExecutableDefinition,
        registry: ExperimentRegistry,
        run_name: str,
        overrides: Mapping[str, Any] | None = None,
    ) -> None:
        """Execution entry point; disabled until a record is authorized.

        The dispatcher gates this method on ``registry.can_dispatch``; no
        handler implements execution in Task 7, so an authorized record would
        still raise here until a reviewed execution task lands.
        """
        raise DispatchRefusedError(
            f"handler {type(self).__name__} does not implement run; execution is "
            "not enabled by the experiment registry"
        )


# Concrete handlers are imported after the base types so their modules can
# import this package without a circular-import failure.
from spectral_detection_posttrain.experiments.handlers.dense_endpoint import (  # noqa: E402
    DenseEndpointHandler,
)
from spectral_detection_posttrain.experiments.handlers.native_contract import (  # noqa: E402
    NativeContractHandler,
)
from spectral_detection_posttrain.experiments.handlers.standard_detection import (  # noqa: E402
    StandardDetectionHandler,
)
