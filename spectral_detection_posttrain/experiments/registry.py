"""Experiment-definition registry: how versioned experiments can be inspected.

``spectral_detection_posttrain/configs/registry/experiments.json`` defines how
a versioned experiment can be inspected or reproduced. It must never imply
scientific authorization: the only authorization signal is the research-line
status in ``research_lines.json``, enforced here through
``STATUS_TO_RUNNABLE`` from ``research_status.py``. There is no second mutable
authorization flag.

Two mutually exclusive record shapes form a strict discriminated union (plan
Section 3.2):

- ``executable_definition``: a versioned experiment with a locked config, a
  legacy entrypoint, and a static handler name. Path-existence and
  handler-membership validation apply.
- ``historical_artifact``: a non-orphan identity for archived/backfilled
  evidence. Its execution fields are explicit JSON ``null``; it can never be
  dispatched.

Unknown or mixed fields are rejected rather than coerced.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Union

from .contracts import (
    EvaluationScope,
    ExperimentDefinition,
    RunnableMode,
    parse_evaluation_scope,
    validate_relative_path,
)
from .research_status import (
    ResearchLineRegistry,
    ResearchStatus,
    load_research_status_registry,
    validate_runnable_for_status,
)

__all__ = [
    "EVIDENCE_KINDS",
    "EXECUTION_FIELDS",
    "HANDLERS",
    "RECORD_KINDS",
    "ExecutableDefinition",
    "ExperimentRecord",
    "ExperimentRegistry",
    "HistoricalArtifact",
    "load_experiment_registry",
]

SCHEMA_VERSION = "experiments.v1"

# Static handler names. Task 7 implements these handlers; the registry only
# validates membership. A handler is a static name, never a dotted/dynamic
# module string.
HANDLERS = frozenset({"standard_detection", "native_contract", "dense_endpoint"})

RECORD_KINDS = frozenset({"executable_definition", "historical_artifact"})

# Fields that carry execution semantics. On historical_artifact records they
# must be present with an explicit JSON null value.
EXECUTION_FIELDS = (
    "capability",
    "config_path",
    "legacy_entrypoint",
    "handler",
    "required_inputs",
    "expected_outputs",
    "gpu_policy",
)

# Closed vocabulary describing what kind of evidence a historical artifact is.
EVIDENCE_KINDS = frozenset(
    {
        "full_val_run",
        "smoke_run",
        "train_only_run",
        "cache_only_analysis",
        "parity_check",
        "post_hoc_analysis",
    }
)

_COMMON_FIELDS = (
    "record_kind",
    "id",
    "research_line",
    "runnable",
    "evaluation_scope",
    "decision_document",
)
_EXECUTABLE_ALLOWED = frozenset(_COMMON_FIELDS) | frozenset(EXECUTION_FIELDS)
_HISTORICAL_ALLOWED = frozenset(_COMMON_FIELDS) | frozenset(EXECUTION_FIELDS) | {
    "evidence_kind",
    "source_evidence_pointer",
}

# Capability is a dotted enum-like label (for example
# ``detection.energy_transport.c3b``), never an import path: lowercase
# segments only, no path separators, no ``.py`` suffix, no ``::``.
_CAPABILITY_LABEL = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")

# Override keys that change training epochs, seed, split, action space, gate,
# or evaluation scope. Only authorized records may accept them (plan 3.2).
RESTRICTED_OVERRIDE_KEYS = frozenset(
    {
        "epochs",
        "training_epochs",
        "seed",
        "data_seed",
        "split",
        "train_split",
        "val_split",
        "action_space",
        "candidate_space",
        "gate",
        "gates",
        "evaluation_scope",
    }
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _check_override_mapping(overrides: Mapping[str, Any]) -> None:
    if not isinstance(overrides, Mapping):
        raise ValueError(
            f"overrides must be a mapping, got {type(overrides).__name__}"
        )
    for key in overrides:
        if not isinstance(key, str) or not key:
            raise ValueError(f"override keys must be non-empty strings, got {key!r}")


@dataclass(frozen=True)
class ExecutableDefinition(ExperimentDefinition):
    """Executable record shape (contracts v1 ``ExperimentDefinition`` plus kind).

    ``handler`` is validated against the static ``HANDLERS`` set; dotted or
    dynamic module strings are rejected. ``resolve_overrides`` enforces the
    frozen-reproduction and non-authorized override guards from plan 3.2.
    """

    record_kind: str = field(default="executable_definition")

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.record_kind != "executable_definition":
            raise ValueError(f"record_kind must be 'executable_definition', got {self.record_kind!r}")
        handler = self.handler
        if not isinstance(handler, str) or not handler:
            raise ValueError(f"handler must be a non-empty string, got {handler!r}")
        if "." in handler or ":" in handler or "/" in handler or "\\" in handler:
            raise ValueError(
                f"dynamic module strings are not allowed as handler: {handler!r}"
            )
        if handler not in HANDLERS:
            raise ValueError(
                f"unknown handler {handler!r}; handler must be one of {sorted(HANDLERS)}"
            )
        if not _CAPABILITY_LABEL.match(self.capability):
            raise ValueError(
                "capability must be a dotted enum-like label "
                f"(e.g. 'detection.energy_transport.c3b'), got {self.capability!r}"
            )

    def resolve_overrides(self, overrides: Mapping[str, Any]) -> dict[str, Any]:
        """Validate dispatcher overrides for this record and return a plain dict.

        A ``frozen_reproduction_only`` record rejects every override key: it
        may run only with its exact locked config. Any non-authorized record
        additionally rejects overrides that change training epochs, seed,
        split, action space, gate, or evaluation scope.
        """
        _check_override_mapping(overrides)
        if self.runnable is RunnableMode.FROZEN_REPRODUCTION_ONLY and overrides:
            raise ValueError(
                f"frozen reproduction record {self.id!r} accepts no overrides; "
                f"rejected key(s): {sorted(overrides)}"
            )
        if self.runnable is not RunnableMode.AUTHORIZED:
            restricted = sorted(set(overrides) & RESTRICTED_OVERRIDE_KEYS)
            if restricted:
                raise ValueError(
                    f"override(s) {restricted} change training epochs, seed, split, "
                    "action space, gate, or evaluation scope and are rejected for "
                    f"non-authorized record {self.id!r}"
                )
        return dict(overrides)


@dataclass(frozen=True)
class HistoricalArtifact:
    """Non-executable record giving archived evidence a non-orphan identity.

    It can never be dispatched or authorize execution. Execution fields are
    not stored at all: the loader requires them to be explicit JSON ``null``
    in the serialized form and then drops them.
    """

    id: str
    research_line: str
    evaluation_scope: EvaluationScope
    evidence_kind: str
    source_evidence_pointer: str
    decision_document: str
    runnable: RunnableMode = RunnableMode.DISABLED
    record_kind: str = field(default="historical_artifact")

    def __post_init__(self) -> None:
        if self.record_kind != "historical_artifact":
            raise ValueError(f"record_kind must be 'historical_artifact', got {self.record_kind!r}")
        for name in ("id", "research_line", "evidence_kind"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string, got {value!r}")
        if self.evidence_kind not in EVIDENCE_KINDS:
            raise ValueError(
                f"unknown evidence_kind {self.evidence_kind!r}; "
                f"must be one of {sorted(EVIDENCE_KINDS)}"
            )
        runnable = self.runnable
        if not isinstance(runnable, RunnableMode):
            if isinstance(runnable, str):
                try:
                    runnable = RunnableMode(runnable)
                except ValueError:
                    raise ValueError(f"unknown runnable mode: {runnable!r}") from None
            else:
                raise ValueError(f"runnable must be a RunnableMode, got {runnable!r}")
            object.__setattr__(self, "runnable", runnable)
        if runnable is not RunnableMode.DISABLED:
            raise ValueError(
                f"historical_artifact records must have runnable='disabled', "
                f"got {runnable.value!r}"
            )
        if not isinstance(self.evaluation_scope, EvaluationScope):
            raise ValueError(
                "evaluation_scope must be an EvaluationScope, "
                f"got {type(self.evaluation_scope).__name__}"
            )
        validate_relative_path(self.source_evidence_pointer)
        validate_relative_path(self.decision_document)

    def resolve_overrides(self, overrides: Mapping[str, Any]) -> dict[str, Any]:
        _check_override_mapping(overrides)
        raise ValueError(
            f"historical artifact {self.id!r} can never be dispatched and "
            "accepts no overrides"
        )


ExperimentRecord = Union[ExecutableDefinition, HistoricalArtifact]


@dataclass(frozen=True)
class ExperimentRegistry:
    """Immutable collection of experiment records with unique IDs.

    ``research_lines`` is the read-only status reference used for the
    status-to-runnable enforcement and for the dispatch guard; it is never
    mutated by this registry.
    """

    records: tuple[ExperimentRecord, ...]
    research_lines: ResearchLineRegistry
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        records = tuple(self.records)
        for record in records:
            if not isinstance(record, (ExecutableDefinition, HistoricalArtifact)):
                raise ValueError(f"records must contain experiment records, got {record!r}")
        object.__setattr__(self, "records", records)
        seen: set[str] = set()
        duplicates = sorted({r.id for r in records if r.id in seen or seen.add(r.id)})
        if duplicates:
            raise ValueError(f"duplicate experiment id(s): {duplicates}")
        if not isinstance(self.research_lines, ResearchLineRegistry):
            raise ValueError("research_lines must be a ResearchLineRegistry")

    def __iter__(self) -> Iterator[ExperimentRecord]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def require(self, experiment_id: str) -> ExperimentRecord:
        for record in self.records:
            if record.id == experiment_id:
                return record
        raise KeyError(experiment_id)

    def can_dispatch(self, record: Union[ExperimentRecord, str]) -> bool:
        """True only for executable records on an active line with authorized mode.

        Historical artifacts and every record on a non-active line return
        False. This is a guard, not an authorization: dispatch itself is
        implemented by the Task 7 handlers.
        """
        if isinstance(record, str):
            record = self.require(record)
        if not isinstance(record, ExecutableDefinition):
            return False
        try:
            line = self.research_lines.get(record.research_line)
        except KeyError:
            return False
        return (
            line.status is ResearchStatus.ACTIVE
            and record.runnable is RunnableMode.AUTHORIZED
        )


_REGISTRY_TOP_KEYS = frozenset({"schema_version", "records"})


def _resolve_research_lines_path(path: Path, explicit: str | Path | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    sibling = path.parent / "research_lines.json"
    if sibling.is_file():
        return sibling
    return _repo_root() / "spectral_detection_posttrain" / "configs" / "registry" / "research_lines.json"


def load_experiment_registry(
    path: str | Path,
    research_lines_path: str | Path | None = None,
) -> ExperimentRegistry:
    """Strictly load and validate ``experiments.json``.

    Unknown top-level or record fields, mixed record shapes, duplicate IDs,
    unknown research lines, invalid runnable modes, runnable modes outside
    the research line's ``STATUS_TO_RUNNABLE`` set, unknown or dynamic
    handlers, invalid relative paths, and committed pointers that do not
    exist on disk are all rejected.
    """
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in experiment registry {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("experiment registry must be a JSON object")
    unknown = set(payload) - _REGISTRY_TOP_KEYS
    if unknown:
        raise ValueError(f"unknown registry field(s): {sorted(unknown)}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported registry schema_version: {payload.get('schema_version')!r}"
        )
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("registry field 'records' must be a list")
    lines = load_research_status_registry(
        _resolve_research_lines_path(path, research_lines_path)
    )
    root = _repo_root()
    records = [_parse_record(raw, lines, root) for raw in raw_records]
    return ExperimentRegistry(records=tuple(records), research_lines=lines)


def _parse_record(
    raw: Any, lines: ResearchLineRegistry, root: Path
) -> ExperimentRecord:
    if not isinstance(raw, Mapping):
        raise ValueError(f"experiment record must be a mapping, got {type(raw).__name__}")
    kind = raw.get("record_kind")
    if kind not in RECORD_KINDS:
        raise ValueError(
            f"record_kind must be one of {sorted(RECORD_KINDS)}, got {kind!r}"
        )
    if kind == "executable_definition":
        return _parse_executable(raw, lines, root)
    return _parse_historical(raw, lines, root)


def _check_line_and_runnable(
    raw: Mapping[str, Any], lines: ResearchLineRegistry
) -> None:
    line_id = raw.get("research_line")
    if not isinstance(line_id, str) or not line_id.strip():
        raise ValueError(f"research_line must be a non-empty string, got {line_id!r}")
    try:
        line = lines.get(line_id)
    except KeyError:
        raise ValueError(f"unknown research_line: {line_id!r}") from None
    runnable = raw.get("runnable")
    try:
        mode = runnable if isinstance(runnable, RunnableMode) else RunnableMode(runnable)
    except ValueError:
        raise ValueError(f"unknown runnable mode: {runnable!r}") from None
    validate_runnable_for_status(line.status, mode)


def _parse_scope(raw: Mapping[str, Any]) -> EvaluationScope:
    scope = raw.get("evaluation_scope")
    if not isinstance(scope, Mapping):
        raise ValueError(
            f"evaluation_scope must be a mapping, got {type(scope).__name__}"
        )
    return parse_evaluation_scope(scope)


def _check_existing_file(pointer: str, what: str, root: Path) -> None:
    validate_relative_path(pointer)
    if not (root / pointer).is_file():
        raise ValueError(f"{what} does not exist on disk: {pointer!r}")


def _string_list(value: Any, name: str) -> None:
    if isinstance(value, str) or not isinstance(value, list):
        raise ValueError(f"{name} must be a list of strings, got {value!r}")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{name} must contain only non-empty strings, got {value!r}")


def _parse_executable(
    raw: Mapping[str, Any], lines: ResearchLineRegistry, root: Path
) -> ExecutableDefinition:
    keys = set(raw)
    unknown = keys - _EXECUTABLE_ALLOWED
    if unknown:
        raise ValueError(
            f"unknown or mixed executable_definition field(s): {sorted(unknown)}"
        )
    missing = _EXECUTABLE_ALLOWED - keys
    if missing:
        raise ValueError(f"missing executable_definition field(s): {sorted(missing)}")
    _check_line_and_runnable(raw, lines)
    _string_list(raw["required_inputs"], "required_inputs")
    _string_list(raw["expected_outputs"], "expected_outputs")
    _check_existing_file(raw["config_path"], "config_path", root)
    _check_existing_file(raw["legacy_entrypoint"], "legacy_entrypoint", root)
    _check_existing_file(raw["decision_document"], "decision_document", root)
    return ExecutableDefinition(
        id=raw["id"],
        research_line=raw["research_line"],
        capability=raw["capability"],
        config_path=raw["config_path"],
        legacy_entrypoint=raw["legacy_entrypoint"],
        handler=raw["handler"],
        runnable=raw["runnable"],
        evaluation_scope=_parse_scope(raw),
        required_inputs=tuple(raw["required_inputs"]),
        expected_outputs=tuple(raw["expected_outputs"]),
        gpu_policy=raw["gpu_policy"],
        decision_document=raw["decision_document"],
    )


def _parse_historical(
    raw: Mapping[str, Any], lines: ResearchLineRegistry, root: Path
) -> HistoricalArtifact:
    keys = set(raw)
    unknown = keys - _HISTORICAL_ALLOWED
    if unknown:
        raise ValueError(
            f"unknown or mixed historical_artifact field(s): {sorted(unknown)}"
        )
    missing = _HISTORICAL_ALLOWED - keys
    if missing:
        raise ValueError(f"missing historical_artifact field(s): {sorted(missing)}")
    _check_line_and_runnable(raw, lines)
    for field_name in EXECUTION_FIELDS:
        if raw[field_name] is not None:
            raise ValueError(
                f"historical_artifact field {field_name!r} must be explicit JSON null, "
                f"got {raw[field_name]!r}"
            )
    _check_existing_file(
        raw["source_evidence_pointer"], "source_evidence_pointer", root
    )
    _check_existing_file(raw["decision_document"], "decision_document", root)
    return HistoricalArtifact(
        id=raw["id"],
        research_line=raw["research_line"],
        runnable=raw["runnable"],
        evaluation_scope=_parse_scope(raw),
        evidence_kind=raw["evidence_kind"],
        source_evidence_pointer=raw["source_evidence_pointer"],
        decision_document=raw["decision_document"],
    )
