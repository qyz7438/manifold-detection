"""Contracts v2 for the repository research-state refactor.

Strict, immutable contract types shared by the research-status registry,
the experiment registry, and the artifact manifest layers. Python 3.10,
standard library only. String enums plus frozen dataclasses; parsing is
strict and rejects unknown fields rather than coercing them.

v2 adds cross-field ``EvaluationScope`` validation (``full_val`` rejects
limits; ``smoke``/``limited`` require a positive explicit limit) and the
``normalize_evaluation_scope`` helper used by config normalization. All v1
symbols remain importable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Mapping

__all__ = [
    "CONTRACTS_VERSION",
    "EVALUATION_SCOPE_KINDS",
    "EvaluationScope",
    "ExperimentDefinition",
    "ResearchStatus",
    "RunnableMode",
    "normalize_evaluation_scope",
    "parse_evaluation_scope",
    "parse_experiment_definition",
    "validate_relative_path",
]

CONTRACTS_VERSION = "v2"


class ResearchStatus(str, Enum):
    """Scientific status of a research line.

    Changed only by a reviewed decision commit; never by a result parser.
    """

    ACTIVE = "active"
    QUEUED = "queued"
    PAUSED = "paused"
    FROZEN = "frozen"
    BLOCKED = "blocked"
    DIAGNOSTIC = "diagnostic"
    BASELINE = "baseline"
    HISTORICAL = "historical"
    INVALIDATED = "invalidated"
    VALIDATED = "validated"


class RunnableMode(str, Enum):
    """Execution mode of an experiment definition.

    A runnable mode describes how a record may be inspected or reproduced;
    it never implies scientific authorization on its own.
    """

    DISABLED = "disabled"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    FROZEN_REPRODUCTION_ONLY = "frozen_reproduction_only"
    AUTHORIZED = "authorized"


EVALUATION_SCOPE_KINDS = (
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

_DRIVE_LETTER = re.compile(r"^[A-Za-z]:")


def validate_relative_path(path: str) -> str:
    """Validate a committed repo-relative pointer and return it unchanged.

    POSIX separators only: absolute paths, drive letters, backslashes,
    ``..`` escapes, and empty or ``.`` segments are rejected.
    """
    if not isinstance(path, str) or not path:
        raise ValueError(f"relative path must be a non-empty string, got {path!r}")
    if "\\" in path:
        raise ValueError(f"relative path must use POSIX separators: {path!r}")
    if path.startswith("/") or _DRIVE_LETTER.match(path):
        raise ValueError(f"absolute paths are not allowed: {path!r}")
    for segment in path.split("/"):
        if segment in ("", ".", ".."):
            raise ValueError(f"invalid relative path segment {segment!r} in {path!r}")
    return path


def _check_count(value: Any, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int or None, got {value!r}")


@dataclass(frozen=True)
class EvaluationScope:
    """How far an evaluation reached.

    Contracts v2 cross-field rules: ``full_val`` must not set
    ``limit_train``/``limit_val``; ``smoke`` and ``limited`` require at
    least one explicit positive limit so a partial run can never be
    mistaken for a complete one.
    """

    kind: Literal[
        "smoke",
        "limited",
        "train_only",
        "cache_only",
        "detector_unseen",
        "full_val",
        "synthetic",
        "oracle",
        "limited_unknown",
    ]
    image_count: int | None = None
    limit_train: int | None = None
    limit_val: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in EVALUATION_SCOPE_KINDS:
            raise ValueError(f"unknown evaluation scope kind: {self.kind!r}")
        _check_count(self.image_count, "image_count")
        _check_count(self.limit_train, "limit_train")
        _check_count(self.limit_val, "limit_val")
        if self.kind == "full_val" and (
            self.limit_train is not None or self.limit_val is not None
        ):
            raise ValueError(
                "full_val evaluation scope must not set limit_train/limit_val, "
                f"got limit_train={self.limit_train!r}, limit_val={self.limit_val!r}"
            )
        if self.kind in ("smoke", "limited"):
            has_positive_limit = any(
                limit is not None and limit > 0
                for limit in (self.limit_train, self.limit_val)
            )
            if not has_positive_limit:
                raise ValueError(
                    f"scope kind {self.kind!r} requires at least one explicit "
                    f"positive limit_train/limit_val, got limit_train={self.limit_train!r}, "
                    f"limit_val={self.limit_val!r}"
                )


_EVALUATION_SCOPE_KEYS = frozenset({"kind", "image_count", "limit_train", "limit_val"})


def parse_evaluation_scope(data: Mapping[str, Any]) -> EvaluationScope:
    """Strictly parse an ``EvaluationScope`` mapping; unknown keys are rejected."""
    if not isinstance(data, Mapping):
        raise ValueError(f"evaluation_scope must be a mapping, got {type(data).__name__}")
    keys = set(data)
    unknown = keys - _EVALUATION_SCOPE_KEYS
    if unknown:
        raise ValueError(f"unknown evaluation_scope field(s): {sorted(unknown)}")
    missing = _EVALUATION_SCOPE_KEYS - keys
    if missing:
        raise ValueError(f"missing evaluation_scope field(s): {sorted(missing)}")
    return EvaluationScope(
        kind=data["kind"],
        image_count=data["image_count"],
        limit_train=data["limit_train"],
        limit_val=data["limit_val"],
    )


def normalize_evaluation_scope(
    raw: Mapping[str, Any] | None,
    *,
    limit_train: int | None,
    limit_val: int | None,
    image_count: int | None,
) -> tuple[EvaluationScope, list[str]]:
    """Normalize an optional raw ``evaluation_scope`` mapping.

    An explicit mapping parses strictly: unknown kinds and unknown keys are
    rejected, and contracts v2 cross-field rules apply. Keys absent from the
    mapping fall back to the supplied ``limit_train``/``limit_val``/
    ``image_count`` values.

    A missing scope normalizes to ``limited_unknown`` built from the supplied
    limits, with an explicit warning that the run is non-formal and cannot
    produce validated manifests.
    """
    if raw is None:
        scope = EvaluationScope(
            kind="limited_unknown",
            image_count=image_count,
            limit_train=limit_train,
            limit_val=limit_val,
        )
        return scope, [
            "evaluation_scope missing from config; normalized to limited_unknown "
            "(non-formal run: cannot produce validated manifests)"
        ]
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"evaluation_scope must be a mapping, got {type(raw).__name__}"
        )
    merged = {
        "kind": raw.get("kind"),
        "image_count": raw.get("image_count", image_count),
        "limit_train": raw.get("limit_train", limit_train),
        "limit_val": raw.get("limit_val", limit_val),
    }
    extras = {key: value for key, value in raw.items() if key not in merged}
    return parse_evaluation_scope({**merged, **extras}), []


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list of strings, got {value!r}")
    items = tuple(value)
    if not all(isinstance(item, str) and item for item in items):
        raise ValueError(f"{name} must contain only non-empty strings, got {value!r}")
    return items


@dataclass(frozen=True)
class ExperimentDefinition:
    """Executable record shape for the experiment registry (contracts v1).

    Basic field validation only; discriminated-union enforcement between
    ``executable_definition`` and ``historical_artifact`` records arrives
    with the experiment registry task.
    """

    id: str
    research_line: str
    capability: str
    config_path: str
    legacy_entrypoint: str
    handler: str
    runnable: RunnableMode
    evaluation_scope: EvaluationScope
    required_inputs: tuple[str, ...]
    expected_outputs: tuple[str, ...]
    gpu_policy: str
    decision_document: str

    def __post_init__(self) -> None:
        for name in ("id", "research_line", "capability", "handler", "gpu_policy"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string, got {value!r}")
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
        if not isinstance(self.evaluation_scope, EvaluationScope):
            raise ValueError(
                "evaluation_scope must be an EvaluationScope, "
                f"got {type(self.evaluation_scope).__name__}"
            )
        object.__setattr__(
            self, "required_inputs", _string_tuple(self.required_inputs, "required_inputs")
        )
        object.__setattr__(
            self, "expected_outputs", _string_tuple(self.expected_outputs, "expected_outputs")
        )
        validate_relative_path(self.config_path)
        validate_relative_path(self.legacy_entrypoint)
        validate_relative_path(self.decision_document)


_EXPERIMENT_DEFINITION_KEYS = frozenset(
    {
        "id",
        "research_line",
        "capability",
        "config_path",
        "legacy_entrypoint",
        "handler",
        "runnable",
        "evaluation_scope",
        "required_inputs",
        "expected_outputs",
        "gpu_policy",
        "decision_document",
    }
)


def parse_experiment_definition(data: Mapping[str, Any]) -> ExperimentDefinition:
    """Strictly parse an ``ExperimentDefinition`` mapping; unknown keys are rejected."""
    if not isinstance(data, Mapping):
        raise ValueError(
            f"experiment definition must be a mapping, got {type(data).__name__}"
        )
    keys = set(data)
    unknown = keys - _EXPERIMENT_DEFINITION_KEYS
    if unknown:
        raise ValueError(f"unknown experiment definition field(s): {sorted(unknown)}")
    missing = _EXPERIMENT_DEFINITION_KEYS - keys
    if missing:
        raise ValueError(f"missing experiment definition field(s): {sorted(missing)}")
    scope = data["evaluation_scope"]
    if isinstance(scope, Mapping):
        scope = parse_evaluation_scope(scope)
    return ExperimentDefinition(
        id=data["id"],
        research_line=data["research_line"],
        capability=data["capability"],
        config_path=data["config_path"],
        legacy_entrypoint=data["legacy_entrypoint"],
        handler=data["handler"],
        runnable=data["runnable"],
        evaluation_scope=scope,
        required_inputs=data["required_inputs"],
        expected_outputs=data["expected_outputs"],
        gpu_policy=data["gpu_policy"],
        decision_document=data["decision_document"],
    )
