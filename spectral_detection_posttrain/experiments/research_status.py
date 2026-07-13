"""Research status registry: the machine-readable authority for line state.

``spectral_detection_posttrain/configs/registry/research_lines.json`` is the
only machine-readable authority for whether a research line is authorized,
frozen, blocked, diagnostic, baseline-only, historical, invalidated, or
validated. Scientific status is changed only by a reviewed decision commit:
result parsers may propose a status transition but may not write one, and
``validate_status_transition`` is an explicit API that a result parser must
never call implicitly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .contracts import ResearchStatus, RunnableMode, validate_relative_path

__all__ = [
    "STATUS_TO_RUNNABLE",
    "ResearchLine",
    "ResearchLineRegistry",
    "can_authorize_new_run",
    "load_research_status_registry",
    "validate_runnable_for_status",
    "validate_status_transition",
]


# Fixed status-to-execution mapping (plan Section 3.1). Not independently
# configurable; only ``active`` may authorize a new run.
STATUS_TO_RUNNABLE: dict[ResearchStatus, frozenset[RunnableMode]] = {
    ResearchStatus.ACTIVE: frozenset(
        {
            RunnableMode.AUTHORIZED,
            RunnableMode.DIAGNOSTIC_ONLY,
            RunnableMode.FROZEN_REPRODUCTION_ONLY,
        }
    ),
    ResearchStatus.QUEUED: frozenset({RunnableMode.DISABLED, RunnableMode.DIAGNOSTIC_ONLY}),
    ResearchStatus.PAUSED: frozenset({RunnableMode.DISABLED, RunnableMode.DIAGNOSTIC_ONLY}),
    ResearchStatus.FROZEN: frozenset(
        {
            RunnableMode.DISABLED,
            RunnableMode.DIAGNOSTIC_ONLY,
            RunnableMode.FROZEN_REPRODUCTION_ONLY,
        }
    ),
    ResearchStatus.BLOCKED: frozenset({RunnableMode.DISABLED, RunnableMode.DIAGNOSTIC_ONLY}),
    ResearchStatus.DIAGNOSTIC: frozenset(
        {RunnableMode.DISABLED, RunnableMode.DIAGNOSTIC_ONLY}
    ),
    ResearchStatus.BASELINE: frozenset(
        {
            RunnableMode.DISABLED,
            RunnableMode.DIAGNOSTIC_ONLY,
            RunnableMode.FROZEN_REPRODUCTION_ONLY,
        }
    ),
    ResearchStatus.HISTORICAL: frozenset(
        {RunnableMode.DISABLED, RunnableMode.FROZEN_REPRODUCTION_ONLY}
    ),
    ResearchStatus.INVALIDATED: frozenset(
        {RunnableMode.DISABLED, RunnableMode.DIAGNOSTIC_ONLY}
    ),
    ResearchStatus.VALIDATED: frozenset(
        {RunnableMode.DISABLED, RunnableMode.FROZEN_REPRODUCTION_ONLY}
    ),
}

_DECISION_REQUIRED_STATUSES = frozenset(
    {
        ResearchStatus.FROZEN,
        ResearchStatus.BLOCKED,
        ResearchStatus.INVALIDATED,
        ResearchStatus.VALIDATED,
    }
)


def _coerce_status(value: Any, name: str = "status") -> ResearchStatus:
    if isinstance(value, ResearchStatus):
        return value
    if isinstance(value, str):
        try:
            return ResearchStatus(value)
        except ValueError:
            raise ValueError(f"unknown research status: {value!r}") from None
    raise ValueError(f"{name} must be a ResearchStatus, got {value!r}")


def validate_runnable_for_status(status: ResearchStatus, runnable: RunnableMode) -> None:
    """Raise ``ValueError`` if *runnable* is not allowed for *status*."""
    status = _coerce_status(status)
    if not isinstance(runnable, RunnableMode):
        if isinstance(runnable, str):
            try:
                runnable = RunnableMode(runnable)
            except ValueError:
                raise ValueError(f"unknown runnable mode: {runnable!r}") from None
        else:
            raise ValueError(f"runnable must be a RunnableMode, got {runnable!r}")
    if runnable not in STATUS_TO_RUNNABLE[status]:
        raise ValueError(
            f"runnable mode {runnable.value!r} is not allowed for status {status.value!r}"
        )


@dataclass(frozen=True)
class ResearchLine:
    """One research line in the status registry.

    ``evidence`` and ``decision_document`` are committed repo-relative
    pointers. ``uncommitted_context`` may list untracked working documents
    (for example ``docs/energy_transport_vs_fpn_sm_analysis.md``) but must
    never be a required evidence pointer.
    """

    id: str
    status: ResearchStatus
    hypothesis: str
    claim_ceiling: str = ""
    evidence: tuple[str, ...] = ()
    decision_document: str = ""
    uncommitted_context: tuple[str, ...] = ()
    notes: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError(f"id must be a non-empty string, got {self.id!r}")
        status = _coerce_status(self.status)
        object.__setattr__(self, "status", status)
        if not isinstance(self.hypothesis, str) or not self.hypothesis.strip():
            raise ValueError("hypothesis must be a non-empty string")
        if status in _DECISION_REQUIRED_STATUSES and not (
            isinstance(self.decision_document, str) and self.decision_document.strip()
        ):
            raise ValueError(
                f"status {status.value!r} requires a non-empty decision_document"
            )
        if not isinstance(self.claim_ceiling, str) or not self.claim_ceiling.strip():
            raise ValueError("claim_ceiling must be a non-empty string")
        object.__setattr__(self, "evidence", _pointer_tuple(self.evidence, "evidence"))
        if not self.evidence:
            raise ValueError("evidence must list at least one committed repo-relative pointer")
        if self.decision_document:
            validate_relative_path(self.decision_document)
        object.__setattr__(
            self,
            "uncommitted_context",
            _pointer_tuple(self.uncommitted_context, "uncommitted_context"),
        )
        if self.notes is not None and not isinstance(self.notes, str):
            raise ValueError(f"notes must be a string or None, got {self.notes!r}")


def _pointer_tuple(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list of repo-relative pointers, got {value!r}")
    items = tuple(value)
    for item in items:
        validate_relative_path(item)
    return items


# The single committed transition matrix (plan Section 3.1). Statuses not
# listed here (frozen, blocked, invalidated, validated, historical, baseline,
# diagnostic) may be reclassified only through a superseding decision document.
_EXPLICIT_TRANSITIONS: dict[ResearchStatus, frozenset[ResearchStatus]] = {
    ResearchStatus.ACTIVE: frozenset(
        {
            ResearchStatus.QUEUED,
            ResearchStatus.PAUSED,
            ResearchStatus.FROZEN,
            ResearchStatus.BLOCKED,
            ResearchStatus.DIAGNOSTIC,
            ResearchStatus.VALIDATED,
        }
    ),
    ResearchStatus.QUEUED: frozenset(
        {
            ResearchStatus.ACTIVE,
            ResearchStatus.PAUSED,
            ResearchStatus.FROZEN,
            ResearchStatus.BLOCKED,
        }
    ),
    ResearchStatus.PAUSED: frozenset(
        {
            ResearchStatus.ACTIVE,
            ResearchStatus.QUEUED,
            ResearchStatus.FROZEN,
            ResearchStatus.BLOCKED,
        }
    ),
}


def validate_status_transition(
    old: ResearchStatus,
    new: ResearchStatus,
    decision_document: str = "",
) -> None:
    """Validate one committed status transition; raise ``ValueError`` if invalid.

    Explicit API only: a result parser must never call this implicitly to
    promote or demote a research line, and no status change is automatic.
    Every accepted transition must still be written by a reviewed decision
    commit. Transitions out of ``frozen``, ``blocked``, ``invalidated``,
    ``validated``, ``historical``, ``baseline``, or ``diagnostic`` require a
    non-empty ``decision_document`` that explicitly supersedes the prior
    evidence; callers and reviewers are responsible for that superseding
    content, this function only enforces the committed matrix and the
    non-empty requirement.
    """
    old = _coerce_status(old, "old")
    new = _coerce_status(new, "new")
    if old is new:
        raise ValueError(
            f"no automatic status change: {old.value!r} -> {new.value!r} is not a transition"
        )
    if old in _EXPLICIT_TRANSITIONS:
        if new not in _EXPLICIT_TRANSITIONS[old]:
            raise ValueError(
                f"transition {old.value!r} -> {new.value!r} is not in the committed matrix"
            )
        return
    if not (isinstance(decision_document, str) and decision_document.strip()):
        raise ValueError(
            f"transition out of {old.value!r} requires a non-empty decision_document "
            "that explicitly supersedes prior evidence"
        )


def can_authorize_new_run(line: ResearchLine) -> bool:
    """True only when the line is ``active``; ``queued`` never means runnable."""
    return line.status is ResearchStatus.ACTIVE


@dataclass(frozen=True)
class ResearchLineRegistry:
    """Immutable collection of research lines with unique IDs."""

    lines: tuple[ResearchLine, ...]
    schema_version: str = "research_lines.v1"

    def __post_init__(self) -> None:
        lines = tuple(self.lines)
        for line in lines:
            if not isinstance(line, ResearchLine):
                raise ValueError(f"lines must contain ResearchLine items, got {line!r}")
        object.__setattr__(self, "lines", lines)
        seen: set[str] = set()
        duplicates = sorted({line.id for line in lines if line.id in seen or seen.add(line.id)})
        if duplicates:
            raise ValueError(f"duplicate research line id(s): {duplicates}")

    def get(self, line_id: str) -> ResearchLine:
        for line in self.lines:
            if line.id == line_id:
                return line
        raise KeyError(line_id)


_REGISTRY_TOP_KEYS = frozenset({"schema_version", "lines"})
_LINE_REQUIRED_KEYS = frozenset(
    {"id", "status", "hypothesis", "claim_ceiling", "evidence", "decision_document"}
)
_LINE_OPTIONAL_KEYS = frozenset({"uncommitted_context", "notes"})


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_research_status_registry(path: str | Path) -> ResearchLineRegistry:
    """Strictly load and validate ``research_lines.json``.

    Unknown keys, missing required keys, duplicate IDs, invalid statuses,
    empty evidence, invalid relative paths, and committed pointers that do
    not exist on disk are all rejected. ``uncommitted_context`` pointers are
    validated for shape but never for existence.
    """
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in research status registry {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("research status registry must be a JSON object")
    unknown = set(payload) - _REGISTRY_TOP_KEYS
    if unknown:
        raise ValueError(f"unknown registry field(s): {sorted(unknown)}")
    if payload.get("schema_version") != "research_lines.v1":
        raise ValueError(
            f"unsupported registry schema_version: {payload.get('schema_version')!r}"
        )
    raw_lines = payload.get("lines")
    if not isinstance(raw_lines, list):
        raise ValueError("registry field 'lines' must be a list")
    root = _repo_root()
    lines = [_parse_line(raw, root) for raw in raw_lines]
    return ResearchLineRegistry(lines=tuple(lines))


def _parse_line(raw: Any, root: Path) -> ResearchLine:
    if not isinstance(raw, Mapping):
        raise ValueError(f"research line entry must be a mapping, got {type(raw).__name__}")
    keys = set(raw)
    unknown = keys - (_LINE_REQUIRED_KEYS | _LINE_OPTIONAL_KEYS)
    if unknown:
        raise ValueError(f"unknown research line field(s): {sorted(unknown)}")
    missing = _LINE_REQUIRED_KEYS - keys
    if missing:
        raise ValueError(f"missing research line field(s): {sorted(missing)}")
    line = ResearchLine(
        id=raw["id"],
        status=raw["status"],
        hypothesis=raw["hypothesis"],
        claim_ceiling=raw["claim_ceiling"],
        evidence=tuple(raw["evidence"]) if isinstance(raw["evidence"], list) else raw["evidence"],
        decision_document=raw["decision_document"],
        uncommitted_context=tuple(raw.get("uncommitted_context", ()))
        if isinstance(raw.get("uncommitted_context", ()), list)
        else raw.get("uncommitted_context", ()),
        notes=raw.get("notes"),
    )
    for pointer in (*line.evidence, line.decision_document):
        if not (root / pointer).is_file():
            raise ValueError(f"committed evidence pointer does not exist on disk: {pointer!r}")
    return line
