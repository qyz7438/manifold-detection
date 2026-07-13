"""Structured trainer protocols for detection post-training (plan Task 15).

This module defines the stable contract that every detection trainer adapter
under ``spectral_detection_posttrain.trainers`` satisfies:

- ``DetectionTrainer.train(request)`` takes a :class:`TrainRequest` and
  returns a :class:`TrainResult`.
- Trainers return *data*: checkpoint artifact references, scalar metrics, and
  plain diagnostics. They never decide research status — status words such as
  "validated" or "promoted" are rejected from trainer output because
  promotion happens only through the reviewed manifest path in
  ``spectral_detection_posttrain.experiments``.

``ArtifactRef`` is the manifest-contract type from
``spectral_detection_posttrain.experiments.artifacts``. It is re-exported
here so trainer consumers have a single import site; do not define a parallel
checkpoint-reference type.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Protocol, runtime_checkable

from spectral_detection_posttrain.experiments.artifacts import ArtifactRef

__all__ = [
    "ArtifactRef",
    "DetectionTrainer",
    "RESEARCH_STATUS_TERMS",
    "TrainRequest",
    "TrainResult",
    "filter_scalar_metrics",
]

#: Tokens that assert a research-status decision. Trainers must never emit
#: them in metrics or diagnostics: status is decided only by the reviewed
#: manifest promotion path, never by trainer output.
RESEARCH_STATUS_TERMS = frozenset(
    {"validated", "authorized", "promoted", "promotion", "approved", "reviewed"}
)

_TOKEN_RE = re.compile(r"[a-z]+")
# ``bool`` is an ``int`` subclass; both are listed explicitly for clarity.
_METRIC_SCALAR_TYPES = (bool, int, float, str)


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


def _reject_research_status(value: Any, context: str) -> None:
    for text in _walk_strings(value):
        offending = set(_TOKEN_RE.findall(text.lower())) & RESEARCH_STATUS_TERMS
        if offending:
            raise ValueError(
                "trainer output must not decide research status: "
                f"{sorted(offending)} found in {context}"
            )


def _validate_non_empty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string, got {value!r}")
    return value


def _validate_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an int or None, got {value!r}")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive, got {value!r}")
    return value


def filter_scalar_metrics(row: Mapping[str, Any]) -> dict[str, float | int | str | bool | None]:
    """Keep only the plain scalar entries of a metrics mapping.

    Nested blocks (per-class dicts, parity summaries, tensors) are dropped so
    the result satisfies the :class:`TrainResult` metrics contract; adapters
    put the dropped detail into ``diagnostics`` instead.
    """
    return {
        key: value
        for key, value in row.items()
        if isinstance(key, str) and (value is None or isinstance(value, _METRIC_SCALAR_TYPES))
    }


@dataclass(frozen=True)
class TrainRequest:
    """Inputs for one training run.

    ``runs_root``/``run_name`` resolve the run directory exactly like the
    maintained entry points: ``<runs_root>/<run_name>``. Adapters must write
    only inside that directory. ``seed``/``epochs``/``limit_*`` override the
    corresponding config-file values when set.
    """

    config_path: str
    run_name: str
    runs_root: str = "runs"
    seed: int | None = None
    epochs: int | None = None
    limit_train: int | None = None
    limit_val: int | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.config_path, "config_path")
        _validate_non_empty_str(self.run_name, "run_name")
        _validate_non_empty_str(self.runs_root, "runs_root")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise ValueError(f"seed must be an int or None, got {self.seed!r}")
        _validate_positive_int(self.epochs, "epochs")
        _validate_positive_int(self.limit_train, "limit_train")
        _validate_positive_int(self.limit_val, "limit_val")


@dataclass(frozen=True)
class TrainResult:
    """Structured output of one training run.

    Data only: checkpoint artifact references, scalar metrics, and plain
    diagnostics. Research-status terms are rejected on construction, so a
    trainer cannot smuggle a promotion decision into its output.
    """

    checkpoints: tuple[ArtifactRef, ...]
    metrics: dict[str, float | int | str | bool | None]
    diagnostics: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoints, (tuple, list)):
            raise ValueError(
                f"checkpoints must be a tuple of ArtifactRef, got {type(self.checkpoints).__name__}"
            )
        checkpoints = tuple(self.checkpoints)
        for ref in checkpoints:
            if not isinstance(ref, ArtifactRef):
                raise ValueError(f"checkpoints entries must be ArtifactRef instances, got {ref!r}")
        object.__setattr__(self, "checkpoints", checkpoints)

        if not isinstance(self.metrics, Mapping):
            raise ValueError(f"metrics must be a mapping, got {type(self.metrics).__name__}")
        metrics = dict(self.metrics)
        for key, value in metrics.items():
            if not isinstance(key, str):
                raise ValueError(f"metrics keys must be strings, got {key!r}")
            if value is not None and not isinstance(value, _METRIC_SCALAR_TYPES):
                raise ValueError(
                    "metrics values must be scalar (float, int, str, bool, or None), "
                    f"got {key!r}: {value!r}"
                )
        object.__setattr__(self, "metrics", metrics)

        if not isinstance(self.diagnostics, Mapping):
            raise ValueError(f"diagnostics must be a mapping, got {type(self.diagnostics).__name__}")
        diagnostics = dict(self.diagnostics)
        for key in diagnostics:
            if not isinstance(key, str):
                raise ValueError(f"diagnostics keys must be strings, got {key!r}")
        object.__setattr__(self, "diagnostics", diagnostics)

        _reject_research_status(metrics, "metrics")
        _reject_research_status(diagnostics, "diagnostics")


@runtime_checkable
class DetectionTrainer(Protocol):
    """A detection trainer runs one request and returns structured data.

    Implementations adapt maintained training code to this contract; they
    return measurements and artifact references and never a research-status
    verdict.
    """

    def train(self, request: TrainRequest) -> TrainResult: ...
