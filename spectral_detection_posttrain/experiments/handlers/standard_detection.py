"""Standard detection handler (refactor Task 7).

Covers ordinary detector train/eval experiments driven through the canonical
runner. No record in the initial registry uses this handler yet; it exists
so the static capability map is complete and future standard-detection
records dispatch without dispatcher changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from spectral_detection_posttrain.experiments.handlers import (
    ExperimentCapability,
    ExperimentHandler,
)

if TYPE_CHECKING:  # pragma: no cover
    from spectral_detection_posttrain.experiments.registry import (
        ExecutableDefinition,
        ExperimentRegistry,
    )

__all__ = ["StandardDetectionHandler"]


class StandardDetectionHandler(ExperimentHandler):
    """Handler for standard detection train/eval experiments."""

    capability: ClassVar[ExperimentCapability] = ExperimentCapability.STANDARD_DETECTION
    family_note: ClassVar[str] = (
        "standard detection experiment: train/eval flows go through the "
        "canonical runner with the locked registered config"
    )

    def validate(
        self, record: ExecutableDefinition, registry: ExperimentRegistry | None = None
    ) -> tuple[str, ...]:
        problems = list(super().validate(record, registry))
        capability = getattr(record, "capability", "")
        if not (isinstance(capability, str) and capability.startswith("detection.")):
            problems.append(
                f"capability {capability!r} does not belong to the detection family"
            )
        return tuple(problems)
