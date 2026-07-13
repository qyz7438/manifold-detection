"""Dense endpoint handler (refactor Task 7).

Covers the dense set-energy endpoint and dense local-delta experiments
(absolute endpoint, clean-val validation, local-delta learner, family
prior). Diagnostic records in this family consume caches and locked configs
and never train a detector; frozen records reproduce only through their
locked legacy entrypoint.
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

__all__ = ["DenseEndpointHandler"]

_DENSE_FAMILY_PREFIX = "detection.energy_transport.dense"


class DenseEndpointHandler(ExperimentHandler):
    """Handler for dense endpoint and dense local-delta experiments."""

    capability: ClassVar[ExperimentCapability] = ExperimentCapability.DENSE_ENDPOINT
    family_note: ClassVar[str] = (
        "dense endpoint experiment: consumes caches and locked configs; "
        "diagnostic records never train a detector"
    )

    def validate(
        self, record: ExecutableDefinition, registry: ExperimentRegistry | None = None
    ) -> tuple[str, ...]:
        problems = list(super().validate(record, registry))
        capability = getattr(record, "capability", "")
        if not (
            isinstance(capability, str) and capability.startswith(_DENSE_FAMILY_PREFIX)
        ):
            problems.append(
                f"capability {capability!r} does not belong to the dense "
                f"endpoint family ({_DENSE_FAMILY_PREFIX}*)"
            )
        return tuple(problems)
