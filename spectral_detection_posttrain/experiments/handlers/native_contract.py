"""Native contract handler (refactor Task 7).

Covers the C1-E1 native action-contract experiments (listwise, budget,
global delta-u, set-context, topology, fine-action, consensus, post-NMS).
Frozen records in this family reproduce only through their locked legacy
entrypoint with the exact registered config hash; the handler never alters
configs.
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

__all__ = ["NativeContractHandler"]

_NATIVE_FAMILY_PREFIX = "detection.energy_transport."


class NativeContractHandler(ExperimentHandler):
    """Handler for native action-contract energy-transport experiments."""

    capability: ClassVar[ExperimentCapability] = ExperimentCapability.NATIVE_CONTRACT
    family_note: ClassVar[str] = (
        "native action-contract experiment: reproduction goes through the "
        "locked legacy entrypoint with the exact registered config hash"
    )

    def validate(
        self, record: ExecutableDefinition, registry: ExperimentRegistry | None = None
    ) -> tuple[str, ...]:
        problems = list(super().validate(record, registry))
        capability = getattr(record, "capability", "")
        if not (
            isinstance(capability, str) and capability.startswith(_NATIVE_FAMILY_PREFIX)
        ):
            problems.append(
                f"capability {capability!r} does not belong to the native "
                f"energy-transport family ({_NATIVE_FAMILY_PREFIX}*)"
            )
        return tuple(problems)
