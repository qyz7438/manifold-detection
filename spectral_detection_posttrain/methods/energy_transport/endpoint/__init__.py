"""Dense set endpoints: train-time teacher energy and additive set-quality heads.

This subpackage is the canonical home of the two endpoint modules migrated
from the flat ``energy_transport`` package in refactor plan Task 14 phase 2:

- ``dense_set_energy``: differentiable dense teacher components (coverage,
  background/class/duplicate risk, calibration error) over native post-NMS
  prediction sets, plus robust scalar summaries for fit-only standardization.
- ``dense_endpoint``: reduced teacher coordinates, robust fit statistics,
  sparse pair features, and the permutation-invariant mean-additive
  ``DenseSetEnergyEndpoint`` head.

The flat sibling modules remain pure forwarding shims; prefer importing from
``energy_transport.endpoint`` in new code. This package is standalone: it may
depend only on itself and the base substrate (torch/torchvision/numpy), never
on other ``energy_transport`` subpackages, ``core``, trainers, experiments,
datasets, scripts, or runs.
"""

from spectral_detection_posttrain.methods.energy_transport.endpoint.dense_set_energy import (
    DenseTeacherComponents,
    DenseTeacherConfig,
    dense_teacher_components,
    robust_scalar_summary,
)
from spectral_detection_posttrain.methods.energy_transport.endpoint.dense_endpoint import (
    DenseEndpointOutput,
    DenseSetEnergyEndpoint,
    RobustTeacherStats,
    build_sparse_pair_features,
    reduced_teacher_values,
)

__all__ = [
    "DenseEndpointOutput",
    "DenseSetEnergyEndpoint",
    "DenseTeacherComponents",
    "DenseTeacherConfig",
    "RobustTeacherStats",
    "build_sparse_pair_features",
    "dense_teacher_components",
    "reduced_teacher_values",
    "robust_scalar_summary",
]
