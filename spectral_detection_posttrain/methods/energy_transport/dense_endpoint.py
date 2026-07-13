"""Compatibility shim: dense set endpoints moved to ``energy_transport.endpoint``.

The implementations now live in
``spectral_detection_posttrain.methods.energy_transport.endpoint.dense_endpoint``.
This module only re-exports the public names so existing flat imports keep
working during the package-boundary refactor (plan Tasks 13/14).
"""

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
    "RobustTeacherStats",
    "build_sparse_pair_features",
    "reduced_teacher_values",
]
