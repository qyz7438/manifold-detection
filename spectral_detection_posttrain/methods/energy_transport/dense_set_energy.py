"""Compatibility shim: dense set endpoints moved to ``energy_transport.endpoint``.

The implementations now live in
``spectral_detection_posttrain.methods.energy_transport.endpoint.dense_set_energy``.
This module only re-exports the public names so existing flat imports keep
working during the package-boundary refactor (plan Tasks 13/14).
"""

from spectral_detection_posttrain.methods.energy_transport.endpoint.dense_set_energy import (
    DenseTeacherComponents,
    DenseTeacherConfig,
    dense_teacher_components,
    robust_scalar_summary,
)

__all__ = [
    "DenseTeacherComponents",
    "DenseTeacherConfig",
    "dense_teacher_components",
    "robust_scalar_summary",
]
