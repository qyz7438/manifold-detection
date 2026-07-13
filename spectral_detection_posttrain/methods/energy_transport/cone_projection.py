"""Compatibility shim: transport diagnostics moved to ``energy_transport.diagnostics``.

The implementations now live in
``spectral_detection_posttrain.methods.energy_transport.diagnostics.cone_projection``.
This module only re-exports the public names so existing flat imports keep
working during the package-boundary refactor (plan Tasks 13/14).
"""

from spectral_detection_posttrain.methods.energy_transport.diagnostics.cone_projection import (
    ConeDecomposition,
    ConeProjectionEndpoint,
    EnergyFn,
    compute_class_prototypes,
    cone_dpog_regularizer,
    cone_residual_alignment_loss,
    cross_entropy_energy,
    decompose_cone_features,
    local_tangent_energy_endpoint,
    normalize_l2,
    project_to_tangent,
)

__all__ = [
    "ConeDecomposition",
    "ConeProjectionEndpoint",
    "EnergyFn",
    "compute_class_prototypes",
    "cone_dpog_regularizer",
    "cone_residual_alignment_loss",
    "cross_entropy_energy",
    "decompose_cone_features",
    "local_tangent_energy_endpoint",
    "normalize_l2",
    "project_to_tangent",
]
