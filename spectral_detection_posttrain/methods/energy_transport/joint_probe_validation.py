"""Compatibility shim: transport diagnostics moved to ``energy_transport.diagnostics``.

The implementations now live in
``spectral_detection_posttrain.methods.energy_transport.diagnostics.joint_probe_validation``.
This module only re-exports the public names so existing flat imports keep
working during the package-boundary refactor (plan Tasks 13/14).
"""

from spectral_detection_posttrain.methods.energy_transport.diagnostics.joint_probe_validation import (
    ConservativeCalibration,
    calibrate_conservative_threshold,
    calibrated_selection_metrics,
    constant_utility_baselines,
    imagewise_pairwise_accuracy,
    paired_bootstrap_mean_difference,
    shuffle_edge_topology,
)

__all__ = [
    "ConservativeCalibration",
    "calibrate_conservative_threshold",
    "calibrated_selection_metrics",
    "constant_utility_baselines",
    "imagewise_pairwise_accuracy",
    "paired_bootstrap_mean_difference",
    "shuffle_edge_topology",
]
