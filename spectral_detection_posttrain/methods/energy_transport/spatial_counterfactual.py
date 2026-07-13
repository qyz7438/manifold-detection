"""Compatibility shim: transport diagnostics moved to ``energy_transport.diagnostics``.

The implementations now live in
``spectral_detection_posttrain.methods.energy_transport.diagnostics.spatial_counterfactual``.
This module only re-exports the public names so existing flat imports keep
working during the package-boundary refactor (plan Tasks 13/14).
"""

from spectral_detection_posttrain.methods.energy_transport.diagnostics.spatial_counterfactual import (
    SpatialCandidateRows,
    FrozenSpatialFeatureMap,
    extract_spatial_candidate_rows,
    move_spatial_candidate_rows,
    replace_action_features,
    spatial_counterfactual_blocks,
    spatial_layout_shuffle,
    within_image_delta_alignment_shuffle,
    fit_spatial_feature_map,
    transform_spatial_features,
    evaluate_spatial_counterfactual_gates,
)

__all__ = [
    "SpatialCandidateRows",
    "FrozenSpatialFeatureMap",
    "extract_spatial_candidate_rows",
    "move_spatial_candidate_rows",
    "replace_action_features",
    "spatial_counterfactual_blocks",
    "spatial_layout_shuffle",
    "within_image_delta_alignment_shuffle",
    "fit_spatial_feature_map",
    "transform_spatial_features",
    "evaluate_spatial_counterfactual_gates",
]
