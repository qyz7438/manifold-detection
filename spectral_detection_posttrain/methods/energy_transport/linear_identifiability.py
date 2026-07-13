"""Compatibility shim: transport diagnostics moved to ``energy_transport.diagnostics``.

The implementations now live in
``spectral_detection_posttrain.methods.energy_transport.diagnostics.linear_identifiability``.
This module only re-exports the public names so existing flat imports keep
working during the package-boundary refactor (plan Tasks 13/14).
"""

from spectral_detection_posttrain.methods.energy_transport.diagnostics.linear_identifiability import (
    LocalCandidateRows,
    FrozenLocalFeatureMap,
    RidgeRegressor,
    extract_local_candidate_rows,
    move_local_candidate_rows,
    fit_local_feature_map,
    transform_local_candidate_rows,
    within_image_shuffle_order,
    image_balanced_weights,
    fit_ridge_regression,
    predict_ridge_regression,
    evaluate_identifiability_gates,
)

__all__ = [
    "LocalCandidateRows",
    "FrozenLocalFeatureMap",
    "RidgeRegressor",
    "extract_local_candidate_rows",
    "move_local_candidate_rows",
    "fit_local_feature_map",
    "transform_local_candidate_rows",
    "within_image_shuffle_order",
    "image_balanced_weights",
    "fit_ridge_regression",
    "predict_ridge_regression",
    "evaluate_identifiability_gates",
]
