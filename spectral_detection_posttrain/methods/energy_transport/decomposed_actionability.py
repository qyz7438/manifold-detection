"""Compatibility shim: transport diagnostics moved to ``energy_transport.diagnostics``.

The implementations now live in
``spectral_detection_posttrain.methods.energy_transport.diagnostics.decomposed_actionability``.
This module only re-exports the public names so existing flat imports keep
working during the package-boundary refactor (plan Tasks 13/14).
"""

from spectral_detection_posttrain.methods.energy_transport.diagnostics.decomposed_actionability import (
    PairwiseRows,
    within_image_pairwise_rows,
    class_image_balanced_weights,
    sign_classification_metrics,
    ranked_abstention_metrics,
    calibrate_ranked_abstention,
    evaluate_decomposed_actionability_gates,
)

__all__ = [
    "PairwiseRows",
    "within_image_pairwise_rows",
    "class_image_balanced_weights",
    "sign_classification_metrics",
    "ranked_abstention_metrics",
    "calibrate_ranked_abstention",
    "evaluate_decomposed_actionability_gates",
]
