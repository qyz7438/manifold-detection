"""Compatibility shim: selection policies moved to ``energy_transport.policy``.

The implementations now live in
``spectral_detection_posttrain.methods.energy_transport.policy.listwise_noop``.
This module only re-exports the public names so existing flat imports keep
working during the package-boundary refactor (plan Tasks 13/14).
"""

from spectral_detection_posttrain.methods.energy_transport.policy.listwise_noop import (
    LinearListwiseModel,
    ListwiseBatch,
    build_listwise_batch,
    calibrate_noop_margin,
    evaluate_listwise_gates,
    fit_listwise_model,
    listwise_decision_metrics,
    paired_gain_stats,
    predict_listwise_scores,
)

__all__ = [
    "LinearListwiseModel",
    "ListwiseBatch",
    "build_listwise_batch",
    "calibrate_noop_margin",
    "evaluate_listwise_gates",
    "fit_listwise_model",
    "listwise_decision_metrics",
    "paired_gain_stats",
    "predict_listwise_scores",
]
