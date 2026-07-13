"""Compatibility shim: geometric constraints moved to ``energy_transport.action``.

The implementations now live in
``spectral_detection_posttrain.methods.energy_transport.action.geometric_constraints``.
This module only re-exports the public names so existing flat imports keep
working during the package-boundary refactor (plan Tasks 13/14).
"""

from spectral_detection_posttrain.methods.energy_transport.action.geometric_constraints import (
    SAMPLE_AMBIGUOUS_BG,
    SAMPLE_CLS_ERR,
    SAMPLE_LOC_ERR,
    SAMPLE_PURE_BG,
    SAMPLE_TP,
    GeometricConstraintConfig,
    bbox_aware_action_loss,
    bbox_aware_direct_loss,
    bbox_aware_linearization_loss,
    bg_proto_action_loss,
    classify_error_modes,
    cls_err_action_loss,
    compute_action_local_prototypes,
    feat_preserve_action_loss,
    fg_bg_sep_action_loss,
    geometric_transport_loss,
    intra_tp_action_loss,
    loc_err_action_loss,
)

__all__ = [
    "GeometricConstraintConfig",
    "SAMPLE_TP",
    "SAMPLE_CLS_ERR",
    "SAMPLE_LOC_ERR",
    "SAMPLE_PURE_BG",
    "SAMPLE_AMBIGUOUS_BG",
    "bbox_aware_action_loss",
    "bbox_aware_direct_loss",
    "bbox_aware_linearization_loss",
    "bg_proto_action_loss",
    "classify_error_modes",
    "cls_err_action_loss",
    "compute_action_local_prototypes",
    "feat_preserve_action_loss",
    "fg_bg_sep_action_loss",
    "geometric_transport_loss",
    "intra_tp_action_loss",
    "loc_err_action_loss",
]
