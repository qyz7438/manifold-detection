"""Action core: bounded local transport primitives for energy-guided ROI transport.

This subpackage is the canonical home of the five action-core modules migrated
from the flat ``energy_transport`` package in refactor plan Task 13:

- ``actions``: action head, bounded score deltas, energy and safety losses.
- ``contracts``: typed action/state/outcome/preference dataclasses.
- ``operators``: box delta decode and image clipping.
- ``preferences``: top-vs-bottom pairwise preference construction.
- ``geometric_constraints``: action-local geometric regularizers.

The flat sibling modules remain pure forwarding shims; prefer importing from
``energy_transport.action`` in new code.
"""

from spectral_detection_posttrain.methods.energy_transport.action.actions import (
    ActionLocalTransportHead,
    ROITransportActions,
    apply_bounded_score_delta,
    rescue_budget_loss,
    summarize_score_actions,
    threshold_preservation_loss,
    transport_action_energy,
)
from spectral_detection_posttrain.methods.energy_transport.action.contracts import (
    ActionOutcome,
    ConstraintConfig,
    PreferenceBatch,
    ROIActionState,
)
from spectral_detection_posttrain.methods.energy_transport.action.operators import (
    apply_box_delta,
    clip_boxes_to_image,
)
from spectral_detection_posttrain.methods.energy_transport.action.preferences import (
    build_top_bottom_preferences,
)
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
    "ActionLocalTransportHead",
    "ROITransportActions",
    "apply_bounded_score_delta",
    "rescue_budget_loss",
    "summarize_score_actions",
    "threshold_preservation_loss",
    "transport_action_energy",
    "ActionOutcome",
    "ConstraintConfig",
    "PreferenceBatch",
    "ROIActionState",
    "apply_box_delta",
    "clip_boxes_to_image",
    "build_top_bottom_preferences",
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
