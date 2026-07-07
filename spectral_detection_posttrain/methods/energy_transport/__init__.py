"""Action-local energy-guided transport primitives for detector post-training.

This package is the maintained entry point for the new research line where the
class-conditioned target manifold is unknown.  It models ROI correction as a
small, constrained action rather than as direct prototype attraction.
"""

from spectral_detection_posttrain.methods.energy_transport.actions import (
    ActionLocalTransportHead,
    ROITransportActions,
    apply_bounded_score_delta,
    rescue_budget_loss,
    summarize_score_actions,
    threshold_preservation_loss,
    transport_action_energy,
)
from spectral_detection_posttrain.methods.energy_transport.cone_projection import (
    ConeDecomposition,
    ConeProjectionEndpoint,
    compute_class_prototypes,
    cone_dpog_regularizer,
    cone_residual_alignment_loss,
    cross_entropy_energy,
    decompose_cone_features,
    local_tangent_energy_endpoint,
)
from spectral_detection_posttrain.methods.energy_transport.contracts import (
    ActionOutcome,
    ConstraintConfig,
    PreferenceBatch,
    ROIActionState,
)
from spectral_detection_posttrain.methods.energy_transport.operators import (
    apply_box_delta,
    clip_boxes_to_image,
)
from spectral_detection_posttrain.methods.energy_transport.preferences import (
    build_top_bottom_preferences,
)
from spectral_detection_posttrain.methods.energy_transport.search import (
    ActionSearchConfig,
    ScoreActionSearchResult,
    apply_score_action_to_prediction,
    select_min_energy_score_actions,
)

__all__ = [
    "ActionLocalTransportHead",
    "ActionOutcome",
    "ActionSearchConfig",
    "ConeDecomposition",
    "ConeProjectionEndpoint",
    "ConstraintConfig",
    "PreferenceBatch",
    "ROIActionState",
    "ROITransportActions",
    "ScoreActionSearchResult",
    "apply_bounded_score_delta",
    "apply_box_delta",
    "apply_score_action_to_prediction",
    "build_top_bottom_preferences",
    "clip_boxes_to_image",
    "compute_class_prototypes",
    "cone_dpog_regularizer",
    "cone_residual_alignment_loss",
    "cross_entropy_energy",
    "decompose_cone_features",
    "local_tangent_energy_endpoint",
    "rescue_budget_loss",
    "select_min_energy_score_actions",
    "summarize_score_actions",
    "threshold_preservation_loss",
    "transport_action_energy",
]
