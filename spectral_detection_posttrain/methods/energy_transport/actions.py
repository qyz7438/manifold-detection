"""Compatibility shim: action primitives moved to ``energy_transport.action``.

The implementations now live in
``spectral_detection_posttrain.methods.energy_transport.action.actions``.
This module only re-exports the public names so existing flat imports keep
working during the package-boundary refactor (plan Tasks 13/14).
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

__all__ = [
    "ActionLocalTransportHead",
    "ROITransportActions",
    "apply_bounded_score_delta",
    "rescue_budget_loss",
    "summarize_score_actions",
    "threshold_preservation_loss",
    "transport_action_energy",
]
