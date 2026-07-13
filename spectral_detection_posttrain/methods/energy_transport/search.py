"""Compatibility shim: selection policies moved to ``energy_transport.policy``.

The implementations now live in
``spectral_detection_posttrain.methods.energy_transport.policy.search``.
This module only re-exports the public names so existing flat imports keep
working during the package-boundary refactor (plan Tasks 13/14).
"""

from spectral_detection_posttrain.methods.energy_transport.policy.search import (
    ActionSearchConfig,
    ScoreActionSearchResult,
    apply_score_action_to_prediction,
    select_min_energy_score_actions,
)

__all__ = [
    "ActionSearchConfig",
    "ScoreActionSearchResult",
    "apply_score_action_to_prediction",
    "select_min_energy_score_actions",
]
