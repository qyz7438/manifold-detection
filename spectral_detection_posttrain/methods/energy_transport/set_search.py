"""Compatibility shim: selection policies moved to ``energy_transport.policy``.

The implementations now live in
``spectral_detection_posttrain.methods.energy_transport.policy.set_search``.
This module only re-exports the public names so existing flat imports keep
working during the package-boundary refactor (plan Tasks 13/14).
"""

from spectral_detection_posttrain.methods.energy_transport.policy.set_search import (
    ActionCandidate,
    PairedBootstrapSummary,
    SetEvaluator,
    SetOutcome,
    SetSearchResult,
    beam_search,
    deterministic_delta_permutation,
    greedy_positive_marginal_selection,
    local_top_b,
    paired_bootstrap_summary,
    set_outcome_from_prediction,
)

__all__ = [
    "ActionCandidate",
    "PairedBootstrapSummary",
    "SetEvaluator",
    "SetOutcome",
    "SetSearchResult",
    "beam_search",
    "deterministic_delta_permutation",
    "greedy_positive_marginal_selection",
    "local_top_b",
    "paired_bootstrap_summary",
    "set_outcome_from_prediction",
]
