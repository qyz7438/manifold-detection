"""Compatibility shim: candidate energy moved to ``energy_transport.native``.

The implementations now live in
``spectral_detection_posttrain.methods.energy_transport.native.candidate_energy``.
This module only re-exports the public names so existing flat imports keep
working during the package-boundary refactor (plan Tasks 13/14).
"""

from spectral_detection_posttrain.methods.energy_transport.native.candidate_energy import (
    CandidateEnergyLossConfig,
    CandidateGainLossConfig,
    CandidateQualityTargets,
    ContextOnlyCandidateEnergyHead,
    SpatialCandidateEnergyHead,
    build_candidate_quality_targets,
    build_symmetric_box_candidates,
    candidate_action_energy_loss,
    candidate_action_gain_loss,
    select_min_energy_box_actions,
)

__all__ = [
    "CandidateEnergyLossConfig",
    "CandidateGainLossConfig",
    "CandidateQualityTargets",
    "ContextOnlyCandidateEnergyHead",
    "SpatialCandidateEnergyHead",
    "build_candidate_quality_targets",
    "build_symmetric_box_candidates",
    "candidate_action_energy_loss",
    "candidate_action_gain_loss",
    "select_min_energy_box_actions",
]
