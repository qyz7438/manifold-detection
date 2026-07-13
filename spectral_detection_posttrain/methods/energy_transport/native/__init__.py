"""Native detection transport: detector-coupled energy heads and contracts.

This subpackage is the canonical home of the four native modules migrated
from the flat ``energy_transport`` package in refactor plan Task 13 phase 2:

- ``native_contract``: detector-visible candidate contract and strict
  zero-action parity gate helpers.
- ``native_topology``: torchvision-equivalent class-expanded NMS topology
  features (clip, threshold, small-box removal, class-wise NMS, top-K).
- ``candidate_energy``: discrete local box candidates, spatial/context energy
  heads, and identity-relative energy selection.
- ``benefit_energy``: pairwise identity-vs-action energy model and gating.

The flat sibling modules remain pure forwarding shims; prefer importing from
``energy_transport.native`` in new code. This package may depend on
``energy_transport.action`` (layering action <- native), never on trainers,
experiments, datasets, scripts, or runs.
"""

from spectral_detection_posttrain.methods.energy_transport.native.native_contract import (
    DetectorNativeCandidates,
    build_detector_native_candidates,
    build_native_c1_deltas,
    evaluate_native_contract_gates,
    validate_strict_parity_artifact,
)
from spectral_detection_posttrain.methods.energy_transport.native.native_topology import (
    NATIVE_TOPOLOGY_FEATURE_NAMES,
    native_action_nms_topology,
)
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
from spectral_detection_posttrain.methods.energy_transport.native.benefit_energy import (
    ActionBenefitEnergyHead,
    ActionBenefitTargets,
    BenefitEnergyLossConfig,
    action_benefit_energy_loss,
    apply_action_benefit_gate,
    build_action_benefit_targets,
)

__all__ = [
    "DetectorNativeCandidates",
    "build_native_c1_deltas",
    "build_detector_native_candidates",
    "validate_strict_parity_artifact",
    "evaluate_native_contract_gates",
    "NATIVE_TOPOLOGY_FEATURE_NAMES",
    "native_action_nms_topology",
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
    "ActionBenefitEnergyHead",
    "ActionBenefitTargets",
    "BenefitEnergyLossConfig",
    "action_benefit_energy_loss",
    "apply_action_benefit_gate",
    "build_action_benefit_targets",
]
