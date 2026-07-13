"""Compatibility shim: benefit energy moved to ``energy_transport.native``.

The implementations now live in
``spectral_detection_posttrain.methods.energy_transport.native.benefit_energy``.
This module only re-exports the public names so existing flat imports keep
working during the package-boundary refactor (plan Tasks 13/14).
"""

from spectral_detection_posttrain.methods.energy_transport.native.benefit_energy import (
    ActionBenefitEnergyHead,
    ActionBenefitTargets,
    BenefitEnergyLossConfig,
    action_benefit_energy_loss,
    apply_action_benefit_gate,
    build_action_benefit_targets,
)

__all__ = [
    "ActionBenefitEnergyHead",
    "ActionBenefitTargets",
    "BenefitEnergyLossConfig",
    "action_benefit_energy_loss",
    "apply_action_benefit_gate",
    "build_action_benefit_targets",
]
