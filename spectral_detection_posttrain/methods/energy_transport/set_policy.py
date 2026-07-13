"""Compatibility shim: selection policies moved to ``energy_transport.policy``.

The implementations now live in
``spectral_detection_posttrain.methods.energy_transport.policy.set_policy``.
This module only re-exports the public names so existing flat imports keep
working during the package-boundary refactor (plan Tasks 13/14).
"""

from spectral_detection_posttrain.methods.energy_transport.policy.set_policy import (
    NMSAwareSetPolicyHead,
    SetPolicyLossConfig,
    SetPolicyOutput,
    SetPolicySelection,
    class_aware_conflict_statistics,
    select_set_policy_actions,
    set_policy_loss,
)

__all__ = [
    "NMSAwareSetPolicyHead",
    "SetPolicyLossConfig",
    "SetPolicyOutput",
    "SetPolicySelection",
    "class_aware_conflict_statistics",
    "select_set_policy_actions",
    "set_policy_loss",
]
