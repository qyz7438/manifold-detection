"""Compatibility shim: selection policies moved to ``energy_transport.policy``.

The implementations now live in
``spectral_detection_posttrain.methods.energy_transport.policy.global_top1``.
This module only re-exports the public names so existing flat imports keep
working during the package-boundary refactor (plan Tasks 13/14).
"""

from spectral_detection_posttrain.methods.energy_transport.policy.global_top1 import (
    ActionTopologyGlobalTop1PolicyHead,
    AdaptiveConsensusGlobalTop1PolicyHead,
    FlattenedGlobalLogits,
    GlobalTop1Output,
    GlobalTop1PolicyHead,
    GlobalTop1Selection,
    GlobalTop1Target,
    NativeActionTopologyGlobalTop1PolicyHead,
    SetContextGlobalTop1PolicyHead,
    action_conditioned_nms_topology,
    build_global_top1_target,
    flatten_observable_action_logits,
    global_top1_balanced_margin_loss,
    global_top1_loss,
    select_adaptive_consensus_action,
    select_global_top1_action,
)

__all__ = [
    "ActionTopologyGlobalTop1PolicyHead",
    "AdaptiveConsensusGlobalTop1PolicyHead",
    "FlattenedGlobalLogits",
    "GlobalTop1Output",
    "GlobalTop1PolicyHead",
    "GlobalTop1Selection",
    "GlobalTop1Target",
    "NativeActionTopologyGlobalTop1PolicyHead",
    "SetContextGlobalTop1PolicyHead",
    "action_conditioned_nms_topology",
    "build_global_top1_target",
    "flatten_observable_action_logits",
    "global_top1_balanced_margin_loss",
    "global_top1_loss",
    "select_adaptive_consensus_action",
    "select_global_top1_action",
]
