"""Compatibility shim: selection policies moved to ``energy_transport.policy``.

The implementations now live in
``spectral_detection_posttrain.methods.energy_transport.policy.joint_delta_u``.
This module only re-exports the public names so existing flat imports keep
working during the package-boundary refactor (plan Tasks 13/14).
"""

from spectral_detection_posttrain.methods.energy_transport.policy.joint_delta_u import (
    GroupHeldoutSplit,
    JointDeltaULossConfig,
    JointDeltaUProbe,
    JointDeltaUSelection,
    ProposalSetEdges,
    build_proposal_set_edges,
    group_heldout_split,
    joint_delta_u_loss,
    joint_delta_u_metrics,
    select_joint_delta_u_actions,
)

__all__ = [
    "GroupHeldoutSplit",
    "JointDeltaULossConfig",
    "JointDeltaUProbe",
    "JointDeltaUSelection",
    "ProposalSetEdges",
    "build_proposal_set_edges",
    "group_heldout_split",
    "joint_delta_u_loss",
    "joint_delta_u_metrics",
    "select_joint_delta_u_actions",
]
