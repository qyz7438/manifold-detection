"""Selection policies: bounded local action search and set-selection heads.

This subpackage is the canonical home of the eight policy modules migrated
from the flat ``energy_transport`` package in refactor plan Task 14 phase 1:

- ``search``: oracle-bounded residual score-action search.
- ``set_search``: deterministic action-set search and paired statistics.
- ``set_policy``: NMS-aware per-proposal set policy head and selection.
- ``global_top1``: global top-1/no-op policy heads, topology, and losses.
- ``listwise_noop``: listwise no-op-margin model and decision gates.
- ``joint_delta_u``: joint Delta-U probe, loss, selection, and metrics.
- ``adaptive_consensus``: proposal-graph consensus deltas.
- ``post_nms_suppress``: post-NMS suppression policy head and selection.

The flat sibling modules remain pure forwarding shims; prefer importing from
``energy_transport.policy`` in new code. This package may depend on
``energy_transport.action`` and ``energy_transport.native`` (layering
action <- native <- policy), never on trainers, experiments, datasets,
scripts, or runs.
"""

from spectral_detection_posttrain.methods.energy_transport.policy.search import (
    ActionSearchConfig,
    ScoreActionSearchResult,
    apply_score_action_to_prediction,
    select_min_energy_score_actions,
)
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
from spectral_detection_posttrain.methods.energy_transport.policy.set_policy import (
    NMSAwareSetPolicyHead,
    SetPolicyLossConfig,
    SetPolicyOutput,
    SetPolicySelection,
    class_aware_conflict_statistics,
    select_set_policy_actions,
    set_policy_loss,
)
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
from spectral_detection_posttrain.methods.energy_transport.policy.listwise_noop import (
    LinearListwiseModel,
    ListwiseBatch,
    build_listwise_batch,
    calibrate_noop_margin,
    evaluate_listwise_gates,
    fit_listwise_model,
    listwise_decision_metrics,
    paired_gain_stats,
    predict_listwise_scores,
)
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
from spectral_detection_posttrain.methods.energy_transport.policy.adaptive_consensus import (
    proposal_graph_consensus_deltas,
)
from spectral_detection_posttrain.methods.energy_transport.policy.post_nms_suppress import (
    PostNMSSuppression,
    PostNMSSuppressPolicyHead,
    build_post_nms_detection_features,
    select_post_nms_suppression,
    suppress_detection,
)

__all__ = [
    "ActionCandidate",
    "ActionSearchConfig",
    "ActionTopologyGlobalTop1PolicyHead",
    "AdaptiveConsensusGlobalTop1PolicyHead",
    "FlattenedGlobalLogits",
    "GlobalTop1Output",
    "GlobalTop1PolicyHead",
    "GlobalTop1Selection",
    "GlobalTop1Target",
    "GroupHeldoutSplit",
    "JointDeltaULossConfig",
    "JointDeltaUProbe",
    "JointDeltaUSelection",
    "LinearListwiseModel",
    "ListwiseBatch",
    "NMSAwareSetPolicyHead",
    "NativeActionTopologyGlobalTop1PolicyHead",
    "PairedBootstrapSummary",
    "PostNMSSuppression",
    "PostNMSSuppressPolicyHead",
    "ProposalSetEdges",
    "ScoreActionSearchResult",
    "SetContextGlobalTop1PolicyHead",
    "SetEvaluator",
    "SetOutcome",
    "SetPolicyLossConfig",
    "SetPolicyOutput",
    "SetPolicySelection",
    "SetSearchResult",
    "action_conditioned_nms_topology",
    "apply_score_action_to_prediction",
    "beam_search",
    "build_global_top1_target",
    "build_listwise_batch",
    "build_post_nms_detection_features",
    "build_proposal_set_edges",
    "calibrate_noop_margin",
    "class_aware_conflict_statistics",
    "deterministic_delta_permutation",
    "evaluate_listwise_gates",
    "fit_listwise_model",
    "flatten_observable_action_logits",
    "global_top1_balanced_margin_loss",
    "global_top1_loss",
    "greedy_positive_marginal_selection",
    "group_heldout_split",
    "joint_delta_u_loss",
    "joint_delta_u_metrics",
    "listwise_decision_metrics",
    "local_top_b",
    "paired_bootstrap_summary",
    "paired_gain_stats",
    "predict_listwise_scores",
    "proposal_graph_consensus_deltas",
    "select_adaptive_consensus_action",
    "select_global_top1_action",
    "select_joint_delta_u_actions",
    "select_min_energy_score_actions",
    "select_post_nms_suppression",
    "select_set_policy_actions",
    "set_outcome_from_prediction",
    "set_policy_loss",
    "suppress_detection",
]
