"""Canonical DPO/action-preference utilities."""

from .action_verifier import (
    ActionBatch,
    ActionVerifierConfig,
    DpoPairs,
    build_action_batch,
    build_dpo_pairs,
    build_rlvr_rewards,
    compute_fft_action_quality,
    compute_manifold_action_quality,
    decode_box_actions,
    dpo_loss_from_log_probs,
    gaussian_log_prob,
    normalize_group_advantage,
)

__all__ = [
    "ActionBatch",
    "ActionVerifierConfig",
    "DpoPairs",
    "build_action_batch",
    "build_dpo_pairs",
    "build_rlvr_rewards",
    "compute_fft_action_quality",
    "compute_manifold_action_quality",
    "decode_box_actions",
    "dpo_loss_from_log_probs",
    "gaussian_log_prob",
    "normalize_group_advantage",
]
