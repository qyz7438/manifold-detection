from __future__ import annotations

import torch

from spectral_detection_posttrain.methods.energy_transport import (
    ActionLocalTransportHead,
    apply_bounded_score_delta,
    rescue_budget_loss,
    summarize_score_actions,
    threshold_preservation_loss,
    transport_action_energy,
)


def test_action_local_transport_head_starts_near_identity() -> None:
    torch.manual_seed(0)
    head = ActionLocalTransportHead(feature_dim=8, hidden_dim=12, max_score_delta=0.2)
    features = torch.randn(5, 8)

    actions = head(features)

    assert actions.feature_delta.shape == (5, 8)
    assert actions.score_delta.shape == (5,)
    assert actions.box_delta.shape == (5, 4)
    assert actions.keep_logit.shape == (5,)
    assert actions.feature_delta.abs().max().item() < 1e-2
    assert actions.score_delta.abs().max().item() < 1e-2
    assert actions.box_delta.abs().max().item() < 1e-2


def test_apply_bounded_score_delta_caps_and_clamps_scores() -> None:
    scores = torch.tensor([0.02, 0.5, 0.95])
    raw_delta = torch.tensor([10.0, -10.0, 10.0])

    updated = apply_bounded_score_delta(scores, raw_delta, max_delta=0.1)

    assert torch.allclose(updated, torch.tensor([0.12, 0.4, 1.0]))


def test_threshold_preservation_loss_penalizes_bad_crossings_only() -> None:
    old_scores = torch.tensor([0.02, 0.04, 0.04])
    score_delta = torch.tensor([0.01, 0.03, 0.04])
    low_quality = torch.tensor([True, True, False])

    loss = threshold_preservation_loss(
        old_scores,
        score_delta,
        low_quality_mask=low_quality,
        threshold=0.05,
    )

    # First stays below threshold, second bad candidate crosses, third is ignored.
    assert loss.item() > 0.0
    zero = threshold_preservation_loss(
        old_scores,
        torch.zeros_like(score_delta),
        low_quality_mask=low_quality,
        threshold=0.05,
    )
    assert zero.item() == 0.0


def test_rescue_budget_loss_counts_soft_threshold_crossings() -> None:
    old_scores = torch.tensor([0.01, 0.02, 0.03, 0.8])
    score_delta = torch.tensor([0.1, 0.1, 0.1, 0.0])
    candidate_mask = torch.tensor([True, True, True, False])

    over_budget = rescue_budget_loss(
        old_scores,
        score_delta,
        candidate_mask=candidate_mask,
        threshold=0.05,
        max_rescues=1.0,
        temperature=0.01,
    )
    under_budget = rescue_budget_loss(
        old_scores,
        torch.zeros_like(score_delta),
        candidate_mask=candidate_mask,
        threshold=0.05,
        max_rescues=1.0,
        temperature=0.01,
    )

    assert over_budget.item() > 0.0
    assert under_budget.item() == 0.0


def test_transport_action_energy_is_differentiable_scalar() -> None:
    feature_delta = torch.randn(4, 8, requires_grad=True)
    score_delta = torch.randn(4, requires_grad=True)
    box_delta = torch.randn(4, 4, requires_grad=True)

    energy = transport_action_energy(
        feature_delta,
        score_delta=score_delta,
        box_delta=box_delta,
        score_weight=2.0,
        box_weight=0.5,
    )
    energy.backward()

    assert energy.ndim == 0
    assert energy.item() > 0.0
    assert feature_delta.grad is not None
    assert score_delta.grad is not None
    assert box_delta.grad is not None


def test_summarize_score_actions_reports_crossings_and_group_shifts() -> None:
    old_scores = torch.tensor([0.02, 0.04, 0.80, 0.03])
    score_delta = torch.tensor([0.01, 0.04, -0.10, 0.05])
    low_quality = torch.tensor([False, True, False, True])
    rescue_candidates = torch.tensor([True, True, False, True])

    summary = summarize_score_actions(
        old_scores,
        score_delta,
        threshold=0.05,
        low_quality_mask=low_quality,
        candidate_mask=rescue_candidates,
    )

    assert summary["num_candidates"] == 3
    assert summary["num_threshold_crossings"] == 2
    assert summary["num_low_quality_crossings"] == 2
    assert summary["mean_score_delta"] == torch.mean(score_delta).item()
    assert summary["mean_low_quality_delta"] == torch.mean(score_delta[low_quality]).item()
