from __future__ import annotations

import torch

from spectral_detection_posttrain.methods.energy_transport import (
    ActionBenefitEnergyHead,
    BenefitEnergyLossConfig,
    ROIActionState,
    ROITransportActions,
    action_benefit_energy_loss,
    apply_action_benefit_gate,
    build_action_benefit_targets,
)


def _state() -> ROIActionState:
    return ROIActionState(
        features=torch.randn(4, 6),
        boxes=torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [0.0, 0.0, 10.0, 10.0],
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 30.0, 30.0],
            ]
        ),
        scores=torch.tensor([0.9, 0.8, 0.04, 0.7]),
        labels=torch.tensor([1, 1, 1, 2]),
        image_indices=torch.tensor([0, 0, 0, 1]),
        proposal_indices=torch.tensor([0, 1, 2, 0]),
        logits=torch.tensor(
            [
                [0.0, 3.0, 0.0],
                [4.0, 3.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        ),
        matched_gt_indices=torch.tensor([0, 1, 2, 0]),
    )


def _actions(box_delta: torch.Tensor) -> ROITransportActions:
    count = box_delta.shape[0]
    return ROITransportActions(
        feature_delta=torch.zeros((count, 6)),
        score_delta=torch.zeros(count),
        box_delta=box_delta,
        keep_logit=torch.zeros(count),
    )


def test_energy_gap_is_exact_zero_for_identity_action() -> None:
    state = _state()
    head = ActionBenefitEnergyHead(feature_dim=6, num_classes=3, hidden_dim=8)
    zero = torch.zeros((state.batch_size, 4))

    gap = head.energy_gap(
        state.features,
        state.logits,
        state.labels,
        state.scores,
        zero,
    )

    assert gap.shape == (state.batch_size,)
    assert torch.equal(gap, torch.zeros_like(gap))


def test_pairwise_loss_prefers_gain_ordering_that_matches_true_iou_gain() -> None:
    true_gain = torch.tensor([0.08, 0.03, -0.04, -0.10])
    base_iou = torch.tensor([0.65, 0.72, 0.80, 0.40])
    class_correct = torch.tensor([True, True, True, False])
    foreground = torch.tensor([True, True, True, True])
    scores = torch.tensor([0.7, 0.8, 0.9, 0.6])
    cfg = BenefitEnergyLossConfig(min_positive_gain=0.005, temperature=0.05)

    aligned = action_benefit_energy_loss(
        torch.tensor([0.08, 0.03, -0.04, -0.10]),
        true_gain=true_gain,
        base_iou=base_iou,
        class_correct=class_correct,
        foreground_dominant=foreground,
        scores=scores,
        config=cfg,
    )
    inverted = action_benefit_energy_loss(
        torch.tensor([-0.08, -0.03, 0.04, 0.10]),
        true_gain=true_gain,
        base_iou=base_iou,
        class_correct=class_correct,
        foreground_dominant=foreground,
        scores=scores,
        config=cfg,
    )

    assert aligned["loss_total"].item() < inverted["loss_total"].item()


def test_gate_keeps_identity_for_background_low_score_and_non_topk_rows() -> None:
    state = _state()
    actions = _actions(torch.ones((4, 4)) * 0.1)

    gated, mask = apply_action_benefit_gate(
        state,
        actions,
        predicted_gain=torch.tensor([0.4, 0.3, 0.2, 0.1]),
        min_predicted_gain=0.0,
        min_score=0.05,
        require_foreground_dominant=True,
        max_actions_per_image=1,
    )

    assert mask.tolist() == [True, False, False, True]
    assert gated.box_delta[0].abs().sum().item() > 0.0
    assert gated.box_delta[1:3].count_nonzero().item() == 0


def test_benefit_targets_measure_gain_and_class_correctness() -> None:
    state = _state()
    actions = _actions(
        torch.tensor(
            [
                [0.1, 0.0, 0.0, 0.0],
                [-0.1, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
    )
    targets = build_action_benefit_targets(
        state,
        actions,
        matched_gt_boxes=torch.tensor(
            [
                [1.0, 0.0, 11.0, 10.0],
                [1.0, 0.0, 11.0, 10.0],
                [1.0, 0.0, 11.0, 10.0],
                [20.0, 20.0, 30.0, 30.0],
            ]
        ),
        matched_gt_labels=torch.tensor([1, 1, 2, 2]),
        image_sizes=[(20, 20), (40, 40)],
    )

    assert targets.class_correct.tolist() == [True, True, False, True]
    assert targets.true_gain[0].item() > 0.0
    assert targets.true_gain[1].item() < 0.0
    assert targets.foreground_dominant.tolist() == [True, False, True, True]
