from __future__ import annotations

import pytest
import torch

from spectral_detection_posttrain.methods.energy_transport import (
    NMSAwareSetPolicyHead,
    SetPolicyLossConfig,
    SetPolicyOutput,
    build_symmetric_box_candidates,
    class_aware_conflict_statistics,
    select_set_policy_actions,
    set_policy_loss,
)


def _candidates() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0],
            [0.0, -0.1, 0.0, 0.0],
        ]
    )


def _head() -> NMSAwareSetPolicyHead:
    return NMSAwareSetPolicyHead(
        in_channels=4,
        num_classes=3,
        candidate_deltas=_candidates(),
        hidden_dim=8,
        spatial_size=3,
    )


def _head_inputs(count: int = 3) -> tuple[torch.Tensor, ...]:
    return (
        torch.randn(count, 4, 3, 3),
        torch.randn(count, 3),
        torch.tensor([1, 1, 2])[:count],
        torch.tensor([0.7, 0.9, 0.5])[:count],
        torch.tensor(
            [[0.0, 0.0, 10.0, 10.0], [1.0, 0.0, 11.0, 10.0], [0.0, 0.0, 10.0, 10.0]]
        )[:count],
    )


def test_head_returns_set_policy_shapes_from_observable_detector_inputs() -> None:
    head = _head()
    features, logits, labels, scores, boxes = _head_inputs()

    output = head(features, logits, labels, scores, boxes, image_size=(20, 20))

    assert output.action_logits.shape == (3, 3)
    assert output.move_logits.shape == (3,)
    assert output.conflict_stats.shape == (3, 4)


def test_head_requires_an_identity_candidate_at_index_zero() -> None:
    with pytest.raises(ValueError, match="identity"):
        NMSAwareSetPolicyHead(
            in_channels=4,
            num_classes=3,
            candidate_deltas=torch.tensor([[0.1, 0.0, 0.0, 0.0]]),
        )


def test_energy_penalty_exactly_offsets_non_identity_action_logits() -> None:
    head = _head()
    features, logits, labels, scores, boxes = _head_inputs()

    no_energy = head(features, logits, labels, scores, boxes, image_size=(20, 20), energy_weight=0.0)
    weighted = head(features, logits, labels, scores, boxes, image_size=(20, 20), energy_weight=0.2)
    expected = 0.2 * _candidates().square().sum(dim=1) / 0.04

    assert torch.allclose(weighted.action_logits - no_energy.action_logits, -expected.expand_as(weighted.action_logits))
    assert torch.allclose(weighted.action_logits[:, 0], no_energy.action_logits[:, 0])


def test_conflict_statistics_include_only_same_class_peers_and_handle_single_box() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 0.0, 11.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
        ]
    )
    stats = class_aware_conflict_statistics(
        boxes,
        labels=torch.tensor([1, 1, 2]),
        scores=torch.tensor([0.7, 0.9, 0.99]),
        image_size=(20, 20),
        nms_threshold=0.5,
    )

    assert stats[0, 0].item() == pytest.approx(9.0 / 11.0)
    assert stats[0, 1].item() == pytest.approx(9.0 / 11.0)
    assert stats[0, 2].item() == pytest.approx(1.0)
    assert stats[0, 3].item() == pytest.approx(9.0 / 11.0)
    assert stats[2].count_nonzero().item() == 0

    one = class_aware_conflict_statistics(
        boxes[:1], labels=torch.tensor([1]), scores=torch.tensor([0.7]), image_size=(20, 20)
    )
    assert one.shape == (1, 4)
    assert one.count_nonzero().item() == 0


def test_set_policy_loss_is_finite_and_backpropagates() -> None:
    action_logits = torch.tensor([[0.1, 1.0, -0.5], [1.0, 0.3, 0.2]], requires_grad=True)
    move_logits = torch.tensor([1.2, -0.4], requires_grad=True)
    output = SetPolicyOutput(action_logits, move_logits, torch.zeros(2, 4))

    losses = set_policy_loss(
        output,
        action_targets=torch.tensor([1, 0]),
        move_targets=torch.tensor([1.0, 0.0]),
        supervision_mask=torch.tensor([True, True]),
        config=SetPolicyLossConfig(identity_weight=0.25, move_positive_weight=2.0),
    )
    losses["loss_total"].backward()

    assert torch.isfinite(losses["loss_total"])
    assert action_logits.grad is not None
    assert move_logits.grad is not None
    assert "action_accuracy" in losses
    assert "move_accuracy" in losses


def test_set_policy_loss_rejects_identity_action_for_selected_positive() -> None:
    output = SetPolicyOutput(torch.zeros(1, 2), torch.zeros(1), torch.zeros(1, 4))

    with pytest.raises(ValueError, match="non-identity"):
        set_policy_loss(
            output,
            action_targets=torch.tensor([0]),
            move_targets=torch.tensor([1.0]),
            supervision_mask=torch.tensor([True]),
            config=SetPolicyLossConfig(),
        )


def test_selection_stops_everything_when_move_gate_does_not_pass() -> None:
    output = SetPolicyOutput(
        action_logits=torch.tensor([[0.0, 8.0, 1.0], [0.0, 7.0, 1.0]]),
        move_logits=torch.tensor([0.0, -0.5]),
        conflict_stats=torch.zeros(2, 4),
    )

    selected = select_set_policy_actions(
        output, _candidates(), torch.tensor([0, 0]), torch.tensor([True, True])
    )

    assert not selected.selected_mask.any()
    assert selected.candidate_indices.tolist() == [0, 0]
    assert selected.box_delta.count_nonzero().item() == 0


def test_selection_respects_per_image_budget_and_skips_unobservable_proposals() -> None:
    output = SetPolicyOutput(
        action_logits=torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 4.0, 0.0], [0.0, 3.0, 0.0], [0.0, 2.0, 0.0], [0.0, 5.0, 0.0], [0.0, 9.0, 0.0]]
        ),
        move_logits=torch.ones(6),
        conflict_stats=torch.zeros(6, 4),
    )

    selected = select_set_policy_actions(
        output,
        _candidates(),
        image_indices=torch.tensor([0, 0, 0, 0, 0, 1]),
        observable_mask=torch.tensor([True, True, True, True, True, False]),
        max_actions_per_image=4,
    )

    assert selected.selected_mask.tolist() == [False, True, True, True, True, False]
    assert selected.candidate_indices.tolist() == [0, 1, 1, 1, 1, 0]
    assert selected.box_delta[0].count_nonzero().item() == 0
    assert selected.box_delta[-1].count_nonzero().item() == 0


def test_fresh_policy_defaults_to_identity_for_every_proposal() -> None:
    torch.manual_seed(3)
    candidates = build_symmetric_box_candidates((0.05, 0.1, 0.2))
    head = NMSAwareSetPolicyHead(
        in_channels=4,
        num_classes=3,
        candidate_deltas=candidates,
        hidden_dim=8,
        spatial_size=2,
    )
    count = 6
    output = head(
        spatial_features=torch.randn(count, 4, 2, 2),
        class_logits=torch.randn(count, 3),
        labels=torch.tensor([1, 1, 1, 2, 2, 2]),
        scores=torch.rand(count),
        boxes=torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [1.0, 1.0, 11.0, 11.0],
                [20.0, 20.0, 30.0, 30.0],
                [0.0, 0.0, 8.0, 8.0],
                [10.0, 10.0, 20.0, 20.0],
                [22.0, 22.0, 30.0, 30.0],
            ]
        ),
        image_size=(32, 32),
    )
    selected = select_set_policy_actions(
        output,
        candidates,
        image_indices=torch.zeros(count, dtype=torch.long),
        observable_mask=torch.ones(count, dtype=torch.bool),
    )

    assert not selected.selected_mask.any()
    assert selected.box_delta.count_nonzero().item() == 0
