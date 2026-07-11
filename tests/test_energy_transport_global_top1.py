from __future__ import annotations

import pytest
import torch

from spectral_detection_posttrain.methods.energy_transport.global_top1 import (
    GlobalTop1Output,
    GlobalTop1PolicyHead,
    GlobalTop1Target,
    build_global_top1_target,
    flatten_observable_action_logits,
    global_top1_loss,
    select_global_top1_action,
)
from spectral_detection_posttrain.methods.energy_transport.native_contract import build_native_c1_deltas


def test_flattened_logits_are_global_noop_then_observable_proposal_actions() -> None:
    output = GlobalTop1Output(
        action_logits=torch.arange(27, dtype=torch.float32).reshape(3, 9),
        noop_logit=torch.tensor(100.0),
        conflict_stats=torch.zeros(3, 4),
    )
    flattened = flatten_observable_action_logits(
        output,
        observable_mask=torch.tensor([True, False, True]),
    )

    assert flattened.logits.shape == (17,)
    assert flattened.logits[0].item() == 100.0
    assert flattened.proposal_indices.tolist() == [-1] + [0] * 8 + [2] * 8
    assert flattened.candidate_indices.tolist() == [0] + list(range(1, 9)) * 2


def test_global_target_is_noop_unless_best_postprocess_delta_is_positive() -> None:
    delta_u = torch.tensor(
        [
            [0.0, -0.1, 0.2, 0.1],
            [0.0, 0.3, -0.2, 0.0],
        ]
    )
    target = build_global_top1_target(
        delta_u,
        observable_mask=torch.tensor([True, True]),
        min_delta_u=0.0,
    )
    assert target.is_noop is False
    assert target.proposal_index == 1
    assert target.candidate_index == 1
    assert target.delta_u == pytest.approx(0.3)

    noop = build_global_top1_target(
        delta_u.new_tensor([[0.0, -0.1, 0.0, -0.2]]),
        observable_mask=torch.tensor([True]),
        min_delta_u=0.0,
    )
    assert noop.is_noop is True
    assert noop.proposal_index == -1
    assert noop.candidate_index == 0


def test_global_selection_materializes_at_most_one_c1_action() -> None:
    deltas = build_native_c1_deltas(0.05)
    output = GlobalTop1Output(
        action_logits=torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        noop_logit=torch.tensor(0.5),
        conflict_stats=torch.zeros(2, 4),
    )
    selected = select_global_top1_action(
        output,
        deltas,
        observable_mask=torch.tensor([True, True]),
    )

    assert selected.is_noop is False
    assert selected.proposal_index == 1
    assert selected.candidate_index == 2
    assert selected.box_delta.shape == (2, 4)
    assert selected.box_delta.count_nonzero().item() == 1
    assert selected.box_delta[1, 0].item() == pytest.approx(-0.05)

    no_action = select_global_top1_action(
        GlobalTop1Output(output.action_logits, torch.tensor(3.0), output.conflict_stats),
        deltas,
        observable_mask=torch.tensor([True, True]),
    )
    assert no_action.is_noop is True
    assert no_action.box_delta.count_nonzero().item() == 0


def test_unified_image_level_loss_backpropagates_to_action_and_noop_scores() -> None:
    deltas = build_native_c1_deltas(0.05)
    policy = GlobalTop1PolicyHead(
        in_channels=2,
        num_classes=3,
        candidate_deltas=deltas,
        hidden_dim=8,
        spatial_size=2,
    )
    observable = torch.tensor([True, True])
    output = policy(
        spatial_features=torch.randn(2, 2, 3, 3),
        class_logits=torch.tensor([[0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]),
        labels=torch.tensor([1, 2]),
        scores=torch.tensor([0.8, 0.7]),
        boxes=torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]]),
        image_size=(4, 4),
        observable_mask=observable,
    )
    loss = global_top1_loss(
        output,
        observable,
        GlobalTop1Target(False, proposal_index=1, candidate_index=2, delta_u=0.5),
    )["loss_total"]
    loss.backward()

    assert torch.isfinite(loss)
    assert policy.noop_bias.grad is not None
    assert policy.proposal_policy.action_head.weight.grad is not None
    assert policy.proposal_policy.move_head.weight.grad is None
