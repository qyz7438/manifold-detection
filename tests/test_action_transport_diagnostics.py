from __future__ import annotations

import pytest
import torch

from spectral_detection_posttrain.eval.action_transport_diagnostics import (
    box_only_actions,
    oracle_accept_improving_box_actions,
    permute_box_actions_within_images,
    proposal_transition_tensors,
    summarize_proposal_transitions,
)
from spectral_detection_posttrain.methods.energy_transport import (
    ROIActionState,
    ROITransportActions,
)


def _actions(box_delta: torch.Tensor) -> ROITransportActions:
    count = box_delta.shape[0]
    return ROITransportActions(
        feature_delta=torch.ones((count, 3)),
        score_delta=torch.ones(count),
        box_delta=box_delta,
        keep_logit=torch.ones(count),
    )


def test_box_only_actions_scales_boxes_and_zeros_other_outputs() -> None:
    actions = _actions(torch.tensor([[0.2, -0.1, 0.0, 0.1]]))

    scaled = box_only_actions(actions, scale=0.25)

    assert torch.allclose(scaled.box_delta, torch.tensor([[0.05, -0.025, 0.0, 0.025]]))
    assert scaled.feature_delta.count_nonzero().item() == 0
    assert scaled.score_delta.count_nonzero().item() == 0
    assert scaled.keep_logit.count_nonzero().item() == 0


def test_permutation_never_crosses_image_boundaries() -> None:
    actions = _actions(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0, 0.0],
                [20.0, 0.0, 0.0, 0.0],
            ]
        )
    )
    image_indices = torch.tensor([0, 0, 1, 1])

    permuted = permute_box_actions_within_images(
        actions,
        image_indices,
        generator=torch.Generator().manual_seed(7),
    )

    assert sorted(permuted.box_delta[:2, 0].tolist()) == [1.0, 2.0]
    assert sorted(permuted.box_delta[2:, 0].tolist()) == [10.0, 20.0]
    assert permuted.score_delta.count_nonzero().item() == 0


def test_transition_summary_counts_ap75_promotions_and_demotions() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 30.0, 30.0],
        ]
    )
    targets = torch.tensor(
        [
            [1.0, 0.0, 11.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 30.0, 30.0],
        ]
    )
    state = ROIActionState(
        features=torch.zeros((3, 3)),
        boxes=boxes,
        scores=torch.tensor([0.8, 0.9, 0.6]),
        labels=torch.tensor([1, 1, 2]),
        image_indices=torch.tensor([0, 0, 0]),
        proposal_indices=torch.tensor([0, 1, 2]),
        logits=torch.tensor([[0.0, 2.0, 0.0], [0.0, 2.0, 0.0], [3.0, 0.0, 2.0]]),
        matched_gt_indices=torch.tensor([0, 1, 2]),
        ious=torch.tensor([0.8181818, 1.0, 1.0]),
    )
    # First box moves onto its target. The second moves away far enough to fall below 0.75.
    deltas = torch.tensor(
        [
            [0.1, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    transitions = proposal_transition_tensors(
        state,
        matched_gt_boxes=targets,
        matched_gt_labels=torch.tensor([1, 1, 2]),
        box_delta=deltas,
        image_sizes=[(40, 40)],
    )
    summary = summarize_proposal_transitions(transitions, threshold=0.75)

    assert summary["all"]["count"] == 3
    assert summary["all"]["demoted_75"] == 1
    assert summary["all"]["improved_count"] == 1
    assert summary["all"]["action_target_cosine_mean"] is not None
    assert summary["all"]["action_target_cosine_mean"] > 0.0
    assert summary["background_dominant"]["count"] == 1
    assert summary["class_correct"]["count"] == 3
    assert summary["iou_0.75_1.01"]["count"] == 3


def test_transition_summary_reports_promotion_from_below_threshold() -> None:
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    target = torch.tensor([[2.0, 0.0, 12.0, 10.0]])
    state = ROIActionState(
        features=torch.zeros((1, 2)),
        boxes=boxes,
        scores=torch.tensor([0.7]),
        labels=torch.tensor([1]),
        image_indices=torch.tensor([0]),
        proposal_indices=torch.tensor([0]),
        logits=torch.tensor([[0.0, 2.0]]),
        matched_gt_indices=torch.tensor([0]),
        ious=torch.tensor([2.0 / 3.0]),
    )

    transitions = proposal_transition_tensors(
        state,
        matched_gt_boxes=target,
        matched_gt_labels=torch.tensor([1]),
        box_delta=torch.tensor([[0.2, 0.0, 0.0, 0.0]]),
        image_sizes=[(20, 20)],
    )
    summary = summarize_proposal_transitions(transitions, threshold=0.75)

    assert summary["all"]["promoted_75"] == 1
    assert summary["all"]["post_iou_mean"] == 1.0
    assert summary["iou_0.5_0.75"]["count"] == 1


def test_oracle_acceptance_keeps_only_class_correct_iou_improvements() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
        ]
    )
    state = ROIActionState(
        features=torch.zeros((3, 2)),
        boxes=boxes,
        scores=torch.ones(3),
        labels=torch.tensor([1, 1, 2]),
        image_indices=torch.zeros(3, dtype=torch.long),
        proposal_indices=torch.arange(3),
        matched_gt_indices=torch.arange(3),
    )
    actions = _actions(
        torch.tensor(
            [
                [0.1, 0.0, 0.0, 0.0],
                [-0.1, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0, 0.0],
            ]
        )
    )

    accepted = oracle_accept_improving_box_actions(
        state,
        actions,
        matched_gt_boxes=torch.tensor(
            [
                [1.0, 0.0, 11.0, 10.0],
                [1.0, 0.0, 11.0, 10.0],
                [1.0, 0.0, 11.0, 10.0],
            ]
        ),
        matched_gt_labels=torch.tensor([1, 1, 1]),
        image_sizes=[(20, 20)],
    )

    assert accepted.box_delta[0, 0].item() == pytest.approx(0.1)
    assert accepted.box_delta[1:].count_nonzero().item() == 0
