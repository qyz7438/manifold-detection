from __future__ import annotations

import torch

from spectral_detection_posttrain.methods.energy_transport.global_top1 import GlobalTop1Output
from spectral_detection_posttrain.methods.energy_transport.post_nms_suppress import (
    PostNMSSuppressPolicyHead,
    build_post_nms_detection_features,
    select_post_nms_suppression,
    suppress_detection,
)


def test_post_nms_features_and_policy_are_permutation_equivariant() -> None:
    boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0], [3.0, 0.0, 4.0, 1.0]])
    scores = torch.tensor([0.9, 0.7, 0.5])
    labels = torch.tensor([1, 1, 2])
    features = build_post_nms_detection_features(boxes, scores, labels, (4, 4), num_classes=3)
    policy = PostNMSSuppressPolicyHead(features.shape[1], hidden_dim=8)
    torch.nn.init.normal_(policy.action_head[-1].weight, std=0.1)
    visible = torch.ones(3, dtype=torch.bool)
    original = policy(features, visible)
    permutation = torch.tensor([2, 0, 1])
    permuted_features = build_post_nms_detection_features(
        boxes[permutation], scores[permutation], labels[permutation], (4, 4), num_classes=3
    )
    permuted = policy(permuted_features, visible[permutation])
    assert torch.allclose(permuted.action_logits, original.action_logits[permutation], atol=1e-6)
    assert torch.allclose(permuted.noop_logit, original.noop_logit, atol=1e-6)


def test_post_nms_selection_and_suppression_remove_exactly_one_detection() -> None:
    output = GlobalTop1Output(
        action_logits=torch.tensor([[0.0, 0.2], [0.0, 1.2], [0.0, 0.8]]),
        noop_logit=torch.tensor(0.5),
        conflict_stats=torch.zeros(3, 4),
    )
    selection = select_post_nms_suppression(output, torch.ones(3, dtype=torch.bool))
    assert not selection.is_noop and selection.detection_index == 1
    prediction = {
        "boxes": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "scores": torch.tensor([0.9, 0.8, 0.7]),
        "labels": torch.tensor([1, 2, 3]),
    }
    suppressed = suppress_detection(prediction, selection.detection_index)
    assert suppressed["boxes"].shape == (2, 4)
    assert torch.equal(suppressed["scores"], torch.tensor([0.9, 0.7]))
    assert torch.equal(suppressed["labels"], torch.tensor([1, 3]))


def test_noop_wins_ties_and_empty_sets() -> None:
    tied = GlobalTop1Output(torch.tensor([[0.0, 0.5]]), torch.tensor(0.5), torch.zeros(1, 4))
    assert select_post_nms_suppression(tied, torch.tensor([True])).is_noop
    empty = GlobalTop1Output(torch.empty(0, 2), torch.tensor(0.0), torch.empty(0, 4))
    assert select_post_nms_suppression(empty, torch.empty(0, dtype=torch.bool)).is_noop
