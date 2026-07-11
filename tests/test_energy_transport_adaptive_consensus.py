from __future__ import annotations

import torch

from spectral_detection_posttrain.methods.energy_transport.adaptive_consensus import (
    proposal_graph_consensus_deltas,
)
from spectral_detection_posttrain.methods.energy_transport.operators import apply_box_delta


def test_consensus_moves_weaker_overlap_toward_stronger_same_class_peer() -> None:
    boxes = torch.tensor([[0.0, 0.0, 4.0, 4.0], [1.0, 0.0, 5.0, 4.0]])
    deltas = proposal_graph_consensus_deltas(
        boxes,
        labels=torch.tensor([1, 1]),
        scores=torch.tensor([0.5, 0.9]),
        observable_mask=torch.tensor([True, True]),
        min_peer_iou=0.3,
        max_abs_delta=0.05,
    )
    moved = apply_box_delta(boxes[:1], deltas[:1])
    assert deltas[0, 0] > 0.0
    assert torch.equal(deltas[1], torch.zeros(4))
    assert (moved - boxes[1:2]).abs().sum() < (boxes[:1] - boxes[1:2]).abs().sum()


def test_consensus_excludes_other_classes_low_overlap_and_unobservable_rows() -> None:
    boxes = torch.tensor(
        [[0.0, 0.0, 4.0, 4.0], [1.0, 0.0, 5.0, 4.0], [20.0, 20.0, 24.0, 24.0]]
    )
    deltas = proposal_graph_consensus_deltas(
        boxes,
        labels=torch.tensor([1, 2, 1]),
        scores=torch.tensor([0.5, 0.9, 0.95]),
        observable_mask=torch.tensor([True, True, False]),
    )
    assert torch.equal(deltas, torch.zeros_like(deltas))


def test_consensus_delta_is_bounded_and_permutation_equivariant() -> None:
    boxes = torch.tensor(
        [[0.0, 0.0, 4.0, 4.0], [1.0, 0.0, 6.0, 5.0], [0.5, 0.0, 5.0, 4.5]]
    )
    labels = torch.tensor([1, 1, 1])
    scores = torch.tensor([0.4, 0.9, 0.7])
    visible = torch.ones(3, dtype=torch.bool)
    original = proposal_graph_consensus_deltas(boxes, labels, scores, visible, max_abs_delta=0.05)
    permutation = torch.tensor([2, 0, 1])
    permuted = proposal_graph_consensus_deltas(
        boxes[permutation], labels[permutation], scores[permutation], visible[permutation], max_abs_delta=0.05
    )
    assert original.abs().amax() <= 0.05 + 1e-7
    assert torch.allclose(permuted, original[permutation], atol=1e-7)
