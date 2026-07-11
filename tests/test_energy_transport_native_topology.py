from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torchvision.ops import batched_nms, box_iou, clip_boxes_to_image, remove_small_boxes

from spectral_detection_posttrain.methods.energy_transport.native_topology import (
    NATIVE_TOPOLOGY_FEATURE_NAMES,
    native_action_nms_topology,
)


def _manual_kept_indices(
    boxes: torch.Tensor,
    probabilities: torch.Tensor,
    image_size: tuple[int, int],
    score_threshold: float,
    nms_threshold: float,
    detections_per_img: int,
) -> set[tuple[int, int]]:
    """Return proposal/class pairs from torchvision's class-expanded postprocess."""
    count, classes = probabilities.shape
    clipped = clip_boxes_to_image(boxes, image_size)
    proposal_ids = torch.arange(count).view(-1, 1).expand(count, classes)[:, 1:].reshape(-1)
    labels = torch.arange(classes).view(1, -1).expand(count, classes)[:, 1:].reshape(-1)
    flat_boxes = clipped[:, 1:].reshape(-1, 4)
    scores = probabilities[:, 1:].reshape(-1)
    keep = torch.where(scores > score_threshold)[0]
    flat_boxes, scores, labels, proposal_ids = (
        flat_boxes[keep],
        scores[keep],
        labels[keep],
        proposal_ids[keep],
    )
    keep = remove_small_boxes(flat_boxes, min_size=1e-2)
    flat_boxes, scores, labels, proposal_ids = (
        flat_boxes[keep],
        scores[keep],
        labels[keep],
        proposal_ids[keep],
    )
    keep = batched_nms(flat_boxes, scores, labels, nms_threshold)[:detections_per_img]
    return set(zip(proposal_ids[keep].tolist(), labels[keep].tolist()))


def _feature_index(name: str) -> int:
    return NATIVE_TOPOLOGY_FEATURE_NAMES.index(name)


def test_identity_action_matches_manual_native_nms_kept_results() -> None:
    boxes = torch.tensor(
        [
            [[0.0, 0.0, 4.0, 4.0], [0.0, 0.0, 4.0, 4.0], [7.0, 7.0, 9.0, 9.0]],
            [[0.0, 0.0, 4.0, 4.0], [0.5, 0.5, 4.5, 4.5], [6.0, 6.0, 9.0, 9.0]],
            [[5.0, 0.0, 8.0, 3.0], [5.0, 0.0, 8.0, 3.0], [5.0, 0.0, 8.0, 3.0]],
        ]
    )
    probabilities = torch.tensor(
        [[0.01, 0.90, 0.12], [0.01, 0.80, 0.95], [0.01, 0.70, 0.20]]
    )
    labels = torch.tensor([1, 1, 1])
    expected = _manual_kept_indices(boxes, probabilities, (10, 10), 0.05, 0.5, 10)

    topology = native_action_nms_topology(
        boxes,
        probabilities,
        labels,
        image_size=(10, 10),
        candidate_deltas=torch.zeros((1, 4)),
        observable_mask=torch.ones(3, dtype=torch.bool),
        score_threshold=0.05,
        nms_threshold=0.5,
        detections_per_img=10,
    )

    assert topology.shape == (3, 1, len(NATIVE_TOPOLOGY_FEATURE_NAMES))
    kept = _feature_index("acted_candidate_kept")
    identity = _feature_index("identity_kept")
    for proposal in range(3):
        manual = float((proposal, int(labels[proposal])) in expected)
        assert topology[proposal, 0, kept].item() == manual
        assert topology[proposal, 0, identity].item() == manual


def test_class_expansion_counts_competing_non_top1_classes() -> None:
    boxes = torch.tensor(
        [
            [[0.0, 0.0, 4.0, 4.0]] * 3,
            [[0.0, 0.0, 4.0, 4.0]] * 3,
        ]
    )
    probabilities = torch.tensor(
        [[0.01, 0.90, 0.80], [0.01, 0.85, 0.95]]
    )
    topology = native_action_nms_topology(
        boxes,
        probabilities,
        action_labels=torch.tensor([2, 1]),
        image_size=(8, 8),
        candidate_deltas=torch.zeros((1, 4)),
        observable_mask=torch.tensor([True, True]),
        score_threshold=0.05,
        nms_threshold=0.5,
        detections_per_img=10,
    )

    assert topology[0, 0, _feature_index("acted_candidate_kept")].item() == 0.0
    assert topology[1, 0, _feature_index("acted_candidate_kept")].item() == 0.0
    assert topology[0, 0, _feature_index("max_iou_higher_score_same_class")].item() == pytest.approx(1.0)
    assert topology[1, 0, _feature_index("max_iou_higher_score_same_class")].item() == pytest.approx(1.0)


def test_identity_candidate_must_be_exact_zero_delta() -> None:
    boxes = torch.tensor([[[0.0, 0.0, 2.0, 2.0]] * 2])
    probabilities = torch.tensor([[0.01, 0.9]])

    with pytest.raises(ValueError, match="candidate_deltas\\[0\\]"):
        native_action_nms_topology(
            boxes,
            probabilities,
            action_labels=torch.tensor([1]),
            image_size=(4, 4),
            candidate_deltas=torch.tensor([[0.01, 0.0, 0.0, 0.0]]),
            observable_mask=torch.tensor([True]),
            score_threshold=0.05,
            nms_threshold=0.5,
            detections_per_img=10,
        )


def test_exact_zero_identity_reuses_baseline_without_box_transform() -> None:
    boxes = torch.tensor([[[0.0, 0.0, 2.0, 2.0]] * 2])
    probabilities = torch.tensor([[0.01, 0.90]])
    with patch(
        "spectral_detection_posttrain.methods.energy_transport.native_topology.apply_box_delta",
        side_effect=AssertionError("identity must not transform boxes"),
    ):
        topology = native_action_nms_topology(
            boxes,
            probabilities,
            action_labels=torch.tensor([1]),
            image_size=(4, 4),
            candidate_deltas=torch.zeros((1, 4)),
            observable_mask=torch.tensor([True]),
            score_threshold=0.05,
            nms_threshold=0.5,
            detections_per_img=10,
        )
    assert topology[0, 0, _feature_index("kept_delta")].item() == 0.0
    assert topology[0, 0, _feature_index("max_iou_delta")].item() == 0.0
    assert topology[0, 0, _feature_index("kept_rank_delta")].item() == 0.0


def test_actions_clip_boxes_and_remove_degenerate_candidates() -> None:
    boxes = torch.tensor([[[0.0, 0.0, 2.0, 2.0]] * 2])
    probabilities = torch.tensor([[0.01, 0.90]])
    topology = native_action_nms_topology(
        boxes,
        probabilities,
        action_labels=torch.tensor([1]),
        image_size=(4, 4),
        candidate_deltas=torch.tensor([[0.0, 0.0, 0.0, 0.0], [-10.0, 0.0, 0.0, 0.0]]),
        observable_mask=torch.tensor([True]),
        score_threshold=0.05,
        nms_threshold=0.5,
        detections_per_img=10,
    )

    assert topology[0, 0, _feature_index("acted_candidate_kept")].item() == 1.0
    assert topology[0, 1, _feature_index("acted_candidate_kept")].item() == 0.0
    assert topology[0, 1, _feature_index("kept_delta")].item() == pytest.approx(-1.0)


def test_topology_is_proposal_permutation_equivariant() -> None:
    boxes = torch.tensor(
        [
            [[0.0, 0.0, 4.0, 4.0]] * 3,
            [[1.0, 0.0, 5.0, 4.0]] * 3,
            [[5.0, 5.0, 8.0, 8.0]] * 3,
        ]
    )
    probabilities = torch.tensor(
        [[0.01, 0.95, 0.20], [0.01, 0.80, 0.90], [0.01, 0.60, 0.70]]
    )
    labels = torch.tensor([1, 2, 2])
    observable = torch.tensor([True, False, True])
    deltas = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.10, 0.0, 0.0, 0.0]])
    original = native_action_nms_topology(
        boxes, probabilities, labels, (10, 10), deltas, observable,
        score_threshold=0.05, nms_threshold=0.5, detections_per_img=10,
    )
    permutation = torch.tensor([2, 0, 1])
    permuted = native_action_nms_topology(
        boxes[permutation], probabilities[permutation], labels[permutation], (10, 10),
        deltas, observable[permutation],
        score_threshold=0.05, nms_threshold=0.5, detections_per_img=10,
    )

    assert torch.allclose(permuted, original[permutation])
    assert torch.equal(original[1], torch.zeros_like(original[1]))
