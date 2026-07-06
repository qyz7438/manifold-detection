from __future__ import annotations

import pytest
import torch

from spectral_detection_posttrain.trainers.detection.roi_state import (
    build_roi_action_state_from_predictions,
    extract_proposal_roi_action_state,
)


def test_build_roi_action_state_concatenates_predictions_and_indices() -> None:
    predictions = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
            "scores": torch.tensor([0.9, 0.2]),
            "labels": torch.tensor([1, 2]),
        },
        {
            "boxes": torch.tensor([[5.0, 5.0, 12.0, 12.0]]),
            "scores": torch.tensor([0.4]),
            "labels": torch.tensor([1]),
        },
    ]
    features = torch.arange(12, dtype=torch.float32).view(3, 4)

    state = build_roi_action_state_from_predictions(predictions, features)

    assert state.batch_size == 3
    assert state.feature_dim == 4
    assert state.image_indices.tolist() == [0, 0, 1]
    assert state.proposal_indices.tolist() == [0, 1, 0]
    assert torch.equal(state.features, features)


def test_build_roi_action_state_adds_class_aware_target_matches() -> None:
    predictions = [
        {
            "boxes": torch.tensor([
                [0.0, 0.0, 10.0, 10.0],
                [0.0, 0.0, 10.0, 10.0],
                [30.0, 30.0, 40.0, 40.0],
            ]),
            "scores": torch.tensor([0.9, 0.8, 0.7]),
            "labels": torch.tensor([1, 2, 1]),
        }
    ]
    targets = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "labels": torch.tensor([1]),
        }
    ]
    features = torch.randn(3, 4)

    state = build_roi_action_state_from_predictions(predictions, features, targets)

    assert state.matched_gt_indices is not None
    assert state.ious is not None
    assert state.matched_gt_indices.tolist() == [0, -1, -1]
    assert state.ious.tolist() == [pytest.approx(1.0), pytest.approx(0.0), pytest.approx(0.0)]


def test_build_roi_action_state_accepts_logits_and_verifier_scores() -> None:
    predictions = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            "scores": torch.tensor([0.2]),
            "labels": torch.tensor([1]),
        }
    ]
    features = torch.randn(1, 3)
    logits = torch.randn(1, 5)
    verifier_scores = torch.tensor([0.7])

    state = build_roi_action_state_from_predictions(
        predictions,
        features,
        logits=logits,
        verifier_scores=verifier_scores,
    )

    assert torch.equal(state.logits, logits)
    assert torch.equal(state.verifier_scores, verifier_scores)


def test_build_roi_action_state_returns_empty_state_for_no_predictions() -> None:
    predictions = [
        {
            "boxes": torch.empty(0, 4),
            "scores": torch.empty(0),
            "labels": torch.empty(0, dtype=torch.long),
        }
    ]
    features = torch.empty(0, 6)

    state = build_roi_action_state_from_predictions(predictions, features)

    assert state.batch_size == 0
    assert state.feature_dim == 6
    assert state.boxes.shape == (0, 4)


def test_build_roi_action_state_rejects_misaligned_feature_count() -> None:
    predictions = [
        {
            "boxes": torch.zeros(2, 4),
            "scores": torch.ones(2),
            "labels": torch.ones(2, dtype=torch.long),
        }
    ]

    with pytest.raises(ValueError, match="roi_features"):
        build_roi_action_state_from_predictions(predictions, torch.zeros(1, 4))


def test_extract_proposal_roi_action_state_uses_model_roi_pipeline() -> None:
    class ImageList:
        def __init__(self, tensors, image_sizes):
            self.tensors = tensors
            self.image_sizes = image_sizes

    class FakeTransform:
        def __call__(self, images, targets=None):
            batch = torch.stack(images, dim=0)
            return ImageList(batch, [tuple(img.shape[-2:]) for img in images]), targets

    class FakeBackbone(torch.nn.Module):
        def forward(self, tensors):
            return {"0": tensors}

    class FakeRPN(torch.nn.Module):
        def forward(self, images, features, targets=None):
            return [
                torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
                torch.tensor([[1.0, 1.0, 5.0, 5.0]]),
            ], {}

    class FakeBoxROIPool(torch.nn.Module):
        def forward(self, features, proposals, image_sizes):
            count = sum(len(p) for p in proposals)
            return torch.ones(count, 2)

    class FakeBoxHead(torch.nn.Module):
        def forward(self, pooled):
            return torch.arange(pooled.shape[0] * 4, dtype=torch.float32).view(-1, 4)

    class FakeBoxPredictor(torch.nn.Module):
        def forward(self, features):
            logits = torch.tensor([
                [0.0, 3.0, 1.0],
                [0.0, 0.5, 2.0],
                [0.0, 4.0, 1.0],
            ])
            bbox = torch.zeros(features.shape[0], 12)
            return logits, bbox

    class FakeROIHeads:
        def __init__(self):
            self.box_roi_pool = FakeBoxROIPool()
            self.box_head = FakeBoxHead()
            self.box_predictor = FakeBoxPredictor()

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transform = FakeTransform()
            self.backbone = FakeBackbone()
            self.rpn = FakeRPN()
            self.roi_heads = FakeROIHeads()

    targets = [
        {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([1])},
        {"boxes": torch.tensor([[1.0, 1.0, 5.0, 5.0]]), "labels": torch.tensor([1])},
    ]

    state = extract_proposal_roi_action_state(
        FakeModel(),
        [torch.zeros(3, 32, 32), torch.zeros(3, 32, 32)],
        targets=targets,
    )

    assert state.batch_size == 3
    assert state.feature_dim == 4
    assert state.labels.tolist() == [1, 2, 1]
    assert state.image_indices.tolist() == [0, 0, 1]
    assert state.proposal_indices.tolist() == [0, 1, 0]
    assert state.logits is not None
    assert state.matched_gt_indices is not None
    assert state.matched_gt_indices.tolist() == [0, -1, 0]
