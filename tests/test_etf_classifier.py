"""Unit tests for the Equiangular Tight Frame (ETF) classifier."""

from __future__ import annotations

import math

import torch

from spectral_detection_posttrain.core.models.etf_predictor import (
    ETFClassifier,
    build_etf_weight,
    replace_cls_score_with_etf,
)


def test_etf_weight_shape():
    W = build_etf_weight(num_classes=5, feature_dim=8)
    assert W.shape == (5, 8)


def test_etf_weight_unit_norm():
    W = build_etf_weight(num_classes=5, feature_dim=10)
    norms = W.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_etf_weight_pairwise_angles():
    num_classes = 5
    W = build_etf_weight(num_classes=num_classes, feature_dim=10)
    cos_sim = W @ W.T
    expected = -1.0 / (num_classes - 1.0)
    # Mask out diagonal.
    off_diag = ~torch.eye(num_classes, dtype=torch.bool)
    assert torch.allclose(cos_sim[off_diag], torch.full((num_classes * (num_classes - 1),), expected), atol=1e-5)


def test_etf_classifier_forward():
    layer = ETFClassifier(feature_dim=8, num_classes=3)
    features = torch.randn(4, 8)
    logits = layer(features)
    assert logits.shape == (4, 3)


def test_etf_classifier_weight_is_frozen():
    layer = ETFClassifier(feature_dim=8, num_classes=3)
    assert "weight" in layer._buffers
    assert not layer.weight.requires_grad


def test_replace_cls_score_with_etf():
    class DummyBoxPredictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cls_score = torch.nn.Linear(8, 3)
            self.bbox_pred = torch.nn.Linear(8, 12)

    predictor = DummyBoxPredictor()
    replace_cls_score_with_etf(predictor, num_classes=3)
    assert isinstance(predictor.cls_score, ETFClassifier)
    assert predictor.cls_score.weight.shape == (3, 8)
    # bbox_pred should be untouched.
    assert isinstance(predictor.bbox_pred, torch.nn.Linear)


def test_etf_classifier_raises_on_small_dim():
    try:
        build_etf_weight(num_classes=5, feature_dim=4)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for feature_dim < num_classes")
