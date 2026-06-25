import math

import torch

from spectral_detection_posttrain.core.models.etf_predictor import (
    ETFClassifier,
    build_etf_weight,
)


def test_build_etf_weight_shape_and_geometry():
    """Foreground ETF rows must form a simplex ETF."""
    W = build_etf_weight(num_foreground_classes=10, feature_dim=1024)
    assert W.shape == (10, 1024)

    cos = W @ W.T
    expected_off_diag = -1.0 / 9.0
    off_diag_mask = ~torch.eye(10, dtype=torch.bool)
    assert torch.allclose(cos[off_diag_mask], torch.full_like(cos[off_diag_mask], expected_off_diag), atol=1e-5)

    norms = W.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_etf_classifier_background_is_not_in_simplex():
    """Class 0 is background and must not participate in the simplex ETF."""
    cls = ETFClassifier(
        feature_dim=1024,
        num_classes=11,
        use_projector=False,
        preserve_logit_scale=False,
    )
    W = cls.weight
    assert W.shape == (11, 1024)

    fg = W[1:]
    cos = fg @ fg.T
    expected_off_diag = -1.0 / 9.0
    off_diag_mask = ~torch.eye(10, dtype=torch.bool)
    assert torch.allclose(cos[off_diag_mask], torch.full_like(cos[off_diag_mask], expected_off_diag), atol=1e-5)


def test_etf_logit_scale_preservation():
    """When preserve_logit_scale=True, foreground rows keep the original mean row norm."""
    original = torch.randn(11, 1024)
    target_scale = 7.5
    original = original / original.norm(dim=1, keepdim=True) * target_scale

    cls = ETFClassifier(
        feature_dim=1024,
        num_classes=11,
        use_projector=False,
        preserve_logit_scale=True,
        original_weight=original,
    )
    fg_norms = cls.weight[1:].norm(dim=1)
    assert torch.allclose(fg_norms, torch.full_like(fg_norms, target_scale), atol=1e-4)
    assert math.isclose(float(cls.logit_scale.item()), target_scale, abs_tol=1e-4)


def test_etf_projector_output_shape():
    """Projector must not change logit shape."""
    cls = ETFClassifier(feature_dim=1024, num_classes=11, use_projector=True)
    x = torch.randn(4, 1024)
    logits = cls(x)
    assert logits.shape == (4, 11)


def test_etf_original_background_mode():
    """background_mode='original' copies the pretrained background row."""
    original = torch.randn(11, 1024)
    cls = ETFClassifier(
        feature_dim=1024,
        num_classes=11,
        use_projector=False,
        preserve_logit_scale=False,
        background_mode="original",
        original_weight=original,
    )
    assert torch.allclose(cls.weight[0], original[0], atol=1e-6)
