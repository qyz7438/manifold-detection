"""Unit tests for Plan D image classification modules."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from spectral_detection_posttrain.methods.classification import (
    OTPrototypeClassifier,
    SpectralClassifierHead,
    SpectralMixup,
    accuracy,
    evaluate_classifier,
)


def test_spectral_classifier_head_forward():
    """Forward pass returns finite logits and supports backpropagation."""
    head = SpectralClassifierHead(in_channels=8, num_classes=5)
    x = torch.randn(2, 8, 16, 16, requires_grad=True)
    logits = head(x)

    assert logits.shape == (2, 5)
    assert torch.isfinite(logits).all()

    loss = logits.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_spectral_classifier_head_branch_outputs():
    """Magnitude-only and phase-only branches produce correct logit shapes."""
    head = SpectralClassifierHead(in_channels=4, num_classes=3)
    x = torch.randn(2, 4, 8, 8)

    logits_mag = head.forward_magnitude(x)
    logits_phase = head.forward_phase(x)

    assert logits_mag.shape == (2, 3)
    assert logits_phase.shape == (2, 3)
    assert torch.isfinite(logits_mag).all()
    assert torch.isfinite(logits_phase).all()


def test_spectral_classifier_head_fusion_weight_in_unit_interval():
    """The learnable fusion weight stays inside [0, 1] after sigmoid."""
    head = SpectralClassifierHead(in_channels=4, num_classes=3)
    alpha = torch.sigmoid(head.fusion_weight)
    assert 0.0 <= alpha.item() <= 1.0


def test_ot_prototype_classifier_forward():
    """Forward returns logits and gradients flow to features and prototypes."""
    clf = OTPrototypeClassifier(
        feature_dim=8, num_classes=3, n_prototypes=4, eps=0.05
    )
    features = torch.randn(4, 8, requires_grad=True)
    logits = clf(features)

    assert logits.shape == (4, 3)
    assert torch.isfinite(logits).all()

    loss = logits.sum()
    loss.backward()
    assert features.grad is not None
    assert clf.prototypes.grad is not None


def test_ot_prototype_classifier_single_prototype_distance():
    """With one prototype the Sinkhorn distance equals the deterministic cost."""
    feature_dim = 4
    p = 2
    clf = OTPrototypeClassifier(
        feature_dim=feature_dim,
        num_classes=2,
        n_prototypes=1,
        eps=0.01,
        p=p,
    )
    feat = torch.randn(1, feature_dim)
    proto0 = clf.prototypes.data[0, 0]

    expected_dist = torch.cdist(feat, proto0.unsqueeze(0), p=p).pow(p).item()
    logits = clf(feat)
    assert abs(-logits[0, 0].item() - expected_dist) < 1e-3


def test_ot_prototype_classifier_update_prototypes():
    """Online prototype update changes prototypes for present classes."""
    clf = OTPrototypeClassifier(
        feature_dim=4, num_classes=2, n_prototypes=2, momentum=0.5
    )
    features = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1])

    old_prototypes = clf.prototypes.data.clone()
    clf.update_prototypes(features, labels)

    assert not torch.equal(clf.prototypes.data, old_prototypes)


def test_spectral_mixup_shapes_and_labels():
    """SpectralMixup preserves image shape and mixes labels."""
    mixup = SpectralMixup(alpha=10.0, eps=0.05, max_iter=10)
    x1 = torch.randn(3, 16, 16)
    x2 = torch.randn(3, 16, 16)
    y1 = torch.tensor([1.0, 0.0, 0.0])
    y2 = torch.tensor([0.0, 1.0, 0.0])

    x_mix, y_mix = mixup(x1, x2, y1, y2)

    assert x_mix.shape == x1.shape
    assert torch.isfinite(x_mix).all()
    # Beta(10, 10) is sharply peaked at 0.5.
    assert torch.allclose(y_mix, 0.5 * (y1 + y2), atol=0.2)


def test_spectral_mixup_invalid_shapes():
    """SpectralMixup raises on mismatched input shapes."""
    mixup = SpectralMixup(alpha=1.0)
    x1 = torch.randn(3, 16, 16)
    x2 = torch.randn(3, 8, 8)
    y = torch.zeros(3)

    with pytest.raises(ValueError):
        mixup(x1, x2, y, y)


def test_accuracy_topk():
    """accuracy computes top-k hits correctly."""
    output = torch.tensor(
        [
            [0.9, 0.1, 0.0],
            [0.2, 0.7, 0.1],
            [0.2, 0.2, 0.6],
        ]
    )
    target = torch.tensor([0, 1, 2])

    top1 = accuracy(output, target, topk=(1,))
    assert top1 == [100.0]

    top1_top2 = accuracy(output, target, topk=(1, 2))
    assert top1_top2 == [100.0, 100.0]


def test_evaluate_classifier():
    """evaluate_classifier returns a valid top-k accuracy dictionary."""
    model = nn.Linear(4, 3)
    dataset = TensorDataset(torch.randn(8, 4), torch.randint(0, 3, (8,)))
    loader = DataLoader(dataset, batch_size=4)

    result = evaluate_classifier(model, loader, "cpu", topk=(1,))

    assert "top1_acc" in result
    assert 0.0 <= result["top1_acc"] <= 100.0
