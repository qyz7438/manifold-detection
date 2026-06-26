"""Unit tests for Plan E multimodal alignment modules."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spectral_detection_posttrain.methods.manifold import (
    AdaptiveRiemannianMetric,
    ChordTransport,
    ComplexSpectralManifold,
)
from spectral_detection_posttrain.methods.multimodal import (
    ChordTextGuidedEdit,
    CrossModalTransport,
    OTImageTextAlignment,
    compute_similarity_matrix,
    evaluate_image_text_retrieval,
    recall_at_k,
)


@pytest.fixture
def manifold_and_transport():
    """Return a paired manifold and Chord transport for Plan E tests."""
    dim = 16
    manifold = ComplexSpectralManifold(in_dim=dim, latent_dim=dim, hidden_dim=dim)
    metric = AdaptiveRiemannianMetric(latent_dim=dim)
    transport = ChordTransport(manifold, metric, delta=0.15, lambda_step=1.0)
    return manifold, transport


def test_ot_alignment_forward_shape_and_gradient():
    """OTImageTextAlignment returns a scalar loss with working gradients."""
    batch_size, feature_dim = 8, 32
    loss_fn = OTImageTextAlignment(feature_dim=feature_dim, eps=0.05, max_iter=20)

    image_features = torch.randn(batch_size, feature_dim, requires_grad=True)
    text_features = torch.randn(batch_size, feature_dim, requires_grad=True)

    loss = loss_fn(image_features, text_features)
    assert loss.shape == ()
    assert torch.isfinite(loss)

    loss.backward()
    assert image_features.grad is not None
    assert text_features.grad is not None
    assert torch.isfinite(image_features.grad).all()
    assert torch.isfinite(text_features.grad).all()


def test_ot_alignment_diagonal_cheaper():
    """Paired features yield a lower OT distance than random pairings."""
    batch_size, feature_dim = 16, 16
    loss_fn = OTImageTextAlignment(feature_dim=feature_dim, eps=0.05, max_iter=30)

    torch.manual_seed(0)
    image_features = torch.randn(batch_size, feature_dim)
    text_features = image_features + 0.1 * torch.randn_like(image_features)

    aligned_loss = loss_fn(image_features, text_features)

    # Shuffle text features to break the pairing.
    shuffled_text = text_features[torch.randperm(batch_size)]
    shuffled_loss = loss_fn(image_features, shuffled_text)

    assert aligned_loss.item() < shuffled_loss.item()


def test_ot_alignment_invalid_shape():
    """OTImageTextAlignment raises on mismatched or non-2-D inputs."""
    loss_fn = OTImageTextAlignment(feature_dim=8)
    with pytest.raises(ValueError):
        loss_fn(torch.randn(4, 8), torch.randn(5, 8))
    with pytest.raises(ValueError):
        loss_fn(torch.randn(4, 8, 1), torch.randn(4, 8, 1))


def test_cross_modal_transport_forward_shape(manifold_and_transport):
    """CrossModalTransport produces refined image features of the same shape."""
    manifold, transport = manifold_and_transport
    text_dim, image_dim = 24, 32
    batch_size = 4

    model = CrossModalTransport(text_dim, image_dim, manifold, transport)
    text_feature = torch.randn(batch_size, text_dim)
    image_feature = torch.randn(batch_size, image_dim)

    refined = model(text_feature, image_feature)
    assert refined.shape == image_feature.shape
    assert torch.isfinite(refined).all()


def test_cross_modal_transport_gradient_flow(manifold_and_transport):
    """CrossModalTransport supports backpropagation through all branches."""
    manifold, transport = manifold_and_transport
    text_dim, image_dim = 12, 16
    batch_size = 3

    model = CrossModalTransport(text_dim, image_dim, manifold, transport)
    text_feature = torch.randn(batch_size, text_dim, requires_grad=True)
    image_feature = torch.randn(batch_size, image_dim, requires_grad=True)

    refined = model(text_feature, image_feature)
    loss = refined.pow(2).sum()
    loss.backward()

    assert text_feature.grad is not None
    assert image_feature.grad is not None
    assert torch.isfinite(text_feature.grad).all()
    assert torch.isfinite(image_feature.grad).all()
    for name, param in model.named_parameters():
        # ChordTransport stores its metric only for API compatibility and does
        # not use it during the forward pass, so its parameters stay gradient-free.
        if "transport.metric" in name:
            continue
        assert param.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} gradient is non-finite"


def test_cross_modal_transport_exposes_energy(manifold_and_transport):
    """CrossModalTransport exposes the underlying Chord transport energy."""
    manifold, transport = manifold_and_transport
    model = CrossModalTransport(8, 16, manifold, transport)

    assert model.transport_energy() is None
    model(torch.randn(2, 8), torch.randn(2, 16))
    energy = model.transport_energy()
    assert energy is not None
    assert energy.shape == (2,)
    assert (energy >= 0).all()


def test_chord_text_guided_edit_forward_shape(manifold_and_transport):
    """ChordTextGuidedEdit returns edited features of the expected shape."""
    manifold, transport = manifold_and_transport
    feature_dim = 16
    batch_size = 2

    image_encoder = nn.Linear(8, feature_dim)
    text_encoder = nn.Embedding(20, feature_dim)

    editor = ChordTextGuidedEdit(image_encoder, text_encoder, transport, feature_dim)
    x_source = torch.randn(batch_size, 8)
    text_source = torch.randint(0, 20, (batch_size,))
    text_target = torch.randint(0, 20, (batch_size,))

    edited = editor(x_source, text_source, text_target)
    assert edited.shape == (batch_size, feature_dim)
    assert torch.isfinite(edited).all()


def test_chord_text_guided_edit_gradient_flow(manifold_and_transport):
    """ChordTextGuidedEdit backpropagates through image/text encoders and transport."""
    manifold, transport = manifold_and_transport
    feature_dim = 16
    batch_size = 2

    image_encoder = nn.Linear(8, feature_dim)
    text_encoder = nn.Embedding(20, feature_dim)

    editor = ChordTextGuidedEdit(image_encoder, text_encoder, transport, feature_dim)
    x_source = torch.randn(batch_size, 8)
    text_source = torch.randint(0, 20, (batch_size,))
    text_target = torch.randint(0, 20, (batch_size,))

    edited = editor(x_source, text_source, text_target)
    loss = edited.pow(2).sum()
    loss.backward()

    for name, param in editor.named_parameters():
        if "transport.metric" in name:
            continue
        assert param.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} gradient is non-finite"


def test_chord_text_guided_edit_mismatched_feature_shapes(manifold_and_transport):
    """ChordTextGuidedEdit raises when encoder outputs have incompatible shapes."""
    manifold, transport = manifold_and_transport

    class BadEncoder(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.randn(x.size(0), 32)

    editor = ChordTextGuidedEdit(
        image_encoder=nn.Linear(8, 16),
        text_encoder=BadEncoder(),
        transport=transport,
        feature_dim=16,
    )
    with pytest.raises(ValueError):
        editor(torch.randn(2, 8), torch.randint(0, 10, (2,)), torch.randint(0, 10, (2,)))


def test_compute_similarity_matrix():
    """compute_similarity_matrix returns cosine similarities in [-1, 1]."""
    image_features = torch.randn(5, 8)
    text_features = torch.randn(5, 8)

    sim = compute_similarity_matrix(image_features, text_features)
    assert sim.shape == (5, 5)
    assert (sim >= -1.0 - 1e-6).all() and (sim <= 1.0 + 1e-6).all()

    # Self-similarity of identical vectors equals one.
    sim_self = compute_similarity_matrix(image_features, image_features)
    assert torch.allclose(torch.diag(sim_self), torch.ones(5), atol=1e-5)


def test_recall_at_k_perfect():
    """recall_at_k returns 1.0 for all k when the diagonal is dominant."""
    n = 8
    features = torch.randn(n, 16)
    similarity = compute_similarity_matrix(features, features)

    recalls = recall_at_k(similarity, k_values=(1, 5, 8))
    assert recalls["R@1"] == 1.0
    assert recalls["R@5"] == 1.0
    assert recalls["R@8"] == 1.0


def test_recall_at_k_zero():
    """recall_at_k returns 0.0 for R@1 when off-diagonal entries dominate."""
    n = 4
    # Image features are orthogonal to text features; add a strong off-diagonal peak.
    image_features = torch.eye(n)
    text_features = torch.eye(n).roll(shifts=1, dims=0)

    similarity = compute_similarity_matrix(image_features, text_features)
    recalls = recall_at_k(similarity, k_values=(1,))
    assert recalls["R@1"] == 0.0


def test_evaluate_image_text_retrieval():
    """evaluate_image_text_retrieval returns symmetric results for identical features."""
    n = 6
    features = torch.randn(n, 10)
    result = evaluate_image_text_retrieval(features, features, k_values=(1, 5))

    assert "image_to_text" in result
    assert "text_to_image" in result
    assert result["image_to_text"]["R@1"] == 1.0
    assert result["text_to_image"]["R@1"] == 1.0
