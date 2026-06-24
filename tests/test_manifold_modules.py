"""Unit tests for the MGL-OPT manifold modules."""

from __future__ import annotations

import math

import pytest
import torch

from spectral_detection_posttrain.methods.manifold.geometry_metrics import (
    compute_effective_rank,
    compute_nc1,
)
from spectral_detection_posttrain.methods.manifold.prototype_bank import PrototypeBank
from spectral_detection_posttrain.methods.manifold.sinkhorn_assigner import SinkhornAssigner
from spectral_detection_posttrain.methods.manifold.transport_head import TransportHead
from spectral_detection_posttrain.methods.manifold.intrinsic_dim import IntrinsicDimEstimator


# ---------------------------------------------------------------------------
# PrototypeBank
# ---------------------------------------------------------------------------

def test_prototype_bank_initializes_correct_shape():
    bank = PrototypeBank(num_classes=3, num_prototypes_per_class=4, feature_dim=16)
    assert bank.prototypes.shape == (3, 4, 16)
    assert bank.ema_sums.shape == (3, 4, 16)
    assert bank.ema_counts.shape == (3, 4)


def test_prototype_bank_compute_distances_shape():
    bank = PrototypeBank(num_classes=2, num_prototypes_per_class=3, feature_dim=8)
    features = torch.randn(5, 8)
    class_ids = torch.tensor([0, 1, 0, 1, 0])
    distances = bank.compute_distances(features, class_ids)
    assert distances.shape == (5, 3)
    assert (distances >= 0).all()


def test_prototype_bank_update_changes_prototypes():
    bank = PrototypeBank(num_classes=2, num_prototypes_per_class=2, feature_dim=4)
    initial = bank.prototypes.clone()

    features = torch.randn(8, 4)
    class_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    assignments = torch.rand(8, 2)
    assignments = assignments / assignments.sum(dim=1, keepdim=True)

    bank.update(features, class_ids, assignments)
    assert not torch.allclose(bank.prototypes, initial)


def test_prototype_bank_initialize_from_centers():
    bank = PrototypeBank(num_classes=2, num_prototypes_per_class=3, feature_dim=4)
    centers = torch.randn(2, 4)
    bank.initialize_from_centers(centers, noise_scale=0.0)

    # With zero noise all prototypes for a class equal the center.
    for c in range(2):
        assert torch.allclose(bank.prototypes[c], centers[c].unsqueeze(0), atol=1e-6)


def test_prototype_bank_get_prototypes():
    bank = PrototypeBank(num_classes=2, num_prototypes_per_class=2, feature_dim=4)
    assert bank.get_prototypes().shape == (2, 2, 4)
    assert bank.get_prototypes(class_id=0).shape == (2, 4)


# ---------------------------------------------------------------------------
# SinkhornAssigner
# ---------------------------------------------------------------------------

def test_sinkhorn_assigner_row_sums_and_column_sums():
    assigner = SinkhornAssigner(eps=0.05, max_iter=100)
    cost = torch.rand(10, 4)
    q = assigner(cost)

    assert q.shape == (10, 4)
    row_sums = q.sum(dim=1)
    col_sums = q.sum(dim=0)

    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)
    assert torch.allclose(col_sums, torch.full_like(col_sums, 10.0 / 4.0), atol=1e-3)


def test_sinkhorn_assigner_gradients_flow():
    assigner = SinkhornAssigner(eps=0.05, max_iter=20)
    cost = torch.rand(6, 3, requires_grad=True)
    q = assigner(cost)
    loss = q.sum()
    loss.backward()
    assert cost.grad is not None
    assert cost.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# TransportHead
# ---------------------------------------------------------------------------

def test_transport_head_output_shape():
    head = TransportHead(feature_dim=16, num_prototypes=4)
    features = torch.randn(5, 16)
    distances = torch.rand(5, 4)
    transport = head(features, distances)
    assert transport.shape == (5, 16)


def test_transport_head_energy_is_scalar():
    head = TransportHead(feature_dim=8, num_prototypes=2)
    features = torch.randn(4, 8)
    distances = torch.rand(4, 2)
    energy = head.transport_energy(features, distances)
    assert energy.ndim == 0
    assert energy.item() >= 0.0


def test_transport_head_gradients_flow():
    head = TransportHead(feature_dim=8, num_prototypes=2)
    features = torch.randn(4, 8, requires_grad=True)
    distances = torch.rand(4, 2)
    transport = head(features, distances)
    loss = (transport ** 2).sum()
    loss.backward()
    assert features.grad is not None
    for p in head.parameters():
        assert p.grad is not None


def test_transport_head_smoothness_penalty():
    head = TransportHead(feature_dim=8, num_prototypes=2)
    features = torch.randn(3, 8, requires_grad=True)
    distances = torch.rand(3, 2)
    energy = head.transport_energy(features, distances, smoothness_weight=0.1)
    energy.backward()
    assert features.grad is not None


# ---------------------------------------------------------------------------
# IntrinsicDimEstimator
# ---------------------------------------------------------------------------

def test_pca_estimator_line_has_id_one():
    est = IntrinsicDimEstimator(method="pca", pca_variance_threshold=0.95)
    # Points on a line in 10-D space.
    t = torch.linspace(-1, 1, 50).unsqueeze(1)
    features = torch.cat([t, torch.zeros(50, 9)], dim=1)
    id_est = est.estimate_id(features)
    assert id_est.item() == pytest.approx(1.0, abs=0.5)


def test_twonn_estimator_line_has_low_id():
    est = IntrinsicDimEstimator(method="twonn")
    # Random points on a line in 10-D space.
    # TwoNN assumes continuous random sampling, so a uniform grid would
    # over-estimate the dimension.
    rng = torch.Generator().manual_seed(42)
    t = torch.randn(200, 1, generator=rng) * 3.0
    features = torch.cat([t, torch.zeros(200, 9)], dim=1)
    id_est = est.estimate_id(features)
    assert 0.5 <= id_est.item() <= 2.5


def test_local_geometry_returns_expected_keys():
    est = IntrinsicDimEstimator(method="pca")
    features = torch.randn(30, 8)
    labels = torch.randint(0, 2, (30,))
    geom = est.local_geometry(features, labels=labels)
    assert "intrinsic_dim" in geom
    assert "radius" in geom
    assert "separability" in geom
    assert geom["radius"].item() >= 0.0


def test_local_geometry_separability_prefers_close_positives():
    est = IntrinsicDimEstimator(method="pca")
    # Positives cluster near zero, negatives far away.
    pos = torch.randn(20, 4) * 0.1
    neg = torch.randn(20, 4) + 3.0
    features = torch.cat([pos, neg], dim=0)
    labels = torch.cat([
        torch.ones(20, dtype=torch.bool),
        torch.zeros(20, dtype=torch.bool),
    ])
    geom = est.local_geometry(features, labels=labels)
    assert geom["separability"].item() > 0.5


def test_effective_rank_full_rank_is_near_dimension():
    torch.manual_seed(0)
    # Random Gaussian data in 8D is approximately full rank.
    features = torch.randn(100, 8)
    eff_rank = compute_effective_rank(features)
    assert 5.0 <= eff_rank.item() <= 8.5


def test_effective_rank_is_nan_for_single_sample():
    features = torch.randn(1, 4)
    eff_rank = compute_effective_rank(features)
    assert math.isnan(eff_rank.item())


def test_nc1_returns_nan_for_single_foreground_class():
    torch.manual_seed(0)
    features = torch.randn(20, 4)
    labels = torch.ones(20, dtype=torch.long)
    nc1 = compute_nc1(features, labels, num_classes=3)
    assert math.isnan(nc1.item())
