from __future__ import annotations

import pytest
import torch

from spectral_detection_posttrain.methods.manifold.geometry_metrics import (
    compute_class_centroids,
    compute_effective_rank,
    compute_inter_class_separation,
    compute_intra_class_compactness,
    compute_manifold_geometry,
    compute_nc1,
    compute_spectral_decay,
    scalar_geometry_report,
)


def test_compute_class_centroids_counts_empty_classes() -> None:
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    labels = torch.tensor([1, 1, 2])
    centroids, counts = compute_class_centroids(features, labels, num_classes=4)

    assert counts.tolist() == [0, 2, 1, 0]
    assert torch.allclose(centroids[0], torch.zeros(2))
    assert torch.allclose(centroids[3], torch.zeros(2))
    assert torch.allclose(centroids[1], torch.tensor([0.5, 0.5]))
    assert torch.allclose(centroids[2], torch.tensor([1.0, 1.0]))


def test_intra_class_compactness_is_low_for_tight_clusters() -> None:
    # Class 1: two points very close to [0, 0].
    features = torch.tensor([[0.1, 0.0], [-0.1, 0.0], [10.0, 10.0]])
    labels = torch.tensor([1, 1, 2])

    compactness = compute_intra_class_compactness(features, labels, num_classes=3, normalize=False)
    # Class 1 mean distance to centroid should be ~0.1.
    assert compactness["intra_mean"] > 0.0
    assert compactness["intra_mean"] < 5.0


def test_inter_class_separation_increases_with_distance() -> None:
    # Background at [0, 0] is skipped when labels is None; only foreground
    # centroids [1, 0] and [3, 0] are compared.
    centroids = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    separation = compute_inter_class_separation(centroids, labels=None, normalize=False)

    assert separation["inter_min"] == pytest.approx(2.0, abs=1e-4)
    assert separation["inter_mean"] == pytest.approx(2.0, abs=1e-4)
    assert separation["inter_max"] == pytest.approx(2.0, abs=1e-4)


def test_geometry_report_detects_dimension_reduction_after_correction() -> None:
    torch.manual_seed(0)
    # 50 samples in 8D, two classes.
    features = torch.randn(50, 8)
    labels = torch.cat([torch.ones(25, dtype=torch.long), torch.full((25,), 2, dtype=torch.long)])

    # Corrected features are projected onto the first two dimensions -> lower ID.
    corrected = features.clone()
    corrected[:, 2:] *= 0.01

    geometry = compute_manifold_geometry(
        features,
        labels,
        num_classes=3,
        corrected_features=corrected,
        method="pca",
        normalize=False,
    )

    assert geometry["id_foreground"] > geometry["id_corrected_foreground"]
    report = scalar_geometry_report(geometry)
    assert "id_foreground" in report
    assert "id_corrected_foreground" in report
    assert report["id_delta_foreground"] < 0


def test_effective_rank_detects_low_rank_structure() -> None:
    torch.manual_seed(0)
    # 30 samples in 10D, but they lie on a 2D subspace.
    basis = torch.randn(10, 2)
    coeffs = torch.randn(30, 2)
    features = coeffs @ basis.T

    eff_rank = compute_effective_rank(features)
    # Should be close to 2 (with some slack due to finite sampling).
    assert 1.5 <= eff_rank.item() <= 3.5


def test_spectral_decay_ratios_sum_to_reasonable_range() -> None:
    torch.manual_seed(0)
    features = torch.randn(40, 16)
    decay = compute_spectral_decay(features)

    assert "spectral_top1_ratio" in decay
    assert "spectral_top5_ratio" in decay
    assert "spectral_top10_ratio" in decay
    assert 0.0 <= decay["spectral_top1_ratio"].item() <= decay["spectral_top5_ratio"].item() <= 1.0
    assert decay["spectral_top10_ratio"].item() <= 1.0


def test_nc1_low_for_tight_clusters_high_for_dispersed_clusters() -> None:
    torch.manual_seed(0)
    # Tight clusters: two classes, small within-class variance, large separation.
    tight = torch.cat([
        torch.randn(25, 4) * 0.1,
        torch.randn(25, 4) * 0.1 + 5.0,
    ], dim=0)
    labels = torch.cat([torch.ones(25, dtype=torch.long), torch.full((25,), 2, dtype=torch.long)])
    nc1_tight = compute_nc1(tight, labels, num_classes=3).item()

    # Dispersed clusters: same centers but large within-class variance.
    dispersed = torch.cat([
        torch.randn(25, 4) * 2.0,
        torch.randn(25, 4) * 2.0 + 5.0,
    ], dim=0)
    nc1_dispersed = compute_nc1(dispersed, labels, num_classes=3).item()

    assert nc1_tight < nc1_dispersed
    assert 0.0 < nc1_tight < 1.0


def test_geometry_report_contains_new_metrics() -> None:
    torch.manual_seed(0)
    features = torch.randn(60, 8)
    labels = torch.cat([
        torch.zeros(20, dtype=torch.long),
        torch.ones(20, dtype=torch.long),
        torch.full((20,), 2, dtype=torch.long),
    ])

    geometry = compute_manifold_geometry(features, labels, num_classes=3)
    report = scalar_geometry_report(geometry)

    required_keys = [
        "effective_rank_overall",
        "effective_rank_foreground",
        "spectral_top1_ratio",
        "spectral_top5_ratio",
        "spectral_top10_ratio",
        "spectral_top1_ratio_foreground",
        "nc1_overall",
        "separability_overall",
    ]
    for key in required_keys:
        assert key in geometry
        assert key in report


def test_separability_auc_above_half_for_separable_data() -> None:
    torch.manual_seed(0)
    # Foreground clusters near class centroids, background far away.
    fg1 = torch.randn(20, 4) * 0.2 + torch.tensor([0.0, 0.0, 0.0, 0.0])
    fg2 = torch.randn(20, 4) * 0.2 + torch.tensor([3.0, 0.0, 0.0, 0.0])
    bg = torch.randn(20, 4) * 0.5 + torch.tensor([6.0, 0.0, 0.0, 0.0])
    features = torch.cat([fg1, fg2, bg], dim=0)
    labels = torch.cat([
        torch.ones(20, dtype=torch.long),
        torch.full((20,), 2, dtype=torch.long),
        torch.zeros(20, dtype=torch.long),
    ])

    geometry = compute_manifold_geometry(features, labels, num_classes=3)
    assert geometry["separability_overall"].item() > 0.5


def test_corrected_geometry_adds_delta_metrics() -> None:
    torch.manual_seed(0)
    features = torch.randn(50, 8)
    labels = torch.cat([
        torch.zeros(15, dtype=torch.long),
        torch.ones(20, dtype=torch.long),
        torch.full((15,), 2, dtype=torch.long),
    ])
    corrected = features.clone()
    corrected[:, 2:] *= 0.01

    geometry = compute_manifold_geometry(
        features, labels, num_classes=3, corrected_features=corrected
    )
    report = scalar_geometry_report(geometry)

    assert "effective_rank_corrected_overall" in geometry
    assert "nc1_corrected" in geometry
    assert "nc1_delta" in geometry
    assert "separability_corrected" in geometry
    assert "effective_rank_corrected_overall" in report
    assert "nc1_corrected" in report
