from __future__ import annotations

import pytest
import torch

from spectral_detection_posttrain.methods.manifold.geometry_metrics import (
    compute_class_centroids,
    compute_inter_class_separation,
    compute_intra_class_compactness,
    compute_manifold_geometry,
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
