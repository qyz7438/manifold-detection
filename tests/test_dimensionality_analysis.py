import numpy as np
import pytest

from spectral_detection_posttrain.analysis.dimensionality import (
    binary_ranking_metrics,
    center_margin_scores,
    pca_dimensionality_summary,
)
from scripts.analyze_nwpu_roi_feature_dimensions import knn_density_ratio_scores


def test_binary_ranking_metrics_reports_auc_ap_and_recall_at_precision():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0], dtype=bool)

    metrics = binary_ranking_metrics(scores, labels, precision_targets=(0.5, 1.0))

    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["recall_at_precision_0.5"] == pytest.approx(1.0)
    assert metrics["recall_at_precision_1"] == pytest.approx(1.0)


def test_pca_dimensionality_summary_reports_effective_rank_and_threshold_dims():
    features = np.array(
        [
            [2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
        ]
    )

    summary = pca_dimensionality_summary(features, max_components=3)

    assert summary["feature_dim"] == 3
    assert summary["sample_count"] == 6
    assert summary["components_for_80pct"] == 1
    assert summary["components_for_95pct"] <= 2
    assert summary["participation_ratio"] < 2.0


def test_center_margin_scores_prefers_points_near_positive_center():
    train_features = np.array([[0.0, 0.0], [0.2, 0.0], [3.0, 3.0], [3.2, 3.0]])
    train_labels = np.array([1, 1, 0, 0], dtype=bool)
    query_features = np.array([[0.1, 0.0], [3.1, 3.0]])

    scores = center_margin_scores(train_features, train_labels, query_features)

    assert scores[0] > scores[1]


def test_knn_density_ratio_scores_handles_high_dimensional_inputs_without_broadcasting():
    rng = np.random.default_rng(42)
    positive = rng.normal(loc=0.0, scale=0.1, size=(3, 256))
    negative = rng.normal(loc=3.0, scale=0.1, size=(4, 256))
    train_features = np.concatenate([positive, negative], axis=0)
    train_labels = np.array([1, 1, 1, 0, 0, 0, 0], dtype=bool)
    query_features = np.stack([positive.mean(axis=0), negative.mean(axis=0)])

    scores = knn_density_ratio_scores(train_features, train_labels, query_features, k=2)

    assert scores.shape == (2,)
    assert scores[0] > scores[1]
