import numpy as np
import pytest
import torch

from spectral_detection_posttrain.analysis.raw_ifft_verifier import (
    apply_selection_policy,
    calibrate_precision_threshold,
    fit_train_effect_scorer,
    parse_legacy_ifft_feature_specs,
    score_legacy_ifft_metric_bank,
    score_scene_legacy_ifft_metric_bank,
    threshold_metrics,
)


def test_rank_sum_scorer_uses_train_positive_direction():
    train = np.array(
        [
            [0.0, 10.0],
            [1.0, 8.0],
            [8.0, 1.0],
            [10.0, 0.0],
        ]
    )
    labels = np.array([False, False, True, True])
    val = np.array(
        [
            [9.0, 0.5],
            [0.5, 9.0],
            [5.0, 5.0],
        ]
    )

    scorer = fit_train_effect_scorer(train, labels, method="rank_sum")
    scores = scorer.score(val)

    assert scores[0] > scores[2] > scores[1]
    assert scorer.weights.tolist() == pytest.approx([1.0, -1.0])


def test_precision_threshold_uses_train_prefix_and_margin():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    labels = np.array([True, True, False, True])

    result = calibrate_precision_threshold(scores, labels, target_precision=0.9, margin=0.05)

    assert result.threshold == pytest.approx(0.85)
    assert result.selected_prefix == 2
    assert result.tp_prefix == 2
    assert result.fp_prefix == 0
    assert result.precision_prefix == pytest.approx(1.0)


def test_selection_policy_applies_primary_guard_and_per_image_top_k():
    scores = np.array([0.95, 0.90, 0.80, 0.70, 0.60])
    primary_scores = np.array([0.2, 0.9, 0.9, 0.9, 0.9])
    image_ids = np.array([1, 1, 1, 2, 2])

    selected = apply_selection_policy(
        scores,
        threshold=0.65,
        primary_scores=primary_scores,
        primary_threshold=0.5,
        image_ids=image_ids,
        top_k_per_image=1,
    )

    assert selected.tolist() == [False, True, False, True, False]


def test_threshold_metrics_reports_precision_recall_and_fp_rate():
    selected = np.array([True, True, False, True])
    labels = np.array([True, False, True, False])

    metrics = threshold_metrics(selected, labels)

    assert metrics["selected"] == 3
    assert metrics["tp"] == 1
    assert metrics["fp"] == 2
    assert metrics["precision"] == pytest.approx(1 / 3)
    assert metrics["recall"] == pytest.approx(1 / 2)
    assert metrics["false_positive_rate"] == pytest.approx(1.0)


def test_parse_legacy_ifft_feature_specs_resolves_feature_indices_and_crops():
    parsed = parse_legacy_ifft_feature_specs(["fft_edge_truncation@64", "phase_edge@11"])

    assert parsed == [(3, 64, "fft_edge_truncation@64"), (1, 11, "phase_edge@11")]


def test_score_legacy_ifft_metric_bank_applies_frozen_train_scaler_and_threshold():
    metric_bank = {
        64: torch.tensor([[10.0, 1.0], [12.0, 3.0]]),
        11: torch.tensor([[5.0, 7.0], [4.0, 9.0]]),
    }
    parsed_specs = [(0, 64, "fake0@64"), (1, 11, "fake1@11")]

    scores = score_legacy_ifft_metric_bank(
        metric_bank,
        parsed_specs,
        mean=torch.tensor([10.0, 7.0]),
        scale=torch.tensor([2.0, 2.0]),
        weights=torch.tensor([0.5, 1.0]),
        threshold=0.25,
    )

    assert scores.tolist() == pytest.approx([-0.25, 1.25])


def test_score_scene_legacy_ifft_metric_bank_routes_by_class_and_fallback():
    metric_bank = {
        64: torch.tensor(
            [
                [10.0, 1.0],
                [14.0, 2.0],
                [30.0, 3.0],
            ]
        ),
        21: torch.tensor(
            [
                [2.0, 5.0],
                [4.0, 7.0],
                [8.0, 11.0],
            ]
        ),
    }
    labels = torch.tensor([8, 10, 5])
    scene_groups = [
        {
            "name": "maritime",
            "classes": [2, 8],
            "enabled": True,
            "parsed_features": [{"feature_index": 0, "crop_size": 64, "spec": "fake@64"}],
            "scaler_mean": [10.0],
            "scaler_scale": [2.0],
            "weights": [1.0],
            "threshold": 0.5,
        },
        {
            "name": "vehicle",
            "classes": [10],
            "enabled": True,
            "parsed_features": [{"feature_index": 1, "crop_size": 21, "spec": "fake@21"}],
            "scaler_mean": [5.0],
            "scaler_scale": [2.0],
            "weights": [2.0],
            "threshold": 1.0,
        },
    ]

    scores = score_scene_legacy_ifft_metric_bank(metric_bank, labels, scene_groups, fallback_score=-99.0)

    assert scores.tolist() == pytest.approx([-0.5, 1.0, -99.0])
