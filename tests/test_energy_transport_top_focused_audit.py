from __future__ import annotations

import pytest
import torch

from spectral_detection_posttrain.methods.energy_transport.top_focused_audit import (
    evaluate_top_focused_gates,
    median_sign_metrics,
    top_focused_rank_metrics,
)


def test_median_sign_metrics_remove_image_offsets_and_exclude_target_ties() -> None:
    scores = torch.tensor([10.0, 11.0, 12.0, -4.0, -3.0, -2.0])
    target = torch.tensor([-1.0, 0.0, 1.0, -2.0, 0.0, 2.0])
    image_ids = torch.tensor([0, 0, 0, 1, 1, 1])

    metrics = median_sign_metrics(scores, target, image_ids)

    assert metrics["valid_candidate_count"] == 4
    assert metrics["valid_image_count"] == 2
    assert metrics["within_image_pair_count"] == 2
    assert metrics["auroc_equal_image"] == pytest.approx(1.0)
    assert metrics["auroc_candidate_weighted"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)


def test_median_sign_uses_true_half_quantile_for_even_candidate_counts() -> None:
    scores = torch.tensor([0.0, 1.0, 2.0, 3.0])
    target = torch.tensor([0.0, 1.0, 2.0, 3.0])
    image_ids = torch.zeros(4, dtype=torch.long)

    metrics = median_sign_metrics(scores, target, image_ids)

    assert metrics["valid_candidate_count"] == 4
    assert metrics["positive_count"] == 2
    assert metrics["negative_count"] == 2
    assert metrics["within_image_pair_count"] == 4


def test_top_focused_rank_metrics_expose_top1_failure() -> None:
    scores = torch.tensor([2.0, 1.0, 0.0, 1.0])
    target = torch.tensor([0.2, -0.1, 0.3, -0.2])
    image_ids = torch.tensor([0, 0, 1, 1])

    metrics = top_focused_rank_metrics(
        scores,
        target,
        image_ids,
        target_epsilon=0.001,
        min_target_gap=0.001,
        lcb_z=0.0,
    )

    assert metrics["cross_boundary_pairwise_equal_image"] == pytest.approx(0.5)
    assert metrics["oracle_top_pairwise_equal_image"] == pytest.approx(0.5)
    assert metrics["predicted_top1_positive_hit_rate"] == pytest.approx(0.5)
    assert metrics["random_top1_positive_hit_rate"] == pytest.approx(0.5)
    assert metrics["top1_positive_hit_lift"] == pytest.approx(0.0)
    assert metrics["top1_mean_delta_u"] == pytest.approx(0.0)
    assert metrics["top1_mean_delta_u_lcb"] == pytest.approx(0.0)


def test_top_focused_gates_require_control_gain_and_actionable_top1() -> None:
    config = {
        "gates": {
            "min_candidates": 500,
            "min_images": 60,
            "min_median_pairs": 100,
            "min_cross_boundary_pairs": 100,
            "min_oracle_top_pairs": 100,
            "min_median_sign_auroc": 0.55,
            "min_rank_pairwise": 0.58,
            "min_control_gain": 0.02,
            "min_top1_positive_hit_lift": 0.0,
            "min_top1_mean_delta_u_lcb": 0.0,
        }
    }
    full = {
        "median_sign": {
            "valid_candidate_count": 500,
            "valid_image_count": 60,
            "within_image_pair_count": 120,
            "auroc_equal_image": 0.65,
            "auroc_candidate_weighted": 0.66,
        },
        "top_rank": {
            "cross_boundary_pair_count": 150,
            "oracle_top_pair_count": 200,
            "cross_boundary_pairwise_equal_image": 0.68,
            "cross_boundary_pairwise_candidate_weighted": 0.69,
            "oracle_top_pairwise_equal_image": 0.67,
            "oracle_top_pairwise_candidate_weighted": 0.68,
            "top1_positive_hit_lift": 0.1,
            "top1_mean_delta_u_lcb": 0.02,
        },
    }
    control = {
        "median_sign": {
            **full["median_sign"],
            "auroc_equal_image": 0.55,
            "auroc_candidate_weighted": 0.56,
        },
        "top_rank": {
            **full["top_rank"],
            "cross_boundary_pairwise_equal_image": 0.52,
            "cross_boundary_pairwise_candidate_weighted": 0.53,
            "oracle_top_pairwise_equal_image": 0.51,
            "oracle_top_pairwise_candidate_weighted": 0.52,
        },
    }
    payload = {
        "support": {"candidate_count": 633, "image_count": 71},
        "arms": {
            "local_full": full,
            "within_image_feature_shuffle": control,
            "within_image_label_shuffle": control,
        },
    }

    gates = evaluate_top_focused_gates(payload, config)

    assert gates["all_passed"] is True
