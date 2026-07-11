from __future__ import annotations

import pytest
import torch

from spectral_detection_posttrain.methods.energy_transport.decomposed_actionability import (
    calibrate_ranked_abstention,
    class_image_balanced_weights,
    evaluate_decomposed_actionability_gates,
    ranked_abstention_metrics,
    sign_classification_metrics,
    within_image_pairwise_rows,
)


def test_pairwise_rows_preserve_image_groups_and_target_order() -> None:
    features = torch.tensor([[0.0], [1.0], [3.0], [7.0]])
    target = torch.tensor([0.2, -0.1, 0.5, 0.8])
    image_ids = torch.tensor([4, 4, 4, 9])

    pairs = within_image_pairwise_rows(
        features, target, image_ids, min_target_gap=0.001
    )

    assert pairs.features.tolist() == [[-1.0], [-3.0], [-2.0]]
    assert pairs.target.tolist() == [1.0, -1.0, -1.0]
    assert pairs.image_ids.tolist() == [4, 4, 4]


def test_class_image_weights_balance_sign_mass_after_image_weighting() -> None:
    sign_target = torch.tensor([1.0, -1.0, -1.0, -1.0, 1.0, -1.0])
    image_ids = torch.tensor([0, 0, 0, 0, 1, 1])

    weights = class_image_balanced_weights(sign_target, image_ids)

    assert weights.sum().item() == pytest.approx(1.0)
    assert weights[sign_target > 0].sum().item() == pytest.approx(
        weights[sign_target < 0].sum().item()
    )


def test_class_image_weights_balance_global_classes_with_negative_only_images() -> None:
    sign_target = torch.tensor([1.0, -1.0, -1.0, -1.0])
    image_ids = torch.tensor([0, 0, 1, 1])

    weights = class_image_balanced_weights(sign_target, image_ids)

    assert weights[sign_target > 0].sum().item() == pytest.approx(0.5)
    assert weights[sign_target < 0].sum().item() == pytest.approx(0.5)


def test_sign_metrics_report_balanced_accuracy_and_two_aurocs() -> None:
    scores = torch.tensor([-2.0, 2.0, -1.0, 1.0, -3.0, 3.0])
    target = torch.tensor([-0.2, 0.4, -0.1, 0.2, -0.3, 0.5])
    image_ids = torch.tensor([0, 0, 1, 1, 2, 2])

    metrics = sign_classification_metrics(
        scores, target, image_ids, target_epsilon=0.001
    )

    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["auroc_equal_image"] == pytest.approx(1.0)
    assert metrics["auroc_candidate_weighted"] == pytest.approx(1.0)
    assert metrics["always_negative_accuracy"] == pytest.approx(0.5)


def test_ranked_abstention_uses_rank_for_choice_and_sign_for_action() -> None:
    rank_scores = torch.tensor([2.0, 1.0, 1.0, 3.0])
    sign_scores = torch.tensor([0.8, 0.9, 0.9, 0.6])
    target = torch.tensor([0.2, 0.5, 0.4, -0.3])
    image_ids = torch.tensor([0, 0, 1, 1])

    metrics = ranked_abstention_metrics(
        rank_scores,
        sign_scores,
        target,
        image_ids,
        sign_threshold=0.7,
        target_epsilon=0.001,
        lcb_z=0.0,
    )

    assert metrics["selected_count"] == 1
    assert metrics["selected_positive_precision"] == pytest.approx(1.0)
    assert metrics["action_image_rate"] == pytest.approx(0.5)
    assert metrics["mean_delta_u_per_image"] == pytest.approx(0.1)
    assert metrics["mean_delta_u_lcb"] == pytest.approx(0.1)


def test_calibration_freezes_safe_sign_threshold_after_rank_choice() -> None:
    rank_scores = torch.tensor([2.0, 1.0] * 4)
    sign_scores = torch.tensor([0.9, -0.5, 0.8, -0.4, 0.2, -0.3, 0.1, -0.2])
    target = torch.tensor([0.2, -0.5, 0.3, -0.5, -0.2, -0.5, -0.1, -0.5])
    image_ids = torch.repeat_interleave(torch.arange(4), 2)

    calibrated = calibrate_ranked_abstention(
        rank_scores,
        sign_scores,
        target,
        image_ids,
        target_epsilon=0.001,
        min_selected_actions=2,
        max_action_image_rate=0.5,
        min_positive_precision_lift=0.1,
        min_mean_delta_u_lcb=0.0,
        lcb_z=0.0,
    )

    assert calibrated["used_identity_fallback"] is False
    assert calibrated["metrics"]["selected_count"] == 2
    assert calibrated["metrics"]["mean_delta_u_lcb"] == pytest.approx(0.125)


def test_decomposed_gates_require_sign_rank_safety_and_control_gains() -> None:
    config = {
        "gates": {
            "min_heldout_candidates": 500,
            "min_heldout_images": 60,
            "min_sign_auroc": 0.58,
            "min_rank_pairwise": 0.58,
            "min_selected_actions": 12,
            "min_mean_delta_u": 0.01,
            "min_mean_delta_u_lcb": 0.0,
            "min_positive_precision_lift": 0.05,
            "max_action_image_rate": 0.25,
            "max_fit_to_heldout_gap": 0.1,
            "max_tune_to_heldout_gap": 0.05,
            "min_gain_over_controls": 0.02,
        }
    }
    full = {
        "sign": {"auroc_equal_image": 0.62, "auroc_candidate_weighted": 0.63},
        "rank": {"pairwise_accuracy": 0.64, "pairwise_accuracy_candidate_weighted": 0.65},
        "abstention": {
            "selected_count": 12,
            "mean_delta_u_per_image": 0.02,
            "mean_delta_u_lcb": 0.01,
            "positive_precision_lift": 0.2,
            "action_image_rate": 0.2,
        },
    }
    control = {
        "sign": {"auroc_equal_image": 0.55, "auroc_candidate_weighted": 0.56},
        "rank": {"pairwise_accuracy": 0.54, "pairwise_accuracy_candidate_weighted": 0.55},
        "abstention": full["abstention"],
    }
    payload = {
        "heldout": {
            "support": {"candidate_count": 633, "image_count": 71},
            "arms": {
                "local_full": full,
                "within_image_feature_shuffle": control,
                "within_image_label_shuffle": control,
            },
        },
        "selection": {
            "local_full": {
                "fit_sign": {"auroc_equal_image": 0.66, "auroc_candidate_weighted": 0.67},
                "tune_sign": {"auroc_equal_image": 0.64, "auroc_candidate_weighted": 0.65},
                "fit_rank": {"pairwise_accuracy": 0.68, "pairwise_accuracy_candidate_weighted": 0.69},
                "tune_rank": {"pairwise_accuracy": 0.66, "pairwise_accuracy_candidate_weighted": 0.67},
                "used_identity_fallback": False,
            }
        },
    }

    gates = evaluate_decomposed_actionability_gates(payload, config)

    assert gates["all_passed"] is True
