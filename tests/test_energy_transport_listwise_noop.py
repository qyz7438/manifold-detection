from __future__ import annotations

import pytest
import torch

from spectral_detection_posttrain.methods.energy_transport.listwise_noop import (
    build_listwise_batch,
    calibrate_noop_margin,
    evaluate_listwise_gates,
    fit_listwise_model,
    listwise_decision_metrics,
    predict_listwise_scores,
)


def test_listwise_batch_targets_oracle_candidate_or_explicit_noop() -> None:
    features = torch.tensor([[2.0], [1.0], [-1.0], [-2.0]])
    target = torch.tensor([0.3, -0.1, -0.1, -0.2])
    image_ids = torch.tensor([0, 0, 1, 1])

    batch = build_listwise_batch(
        features, target, image_ids, target_epsilon=0.001
    )

    assert batch.features.shape == (2, 2, 1)
    assert batch.candidate_mask.tolist() == [[True, True], [True, True]]
    assert batch.target_indices.tolist() == [0, 2]


def test_linear_listwise_model_learns_candidate_vs_noop_decision() -> None:
    features = torch.tensor(
        [[2.0], [1.0], [2.0], [1.0], [-1.0], [-2.0], [-1.0], [-2.0]]
    )
    target = torch.tensor([0.3, -0.1, 0.4, -0.2, -0.1, -0.2, -0.2, -0.3])
    image_ids = torch.repeat_interleave(torch.arange(4), 2)

    model = fit_listwise_model(
        features,
        target,
        image_ids,
        target_epsilon=0.001,
        l2=0.01,
        max_iter=80,
    )
    scores = predict_listwise_scores(model, features)

    assert scores[:4].reshape(2, 2).max(dim=1).values.min() > model.noop_logit
    assert scores[4:].reshape(2, 2).max(dim=1).values.max() < model.noop_logit


def test_decision_metrics_include_noop_and_paired_random_expectation() -> None:
    scores = torch.tensor([2.0, 1.0, -1.0, -2.0])
    target = torch.tensor([0.2, -0.1, 0.3, -0.2])
    image_ids = torch.tensor([0, 0, 1, 1])

    metrics = listwise_decision_metrics(
        scores,
        noop_logit=0.0,
        noop_margin=0.0,
        target=target,
        image_ids=image_ids,
        target_epsilon=0.001,
        lcb_z=0.0,
    )

    assert metrics["selected_count"] == 1
    assert metrics["mean_delta_u"] == pytest.approx(0.1)
    assert metrics["mean_delta_u_lcb"] == pytest.approx(0.1)
    assert metrics["decision_accuracy"] == pytest.approx(0.5)
    assert metrics["per_image_delta_u"] == pytest.approx([0.2, 0.0])


def test_calibration_selects_safe_noop_margin_on_tune() -> None:
    scores = torch.tensor([2.0, 1.0, 1.5, 0.5, 0.2, 0.1, 0.1, 0.0])
    target = torch.tensor([0.3, -0.1, 0.2, -0.2, -0.2, -0.3, -0.1, -0.2])
    image_ids = torch.repeat_interleave(torch.arange(4), 2)

    calibrated = calibrate_noop_margin(
        scores,
        noop_logit=0.0,
        target=target,
        image_ids=image_ids,
        target_epsilon=0.001,
        min_selected_actions=2,
        max_action_image_rate=0.5,
        min_positive_precision_lift=0.05,
        min_mean_delta_u_lcb=0.0,
        lcb_z=0.0,
    )

    assert calibrated["used_identity_fallback"] is False
    assert calibrated["metrics"]["selected_count"] == 2
    assert calibrated["metrics"]["mean_delta_u_lcb"] == pytest.approx(0.125)


def test_listwise_gates_require_safety_baselines_controls_and_gap() -> None:
    config = {
        "gates": {
            "min_candidates": 500,
            "min_images": 60,
            "min_selected_actions": 12,
            "max_action_image_rate": 0.25,
            "min_positive_precision_lift": 0.05,
            "min_mean_delta_u_lcb": 0.0,
            "min_paired_gain_lcb": 0.0,
            "max_fit_to_outer_mean_gap": 0.1,
            "max_tune_to_outer_mean_gap": 0.05,
        }
    }
    full = {
        "selected_count": 14,
        "action_image_rate": 0.2,
        "positive_precision_lift": 0.3,
        "mean_delta_u": 0.08,
        "mean_delta_u_lcb": 0.02,
    }
    control = {**full, "mean_delta_u": 0.01, "mean_delta_u_lcb": -0.01}
    payload = {
        "support": {"candidate_count": 633, "image_count": 71},
        "arms": {
            "local_full": full,
            "within_image_feature_shuffle": control,
            "within_image_label_shuffle": control,
        },
        "selection": {
            "local_full": {
                "used_identity_fallback": False,
                "fit_metrics": {"mean_delta_u": 0.12},
                "tune_metrics": {"mean_delta_u": 0.1},
            }
        },
        "paired_gains": {
            "vs_rate_matched_random": {"lcb": 0.01},
            "vs_feature_shuffle": {"lcb": 0.02},
            "vs_label_shuffle": {"lcb": 0.03},
        },
    }

    assert evaluate_listwise_gates(payload, config)["all_passed"] is True
