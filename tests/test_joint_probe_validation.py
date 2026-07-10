from __future__ import annotations

import math

import pytest
import torch

from spectral_detection_posttrain.methods.energy_transport import (
    ProposalSetEdges,
    calibrate_conservative_threshold,
    calibrated_selection_metrics,
    constant_utility_baselines,
    imagewise_pairwise_accuracy,
    paired_bootstrap_mean_difference,
    shuffle_edge_topology,
)


def test_edge_topology_shuffle_is_deterministic_and_preserves_groups() -> None:
    edges = ProposalSetEdges(
        edge_index=torch.tensor(
            [[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3]], dtype=torch.long
        ),
        edge_features=torch.arange(48, dtype=torch.float32).reshape(6, 8),
        node_count=6,
    )
    labels = torch.tensor([1, 1, 1, 2, 2, 2])
    image_indices = torch.tensor([0, 0, 0, 0, 0, 0])

    first = shuffle_edge_topology(edges, labels, image_indices, seed=17)
    second = shuffle_edge_topology(edges, labels, image_indices, seed=17)

    assert torch.equal(first.edge_index, second.edge_index)
    assert torch.equal(first.edge_features, edges.edge_features)
    assert torch.equal(first.edge_index[0], edges.edge_index[0])
    assert not torch.equal(first.edge_index[1], edges.edge_index[1])
    assert sorted(first.edge_index[1].tolist()) == sorted(edges.edge_index[1].tolist())
    assert torch.equal(labels[first.edge_index[0]], labels[first.edge_index[1]])
    assert torch.equal(
        image_indices[first.edge_index[0]], image_indices[first.edge_index[1]]
    )


def test_edge_topology_shuffle_never_crosses_images() -> None:
    edges = ProposalSetEdges(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long),
        edge_features=torch.randn(4, 8),
        node_count=4,
    )
    labels = torch.ones(4, dtype=torch.long)
    image_indices = torch.tensor([0, 0, 1, 1])

    shuffled = shuffle_edge_topology(edges, labels, image_indices, seed=3)

    assert torch.equal(
        image_indices[shuffled.edge_index[0]], image_indices[shuffled.edge_index[1]]
    )


def _calibration_rows() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predicted = torch.tensor(
        [0.90, 0.10, 0.80, 0.20, 0.70, 0.30, 0.40, 0.10, 0.35, 0.05, 0.25, 0.15]
    )
    target = torch.tensor(
        [1.20, -0.05, 1.00, -0.04, 0.80, -0.03, -0.20, -0.02, -0.15, -0.01, -0.10, -0.02]
    )
    image_ids = torch.repeat_interleave(torch.arange(6), 2)
    return predicted, target, image_ids


def test_conservative_calibration_selects_only_supported_actions() -> None:
    predicted, target, image_ids = _calibration_rows()

    calibrated = calibrate_conservative_threshold(
        predicted,
        target,
        image_ids,
        target_epsilon=1e-3,
        min_selected_actions=2,
        max_action_image_rate=0.5,
        min_positive_precision_lift=0.20,
        min_mean_delta_u_lcb=0.0,
        lcb_z=1.0,
    )

    assert math.isfinite(calibrated.threshold)
    assert calibrated.used_identity_fallback is False
    assert calibrated.metrics["selected_count"] == 3
    assert calibrated.metrics["action_image_rate"] == pytest.approx(0.5)
    assert calibrated.metrics["selected_positive_precision"] == pytest.approx(1.0)
    assert calibrated.metrics["mean_delta_u_per_image"] > 0.0
    assert calibrated.metrics["mean_delta_u_lcb"] > 0.0


def test_conservative_calibration_falls_back_to_identity_when_unsafe() -> None:
    predicted = torch.tensor([0.8, 0.4, 0.7, 0.3, 0.6, 0.2])
    target = torch.tensor([-0.3, -0.1, -0.2, -0.1, -0.4, -0.2])
    image_ids = torch.repeat_interleave(torch.arange(3), 2)

    calibrated = calibrate_conservative_threshold(
        predicted,
        target,
        image_ids,
        target_epsilon=1e-3,
        min_selected_actions=2,
        max_action_image_rate=1.0,
        min_positive_precision_lift=0.05,
        min_mean_delta_u_lcb=0.0,
        lcb_z=1.0,
    )

    assert math.isinf(calibrated.threshold)
    assert calibrated.used_identity_fallback is True
    assert calibrated.metrics["selected_count"] == 0
    assert calibrated.metrics["mean_delta_u_per_image"] == pytest.approx(0.0)


def test_calibration_respects_configured_lcb_floor() -> None:
    predicted, target, image_ids = _calibration_rows()

    calibrated = calibrate_conservative_threshold(
        predicted,
        target,
        image_ids,
        target_epsilon=1e-3,
        min_selected_actions=2,
        max_action_image_rate=0.5,
        min_positive_precision_lift=0.20,
        min_mean_delta_u_lcb=1.0,
        lcb_z=1.0,
    )

    assert calibrated.used_identity_fallback is True
    assert math.isinf(calibrated.threshold)


def test_calibrated_metrics_apply_one_action_budget_per_image() -> None:
    predicted, target, image_ids = _calibration_rows()

    metrics = calibrated_selection_metrics(
        predicted,
        target,
        image_ids,
        threshold=0.5,
        target_epsilon=1e-3,
        lcb_z=1.0,
    )

    assert metrics["image_count"] == 6
    assert metrics["selected_count"] == 3
    assert metrics["selected_positive_count"] == 3
    assert metrics["selected_positive_precision"] == pytest.approx(1.0)
    assert metrics["mean_delta_u_per_image"] == pytest.approx(0.5)
    assert metrics["mean_delta_u_selected"] == pytest.approx(1.0)
    assert metrics["oracle_regret_mean"] == pytest.approx(0.0)


def test_constant_baselines_expose_imbalance_and_zero_mae() -> None:
    target = torch.tensor([1.0, -0.2, -0.1, -0.3])
    image_ids = torch.tensor([1, 1, 2, 2])

    baselines = constant_utility_baselines(target, image_ids, target_epsilon=1e-3)

    assert baselines["positive_prevalence"] == pytest.approx(0.25)
    assert baselines["always_negative_sign_accuracy"] == pytest.approx(0.75)
    assert baselines["zero_utility_mae"] == pytest.approx(0.4)
    assert baselines["zero_utility_pairwise_accuracy"] == pytest.approx(0.5)


def test_image_paired_bootstrap_reports_positive_context_gain() -> None:
    left = {1: 0.70, 2: 0.65, 3: 0.80, 4: 0.75}
    right = {1: 0.50, 2: 0.45, 3: 0.55, 4: 0.50}

    result = paired_bootstrap_mean_difference(
        left, right, resamples=1000, seed=42, confidence=0.95
    )

    assert result["estimate"] == pytest.approx(0.225)
    assert result["ci_low"] > 0.0
    assert result["n_images"] == 4
    assert result["resamples"] == 1000


def test_pairwise_statistics_are_equal_image_and_tie_aware() -> None:
    predicted = torch.tensor([0.0, 0.0, 0.8, 0.1])
    target = torch.tensor([1.0, 0.0, 1.0, 0.0])
    image_ids = torch.tensor([10, 10, 20, 20])

    stats = imagewise_pairwise_accuracy(
        predicted, target, image_ids, target_epsilon=1e-3
    )

    assert stats == {10: pytest.approx(0.5), 20: pytest.approx(1.0)}
