from __future__ import annotations

import pytest
import torch

from spectral_detection_posttrain.methods.energy_transport.spatial_counterfactual import (
    SpatialCandidateRows,
    evaluate_spatial_counterfactual_gates,
    fit_spatial_feature_map,
    spatial_counterfactual_blocks,
    spatial_layout_shuffle,
    transform_spatial_features,
    within_image_delta_alignment_shuffle,
)


def test_action_aligned_blocks_encode_edges_ring_quadrants_and_delta() -> None:
    spatial = torch.arange(1.0, 50.0).reshape(1, 1, 7, 7)
    action = torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.25]])

    blocks = spatial_counterfactual_blocks(spatial, action)

    assert blocks.shape == (1, 9, 1)
    assert blocks.flatten().tolist() == pytest.approx(
        [25.0, 6.0, 42.0, 0.0, 0.0, 0.0, 24.0, 18.0, 0.6]
    )


def test_layout_shuffle_is_deterministic_and_preserves_channel_global_mean() -> None:
    spatial = torch.arange(2 * 2 * 7 * 7, dtype=torch.float32).reshape(2, 2, 7, 7)

    first, first_fraction = spatial_layout_shuffle(spatial, seed=31)
    second, second_fraction = spatial_layout_shuffle(spatial, seed=31)

    assert torch.equal(first, second)
    assert first_fraction == pytest.approx(second_fraction)
    assert first_fraction > 0.8
    assert torch.equal(first.mean(dim=(-2, -1)), spatial.mean(dim=(-2, -1)))
    assert not torch.equal(first, spatial)


def test_delta_alignment_control_changes_only_the_aligned_block() -> None:
    spatial = torch.arange(1.0, 50.0).reshape(1, 1, 7, 7)
    action = torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.25]])
    shuffled_delta = torch.tensor([[0.0, 0.1, 0.0, 0.0]])

    original = spatial_counterfactual_blocks(spatial, action)
    controlled = spatial_counterfactual_blocks(
        spatial, action, alignment_delta=shuffled_delta
    )

    assert torch.equal(original[:, :8], controlled[:, :8])
    assert original[:, 8].item() == pytest.approx(0.6)
    assert controlled[:, 8].item() == pytest.approx(4.2)


def test_within_image_delta_shuffle_maximizes_changed_vectors() -> None:
    image_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    delta = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
        ]
    )

    shuffled, changed_fraction = within_image_delta_alignment_shuffle(
        delta, image_ids
    )

    assert changed_fraction == pytest.approx(0.75)
    for image_id in torch.unique(image_ids):
        mask = image_ids == image_id
        assert sorted(map(tuple, shuffled[mask].tolist())) == sorted(
            map(tuple, delta[mask].tolist())
        )


def _rows() -> SpatialCandidateRows:
    generator = torch.Generator().manual_seed(5)
    return SpatialCandidateRows(
        spatial_features=torch.randn(8, 2, 7, 7, generator=generator),
        detector_features=torch.randn(8, 2, generator=generator),
        action_features=torch.tensor(
            [
                [0.05, 0.0, 0.0, 0.0, 0.1],
                [0.10, 0.0, 0.0, 0.0, 0.2],
                [0.0, 0.05, 0.0, 0.0, 0.1],
                [0.0, 0.10, 0.0, 0.0, 0.2],
                [0.0, 0.0, 0.05, 0.0, 0.1],
                [0.0, 0.0, 0.10, 0.0, 0.2],
                [0.0, 0.0, 0.0, 0.05, 0.1],
                [0.0, 0.0, 0.0, 0.10, 0.2],
            ]
        ),
        target=torch.linspace(-0.2, 0.5, 8),
        image_ids=torch.tensor([1, 1, 2, 2, 3, 3, 4, 4]),
    )


def test_two_stage_spatial_map_has_locked_dimension_and_transforms_controls() -> None:
    rows = _rows()
    blocks = spatial_counterfactual_blocks(rows.spatial_features, rows.action_features)
    feature_map = fit_spatial_feature_map(
        rows,
        blocks,
        channel_pca_dim=2,
        final_pca_dim=3,
    )
    features = transform_spatial_features(rows, blocks, feature_map)
    shuffled_spatial, _ = spatial_layout_shuffle(rows.spatial_features, seed=7)
    shuffled_blocks = spatial_counterfactual_blocks(
        shuffled_spatial, rows.action_features
    )
    shuffled_features = transform_spatial_features(rows, shuffled_blocks, feature_map)

    assert features.shape == (8, 3 + 2 + 5)
    assert shuffled_features.shape == features.shape
    assert not torch.allclose(features, shuffled_features)


def _gate_payload() -> dict:
    raw = {
        "pairwise_accuracy": 0.63,
        "pairwise_accuracy_candidate_weighted": 0.62,
    }
    return {
        "heldout": {
            "support": {"candidate_count": 633, "image_count": 71},
            "arms": {
                "structured_full": {
                    "raw": raw,
                    "frozen_threshold": {
                        "selected_count": 14,
                        "mean_delta_u_per_image": 0.025,
                        "mean_delta_u_lcb": 0.012,
                        "positive_precision_lift": 0.09,
                        "action_image_rate": 0.20,
                    },
                },
                "global_pooled": {"raw": {"pairwise_accuracy": 0.56, "pairwise_accuracy_candidate_weighted": 0.55}},
                "spatial_layout_shuffle": {"raw": {"pairwise_accuracy": 0.54, "pairwise_accuracy_candidate_weighted": 0.53}},
                "delta_alignment_shuffle": {"raw": {"pairwise_accuracy": 0.52, "pairwise_accuracy_candidate_weighted": 0.51}},
            },
        },
        "selection": {
            "structured_full": {
                "used_identity_fallback": False,
                "fit_raw": {"pairwise_accuracy": 0.68, "pairwise_accuracy_candidate_weighted": 0.67},
                "tune_raw": {"pairwise_accuracy": 0.65, "pairwise_accuracy_candidate_weighted": 0.64},
            }
        },
    }


def _gate_config() -> dict:
    return {
        "gates": {
            "min_heldout_candidates": 500,
            "min_heldout_images": 60,
            "min_heldout_pairwise": 0.58,
            "min_selected_actions": 12,
            "min_mean_delta_u": 0.01,
            "min_mean_delta_u_lcb": 0.0,
            "min_positive_precision_lift": 0.05,
            "max_action_image_rate": 0.25,
            "max_fit_to_heldout_pairwise_gap": 0.10,
            "max_tune_to_heldout_pairwise_gap": 0.05,
            "min_gain_over_controls": 0.02,
        }
    }


def test_spatial_gates_require_ranking_safety_transfer_and_all_control_gains() -> None:
    passed = evaluate_spatial_counterfactual_gates(_gate_payload(), _gate_config())

    assert passed["all_passed"] is True

    failed = _gate_payload()
    failed["heldout"]["arms"]["global_pooled"]["raw"]["pairwise_accuracy"] = 0.62
    result = evaluate_spatial_counterfactual_gates(failed, _gate_config())
    assert result["all_passed"] is False
    assert result["gates"]["G4_control_gain"] is False
