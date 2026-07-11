from __future__ import annotations

import pytest
import torch

from spectral_detection_posttrain.methods.energy_transport.linear_identifiability import (
    LocalCandidateRows,
    evaluate_identifiability_gates,
    extract_local_candidate_rows,
    fit_local_feature_map,
    fit_ridge_regression,
    image_balanced_weights,
    predict_ridge_regression,
    transform_local_candidate_rows,
    within_image_shuffle_order,
)


def _record(target: float = 0.4) -> tuple[dict, dict]:
    detector = {
        "image_id": 7,
        "spatial_features": torch.tensor(
            [
                [[[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]]],
                [[[10.0, 12.0], [14.0, 16.0]], [[20.0, 22.0], [24.0, 26.0]]],
            ]
        ),
        "class_logits": torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
        "predicted_labels": torch.tensor([1, 2]),
        "scores": torch.tensor([0.6, 0.8]),
        "boxes": torch.tensor([[0.0, 0.0, 20.0, 10.0], [10.0, 5.0, 30.0, 25.0]]),
        "image_size": (50, 100),
        "action_targets": torch.zeros(2),
        "move_targets": torch.zeros(2),
    }
    trace = {
        "image_id": 7,
        "proposal_count": 2,
        "action_ids": ("p1_c2",),
        "proposal_indices": torch.tensor([1]),
        "candidate_indices": torch.tensor([2]),
        "box_deltas": torch.tensor([[0.05, 0.0, -0.05, 0.0]]),
        "action_energies": torch.tensor([0.5]),
        "identity_utility": 1.0,
        "singleton_delta_u": torch.tensor([target]),
    }
    return detector, trace


def test_local_candidate_features_do_not_depend_on_delta_u_target() -> None:
    first = extract_local_candidate_rows([_record(0.4)], num_classes=3)
    second = extract_local_candidate_rows([_record(-0.8)], num_classes=3)

    assert first.roi_features.shape == (1, 2)
    assert first.roi_features.tolist() == [[13.0, 23.0]]
    assert first.action_features.tolist()[0] == pytest.approx(
        [0.05, 0.0, -0.05, 0.0, 0.5]
    )
    assert first.image_ids.tolist() == [7]
    assert first.target.tolist() == pytest.approx([0.4])
    assert torch.equal(first.roi_features, second.roi_features)
    assert torch.equal(first.detector_features, second.detector_features)
    assert torch.equal(first.action_features, second.action_features)
    assert not torch.equal(first.target, second.target)


def _toy_rows() -> LocalCandidateRows:
    return LocalCandidateRows(
        roi_features=torch.tensor(
            [[1.0, 0.0], [1.2, 0.1], [0.0, 1.0], [0.1, 1.2], [2.0, 2.0], [2.1, 2.2]]
        ),
        detector_features=torch.tensor(
            [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5]]
        ),
        action_features=torch.tensor(
            [
                [0.05, 0.0, 0.0, 0.0, 0.1],
                [0.10, 0.0, 0.0, 0.0, 0.2],
                [0.0, 0.05, 0.0, 0.0, 0.1],
                [0.0, 0.10, 0.0, 0.0, 0.2],
                [0.0, 0.0, 0.05, 0.0, 0.1],
                [0.0, 0.0, 0.10, 0.0, 0.2],
            ]
        ),
        target=torch.tensor([0.1, 0.2, -0.1, -0.2, 0.3, 0.4]),
        image_ids=torch.tensor([1, 1, 2, 2, 3, 3]),
    )


def test_feature_shuffle_is_deterministic_within_image_and_has_no_fixed_rows() -> None:
    rows = _toy_rows()

    first, first_fraction = within_image_shuffle_order(rows.image_ids, seed=19)
    second, second_fraction = within_image_shuffle_order(rows.image_ids, seed=19)

    assert torch.equal(first, second)
    assert first_fraction == pytest.approx(second_fraction)
    assert first_fraction == pytest.approx(1.0)
    assert sorted(first.tolist()) == list(range(6))
    assert torch.equal(rows.image_ids[first], rows.image_ids)
    assert torch.all(first.ne(torch.arange(6)))


def test_image_balanced_weights_give_each_image_equal_total_mass() -> None:
    image_ids = torch.tensor([1, 1, 1, 2, 3, 3])

    weights = image_balanced_weights(image_ids)

    totals = [float(weights[image_ids.eq(value)].sum()) for value in (1, 2, 3)]
    assert totals == pytest.approx([1.0, 1.0, 1.0])


def test_frozen_feature_map_and_ridge_recover_linear_signal() -> None:
    rows = _toy_rows()
    feature_map = fit_local_feature_map(rows, pca_dim=2)
    features = transform_local_candidate_rows(rows, feature_map)
    target = 1.5 * features[:, 0] - 0.7 * features[:, -1] + 0.25

    model = fit_ridge_regression(
        features,
        target,
        sample_weights=image_balanced_weights(rows.image_ids),
        l2=1e-6,
    )
    predicted = predict_ridge_regression(model, features)

    assert features.shape[1] == 2 + 2 * 4 + 1 + 5
    assert torch.mean(torch.abs(predicted - target)).item() < 1e-3


def _gate_payload() -> dict:
    raw = {
        "pairwise_accuracy": 0.62,
        "pairwise_accuracy_candidate_weighted": 0.61,
    }
    return {
        "heldout": {
            "support": {"candidate_count": 650, "image_count": 70},
            "arms": {
                "local_full": {
                    "raw": raw,
                    "frozen_threshold": {
                        "selected_count": 14,
                        "mean_delta_u_per_image": 0.02,
                        "mean_delta_u_lcb": 0.01,
                        "positive_precision_lift": 0.08,
                        "action_image_rate": 0.20,
                    },
                },
                "within_image_feature_shuffle": {
                    "raw": {
                        "pairwise_accuracy": 0.55,
                        "pairwise_accuracy_candidate_weighted": 0.54,
                    }
                },
                "within_image_label_shuffle": {
                    "raw": {
                        "pairwise_accuracy": 0.51,
                        "pairwise_accuracy_candidate_weighted": 0.50,
                    }
                },
            },
        },
        "selection": {
            "local_full": {
                "used_identity_fallback": False,
                "tune_raw": {
                    "pairwise_accuracy": 0.64,
                    "pairwise_accuracy_candidate_weighted": 0.63,
                },
                "fit_raw": {
                    "pairwise_accuracy": 0.68,
                    "pairwise_accuracy_candidate_weighted": 0.67,
                },
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


def test_identifiability_gates_require_safety_transfer_and_control_gain() -> None:
    passed = evaluate_identifiability_gates(_gate_payload(), _gate_config())

    assert passed["all_passed"] is True
    assert all(passed["gates"].values())

    unsafe = _gate_payload()
    unsafe["heldout"]["arms"]["local_full"]["frozen_threshold"][
        "mean_delta_u_lcb"
    ] = -0.01
    assert evaluate_identifiability_gates(unsafe, _gate_config())["gates"][
        "G2_frozen_threshold_safety"
    ] is False

    no_control_gain = _gate_payload()
    no_control_gain["heldout"]["arms"]["within_image_feature_shuffle"]["raw"][
        "pairwise_accuracy"
    ] = 0.61
    assert evaluate_identifiability_gates(no_control_gain, _gate_config())["gates"][
        "G4_control_gain"
    ] is False
