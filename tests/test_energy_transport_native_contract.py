from __future__ import annotations

import inspect

import pytest
import torch

from spectral_detection_posttrain.methods.energy_transport.native_contract import (
    build_detector_native_candidates,
    build_native_c1_deltas,
    evaluate_native_contract_gates,
    validate_strict_parity_artifact,
)


def test_c1_deltas_are_noop_plus_eight_bounded_step005_actions() -> None:
    deltas = build_native_c1_deltas(step=0.05)

    assert deltas.shape == (9, 4)
    assert torch.equal(deltas[0], torch.zeros(4))
    assert deltas[1:].abs().max().item() == pytest.approx(0.05)
    assert torch.all(deltas[1:].count_nonzero(dim=1) == 1)
    assert torch.unique(deltas, dim=0).shape[0] == 9


def test_detector_native_builder_has_no_target_input_and_filters_observables() -> None:
    assert "target" not in inspect.signature(build_detector_native_candidates).parameters
    assert "labels" not in inspect.signature(build_detector_native_candidates).parameters
    batch = build_detector_native_candidates(
        spatial_features=torch.randn(3, 2, 7, 7),
        class_logits=torch.tensor([[0.0, 2.0], [2.0, 0.0], [0.0, 2.0]]),
        scores=torch.tensor([0.8, 0.9, 0.01]),
        boxes=torch.tensor([[0.0, 0.0, 2.0, 2.0]] * 3),
        image_indices=torch.tensor([0, 0, 1]),
        score_threshold=0.05,
    )

    assert batch.proposal_indices.tolist() == [0]
    assert batch.image_indices.tolist() == [0]
    assert batch.spatial_features.shape == (1, 2, 7, 7)


def test_strict_parity_requires_aggregate_strict_and_exact_zero() -> None:
    artifact = {
        "completed": True,
        "aggregate_zero_action_parity": {"passed": True},
        "strict_zero_action_parity": {
            "passed": True,
            "images": 196,
            "mismatched_images": 0,
            "actions_are_exact_zero": True,
            "max_box_abs_error": 0.0,
            "max_score_abs_error": 0.0,
        },
    }

    assert validate_strict_parity_artifact(artifact)["passed"] is True
    artifact["strict_zero_action_parity"]["actions_are_exact_zero"] = False
    assert validate_strict_parity_artifact(artifact)["passed"] is False


def test_strict_parity_rejects_empty_coverage_and_bool_numeric_fields() -> None:
    artifact = {
        "completed": True,
        "aggregate_zero_action_parity": {"passed": True},
        "strict_zero_action_parity": {
            "passed": True,
            "images": 0,
            "mismatched_images": False,
            "actions_are_exact_zero": True,
            "max_box_abs_error": False,
            "max_score_abs_error": False,
        },
    }

    assert validate_strict_parity_artifact(artifact)["passed"] is False


def test_native_contract_gates_require_two_parities_and_budget_one() -> None:
    payload = {
        "parity": {
            "baseline": {"passed": True},
            "fullft": {"passed": True},
        },
        "candidate_contract": {
            "candidate_count": 9,
            "noop_is_first": True,
            "max_abs_delta": 0.05,
            "max_actions_per_image": 1,
            "detector_only": True,
            "budget_enforced": True,
            "score_threshold": 0.05,
            "candidate_source": "detector_visible_proposals_only",
            "training_gt_scope": "utility_labels_only_never_candidate_filter",
        },
    }
    config = {
        "gates": {
            "required_candidate_count": 9,
            "required_max_abs_delta": 0.05,
            "required_max_actions_per_image": 1,
            "required_score_threshold": 0.05,
            "required_candidate_source": "detector_visible_proposals_only",
            "required_training_gt_scope": "utility_labels_only_never_candidate_filter",
        }
    }

    assert evaluate_native_contract_gates(payload, config)["all_passed"] is True

    payload["candidate_contract"]["score_threshold"] = 0.5
    assert evaluate_native_contract_gates(payload, config)["all_passed"] is False
