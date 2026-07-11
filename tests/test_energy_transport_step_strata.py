from __future__ import annotations

import pytest
import torch

from spectral_detection_posttrain.methods.energy_transport.step_strata import (
    StepActionRows,
    discover_stable_strata,
    evaluate_step_strata_gates,
    extract_step_action_rows,
    select_primary_stratum,
    strata_masks,
    summarize_stratum,
    target_permutation_control,
)


def _record() -> tuple[dict, dict]:
    detector = {
        "image_id": 9,
        "spatial_features": torch.zeros(2, 2, 2, 2),
        "class_logits": torch.zeros(2, 3),
        "predicted_labels": torch.tensor([1, 2]),
        "scores": torch.tensor([0.5, 0.8]),
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 50.0, 50.0]]),
        "image_size": (100, 100),
        "action_targets": torch.zeros(2),
        "move_targets": torch.zeros(2),
    }
    trace = {
        "image_id": 9,
        "proposal_count": 2,
        "action_ids": ("a", "b", "c"),
        "proposal_indices": torch.tensor([0, 1, 1]),
        "candidate_indices": torch.tensor([1, 2, 25]),
        "box_deltas": torch.tensor(
            [
                [0.05, 0.0, 0.0, 0.0],
                [0.0, 0.0, -0.05, 0.05],
                [0.1, 0.0, 0.0, 0.0],
            ]
        ),
        "action_energies": torch.tensor([0.0625, 0.125, 0.25]),
        "identity_utility": 0.0,
        "singleton_delta_u": torch.tensor([0.2, -0.1, 0.5]),
    }
    return detector, trace


def test_step_rows_filter_exact_action_scale_and_encode_strata() -> None:
    rows = extract_step_action_rows([_record()], step=0.05)

    assert rows.target.tolist() == pytest.approx([0.2, -0.1])
    assert rows.family == ("translation", "scale")
    assert rows.direction == ("+1,0,0,0", "0,0,-1,+1")
    assert rows.class_ids.tolist() == [1, 2]
    assert rows.scale_bin == ("small", "large")
    masks = strata_masks(rows)
    assert masks["family=translation"].tolist() == [True, False]
    assert masks["class=2"].tolist() == [False, True]


def _rows() -> StepActionRows:
    return StepActionRows(
        target=torch.tensor([0.3, -0.1, 0.2, 0.4]),
        image_ids=torch.tensor([1, 1, 2, 2]),
        family=("translation", "scale", "translation", "translation"),
        direction=("d1", "d2", "d1", "d1"),
        class_ids=torch.tensor([1, 1, 2, 2]),
        scale_bin=("small", "small", "large", "large"),
    )


def test_stratum_summary_is_image_balanced_and_uses_same_image_uniform_control() -> None:
    rows = _rows()
    mask = torch.tensor([True, False, True, True])
    summary = summarize_stratum(
        rows,
        mask,
        manifest_image_ids=[1, 2, 3],
        target_epsilon=1e-3,
        lcb_z=1.645,
    )

    assert summary["candidate_count"] == 3
    assert summary["image_count"] == 2
    assert summary["action_image_rate"] == pytest.approx(2.0 / 3.0)
    assert summary["mean_delta_u_per_manifest_image"] == pytest.approx(0.2)
    assert summary["mean_delta_u_selected_images"] == pytest.approx(0.3)
    assert summary["selected_positive_precision"] == pytest.approx(1.0)
    assert summary["positive_precision_lift"] == pytest.approx(0.25)
    assert summary["paired_uniform_gain_per_manifest_image"] == pytest.approx(
        1.0 / 15.0
    )


def test_discovery_requires_independent_fit_and_tune_positive_lcb() -> None:
    fit = {
        "family=translation": {"candidate_count": 30, "image_count": 15, "mean_delta_u_lcb": 0.02, "paired_uniform_gain_lcb": 0.01},
        "family=scale": {"candidate_count": 40, "image_count": 16, "mean_delta_u_lcb": 0.03, "paired_uniform_gain_lcb": 0.02},
    }
    tune = {
        "family=translation": {"candidate_count": 12, "image_count": 7, "mean_delta_u_lcb": 0.01, "paired_uniform_gain_lcb": 0.008},
        "family=scale": {"candidate_count": 14, "image_count": 8, "mean_delta_u_lcb": -0.01, "paired_uniform_gain_lcb": -0.01},
    }

    selected = discover_stable_strata(
        fit,
        tune,
        min_fit_candidates=20,
        min_fit_images=10,
        min_tune_candidates=10,
        min_tune_images=5,
    )

    assert selected == ["family=translation"]
    assert select_primary_stratum(selected, tune) == "family=translation"


def test_target_permutation_control_is_deterministic() -> None:
    rows = _rows()
    mask = torch.tensor([True, False, True, True])

    permuted_first = target_permutation_control(
        rows,
        mask,
        manifest_image_ids=[1, 2, 3],
        trials=200,
        seed=23,
    )
    permuted_second = target_permutation_control(
        rows,
        mask,
        manifest_image_ids=[1, 2, 3],
        trials=200,
        seed=23,
    )

    assert permuted_first == permuted_second
    assert permuted_first["paired_uniform_gain_p95"] >= permuted_first[
        "paired_uniform_gain_mean"
    ]


def _gate_payload() -> dict:
    return {
        "outer_support": {"candidate_count": 260, "image_count": 65},
        "selected_strata": ["family=translation"],
        "primary_stratum": "family=translation",
        "strata": {
            "family=translation": {
                "fit": {"mean_delta_u_lcb": 0.02},
                "tune": {"mean_delta_u_lcb": 0.01},
                "outer": {
                    "candidate_count": 30,
                    "image_count": 12,
                    "mean_delta_u_per_manifest_image": 0.03,
                    "mean_delta_u_lcb": 0.02,
                    "paired_uniform_gain_per_manifest_image": 0.02,
                    "paired_uniform_gain_lcb": 0.01,
                    "positive_precision_lift": 0.10,
                },
                "target_permutation": {"paired_uniform_gain_p95": 0.004},
            }
        },
    }


def _gate_config() -> dict:
    return {
        "gates": {
            "min_outer_candidates": 200,
            "min_outer_images": 50,
            "min_outer_stratum_candidates": 20,
            "min_outer_stratum_images": 10,
            "min_outer_mean_delta_u": 0.01,
            "min_outer_mean_delta_u_lcb": 0.0,
            "min_paired_uniform_gain": 0.01,
            "min_paired_uniform_gain_lcb": 0.0,
            "min_positive_precision_lift": 0.05,
            "min_gain_over_permutation_p95": 0.005,
        }
    }


def test_step_strata_gates_require_one_stratum_to_beat_both_controls() -> None:
    passed = evaluate_step_strata_gates(_gate_payload(), _gate_config())

    assert passed["all_passed"] is True
    assert passed["winning_strata"] == ["family=translation"]

    failed = _gate_payload()
    failed["strata"]["family=translation"]["target_permutation"][
        "paired_uniform_gain_p95"
    ] = 0.018
    result = evaluate_step_strata_gates(failed, _gate_config())
    assert result["all_passed"] is False
    assert result["winning_strata"] == []
