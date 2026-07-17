"""Tests for the locked oracle-utility image-weighting protocol."""

from __future__ import annotations

import math

import pytest

from spectral_detection_posttrain.methods.energy_transport.endpoint.awr_weighting import (
    BudgetCounters,
    assert_equal_budget,
    build_oracle_utility_weights,
    derive_oracle_utilities,
    deterministic_weight_permutation,
)


FIT_IDS = (11, 12, 13, 14)


def _rows() -> list[dict[str, object]]:
    return [
        {"image_id": 11, "candidate_index": 0, "family": "identity_permutation", "q_teacher": 0.0},
        {"image_id": 11, "candidate_index": 0, "family": "translate_left", "q_teacher": -0.3},
        {"image_id": 11, "candidate_index": 0, "family": "scale_up", "q_teacher": 0.2},
        {"image_id": 11, "candidate_index": 1, "family": "translate_left", "q_teacher": 0.7},
        {"image_id": 12, "candidate_index": 0, "family": "identity_permutation", "q_teacher": 0.0},
        {"image_id": 12, "candidate_index": 0, "family": "translate_left", "q_teacher": -0.1},
        {"image_id": 13, "candidate_index": 0, "family": "identity_permutation", "q_teacher": 0.0},
    ]


def test_derives_candidate_and_image_utilities_from_nonidentity_q_teacher() -> None:
    utilities = derive_oracle_utilities(FIT_IDS, _rows())

    assert utilities.candidate_utilities == {
        (11, 0): 0.2,
        (11, 1): 0.7,
        (12, 0): -0.1,
    }
    assert utilities.image_utilities == {11: 0.7, 12: 0.0, 13: 0.0, 14: 0.0}
    assert utilities.candidate_counts == {11: 2, 12: 1, 13: 0, 14: 0}
    assert utilities.positive_candidate_count == 2


def test_accepts_nested_cache_records_and_requires_identity_q_teacher_zero() -> None:
    nested = [
        {
            "image_id": 11,
            "actions": [
                {"candidate_index": 0, "family": "identity_permutation", "q_teacher": 0.0},
                {"candidate_index": 0, "family": "translate_left", "q_teacher": 0.5},
            ],
        }
    ]
    assert derive_oracle_utilities((11,), nested).image_utilities == {11: 0.5}

    marked_identity = [
        {"image_id": 11, "candidate_index": 0, "family": "noop", "identity": True, "q_teacher": 0.0},
        {"image_id": 11, "candidate_index": 0, "family": "move", "identity": False, "q_teacher": 0.5},
    ]
    assert derive_oracle_utilities((11,), marked_identity).image_utilities == {11: 0.5}

    invalid = _rows()
    invalid[0] = {**invalid[0], "q_teacher": 1e-4}
    with pytest.raises(ValueError, match="identity.*zero"):
        derive_oracle_utilities(FIT_IDS, invalid)


@pytest.mark.parametrize(
    "rows, match",
    [
        (_rows() + [_rows()[1]], "duplicate"),
        ([{"image_id": 999, "candidate_index": 0, "family": "move", "q_teacher": 0.2}], "unknown"),
        ([{"image_id": 11, "candidate_index": 0, "family": "move", "q_teacher": float("nan")}], "finite"),
        ([{"image_id": 11, "candidate_index": 0, "family": "move", "q_teacher": float("inf")}], "finite"),
    ],
)
def test_rejects_ambiguous_or_invalid_cache_rows(
    rows: list[dict[str, object]], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        derive_oracle_utilities(FIT_IDS, rows)


def test_builds_locked_raw_exponential_weights_with_global_fit_mean_diagnostics() -> None:
    plan = build_oracle_utility_weights({11: 0.0, 12: math.log(20.0), 13: 100.0})

    assert plan.raw_weights == {11: 1.0, 12: 20.0, 13: 20.0}
    assert plan.weights[11] == pytest.approx(3.0 / 41.0)
    assert plan.weights[12] == pytest.approx(60.0 / 41.0)
    assert plan.weights[13] == pytest.approx(60.0 / 41.0)
    assert sum(plan.weights.values()) / len(plan.weights) == pytest.approx(1.0, abs=1e-12)
    assert plan.diagnostics.raw_mean == pytest.approx(41.0 / 3.0)
    assert plan.diagnostics.normalized_mean == pytest.approx(1.0, abs=1e-12)
    assert plan.diagnostics.saturation_fraction == pytest.approx(2.0 / 3.0)
    assert plan.diagnostics.effective_sample_size == pytest.approx(1681.0 / 801.0)


def test_weight_builder_rejects_nonfinite_utility_and_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="finite"):
        build_oracle_utility_weights({11: float("nan")})
    with pytest.raises(ValueError, match="lambda"):
        build_oracle_utility_weights({11: 0.0}, lambda_=0.0)
    with pytest.raises(ValueError, match="weight_max"):
        build_oracle_utility_weights({11: 0.0}, weight_max=0.0)


def test_deterministic_weight_permutation_preserves_ids_and_exact_multiset() -> None:
    weights = {11: 0.1, 12: 0.2, 13: 0.2, 14: 1.7}

    first = deterministic_weight_permutation(weights, seed=2024)
    second = deterministic_weight_permutation(weights, seed=2024)

    assert first == second
    assert tuple(first) == tuple(weights)
    assert sorted(first.values()) == sorted(weights.values())


def test_budget_assertion_requires_exact_u_w_f_s_counters() -> None:
    matched = BudgetCounters(
        image_exposures=2_500,
        logical_batches=320,
        optimizer_steps=320,
        scheduler_steps=0,
    )
    assert_equal_budget({"U": matched, "W": matched, "F": matched, "S": matched})

    with pytest.raises(AssertionError, match="optimizer_steps"):
        assert_equal_budget(
            {
                "U": matched,
                "W": matched,
                "F": matched,
                "S": BudgetCounters(2_500, 320, 319, 0),
            }
        )
    with pytest.raises(AssertionError, match="missing"):
        assert_equal_budget({"U": matched, "W": matched, "F": matched})
