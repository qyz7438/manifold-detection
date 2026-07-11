from __future__ import annotations

import torch

from spectral_detection_posttrain.methods.energy_transport.dense_set_energy import (
    DenseTeacherConfig,
    dense_teacher_components,
    robust_scalar_summary,
)


def _prediction(boxes, scores, labels):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _target(boxes, labels):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def test_coverage_is_continuous_and_monotone_near_iou_threshold():
    target = _target([[0, 0, 10, 10]], [1])
    below = dense_teacher_components(
        _prediction([[0, 0, 10, 7.49]], [0.9], [1]), target
    )
    above = dense_teacher_components(
        _prediction([[0, 0, 10, 7.51]], [0.9], [1]), target
    )

    assert above.coverage > below.coverage
    assert float(above.coverage - below.coverage) < 0.1


def test_wrong_class_and_background_risks_are_separate():
    target = _target([[0, 0, 10, 10]], [1])
    wrong_class = dense_teacher_components(
        _prediction([[0, 0, 10, 10]], [0.9], [2]), target
    )
    background = dense_teacher_components(
        _prediction([[20, 20, 30, 30]], [0.9], [1]), target
    )

    assert wrong_class.class_risk > wrong_class.background_risk
    assert background.background_risk > background.class_risk


def test_duplicate_energy_is_permutation_invariant():
    target = _target([[0, 0, 10, 10]], [1])
    first = _prediction(
        [[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]],
        [0.9, 0.8, 0.7],
        [1, 1, 2],
    )
    permutation = torch.tensor([2, 0, 1])
    second = {key: value[permutation] for key, value in first.items()}

    left = dense_teacher_components(first, target)
    right = dense_teacher_components(second, target)

    assert torch.allclose(left.duplicate_risk, right.duplicate_risk)
    assert left.duplicate_edge_count == right.duplicate_edge_count == 1


def test_empty_prediction_and_target_sets_are_finite():
    config = DenseTeacherConfig()
    empty_prediction = _prediction([], [], [])
    empty_target = _target([], [])

    no_predictions = dense_teacher_components(empty_prediction, _target([[0, 0, 1, 1]], [1]), config)
    no_targets = dense_teacher_components(_prediction([[0, 0, 1, 1]], [0.8], [1]), empty_target, config)
    both_empty = dense_teacher_components(empty_prediction, empty_target, config)

    for result in (no_predictions, no_targets, both_empty):
        values = torch.stack(
            (
                result.coverage,
                result.background_risk,
                result.class_risk,
                result.duplicate_risk,
                result.calibration_error,
            )
        )
        assert torch.isfinite(values).all()
    assert no_predictions.coverage == 0
    assert no_targets.background_risk > 0
    assert both_empty.prediction_count == both_empty.ground_truth_count == 0


def test_robust_summary_exposes_iqr_degeneracy():
    spread = robust_scalar_summary([0.0, 1.0, 2.0, 3.0, 4.0], min_iqr=1e-6)
    constant = robust_scalar_summary([2.0, 2.0, 2.0], min_iqr=1e-6)

    assert spread["median"] == 2.0
    assert spread["q25"] == 1.0
    assert spread["q75"] == 3.0
    assert spread["iqr"] == 2.0
    assert spread["degenerate"] is False
    assert constant["iqr"] == 0.0
    assert constant["degenerate"] is True
