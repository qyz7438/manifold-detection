from __future__ import annotations

import pytest
import torch

from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.eval.paired_bootstrap import (
    hierarchical_paired_bootstrap_ap75,
    paired_bootstrap_ap75,
)


def _target() -> dict[str, torch.Tensor]:
    return {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "labels": torch.tensor([1]),
    }


def _true_positive(score: float) -> dict[str, torch.Tensor]:
    return {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "scores": torch.tensor([score]),
        "labels": torch.tensor([1]),
    }


def _false_positive_then_true_positive() -> dict[str, torch.Tensor]:
    return {
        "boxes": torch.tensor(
            [[20.0, 20.0, 30.0, 30.0], [0.0, 0.0, 10.0, 10.0]]
        ),
        "scores": torch.tensor([0.95, 0.70]),
        "labels": torch.tensor([1, 1]),
    }


def _empty_prediction() -> dict[str, torch.Tensor]:
    return {
        "boxes": torch.empty((0, 4)),
        "scores": torch.empty((0,)),
        "labels": torch.empty((0,), dtype=torch.long),
    }


def _three_image_fixture() -> tuple[
    dict[int, dict[str, torch.Tensor]],
    dict[int, dict[str, torch.Tensor]],
    dict[int, dict[str, torch.Tensor]],
]:
    predictions = {
        0: _true_positive(0.90),
        1: _false_positive_then_true_positive(),
        2: _false_positive_then_true_positive(),
    }
    targets = {image_id: _target() for image_id in predictions}
    return predictions, predictions.copy(), targets


def test_paired_bootstrap_rejects_empty_or_mismatched_image_ids() -> None:
    prediction = {"image": _true_positive(0.9)}
    target = {"image": _target()}

    with pytest.raises(ValueError, match="non-empty"):
        paired_bootstrap_ap75({}, prediction, target, n_resamples=2)

    with pytest.raises(ValueError, match="same image IDs"):
        paired_bootstrap_ap75(prediction, {"other": _true_positive(0.9)}, target, n_resamples=2)

    with pytest.raises(ValueError, match="same image IDs"):
        paired_bootstrap_ap75(prediction, prediction, {"other": _target()}, n_resamples=2)


def test_paired_bootstrap_estimate_is_global_ap75_not_mean_per_image_ap() -> None:
    predictions_a = {
        "easy": _true_positive(0.90),
        "hard": _false_positive_then_true_positive(),
    }
    predictions_b = {image_id: _empty_prediction() for image_id in predictions_a}
    targets = {image_id: _target() for image_id in predictions_a}

    result = paired_bootstrap_ap75(
        predictions_a, predictions_b, targets, n_resamples=8, seed=42
    )
    expected_global = evaluate_detection_predictions(
        list(predictions_a.values()),
        list(targets.values()),
        iou_threshold=0.75,
        score_threshold=0.05,
    )["ap75"]
    mean_per_image = sum(
        evaluate_detection_predictions([prediction], [targets[image_id]], iou_threshold=0.75)["ap75"]
        for image_id, prediction in predictions_a.items()
    ) / len(predictions_a)

    assert expected_global == pytest.approx(2.0 / 3.0)
    assert mean_per_image == pytest.approx(0.75)
    assert result.estimate == pytest.approx(expected_global)
    assert result.estimate != pytest.approx(mean_per_image)


def test_paired_bootstrap_uses_identical_indices_and_is_seeded() -> None:
    predictions_a, predictions_b, targets = _three_image_fixture()

    first = paired_bootstrap_ap75(predictions_a, predictions_b, targets, n_resamples=32, seed=7)
    second = paired_bootstrap_ap75(predictions_a, predictions_b, targets, n_resamples=32, seed=7)

    assert first == second
    assert first.estimate == pytest.approx(0.0)
    assert first.ci_low == pytest.approx(first.ci_high)
    assert first.resamples == 32
    assert first.n_images == 3


def test_paired_bootstrap_counts_duplicate_sampled_images_as_independent_images() -> None:
    predictions_a, predictions_b, targets = _three_image_fixture()
    predictions_b = {image_id: _empty_prediction() for image_id in predictions_b}
    sampled_indices = torch.tensor([2, 0, 2])
    expected = evaluate_detection_predictions(
        [predictions_a[index] for index in sampled_indices.tolist()],
        [targets[index] for index in sampled_indices.tolist()],
        iou_threshold=0.75,
        score_threshold=0.05,
    )["ap75"]
    deduplicated = evaluate_detection_predictions(
        [predictions_a[2], predictions_a[0]],
        [targets[2], targets[0]],
        iou_threshold=0.75,
        score_threshold=0.05,
    )["ap75"]

    result = paired_bootstrap_ap75(
        predictions_a, predictions_b, targets, n_resamples=1, seed=0
    )

    assert expected != pytest.approx(deduplicated)
    assert result.ci_low == pytest.approx(expected)
    assert result.ci_high == pytest.approx(expected)


def test_hierarchical_bootstrap_rejects_too_few_or_mismatched_seeds() -> None:
    arm_a = {42: {"image": _true_positive(0.9)}}
    arm_b = {42: {"image": _empty_prediction()}}
    targets = {42: {"image": _target()}}

    with pytest.raises(ValueError, match="at least 2 training seeds"):
        hierarchical_paired_bootstrap_ap75(arm_a, arm_b, targets, n_resamples=2)

    arm_a[2024] = {"image": _true_positive(0.9)}
    arm_b[2024] = {"other": _empty_prediction()}
    targets[2024] = {"image": _target()}
    with pytest.raises(ValueError, match="same image IDs.*2024"):
        hierarchical_paired_bootstrap_ap75(arm_a, arm_b, targets, n_resamples=2)

    arm_b[2024] = {"image": _empty_prediction()}
    with pytest.raises(ValueError, match="same training seeds"):
        hierarchical_paired_bootstrap_ap75(
            arm_a,
            {42: arm_b[42], 999: arm_b[2024]},
            targets,
            n_resamples=2,
        )


def test_hierarchical_bootstrap_is_deterministic_and_captures_seed_variance() -> None:
    targets = {
        training_seed: {image_id: _target() for image_id in ("a", "b")}
        for training_seed in (42, 2024)
    }
    arm_a = {
        42: {image_id: _true_positive(0.9) for image_id in ("a", "b")},
        2024: {image_id: _empty_prediction() for image_id in ("a", "b")},
    }
    arm_b = {
        training_seed: {image_id: _empty_prediction() for image_id in ("a", "b")}
        for training_seed in (42, 2024)
    }

    first = hierarchical_paired_bootstrap_ap75(
        arm_a, arm_b, targets, n_resamples=256, seed=17
    )
    second = hierarchical_paired_bootstrap_ap75(
        arm_a, arm_b, targets, n_resamples=256, seed=17
    )

    assert first == second
    assert first.estimate == pytest.approx(0.5)
    assert first.ci_low == pytest.approx(0.0)
    assert first.ci_high == pytest.approx(1.0)
    assert first.ci_low < first.estimate < first.ci_high
    assert first.n_seeds == 2
    assert first.resamples == 256


def test_hierarchical_bootstrap_recomputes_global_ap_for_each_selected_seed() -> None:
    targets = {
        training_seed: {image_id: _target() for image_id in ("easy", "hard")}
        for training_seed in (11, 22)
    }
    arm_a = {
        11: {
            "easy": _true_positive(0.90),
            "hard": _false_positive_then_true_positive(),
        },
        22: {image_id: _empty_prediction() for image_id in ("easy", "hard")},
    }
    arm_b = {
        11: {image_id: _empty_prediction() for image_id in ("easy", "hard")},
        22: {image_id: _true_positive(0.90) for image_id in ("easy", "hard")},
    }

    result = hierarchical_paired_bootstrap_ap75(
        arm_a, arm_b, targets, n_resamples=1, seed=0
    )

    # Seed 0 selects positions [hard, easy], giving global AP75 2/3.
    # Seed 1 selects [hard, hard], giving an arm-A-minus-arm-B delta of -1.
    expected_nested_delta = ((2.0 / 3.0) - 1.0) / 2.0
    mean_per_image_delta = (0.75 - 1.0) / 2.0
    assert result.ci_low == pytest.approx(expected_nested_delta)
    assert result.ci_high == pytest.approx(expected_nested_delta)
    assert result.ci_low != pytest.approx(mean_per_image_delta)
