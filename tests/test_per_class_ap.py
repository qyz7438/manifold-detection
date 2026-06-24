from __future__ import annotations

import pytest
import torch

from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions


def test_per_class_ap_distinguishes_classes() -> None:
    """Class 1 predictions are perfect, class 2 predictions miss."""
    predictions = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
            "scores": torch.tensor([0.9, 0.9]),
            "labels": torch.tensor([1, 2]),
        }
    ]
    targets = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
            "labels": torch.tensor([1, 2]),
        }
    ]

    metrics = evaluate_detection_predictions(
        predictions, targets, per_class=True, num_classes=3
    )

    assert metrics["per_class_ap50"]["1"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["per_class_ap50"]["2"] == pytest.approx(1.0, abs=1e-4)


def test_per_class_ap_punishes_class_confusion() -> None:
    predictions = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([2]),  # predicted as class 2
        }
    ]
    targets = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "labels": torch.tensor([1]),  # GT is class 1
        }
    ]

    metrics = evaluate_detection_predictions(
        predictions, targets, per_class=True, num_classes=3
    )

    assert metrics["per_class_ap50"]["1"] == pytest.approx(0.0, abs=1e-4)
    assert metrics["per_class_ap50"]["2"] == pytest.approx(0.0, abs=1e-4)
