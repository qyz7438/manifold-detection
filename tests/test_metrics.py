import torch

from mfvpt.utils.metrics import (
    accuracy,
    expected_calibration_error,
    high_confidence_error_rate,
    prediction_consistency,
)


def test_accuracy_perfect_and_wrong():
    labels = torch.tensor([0, 1])
    perfect = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    wrong = torch.tensor([[0.0, 10.0], [10.0, 0.0]])
    assert accuracy(perfect, labels) == 1.0
    assert accuracy(wrong, labels) == 0.0


def test_prediction_consistency():
    clean = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    same = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
    flipped = torch.tensor([[0.0, 5.0], [5.0, 0.0]])
    assert prediction_consistency(clean, same) == 1.0
    assert prediction_consistency(clean, flipped) == 0.0


def test_high_confidence_error_rate():
    labels = torch.tensor([1, 0])
    logits = torch.tensor([[10.0, 0.0], [10.0, 0.0]])
    assert high_confidence_error_rate(logits, labels, threshold=0.9) == 0.5


def test_ece_returns_float_and_is_small_for_near_perfect():
    labels = torch.tensor([0, 1])
    logits = torch.tensor([[100.0, 0.0], [0.0, 100.0]])
    ece = expected_calibration_error(logits, labels, n_bins=10)
    assert isinstance(ece, float)
    assert ece < 1e-4
