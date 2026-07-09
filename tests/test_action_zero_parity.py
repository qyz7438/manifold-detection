from __future__ import annotations

import sys

from scripts import train_energy_transport_action


def _metrics(*, ap50: float, ap75: float, predictions: int) -> dict:
    return {
        "ap50": ap50,
        "ap75": ap75,
        "precision": 0.4,
        "recall": 0.6,
        "false_positive_rate": 0.6,
        "ece": 0.04,
        "num_predictions": predictions,
    }


def test_zero_action_parity_passes_within_metric_and_count_tolerances() -> None:
    assert hasattr(train_energy_transport_action, "summarize_zero_action_parity")
    default = _metrics(ap50=0.50, ap75=0.30, predictions=1000)
    initial = _metrics(ap50=0.5005, ap75=0.2995, predictions=1004)

    result = train_energy_transport_action.summarize_zero_action_parity(default, initial)

    assert result["passed"] is True
    assert result["deltas"]["ap50"] == 0.0005
    assert result["prediction_relative_delta"] == 0.004


def test_zero_action_parity_fails_for_postprocessing_metric_shift() -> None:
    assert hasattr(train_energy_transport_action, "summarize_zero_action_parity")
    default = _metrics(ap50=0.50, ap75=0.30, predictions=1000)
    initial = _metrics(ap50=0.51, ap75=0.29, predictions=1040)

    result = train_energy_transport_action.summarize_zero_action_parity(default, initial)

    assert result["passed"] is False
    assert set(result["failed_checks"]) == {"ap50", "ap75", "num_predictions"}


def test_action_cli_accepts_reproducible_parity_controls(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_energy_transport_action.py",
            "--run-name",
            "parity",
            "--data-seed",
            "7",
            "--require-clean-git",
            "--parity-ap-tolerance",
            "0.002",
            "--parity-prediction-relative-tolerance",
            "0.01",
        ],
    )

    args = train_energy_transport_action.parse_args()

    assert args.data_seed == 7
    assert args.require_clean_git is True
    assert args.parity_ap_tolerance == 0.002
    assert args.parity_prediction_relative_tolerance == 0.01
