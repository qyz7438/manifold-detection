from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts import train_energy_transport_action


PARITY_LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "experiments"
    / "run_action_zero_parity_control.sh"
)
STRONG_ACTION_MATRIX = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "experiments"
    / "run_nwpu_strong_native_action_matrix_s42.sh"
)


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
    assert args.postprocess_mode == "native"


def test_parity_launcher_waits_for_c0_and_uses_gpu2_memory_gate() -> None:
    assert PARITY_LAUNCHER.exists()
    launcher = PARITY_LAUNCHER.read_text(encoding="utf-8")

    assert 'GPU_ID="${GPU_ID:-2}"' in launcher
    assert 'MIN_FREE_MB="${MIN_FREE_MB:-8192}"' in launcher
    assert '[[ "${GPU_ID}" != "2" ]]' in launcher
    assert '"${free_mb}" -gt "${MIN_FREE_MB}"' in launcher
    assert "ctrl_full_ft18_from12best_fullnw0_nwpu_s42_bs8_ep18" in launcher
    assert '--epochs 0' in launcher
    assert '--action-score-threshold 0.05' in launcher
    assert '--detections-per-img 100' in launcher
    assert 'RUN_TAG="${RUN_TAG:-nativefix}"' in launcher
    assert '--postprocess-mode native' in launcher


def test_strict_output_parity_requires_per_image_identity() -> None:
    assert hasattr(train_energy_transport_action, "summarize_strict_output_parity")
    native = [
        {
            "boxes": train_energy_transport_action.torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            "scores": train_energy_transport_action.torch.tensor([0.8]),
            "labels": train_energy_transport_action.torch.tensor([2]),
        }
    ]
    equal = [{key: value.clone() for key, value in native[0].items()}]
    shifted = [{key: value.clone() for key, value in native[0].items()}]
    shifted[0]["boxes"][0, 0] += 0.01

    passed = train_energy_transport_action.summarize_strict_output_parity(native, equal)
    failed = train_energy_transport_action.summarize_strict_output_parity(native, shifted)

    assert passed["passed"] is True
    assert passed["mismatched_images"] == 0
    assert failed["passed"] is False
    assert failed["max_box_abs_error"] == pytest.approx(0.01)


def test_strong_action_matrix_requires_strict_parity_gate() -> None:
    assert STRONG_ACTION_MATRIX.exists()
    launcher = STRONG_ACTION_MATRIX.read_text(encoding="utf-8")

    assert "nwpu_mob_strong_cosine_s42_bs8_36ep" in launcher
    assert "det_action_zero_parity_nativefix_fullft18best_s42" in launcher
    assert 'strict_zero_action_parity' in launcher
    assert '"passed"' in launcher
    assert "ctrl_strong_full_ft4_nwpu_s42" in launcher
    assert 'run_action "${strong_checkpoint}" "boxonly" 0.0' in launcher
    assert 'run_action "${strong_checkpoint}" "preserve2" 2.0' in launcher
    assert '--postprocess-mode native' in launcher
