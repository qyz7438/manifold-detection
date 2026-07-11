from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_nwpu_joint_delta_u_fit_replay.py"
LAUNCHER = ROOT / "scripts" / "experiments" / "run_nwpu_joint_delta_u_fit_replay_s42.sh"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_nwpu_joint_delta_u_fit_replay", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raw(*, pairwise: float, weighted: float, mae: float, sign: float) -> dict:
    return {
        "pairwise_accuracy": pairwise,
        "pairwise_accuracy_candidate_weighted": weighted,
        "mae": mae,
        "sign_accuracy": sign,
    }


def test_fit_replay_distinguishes_learned_fit_signal_from_heldout_failure() -> None:
    module = _load_module()

    summary = module.classify_fit_replay(
        fit_raw=_raw(pairwise=0.72, weighted=0.69, mae=0.10, sign=0.91),
        fit_calibration={"used_identity_fallback": False},
        fit_baselines={
            "zero_utility_pairwise_accuracy": 0.5,
            "zero_utility_mae": 0.18,
            "always_negative_sign_accuracy": 0.86,
        },
        calibration_raw=_raw(pairwise=0.54, weighted=0.53, mae=0.20, sign=0.52),
        validation_raw=_raw(pairwise=0.51, weighted=0.50, mae=0.21, sign=0.51),
    )

    assert summary["diagnosis"] == "fit_signal_learned_but_heldout_failed"
    assert summary["fit_threshold_feasible"] is True
    assert summary["fit_pairwise_gain_over_zero"] == pytest.approx(0.22)
    assert summary["fit_mae_gain_over_zero"] == pytest.approx(0.08)
    assert summary["fit_sign_gain_over_always_negative"] == pytest.approx(0.05)
    assert summary["fit_to_validation_pairwise_gap"] == pytest.approx(0.21)


def test_fit_replay_rejects_loss_only_learning_without_metric_signal() -> None:
    module = _load_module()

    summary = module.classify_fit_replay(
        fit_raw=_raw(pairwise=0.53, weighted=0.52, mae=0.24, sign=0.55),
        fit_calibration={"used_identity_fallback": True},
        fit_baselines={
            "zero_utility_pairwise_accuracy": 0.5,
            "zero_utility_mae": 0.16,
            "always_negative_sign_accuracy": 0.88,
        },
        calibration_raw=_raw(pairwise=0.52, weighted=0.51, mae=0.23, sign=0.54),
        validation_raw=_raw(pairwise=0.50, weighted=0.50, mae=0.22, sign=0.52),
    )

    assert summary["diagnosis"] == "fit_signal_not_learned"
    assert summary["fit_threshold_feasible"] is False
    assert summary["fit_mae_gain_over_zero"] < 0.0


def test_unconditional_top1_keeps_negative_true_delta_u() -> None:
    module = _load_module()

    metrics = module.unconditional_top1_metrics(
        predicted=torch.tensor([0.4, 0.2, -0.1, -0.2]),
        target=torch.tensor([-0.6, 0.3, 0.2, 0.1]),
        image_ids=torch.tensor([1, 1, 2, 2]),
        target_epsilon=1e-3,
        lcb_z=1.645,
    )

    assert metrics["selected_count"] == 2
    assert metrics["mean_delta_u_per_image"] == pytest.approx(-0.2)
    assert metrics["mean_delta_u_selected"] == pytest.approx(-0.2)


def test_replay_script_is_posthoc_and_does_not_retrain_or_revalidate() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "train_joint_arm(" not in source
    assert "build_probe_cache_records(" not in source
    assert '"posthoc_fit_replay"' in source
    assert '"no_retraining": True' in source
    assert '"clean_validation_reused_from_artifact": True' in source
    assert '"calibration_frozen_threshold"' in source
    assert '"clean_validation_frozen_threshold"' in source


def test_fit_replay_launcher_is_gpu2_only_and_checks_free_memory() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "CUDA_VISIBLE_DEVICES=2" in source
    assert "memory.free" in source
    assert "FREE_MB <= 8192" in source
    assert "analyze_nwpu_joint_delta_u_fit_replay.py" in source
    assert "--require-clean-git" in source
    assert "train_joint_arm" not in source
