from __future__ import annotations

import pytest

from spectral_detection_posttrain.experiments.nni_raw_ifft_posttrain_trial import (
    build_objective,
    build_round2129_command,
    expand_preset,
)


def test_expand_preset_unwraps_nni_choice_payload():
    params = {"preset": {"name": "bbox01", "bbox_rescue_loss_weight": 0.1}, "ignored": 1}

    expanded = expand_preset(params)

    assert expanded == {"name": "bbox01", "bbox_rescue_loss_weight": 0.1}


def test_build_round2129_command_uses_15_epochs_and_raw_ifft_defaults():
    params = {
        "name": "three_bbox01",
        "feature_set": "three",
        "bbox_rescue_loss_weight": 0.1,
        "rescue_loss_weight": 0.005,
        "kl_weight": 0.5,
        "det_loss_weight": 0.1,
        "policy_loss_weight": 0.001,
        "target_precision": 0.8,
        "margin_std_frac": 0.0,
        "bbox_temperature": 0.2,
        "lr": 3e-5,
    }

    command, run_name = build_round2129_command(params, run_prefix="nni_raw_ifft", epochs=15)

    assert run_name == "nni_raw_ifft/three_bbox01"
    assert "--epochs" in command
    assert command[command.index("--epochs") + 1] == "15"
    assert "--rescue-verifier-mode" in command
    assert command[command.index("--rescue-verifier-mode") + 1] == "raw_ifft"
    assert "--rescue-raw-ifft-features" in command
    feature_index = command.index("--rescue-raw-ifft-features")
    assert command[feature_index + 1 : feature_index + 4] == [
        "fft_edge_truncation@64",
        "phase_edge@64",
        "phase_abs_high@11",
    ]
    assert command[command.index("--bbox-rescue-loss-weight") + 1] == "0.1"
    assert command[command.index("--rescue-loss-weight") + 1] == "0.005"


def test_build_round2129_command_passes_predictor_learning_rates():
    params = {
        "name": "predictor_lr_split",
        "feature_set": "three",
        "trainable_mode": "predictor",
        "lr": 3e-5,
        "adapter_lr": 3e-5,
        "predictor_lr": 1e-5,
        "cls_score_lr": 5e-6,
    }

    command, _ = build_round2129_command(params, run_prefix="nni_raw_ifft", epochs=15)

    assert command[command.index("--trainable-mode") + 1] == "predictor"
    assert "--predictor-lr" in command
    assert command[command.index("--predictor-lr") + 1] == "1e-05"
    assert "--cls-score-lr" in command
    assert command[command.index("--cls-score-lr") + 1] == "5e-06"


def test_build_round2129_command_passes_confidence_rescue_extensions():
    params = {
        "name": "conf_ext",
        "feature_set": "three",
        "kl_cls_weight": 0.2,
        "kl_box_weight": 1.0,
        "confidence_crossing_loss_weight": 0.05,
        "confidence_crossing_margin": 0.02,
        "score_budget": True,
        "score_budget_loss_weight": 0.02,
    }

    command, _ = build_round2129_command(params, run_prefix="nni_raw_ifft", epochs=15)

    assert "--kl-cls-weight" in command
    assert command[command.index("--kl-cls-weight") + 1] == "0.2"
    assert "--kl-box-weight" in command
    assert command[command.index("--kl-box-weight") + 1] == "1.0"
    assert "--confidence-crossing-loss-weight" in command
    assert command[command.index("--confidence-crossing-loss-weight") + 1] == "0.05"
    assert "--confidence-crossing-margin" in command
    assert command[command.index("--confidence-crossing-margin") + 1] == "0.02"
    assert command[command.index("--score-budget-loss-weight") + 1] == "0.02"


def test_build_round2129_command_passes_hd_fusion_verifier_options():
    params = {
        "name": "hd_fusion",
        "feature_set": "three",
        "verifier_mode": "raw_ifft_hd_fusion",
        "hd_fusion_pca_components": 96,
        "hd_fusion_hd_scorer": "logistic",
        "hd_fusion_method": "train_effect",
    }

    command, _ = build_round2129_command(params, run_prefix="nni_raw_ifft", epochs=15)

    assert command[command.index("--rescue-verifier-mode") + 1] == "raw_ifft_hd_fusion"
    assert "--rescue-hd-fusion-pca-components" in command
    assert command[command.index("--rescue-hd-fusion-pca-components") + 1] == "96"
    assert "--rescue-hd-fusion-hd-scorer" in command
    assert command[command.index("--rescue-hd-fusion-hd-scorer") + 1] == "logistic"
    assert "--rescue-hd-fusion-method" in command
    assert command[command.index("--rescue-hd-fusion-method") + 1] == "train_effect"


def test_build_round2129_command_passes_scene_verifier_options():
    params = {
        "name": "scene_fft",
        "feature_set": "scene",
        "verifier_mode": "raw_ifft_scene",
        "scene_groups": ["maritime", "vehicle"],
        "scene_target_precision": 0.7,
        "scene_min_positives": 2,
    }

    command, _ = build_round2129_command(params, run_prefix="nni_raw_ifft", epochs=15)

    assert command[command.index("--rescue-verifier-mode") + 1] == "raw_ifft_scene"
    assert "--rescue-raw-ifft-scene-groups" in command
    group_index = command.index("--rescue-raw-ifft-scene-groups")
    assert command[group_index + 1 : group_index + 3] == ["maritime", "vehicle"]
    assert command[command.index("--rescue-raw-ifft-scene-target-precision") + 1] == "0.7"
    assert command[command.index("--rescue-raw-ifft-scene-min-positives") + 1] == "2"


def test_build_objective_rewards_safe_ap75_gain():
    baseline = {"ap50": 0.65, "ap75": 0.29, "false_positive_rate": 0.48, "ece": 0.08, "num_predictions": 1000}
    final = {"ap50": 0.65, "ap75": 0.31, "false_positive_rate": 0.49, "ece": 0.081, "num_predictions": 1010}
    history = {
        "history": [
            {
                "bbox_rescue_count": 10,
                "grad_bbox_rescue_bbox_adapter_l2": 0.01,
                "lchi_conf_delta_mean": 0.04,
                "lchi_conf_cross_score_threshold_count": 3,
                "verifier_positive_lchi_conf_delta_mean": 0.06,
                "verifier_positive_lchi_conf_cross_score_threshold_count": 2,
                "confidence_crossing_count": 4,
                "confidence_crossing_active_count": 1,
            }
        ]
    }

    objective = build_objective(final, baseline, history)

    assert objective["constraint_failed"] == ""
    assert objective["default"] > 0.0
    assert objective["delta_ap75"] == pytest.approx(0.02)
    assert objective["lchi_conf_delta_mean"] == pytest.approx(0.04)
    assert objective["lchi_conf_cross_score_threshold_count"] == 3
    assert objective["verifier_positive_lchi_conf_delta_mean"] == pytest.approx(0.06)
    assert objective["verifier_positive_lchi_conf_cross_score_threshold_count"] == 2
    assert objective["confidence_crossing_count"] == 4
    assert objective["confidence_crossing_active_count"] == 1


def test_build_objective_rejects_fp_rate_regression():
    baseline = {"ap50": 0.65, "ap75": 0.29, "false_positive_rate": 0.48, "ece": 0.08, "num_predictions": 1000}
    final = {"ap50": 0.65, "ap75": 0.32, "false_positive_rate": 0.53, "ece": 0.081, "num_predictions": 1010}
    history = {"history": [{"bbox_rescue_count": 10, "grad_bbox_rescue_bbox_adapter_l2": 0.01}]}

    objective = build_objective(final, baseline, history)

    assert objective["default"] == -1.0
    assert objective["constraint_failed"] == "false_positive_rate"
