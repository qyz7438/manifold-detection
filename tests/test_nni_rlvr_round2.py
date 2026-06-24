from spectral_detection_posttrain.experiments.nni_rlvr_trial import compute_round2_objective, expand_preset


def test_expand_preset_returns_single_trial_dict():
    params = {"preset": {"name": "ramp_mid", "signal": "ramp", "reward_lambda": 0.1, "policy_loss_weight": 0.3}}
    expanded = expand_preset(params)

    assert expanded["name"] == "ramp_mid"
    assert expanded["signal"] == "ramp"
    assert expanded["reward_lambda"] == 0.1


def test_compute_round2_objective_rejects_ap50_collapse():
    baseline = {
        "clean": {"ap50": 0.86, "ap75": 0.62, "recall": 0.88, "ece": 0.06},
        "object_edge_checkerboard": {"ap50": 0.84, "ap75": 0.55, "recall": 0.86, "ece": 0.05},
    }
    metrics = {
        "clean": {"ap50": 0.70, "ap75": 0.50, "recall": 0.80, "ece": 0.04},
        "object_edge_checkerboard": {"ap50": 0.83, "ap75": 0.57, "recall": 0.86, "ece": 0.04},
    }

    result = compute_round2_objective(metrics, baseline)
    assert result["default"] == -1.0
    assert result["constraint_failed"] == "ap50_clean"


def test_compute_round2_objective_passes_with_good_metrics():
    baseline = {
        "clean": {"ap50": 0.86, "ap75": 0.62, "recall": 0.88, "ece": 0.06},
        "object_edge_checkerboard": {"ap50": 0.84, "ap75": 0.55, "recall": 0.86, "ece": 0.05},
    }
    metrics = {
        "clean": {"ap50": 0.85, "ap75": 0.60, "recall": 0.87, "ece": 0.05},
        "object_edge_checkerboard": {"ap50": 0.82, "ap75": 0.54, "recall": 0.85, "ece": 0.04},
    }

    result = compute_round2_objective(metrics, baseline)
    assert result["default"] > 0.0
    assert result["constraint_failed"] == ""
