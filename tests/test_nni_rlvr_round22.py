from spectral_detection_posttrain.experiments.nni_rlvr_trial import compute_round22_objective, expand_preset


def test_expand_round22_preset_preserves_new_controls():
    params = {
        "preset": {
            "name": "signed_iou_001",
            "signal": "none",
            "policy_loss_weight": 0.001,
            "det_loss_weight": 0.0,
            "baseline_kl_weight": 1.0,
            "rollout_source": "baseline",
            "policy_objective": "signed",
        }
    }
    expanded = expand_preset(params)

    assert expanded["det_loss_weight"] == 0.0
    assert expanded["baseline_kl_weight"] == 1.0
    assert expanded["rollout_source"] == "baseline"
    assert expanded["policy_objective"] == "signed"


def test_round22_objective_rejects_prediction_explosion():
    baseline = {
        "clean": {"ap50": 0.88, "ap75": 0.64, "recall": 0.90, "precision": 0.70, "high_conf_fp_count": 2, "num_predictions": 122},
        "object_edge_checkerboard": {"ap50": 0.86, "ap75": 0.50, "recall": 0.88, "precision": 0.67, "high_conf_fp_count": 4, "num_predictions": 125},
    }
    metrics = {
        "clean": {"ap50": 0.84, "ap75": 0.60, "recall": 0.87, "precision": 0.30, "high_conf_fp_count": 2, "num_predictions": 300},
        "object_edge_checkerboard": {"ap50": 0.82, "ap75": 0.47, "recall": 0.84, "precision": 0.31, "high_conf_fp_count": 4, "num_predictions": 290},
    }

    result = compute_round22_objective(metrics, baseline)

    assert result["default"] == -1.0
    assert result["constraint_failed"] == "num_predictions_clean"
