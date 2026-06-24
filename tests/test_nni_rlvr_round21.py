from spectral_detection_posttrain.experiments.nni_rlvr_trial import compute_round2_objective, expand_preset


def test_preset_expands_unfreeze_cls():
    params = {"preset": {"name": "ramp_cls_01", "signal": "ramp", "reward_lambda": 0.1,
                          "policy_loss_weight": 0.01, "box_loss_weight": 0.0,
                          "unfreeze": "cls", "optimizer": "adamw",
                          "temperature": 1.0, "max_candidates": 40,
                          "reward_score_threshold": 0.2}}
    expanded = expand_preset(params)
    assert expanded["unfreeze"] == "cls"
    assert expanded["box_loss_weight"] == 0.0
    assert expanded["reward_score_threshold"] == 0.2
    assert expanded["max_candidates"] == 40


def test_objective_rejects_edge_ap50_collapse():
    baseline = {
        "clean": {"ap50": 0.86, "ap75": 0.62, "recall": 0.88, "ece": 0.06},
        "object_edge_checkerboard": {"ap50": 0.84, "ap75": 0.55, "recall": 0.86, "ece": 0.05},
    }
    # clean passes but edge fails
    metrics = {
        "clean": {"ap50": 0.85, "ap75": 0.60, "recall": 0.87, "ece": 0.05},
        "object_edge_checkerboard": {"ap50": 0.70, "ap75": 0.40, "recall": 0.80, "ece": 0.04},
    }
    result = compute_round2_objective(metrics, baseline)
    assert result["default"] == -1.0
    assert result["constraint_failed"] == "ap50_object_edge_checkerboard"
