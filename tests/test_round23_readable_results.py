from spectral_detection_posttrain.experiments.nni_rlvr_trial import (
    REQUIRED_ROUND23_RESULT_FIELDS,
    build_round23_result_row,
    validate_expected_presets,
)


def test_round23_result_row_contains_identity_params_and_metrics():
    params = {
        "name": "signed_iou_0003_kl10",
        "signal": "none",
        "reward_lambda": 0.0,
        "policy_loss_weight": 0.0003,
        "det_loss_weight": 0.0,
        "baseline_kl_weight": 10.0,
        "box_loss_weight": 0.0,
        "unfreeze": "cls",
        "optimizer": "adamw",
        "temperature": 1.0,
        "max_candidates": 40,
        "reward_score_threshold": 0.2,
        "rollout_source": "baseline",
        "policy_objective": "signed",
    }
    metrics = {
        "clean": {
            "ap50": 0.85, "ap75": 0.60, "precision": 0.66, "recall": 0.87,
            "num_predictions": 120, "high_conf_fp_count": 2, "ece": 0.06,
        },
        "object_edge_checkerboard": {
            "ap50": 0.84, "ap75": 0.55, "precision": 0.63, "recall": 0.86,
            "num_predictions": 130, "high_conf_fp_count": 4, "ece": 0.07,
        },
    }
    row = build_round23_result_row(
        params=params, metrics=metrics,
        objective={"default": 2.1, "constraint_failed": ""},
        run_name="nni_rlvr_round23/rlvr_signed_iou",
        checkpoint="runs/x/checkpoint_best.pth",
        eval_status="ok",
    )

    for field in REQUIRED_ROUND23_RESULT_FIELDS:
        assert field in row
    assert row["name"] == "signed_iou_0003_kl10"
    assert row["clean_num_predictions"] == 120
    assert row["edge_ap50"] == 0.84


def test_round23_result_row_survives_missing_eval():
    params = {"name": "broken", "signal": "none"}
    row = build_round23_result_row(
        params=params, metrics={},
        objective={"default": -1.0, "constraint_failed": "eval_missing"},
        run_name="runs/broken", checkpoint="", eval_status="failed",
    )

    assert row["name"] == "broken"
    assert row["eval_status"] == "failed"
    assert row["clean_ap50"] is None


def test_validate_expected_presets_detects_missing_name():
    expected = ["null_no_update", "det_only_cls", "signed_iou"]
    rows = [{"name": "null_no_update"}, {"name": "signed_iou"}]

    result = validate_expected_presets(expected, rows)

    assert result["missing"] == ["det_only_cls"]
    assert result["complete"] is False
