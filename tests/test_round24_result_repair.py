from spectral_detection_posttrain.experiments.nni_rlvr_trial import (
    build_round23_result_row,
    collect_eval_status,
)


def test_collect_eval_status_ok_when_clean_and_edge_exist():
    metrics = {
        "clean": {"ap50": 0.87},
        "object_edge_checkerboard": {"ap50": 0.86},
    }
    assert collect_eval_status(metrics) == "ok"


def test_collect_eval_status_names_missing_clean():
    metrics = {"object_edge_checkerboard": {"ap50": 0.86}}
    assert collect_eval_status(metrics) == "missing_clean"


def test_repaired_row_keeps_name_even_when_eval_missing():
    row = build_round23_result_row(
        params={"name": "signed_ramp_0003_kl10", "signal": "ramp"},
        metrics={"object_edge_checkerboard": {"ap50": 0.87}},
        objective={"default": -1.0, "constraint_failed": "missing_clean"},
        run_name="rlvr_signed_ramp_0003_kl10_cls_adamw",
        checkpoint="runs/x/checkpoint_best.pth",
        eval_status="missing_clean",
    )

    assert row["name"] == "signed_ramp_0003_kl10"
    assert row["eval_status"] == "missing_clean"
    assert row["clean_ap50"] is None
    assert row["edge_ap50"] == 0.87
