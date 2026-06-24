import pytest

from spectral_detection_posttrain.analysis.round30_results import (
    FOUR_SCENES,
    build_round30_result_row,
    compute_pair_delta,
    scene_metric_key,
)


def _metrics(ap75: float) -> dict:
    return {
        "ap50": 0.88, "ap75": ap75, "precision": 0.63, "recall": 0.90,
        "ece": 0.04, "high_conf_fp_count": 2, "high_conf_fp_rate": 0.03,
        "num_predictions": 130,
    }


def test_scene_metric_key_uses_short_prefixes():
    assert scene_metric_key("object_edge_checkerboard", "ap75") == "edge_ap75"
    assert scene_metric_key("object_inside_checkerboard", "ap75") == "inside_ap75"
    assert scene_metric_key("near_object_checkerboard", "ap75") == "near_ap75"


def test_build_round30_result_row_contains_all_four_scenes():
    metrics = {scene: _metrics(0.6 + idx * 0.01) for idx, scene in enumerate(FOUR_SCENES)}
    params = {"name": "signed_amp_l0p1", "signal": "ramp", "reward_lambda": 0.1,
              "struct_weight": 0.0, "policy_loss_weight": 0.0003,
              "baseline_kl_weight": 10.0, "seed": 42}
    row = build_round30_result_row(
        params=params, metrics=metrics, objective={"default": 3.0, "constraint_failed": ""},
        run_name="rlvr_x", checkpoint="runs/x/checkpoint_best.pth", eval_status="ok",
    )
    assert row["clean_ap75"] == pytest.approx(0.60)
    assert row["edge_ap75"] == pytest.approx(0.61)
    assert row["inside_ap75"] == pytest.approx(0.62)
    assert row["near_ap75"] == pytest.approx(0.63)
    assert row["seed"] == 42


def test_compute_pair_delta_averages_scene_deltas():
    real = {"clean_ap75": 0.66, "edge_ap75": 0.58, "inside_ap75": 0.67, "near_ap75": 0.61}
    shuffled = {"clean_ap75": 0.64, "edge_ap75": 0.56, "inside_ap75": 0.65, "near_ap75": 0.60}
    delta = compute_pair_delta(real, shuffled, metric="ap75")
    assert delta["mean_delta"] == pytest.approx((0.02 + 0.02 + 0.02 + 0.01) / 4.0)
    assert delta["positive_scene_count"] == 4
