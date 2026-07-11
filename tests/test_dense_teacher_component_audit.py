from __future__ import annotations

from scripts.analyze_dense_teacher_components import load_config, summarize_rows


def test_dense_teacher_stats_config_is_locked_to_inner_fit():
    config = load_config()

    assert config["dataset"]["split"] == "inner_fit"
    assert config["dataset"]["forbidden_splits"] == [
        "inner_tune",
        "outer_train_heldout",
        "detector_validation",
    ]
    assert config["teacher"]["component_weights"] == "not_fit_in_this_audit"


def test_summarize_rows_detects_duplicate_support_failure_only():
    rows = [
        {
            "coverage": float(index),
            "background_risk": float(index + 1),
            "class_risk": float(index + 2),
            "duplicate_risk": 0.0,
            "calibration_error": float(index + 3),
            "prediction_count": 2,
            "ground_truth_count": 1,
            "duplicate_edge_count": 0,
        }
        for index in range(4)
    ]
    config = load_config()
    config["gates"]["exact_image_count"] = 4
    config["gates"]["min_images_with_predictions"] = 4
    config["gates"]["min_images_with_duplicate_edges"] = 1

    summary = summarize_rows(rows, config)

    assert summary["gates"]["required_components_non_degenerate"] is True
    assert summary["gates"]["duplicate_support"] is False
    assert summary["decision"] == "keep_unary_move_duplicate_to_prenms"
