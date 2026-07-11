from __future__ import annotations

import torch

from scripts.analyze_dense_teacher_confounding import (
    load_config,
    residualized_correlation,
    support_normalized_rows,
)


def test_confounding_config_locks_artifact_only_boundary():
    config = load_config()

    assert config["source"]["forbidden_new_data"] is True
    assert config["analysis"]["max_abs_penalty_correlation"] == 0.85


def test_residualized_correlation_removes_shared_count_effect():
    count = torch.arange(1, 21, dtype=torch.float64)
    controls = torch.stack((count, torch.ones_like(count)), dim=1)
    alternating = torch.tensor([1.0, -1.0] * 10, dtype=torch.float64)
    values = torch.stack((2.0 * count + alternating, 3.0 * count - alternating), dim=1)

    raw = torch.corrcoef(values.T)[0, 1]
    residual = residualized_correlation(values, controls)

    assert raw > 0.95
    assert residual[0][1] < -0.99


def test_support_normalization_uses_natural_component_denominators():
    rows = [
        {
            "coverage": 8.0,
            "background_risk": 6.0,
            "class_risk": 3.0,
            "duplicate_risk": 2.0,
            "calibration_error": 4.0,
            "prediction_count": 2,
            "ground_truth_count": 4,
            "duplicate_edge_count": 4,
        }
    ]

    normalized = support_normalized_rows(rows)[0]

    assert normalized["coverage_per_gt"] == 2.0
    assert normalized["background_risk_per_prediction"] == 3.0
    assert normalized["class_risk_per_prediction"] == 1.5
    assert normalized["calibration_error_per_prediction"] == 2.0
    assert normalized["duplicate_risk_per_edge"] == 0.5
