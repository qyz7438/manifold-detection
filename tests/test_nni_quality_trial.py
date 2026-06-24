from __future__ import annotations

from spectral_detection_posttrain.experiments.nni_quality_trial import FEATURE_MODE_MAP, _objective
import pytest


def test_feature_mode_map_matches_matrix_names() -> None:
    assert FEATURE_MODE_MAP["ROI-only"] == "roi"
    assert FEATURE_MODE_MAP["ROI+Amp"] == "roi_amp"
    assert FEATURE_MODE_MAP["ROI+Amp+Struct"] == "roi_amp_structure"


def test_objective_uses_requested_metrics() -> None:
    metrics = {
        "ap50": 0.8,
        "precision_at_recall_0_85": 0.7,
        "ece": 0.1,
        "high_conf_fp_rate": 0.05,
    }
    assert _objective(metrics, fixed_recall=0.85) == pytest.approx(1.35)
