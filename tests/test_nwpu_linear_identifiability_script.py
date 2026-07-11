from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_nwpu_linear_identifiability.py"
CONFIG = (
    ROOT
    / "spectral_detection_posttrain"
    / "configs"
    / "versions"
    / "det.energy.linear_identifiability.001.json"
)
LAUNCHER = (
    ROOT
    / "scripts"
    / "experiments"
    / "run_nwpu_linear_identifiability_s42.sh"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "probe_nwpu_linear_identifiability", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_linear_probe_config_is_train_cache_only_and_nested() -> None:
    module = _load_module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert module.load_locked_config() == config
    assert config["version_id"] == "det.energy.linear_identifiability.001"
    assert config["probe"]["uses_clean_validation"] is False
    assert config["probe"]["scope"] == "train_cache_nested_fit_tune_outer_heldout"
    assert config["split"]["outer_heldout_fraction"] == 0.2
    assert config["split"]["inner_tune_fraction"] == 0.2
    assert len(config["ridge"]["lambdas"]) >= 7
    assert config["arms"] == [
        "local_full",
        "within_image_feature_shuffle",
        "within_image_label_shuffle",
    ]


def test_linear_probe_runner_does_not_touch_detector_or_clean_validation() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "build_detector" not in source
    assert "build_probe_cache_records" not in source
    assert "clean_validation" not in source
    assert "scikit" not in source.lower()
    assert '"heldout_evaluated_once": True' in source


def test_ridge_selection_prefers_safe_tune_threshold_before_raw_ranking() -> None:
    module = _load_module()
    candidates = [
        {
            "l2": 0.01,
            "used_identity_fallback": True,
            "tune_calibration": {"metrics": {"mean_delta_u_lcb": 0.0}},
            "tune_raw": {
                "pairwise_accuracy": 0.80,
                "pairwise_accuracy_candidate_weighted": 0.79,
                "mae": 0.10,
            },
        },
        {
            "l2": 1.0,
            "used_identity_fallback": False,
            "tune_calibration": {"metrics": {"mean_delta_u_lcb": 0.02}},
            "tune_raw": {
                "pairwise_accuracy": 0.62,
                "pairwise_accuracy_candidate_weighted": 0.61,
                "mae": 0.16,
            },
        },
        {
            "l2": 10.0,
            "used_identity_fallback": False,
            "tune_calibration": {"metrics": {"mean_delta_u_lcb": 0.03}},
            "tune_raw": {
                "pairwise_accuracy": 0.60,
                "pairwise_accuracy_candidate_weighted": 0.59,
                "mae": 0.18,
            },
        },
    ]

    assert module.select_ridge_candidate(candidates)["l2"] == 10.0


def test_linear_probe_launcher_is_gpu2_only_and_idempotent() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "CUDA_VISIBLE_DEVICES=2" in source
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in source
    assert "memory.free" in source
    assert '"$GPU_FREE" -le 8192' in source
    assert "probe_nwpu_linear_identifiability.py" in source
    assert "--require-clean-git" in source
    assert "eval_metrics.json" in source
