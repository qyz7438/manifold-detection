from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_nwpu_decomposed_actionability.py"
CONFIG = (
    ROOT
    / "spectral_detection_posttrain"
    / "configs"
    / "versions"
    / "det.energy.decomposed_actionability.001.json"
)
LAUNCHER = ROOT / "scripts" / "experiments" / "run_nwpu_decomposed_actionability_s42.sh"


def _load_module():
    spec = importlib.util.spec_from_file_location("probe_nwpu_decomposed", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_locks_researcher_adaptive_three_part_protocol() -> None:
    module = _load_module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert module.load_locked_config() == config
    assert config["probe"]["researcher_adaptive"] is True
    assert config["probe"]["outer_split_seen_in_prior_analyses"] is True
    assert config["probe"]["uses_detector_validation"] is False
    assert config["features"]["final_feature_dim"] == 68
    assert config["arms"] == [
        "local_full",
        "within_image_feature_shuffle",
        "within_image_label_shuffle",
    ]


def test_runner_separates_sign_rank_and_frozen_abstention_without_detector() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "build_detector" not in source
    assert "clean_validation" not in source
    assert "fit_sign_head" in source
    assert "fit_rank_head" in source
    assert "calibrate_ranked_abstention" in source
    assert '"selected_sign_l2"' in source
    assert '"selected_rank_l2"' in source
    assert '"outer_evaluated_after_freeze": True' in source
    assert "outer_label_order" not in source
    assert "if outer_feature_fraction < float(" in source


def test_launcher_is_gpu2_only_idempotent_and_strictly_gated() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "CUDA_VISIBLE_DEVICES=2" in source
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in source
    assert '"$GPU_FREE" -le 8192' in source
    assert "probe_nwpu_decomposed_actionability.py" in source
    assert "eval_metrics.json" in source
