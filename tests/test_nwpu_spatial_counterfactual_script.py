from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_nwpu_spatial_counterfactual.py"
CONFIG = (
    ROOT
    / "spectral_detection_posttrain"
    / "configs"
    / "versions"
    / "det.energy.spatial_counterfactual.001.json"
)
LAUNCHER = ROOT / "scripts" / "experiments" / "run_nwpu_spatial_counterfactual_s42.sh"


def _load_module():
    spec = importlib.util.spec_from_file_location("probe_nwpu_spatial_counterfactual", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_spatial_config_is_adaptive_nested_and_locks_four_arms() -> None:
    module = _load_module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert module.load_locked_config() == config
    assert config["version_id"] == "det.energy.spatial_counterfactual.001"
    assert config["probe"]["researcher_adaptive"] is True
    assert config["probe"]["uses_detector_validation"] is False
    assert config["arms"] == [
        "structured_full",
        "global_pooled",
        "spatial_layout_shuffle",
        "delta_alignment_shuffle",
    ]
    assert config["features"]["spatial_blocks"] == 9


def test_spatial_runner_does_not_build_detector_or_touch_detector_validation() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "build_detector" not in source
    assert "build_probe_cache_records" not in source
    assert "clean_validation" not in source
    assert '"outer_evaluated_after_selection": True' in source


def test_delta_control_only_shuffles_alignment_delta() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "replace_action_features" not in source
    assert '"delta_alignment_shuffle"' in source
    assert "alignment_delta=fit_alignment_delta" in source
    assert "alignment_delta=tune_alignment_delta" in source
    assert "alignment_delta=outer_alignment_delta" in source
    assert "within_image_shuffle_order" not in source
    assert "within_image_delta_alignment_shuffle(" in source


def test_spatial_launcher_is_gpu2_only_and_idempotent() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "CUDA_VISIBLE_DEVICES=2" in source
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in source
    assert "memory.free" in source
    assert '"$GPU_FREE" -le 8192' in source
    assert "probe_nwpu_spatial_counterfactual.py" in source
    assert "eval_metrics.json" in source
