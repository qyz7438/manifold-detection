from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_nwpu_top_focused_b3.py"
CONFIG = (
    ROOT
    / "spectral_detection_posttrain"
    / "configs"
    / "versions"
    / "det.energy.top_focused_audit.001.json"
)
LAUNCHER = ROOT / "scripts" / "experiments" / "run_nwpu_top_focused_audit_s42.sh"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_nwpu_top_focused", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_locks_zero_training_reused_outer_protocol() -> None:
    module = _load_module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert module.load_locked_config() == config
    assert config["probe"]["performs_new_training"] is False
    assert config["probe"]["outer_split_seen_in_prior_analyses"] is True
    assert config["probe"]["uses_detector_validation"] is False


def test_audit_loads_frozen_checkpoint_without_fitting_or_detector() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "torch.load" in source
    assert "fit_ridge_regression" not in source
    assert "fit_spatial_feature_map" not in source
    assert "build_detector" not in source
    assert "clean_validation" not in source
    assert "median_sign_metrics" in source
    assert "top_focused_rank_metrics" in source


def test_launcher_is_gpu2_only_and_strictly_gated() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "CUDA_VISIBLE_DEVICES=2" in source
    assert '"$GPU_FREE" -le 8192' in source
    assert "audit_nwpu_top_focused_b3.py" in source
    assert "eval_metrics.json" in source
