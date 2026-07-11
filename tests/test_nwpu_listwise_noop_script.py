from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_nwpu_listwise_noop.py"
CONFIG = (
    ROOT
    / "spectral_detection_posttrain"
    / "configs"
    / "versions"
    / "det.energy.listwise_noop.001.json"
)
LAUNCHER = ROOT / "scripts" / "experiments" / "run_nwpu_listwise_noop_s42.sh"


def _load_module():
    spec = importlib.util.spec_from_file_location("probe_nwpu_listwise_noop", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_locks_direct_listwise_noop_protocol() -> None:
    module = _load_module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert module.load_locked_config() == config
    assert config["targets"]["target_rule"] == "argmax_positive_delta_u_else_noop"
    assert config["probe"]["uses_detector_validation"] is False
    assert config["probe"]["outer_split_seen_in_prior_analyses"] is True


def test_runner_uses_listwise_objective_and_final_utility_only() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "fit_listwise_model" in source
    assert "calibrate_noop_margin" in source
    assert "paired_gain_stats" in source
    assert "fit_ridge_regression" not in source
    assert "pairwise_accuracy" not in source
    assert "build_detector" not in source
    assert "clean_validation" not in source


def test_launcher_is_gpu2_only_idempotent_and_strictly_gated() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "CUDA_VISIBLE_DEVICES=2" in source
    assert '"$GPU_FREE" -le 8192' in source
    assert "probe_nwpu_listwise_noop.py" in source
    assert "eval_metrics.json" in source
