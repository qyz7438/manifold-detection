from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_nwpu_step005_strata.py"
CONFIG = (
    ROOT
    / "spectral_detection_posttrain"
    / "configs"
    / "versions"
    / "det.energy.step005_audit.001.json"
)
LAUNCHER = ROOT / "scripts" / "experiments" / "run_nwpu_step005_audit_s42.sh"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_nwpu_step005_strata", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_step005_config_is_adaptive_train_cache_only() -> None:
    module = _load_module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert module.load_locked_config() == config
    assert config["version_id"] == "det.energy.step005_audit.001"
    assert config["probe"]["researcher_adaptive"] is True
    assert config["probe"]["uses_detector_validation"] is False
    assert config["action"]["step"] == 0.05
    assert config["strata_axes"] == ["family", "direction", "class", "scale"]


def test_step005_runner_does_not_train_or_touch_detector_validation() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "build_detector" not in source
    assert "train_joint_arm" not in source
    assert "clean_validation" not in source
    assert '"outer_evaluated_after_freeze": True' in source


def test_step005_launcher_is_gpu2_only_and_idempotent() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "CUDA_VISIBLE_DEVICES=2" in source
    assert "memory.free" in source
    assert '"$GPU_FREE" -le 8192' in source
    assert "analyze_nwpu_step005_strata.py" in source
    assert "eval_metrics.json" in source
    assert "--require-clean-git" in source
