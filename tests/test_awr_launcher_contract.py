from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "experiments" / "run_awr_weighted_boxhead_group.sh"
CONFIG = ROOT / "spectral_detection_posttrain" / "configs" / "versions" / "det.energy.oracle_utility_boxhead.001.json"


def test_launcher_locks_manifold_workspace_gpu2_and_memory_gate() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "/home/ps/lzz/manifold-detection-energy-transport" in text
    assert "/home/ps/lzz/RLimage" not in text
    assert "CUDA_VISIBLE_DEVICES=2" in text
    assert "free_mib <= 8192" in text
    assert "nvidia-smi --query-gpu=memory.free" in text
    assert "require_gpu2_memory" in text
    assert "kill" not in text
    assert "pgrep" not in text


def test_launcher_runs_locked_arms_seeds_and_group_analysis() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "for seed in 42 2024 999" in text
    assert "for arm in Z U W F S" in text
    assert "analyze_group.py" in text
    assert "--resamples 10000" in text


def test_version_config_matches_preregistered_constants() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["status"] == "preregistered_not_run"
    assert payload["fallback_utility_source"] is None
    assert payload["training_seeds"] == [42, 2024, 999]
    assert payload["train"]["epochs"] == 10
    assert payload["train"]["logical_batch_size"] == 8
    assert payload["weighting"]["normalization"] == "global_fit_mean"
    assert payload["bootstrap"] == {
        "resamples": 10000,
        "seed": 42,
        "unit": "paired_image_with_hierarchical_training_seed_resampling",
        "metric": "global_ap75",
    }
    assert payload["gpu_policy"]["cuda_visible_devices"] == "2"
    assert payload["gpu_policy"]["minimum_free_mib_exclusive"] == 8192
