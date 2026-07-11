from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_nwpu_native_c1_contract.py"
CONFIG = (
    ROOT
    / "spectral_detection_posttrain"
    / "configs"
    / "versions"
    / "det.energy.native_listwise.c1.001.json"
)
LAUNCHER = ROOT / "scripts" / "experiments" / "run_nwpu_native_c1_contract_s42.sh"


def _load_module():
    spec = importlib.util.spec_from_file_location("verify_nwpu_native_c1_contract", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parity() -> dict:
    return {
        "completed": True,
        "aggregate_zero_action_parity": {"passed": True},
        "strict_zero_action_parity": {
            "passed": True,
            "images": 196,
            "mismatched_images": 0,
            "actions_are_exact_zero": True,
            "max_box_abs_error": 0.0,
            "max_score_abs_error": 0.0,
        },
    }


def test_config_locks_detector_only_budget_one_protocol() -> None:
    module = _load_module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert module.load_locked_config() == config
    assert config["candidate_contract"]["candidate_source"] == "detector_visible_proposals_only"
    assert config["candidate_contract"]["training_gt_scope"] == "utility_labels_only_never_candidate_filter"
    assert config["candidate_contract"]["max_actions_per_image"] == 1


def test_report_requires_both_strict_parities_and_exact_c1_actions() -> None:
    module = _load_module()
    config = module.load_locked_config()
    report = module.build_contract_report(config, _parity(), _parity())

    assert report["completed"] is True
    assert report["candidate_contract"]["candidate_count"] == 9
    assert report["candidate_contract"]["noop_is_first"] is True
    assert report["candidate_contract"]["detector_only"] is True
    assert report["gates"]["all_passed"] is True
    assert report["claim_boundary"] == config["claim_boundary"]


def test_launcher_is_gpu2_only_idempotent_and_strictly_gated() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "CUDA_VISIBLE_DEVICES=2" in source
    assert '"$GPU_FREE" -le 8192' in source
    assert "verify_nwpu_native_c1_contract.py" in source
    assert "eval_metrics.json" in source
