from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval_nwpu_m1_move_gate_bypass.py"
LAUNCHER = ROOT / "scripts" / "experiments" / "run_nwpu_m1_move_gate_bypass.sh"
LOCKED_CONFIG = ROOT / "spectral_detection_posttrain" / "configs" / "versions" / "det.energy.set_policy.movegate_diag.001.json"
M1_SCRIPT = ROOT / "scripts" / "train_nwpu_m1_set_policy.py"


def _load_module():
    assert SCRIPT.exists(), "move-gate bypass diagnostic script has not been implemented"
    spec = importlib.util.spec_from_file_location("eval_nwpu_m1_move_gate_bypass", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_locked_config_contains_all_m1_move_gate_bypass_locks() -> None:
    module = _load_module()
    config = json.loads(LOCKED_CONFIG.read_text(encoding="utf-8"))

    assert module.sha256_file(LOCKED_CONFIG) == module.LOCKED_CONFIG_SHA256
    assert config["version_id"] == "det.energy.set_policy.movegate_diag.001"
    assert config["parent_version_id"] == "det.energy.set_policy.m1.001"
    assert config["detector"]["checkpoint_sha256"] == "de126708d063a82dd5c721799cd124fc4e52d28866139e103b1cd65427742027"
    assert config["dataset"]["annotation_sha256"] == "dde27d9362d9c6aa358a1cab160c9e075e57405d3fe876de049973c6e6150f0e"
    assert config["source_m1"]["result_sha256"] == "9b089b6625d0dfc9d9fe6a1553be6d5646b78a8179c7de15e506b2e0215a88e5"
    assert config["policy"]["checkpoint_sha256"] == "54a34c463ea5ed2fea58e7fa9534fc059867d2b7cf1444b32255aadb9055d9cd"
    assert config["dataset"]["validation_images"] == 196
    assert config["dataset"]["val_image_ids_sha256"] == "49f05cc9fa82ccaf924ff3be58a8f6376387c225e020ea6138182a4637219684"
    assert config["diagnostic"]["eligibility"] == "observable & action_margin > 0"
    assert config["diagnostic"]["ranking"] == "move_logit + action_margin"
    assert config["diagnostic"]["action_budget"] == 4
    assert config["diagnostic"]["energy_weight"] == pytest.approx(0.05)
    assert config["detector"]["score_threshold"] == pytest.approx(0.05)
    assert config["detector"]["nms_threshold"] == pytest.approx(0.5)
    assert config["detector"]["detections_per_image"] == 100


def test_config_and_script_expose_exactly_one_diagnostic_arm() -> None:
    module = _load_module()
    config = json.loads(LOCKED_CONFIG.read_text(encoding="utf-8"))
    source = SCRIPT.read_text(encoding="utf-8")

    assert config["arms"] == ["move_gate_bypass"]
    assert config["output_modes"] == ["identity", "learned_set", "move_gate_bypass"]
    assert source.count("include_move_gate_bypass=True") == 1
    assert "train_policy(" not in source
    assert "--rebuild-cache" not in source
    assert "energy_zero" not in source
    assert "shuffled_spatial" not in source
    assert module.DIAGNOSTIC_ARM == "move_gate_bypass"


def test_source_m1_requirements_and_learned_set_reproduction_are_strict() -> None:
    module = _load_module()
    source_payload = {
        "completed": True,
        "gates": {"gates": {"G0_parity": True, "G1_gt_free_eval": True, "G2_learned_vs_identity": False}},
        "metrics": {"learned_set": {"ap50": 0.6, "ap75": 0.3, "false_positive_rate": 0.4, "num_predictions": 1274}},
    }
    evaluation = {"metrics": {"learned_set": {"ap50": 0.6, "ap75": 0.3, "false_positive_rate": 0.4, "num_predictions": 1274}}}

    assert module.validate_source_m1_payload(source_payload) is True
    assert module.verify_learned_set_reproduction(evaluation, source_payload) is True

    changed = json.loads(json.dumps(evaluation))
    changed["metrics"]["learned_set"]["ap75"] += 1e-8
    with pytest.raises(ValueError, match="learned_set|parity"):
        module.verify_learned_set_reproduction(changed, source_payload)


def test_diagnosis_rule_has_required_boundaries_and_always_marks_posthoc() -> None:
    module = _load_module()
    identity = {"ap50": 0.60, "ap75": 0.30, "false_positive_rate": 0.40, "num_predictions": 100}
    sufficient = {"ap50": 0.599, "ap75": 0.302, "false_positive_rate": 0.42, "num_predictions": 105}
    result = module.diagnose_move_gate_bypass(identity, sufficient)
    assert result["diagnosis"] == "move_gate_is_primary_bottleneck"
    assert result["completed"] is True
    assert result["posthoc_diagnostic"] is True
    assert result["not_clean_gain"] is True

    insufficient = {**sufficient, "ap75": 0.3019, "num_predictions": 106}
    assert module.diagnose_move_gate_bypass(identity, insufficient)["diagnosis"] == "move_gate_bypass_not_sufficient"


def test_gt_free_call_contract_is_static_and_uses_the_existing_api() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "evaluate_validation"]
    assert len(calls) == 1
    call = calls[0]
    assert any(keyword.arg == "include_move_gate_bypass" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True for keyword in call.keywords)
    assert "train_nwpu_m1_set_policy.evaluate_validation" in source

    m1_source = M1_SCRIPT.read_text(encoding="utf-8")
    eval_source = m1_source[m1_source.index("def evaluate_validation"):]
    assert "targets=None" in eval_source
    assert "target_for_metrics" in eval_source


def test_launcher_is_gpu2_only_strictly_memory_gated_and_clean_full_val() -> None:
    assert LAUNCHER.exists(), "move-gate bypass launcher has not been implemented"
    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'ROOT="/home/ps/lzz/manifold-detection-energy-transport"' in source
    assert "/home/ps/lzz/RLimage" not in source
    assert "CUDA_VISIBLE_DEVICES=2" in source
    assert "memory.free" in source
    assert "free_mb <= 8192" in source
    assert "nwpu_m1_movegate_bypass_diag_s42_fullval" in source
    assert "det.energy.set_policy.movegate_diag.001.json" in source
    assert "--require-full-validation" in source
    assert "--require-clean-git" in source
    assert "--limit-val" not in source
    assert "train_nwpu_m1_set_policy.py" not in source
