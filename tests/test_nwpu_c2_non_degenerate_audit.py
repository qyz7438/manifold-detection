from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_nwpu_c2_non_degenerate.py"


def _module():
    spec = importlib.util.spec_from_file_location("c2_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_identity_collapse_cannot_pass_zero_delta_gates() -> None:
    module = _module()
    metric = {"ap75": 0.4}
    diagnostic = {"selected_count": 0, "move_gate_positive": 0}
    payload = {
        "evaluations": {
            "local_full": {"metrics": {"identity": metric, "learned_set": metric}, "diagnostics": {"learned_set": diagnostic}},
            "feature_shuffle": {"metrics": {"learned_set": metric}},
            "utility_shuffle": {"metrics": {"learned_set": metric}},
        }
    }
    result = module.audit(payload, module.load_config())
    assert result["all_passed"] is False
    assert result["scientific_status"] == "identity_collapse"
    assert not any(result["gates"].values())
