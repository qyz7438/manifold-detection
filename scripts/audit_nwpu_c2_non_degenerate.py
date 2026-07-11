from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "spectral_detection_posttrain" / "configs" / "versions" / "det.energy.native_budget1.c2.audit.001.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def audit(payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    evaluations = payload["evaluations"]
    full = evaluations["local_full"]
    learned = full["metrics"]["learned_set"]
    identity = full["metrics"]["identity"]
    diagnostics = full["diagnostics"]["learned_set"]
    delta_identity = float(learned["ap75"] - identity["ap75"])
    delta_controls = {
        arm: float(learned["ap75"] - evaluations[arm]["metrics"]["learned_set"]["ap75"])
        for arm in ("feature_shuffle", "utility_shuffle")
    }
    required = config["gates"]
    gates = {
        "nonzero_selection": int(diagnostics["selected_count"]) >= int(required["min_selected_count"]),
        "nonzero_move_gate": int(diagnostics["move_gate_positive"]) >= int(required["min_move_gate_positive"]),
        "positive_ap75_vs_identity": delta_identity >= float(required["min_ap75_delta_vs_identity"]),
        "positive_ap75_vs_controls": all(
            value >= float(required["min_ap75_delta_vs_each_control"])
            for value in delta_controls.values()
        ),
    }
    return {
        "completed": True,
        "version_id": config["version_id"],
        "source_sha256": config["source"]["sha256"],
        "scientific_status": "identity_collapse" if not all(gates.values()) else "non_degenerate",
        "all_passed": all(gates.values()),
        "gates": gates,
        "selected_count": int(diagnostics["selected_count"]),
        "move_gate_positive": int(diagnostics["move_gate_positive"]),
        "ap75_delta_vs_identity": delta_identity,
        "ap75_delta_vs_controls": delta_controls,
        "claim_boundary": config["claim_boundary"],
    }


def main() -> int:
    config = load_config()
    source = ROOT / config["source"]["path"]
    actual = sha256_file(source)
    if actual != config["source"]["sha256"]:
        raise RuntimeError(f"C2 source hash mismatch: {actual}")
    result = audit(json.loads(source.read_text(encoding="utf-8")), config)
    output = source.parent / "non_degenerate_gate_audit.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if not result["all_passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
