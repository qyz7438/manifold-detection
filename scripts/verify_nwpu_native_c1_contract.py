from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spectral_detection_posttrain.methods.energy_transport.native_contract import (
    build_detector_native_candidates,
    build_native_c1_deltas,
    evaluate_native_contract_gates,
    validate_strict_parity_artifact,
)
from spectral_detection_posttrain.methods.energy_transport.set_policy import (
    SetPolicyOutput,
    select_set_policy_actions,
)

import torch


CONFIG_PATH = (
    ROOT
    / "spectral_detection_posttrain"
    / "configs"
    / "versions"
    / "det.energy.native_listwise.c1.001.json"
)


def load_locked_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_contract_report(
    config: dict[str, Any],
    baseline_artifact: dict[str, Any],
    fullft_artifact: dict[str, Any],
) -> dict[str, Any]:
    baseline = validate_strict_parity_artifact(baseline_artifact)
    fullft = validate_strict_parity_artifact(fullft_artifact)
    deltas = build_native_c1_deltas(float(config["candidate_contract"]["step"]))
    action_logits = torch.zeros((4, deltas.shape[0]), dtype=torch.float32)
    action_logits[:, 1] = 1.0
    selection = select_set_policy_actions(
        SetPolicyOutput(
            action_logits=action_logits,
            move_logits=torch.ones(4),
            conflict_stats=torch.zeros((4, 4)),
        ),
        deltas,
        image_indices=torch.zeros(4, dtype=torch.long),
        observable_mask=torch.ones(4, dtype=torch.bool),
        max_actions_per_image=1,
    )
    builder_parameters = inspect.signature(build_detector_native_candidates).parameters
    detector_only = "target" not in builder_parameters and "labels" not in builder_parameters
    candidate_contract = {
        "candidate_count": int(deltas.shape[0]),
        "noop_is_first": bool(deltas[0].count_nonzero().item() == 0),
        "max_abs_delta": float(config["candidate_contract"]["step"]),
        "single_coordinate_non_noop": bool((deltas[1:].count_nonzero(dim=1) == 1).all().item()),
        "max_actions_per_image": int(config["candidate_contract"]["max_actions_per_image"]),
        "budget_enforced": int(selection.selected_mask.sum().item()) == 1,
        "score_threshold": float(config["candidate_contract"]["score_threshold"]),
        "detector_only": detector_only,
        "candidate_source": "detector_visible_proposals_only" if detector_only else "invalid",
        "training_gt_scope": "utility_labels_only_never_candidate_filter",
    }
    payload = {
        "parity": {"baseline": baseline, "fullft": fullft},
        "candidate_contract": candidate_contract,
    }
    gates = evaluate_native_contract_gates(payload, config)
    gates["checks"]["single_coordinate_non_noop"] = candidate_contract[
        "single_coordinate_non_noop"
    ]
    gates["all_passed"] = all(gates["checks"].values())
    return {
        "completed": True,
        "version_id": config["version_id"],
        "claim_boundary": config["claim_boundary"],
        **payload,
        "gates": gates,
        "prior_native_m1": config["sources"]["native_m1"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the locked NWPU native C1 contract")
    parser.add_argument("--run-name", default="nwpu_native_c1_contract_s42")
    return parser.parse_args()


def _load_locked_source(source: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    path = ROOT / source["path"]
    actual = sha256_file(path)
    if actual != source["sha256"]:
        raise RuntimeError(f"source hash mismatch for {path}: {actual}")
    return json.loads(path.read_text(encoding="utf-8")), {
        "path": source["path"],
        "sha256": actual,
    }


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def main() -> None:
    args = parse_args()
    config = load_locked_config()
    baseline, baseline_source = _load_locked_source(config["sources"]["parity_baseline"])
    fullft, fullft_source = _load_locked_source(config["sources"]["parity_fullft"])
    _, m1_source = _load_locked_source(config["sources"]["native_m1"])
    report = build_contract_report(config, baseline, fullft)
    report["git_commit"] = _git_commit()
    report["config_sha256"] = sha256_file(CONFIG_PATH)
    report["verified_sources"] = {
        "parity_baseline": baseline_source,
        "parity_fullft": fullft_source,
        "native_m1": m1_source,
    }

    run_dir = ROOT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "eval_metrics.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["gates"]["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
