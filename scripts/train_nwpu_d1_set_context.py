"""Train D1 proposal-set context under the locked C3 endpoint and controls."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / "spectral_detection_posttrain" / "configs" / "versions" / "det.energy.set_context.d1.001.json"
CONFIG_SHA256 = "441dfca02683b63d8408d81c3919dd970b4ca8718685b63cc89de363acedf357"
C3_SCRIPT = ROOT / "scripts" / "train_nwpu_c3_global_delta_u.py"
EXPECTED_POLICY = {"architecture": "set_context"}


def _load_c3():
    spec = importlib.util.spec_from_file_location("d1_c3_source", C3_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load C3 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c3 = _load_c3()


def load_config() -> dict[str, Any]:
    if c3.m1.sha256_file(CONFIG) != CONFIG_SHA256:
        raise ValueError("canonical D1 config SHA256 mismatch")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("version_id") != "det.energy.set_context.d1.001":
        raise ValueError("D1 version mismatch")
    if config["policy_override"].get("architecture") != EXPECTED_POLICY["architecture"]:
        raise ValueError("D1 set-context architecture drift")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "nwpu_d1_set_context_smoke_s42")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config()
    base = c3.load_locked_config()
    if c3.CONFIG_SHA256 != config["source"]["base_config_sha256"]:
        raise RuntimeError("D1 base config hash mismatch")
    source_result = ROOT / config["source"]["c3b_result"]
    source_cache = ROOT / config["source"]["cache"]
    if c3.m1.sha256_file(source_result) != config["source"]["c3b_result_sha256"]:
        raise RuntimeError("D1 C3b source result hash mismatch")
    if c3.m1.sha256_file(source_cache) != config["source"]["cache_sha256"]:
        raise RuntimeError("D1 global_delta_u_cache.pt hash mismatch")
    if args.require_clean_git and subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("--require-clean-git requested but repository is dirty")

    payload = torch.load(source_cache, map_location="cpu")
    if payload.get("format") != "c3_global_delta_u_v1" or not isinstance(payload.get("records"), list):
        raise RuntimeError("locked D1 cache format mismatch")
    if payload.get("metadata", {}).get("config_sha256") != config["source"]["base_config_sha256"]:
        raise RuntimeError("locked D1 cache provenance mismatch")
    records = payload["records"]
    effective = copy.deepcopy(base)
    effective["policy"].update(config["policy_override"])
    effective["training"].update(config["training_override"])
    support = c3.label_support(records, float(effective["utility"]["min_delta_u"]))

    from spectral_detection_posttrain.datasets import build_nwpu_vhr10_loaders
    from spectral_detection_posttrain.models import build_detector
    from spectral_detection_posttrain.utils.io import load_checkpoint
    from spectral_detection_posttrain.utils.seed import resolve_device, set_seed

    checkpoint = (args.checkpoint or ROOT / effective["detector"]["checkpoint"]).resolve()
    annotation = (args.annotation or ROOT / "data" / "NWPU_VHR10_coco.json").resolve()
    data_root = (args.data_root or ROOT / "data" / "NWPU VHR-10 dataset").resolve()
    c3.m1._verify_input_hashes(effective, checkpoint, annotation)
    device = resolve_device({"device": args.device or ("cuda" if torch.cuda.is_available() else "cpu")})
    detector_config = c3.m1._make_detector_config(effective, data_root, annotation)
    detector = build_detector(detector_config).to(device)
    load_checkpoint(detector, checkpoint, device)
    detector.eval()
    for parameter in detector.parameters():
        parameter.requires_grad_(False)
    _, val_loader = build_nwpu_vhr10_loaders(detector_config, limit_train=1, limit_val=32, batch_size=1)
    c3.verify_locked_manifest(effective, c3.m1.split_manifest(val_loader), "val", 32)

    c3.CONFIG_SHA256 = CONFIG_SHA256
    policies: dict[str, torch.nn.Module] = {}
    training: dict[str, Any] = {}
    for arm in ("local_full", "feature_shuffle", "utility_shuffle"):
        set_seed(int(effective["controls"]["same_initialization_seed"]))
        policies[arm], training[arm] = c3.train_arm(records, arm, effective, device, args.run_dir / arm)
    evaluation = c3.evaluate_policies(detector, policies, val_loader, effective, device)
    identity = evaluation["metrics"]["identity"]
    learned = evaluation["metrics"]["local_full"]
    delta50 = float(learned["ap50"] - identity["ap50"])
    delta75 = float(learned["ap75"] - identity["ap75"])
    diagnostic = evaluation["diagnostics"]["local_full"]
    required = config["gates"]
    gates = {
        "native_parity": evaluation["parity"]["passed"] and evaluation["parity"]["mismatched_images"] == 0,
        "non_degenerate": diagnostic["selected_count"] >= int(required["min_selected_count"]) and float(required["min_action_image_rate"]) <= diagnostic["action_image_rate"] <= float(required["max_action_image_rate"]),
        "detector_delta": delta75 > float(required["min_ap75_delta_vs_identity_exclusive"]) and delta50 >= float(required["min_ap50_delta_vs_identity"]),
        "control_delta": all(learned["ap75"] > evaluation["metrics"][arm]["ap75"] for arm in ("feature_shuffle", "utility_shuffle")),
    }
    all_passed = all(gates.values())
    c3b = json.loads(source_result.read_text(encoding="utf-8"))
    result = {
        "completed": True,
        "scientific_status": "larger_train_confirmation_allowed" if all_passed else "d1_set_context_frozen",
        "version_id": config["version_id"],
        "experiment_scope": "set_context_smoke_32_32",
        "config_sha256": CONFIG_SHA256,
        "base_config_sha256": config["source"]["base_config_sha256"],
        "source_cache_sha256": config["source"]["cache_sha256"],
        "source_c3b_result_sha256": config["source"]["c3b_result_sha256"],
        "source_c3b_summary": c3b.get("summary"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "support": support,
        "training": training,
        "evaluation": evaluation,
        "summary": {
            "ap50_delta_vs_identity": delta50,
            "ap75_delta_vs_identity": delta75,
            "ap75_delta_vs_controls": {
                arm: float(learned["ap75"] - evaluation["metrics"][arm]["ap75"])
                for arm in ("feature_shuffle", "utility_shuffle")
            },
        },
        "gates": {"all_passed": all_passed, "gates": gates},
        "decision_rule": config["decision_rule"],
        "claim_boundary": config["claim_boundary"],
        "validated_claim": False,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "eval_metrics.json").write_text(json.dumps(c3.m1._json_value(result), indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    result = run(parse_args())
    print(json.dumps({"status": result["scientific_status"], "summary": result["summary"], "gates": result["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
