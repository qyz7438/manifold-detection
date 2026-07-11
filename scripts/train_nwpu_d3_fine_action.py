"""Run the single preregistered D3 fine-action whole-image Delta-U smoke."""

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
CONFIG = ROOT / "spectral_detection_posttrain" / "configs" / "versions" / "det.energy.fine_action.d3.001.json"
CONFIG_SHA256 = "0d76a612045dfb2be24c131a6672566f0dfb7a19c5d1c7c46b3dd21112425ea9"
C3_SCRIPT = ROOT / "scripts" / "train_nwpu_c3_global_delta_u.py"
EXPECTED_CANDIDATE = {"step": 0.02, "candidate_count": 9}
EXPECTED_POLICY = {"loss_type": "balanced_margin"}
EXPECTED_ENDPOINT = "native_decode_threshold_class_matching_nms"


def _load_c3():
    spec = importlib.util.spec_from_file_location("d3_c3_source", C3_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load C3 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c3 = _load_c3()


def load_config() -> dict[str, Any]:
    if c3.m1.sha256_file(CONFIG) != CONFIG_SHA256:
        raise ValueError("canonical D3 config SHA256 mismatch")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("version_id") != "det.energy.fine_action.d3.001":
        raise ValueError("D3 version mismatch")
    for key, expected in EXPECTED_CANDIDATE.items():
        if config["candidate_override"].get(key) != expected:
            raise ValueError(f"D3 candidate {key} drift")
    if config["policy_override"].get("loss_type") != EXPECTED_POLICY["loss_type"]:
        raise ValueError("D3 balanced loss drift")
    if config.get("arms") != ["local_full", "feature_shuffle", "utility_shuffle"]:
        raise ValueError("D3 controls drift")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "nwpu_d3_fine_action_smoke_s42")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args()


def _write_result(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(c3.m1._json_value(result), indent=2) + "\n", encoding="utf-8")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config()
    base = c3.load_locked_config()
    if c3.CONFIG_SHA256 != config["source"]["base_config_sha256"]:
        raise RuntimeError("D3 base config hash mismatch")
    c3b_path = ROOT / config["source"]["c3b_result"]
    d2b_path = ROOT / config["source"]["d2b_result"]
    if c3.m1.sha256_file(c3b_path) != config["source"]["c3b_result_sha256"]:
        raise RuntimeError("D3 C3b source hash mismatch")
    if c3.m1.sha256_file(d2b_path) != config["source"]["d2b_result_sha256"]:
        raise RuntimeError("D3 D2b source hash mismatch")
    c3b = json.loads(c3b_path.read_text(encoding="utf-8"))
    d2b = json.loads(d2b_path.read_text(encoding="utf-8"))
    if not c3b.get("completed") or c3b.get("scientific_status") != "current_route_frozen":
        raise RuntimeError("D3 requires the completed frozen C3b result")
    if not d2b.get("completed") or d2b.get("scientific_status") != "d2b_native_topology_frozen_move_d3":
        raise RuntimeError("D3 requires the completed frozen D2b result")
    git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    if args.require_clean_git and git_dirty:
        raise RuntimeError("--require-clean-git requested but repository is dirty")

    effective = copy.deepcopy(base)
    effective["candidate_pool"].update(config["candidate_override"])
    effective["policy"].update(config["policy_override"])
    effective["training"].update(config["training_override"])
    effective["arms"] = list(config["arms"])
    if effective["utility"]["endpoint"] != EXPECTED_ENDPOINT:
        raise RuntimeError("D3 utility endpoint drift")

    from spectral_detection_posttrain.datasets import build_nwpu_vhr10_loaders
    from spectral_detection_posttrain.models import build_detector
    from spectral_detection_posttrain.utils.io import load_checkpoint
    from spectral_detection_posttrain.utils.seed import resolve_device, set_seed

    checkpoint = (args.checkpoint or ROOT / effective["detector"]["checkpoint"]).resolve()
    annotation = (args.annotation or ROOT / "data" / "NWPU_VHR10_coco.json").resolve()
    data_root = (args.data_root or ROOT / "data" / "NWPU VHR-10 dataset").resolve()
    hashes = c3.m1._verify_input_hashes(effective, checkpoint, annotation)
    set_seed(int(effective["dataset"]["seed"]))
    device = resolve_device({"device": args.device or ("cuda" if torch.cuda.is_available() else "cpu")})
    detector_config = c3.m1._make_detector_config(effective, data_root, annotation)
    detector = build_detector(detector_config).to(device)
    load_checkpoint(detector, checkpoint, device)
    detector.eval()
    for parameter in detector.parameters():
        parameter.requires_grad_(False)
    train_loader, val_loader = build_nwpu_vhr10_loaders(detector_config, limit_train=32, limit_val=32, batch_size=1)
    train_manifest = c3.m1.split_manifest(train_loader)
    val_manifest = c3.m1.split_manifest(val_loader)
    c3.verify_locked_manifest(effective, train_manifest, "train", 32)
    c3.verify_locked_manifest(effective, val_manifest, "val", 32)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.run_dir / "global_delta_u_cache.pt"
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    expected_cache_provenance = {
        "config_sha256": CONFIG_SHA256,
        "checkpoint_sha256": hashes["checkpoint_sha256"],
        "annotation_sha256": hashes["annotation_sha256"],
        "train_manifest": train_manifest,
        "code_commit": code_commit,
        "cache_schema_version": int(effective["cache"]["schema_version"]),
        "candidate_builder": "native_c1_global_delta_u_step002_v1",
        "candidate_step": float(effective["candidate_pool"]["step"]),
    }
    if not cache_path.exists():
        cache_metadata = c3.build_train_cache(
            detector,
            train_loader,
            effective,
            device,
            cache_path,
            expected_cache_provenance,
        )
    else:
        cache_metadata = {**expected_cache_provenance, "reused": True}
    records = c3._read_cache(cache_path, expected_cache_provenance)
    cache_sha256 = c3.m1.sha256_file(cache_path)
    support = c3.label_support(records, float(effective["utility"]["min_delta_u"]))
    support_passed = support["noop_images"] >= int(config["gates"]["min_noop_images"])
    support_passed = support_passed and support["action_images"] >= int(config["gates"]["min_action_images"])
    if not support_passed:
        return _write_result(
            args.run_dir / "eval_metrics.json",
            {
                "completed": True,
                "scientific_status": "fixed_grid_action_branch_frozen",
                "version_id": config["version_id"],
                "experiment_scope": "fine_action_smoke_32_train_support_failed",
                "config_sha256": CONFIG_SHA256,
                "source_c3b_result_sha256": config["source"]["c3b_result_sha256"],
                "source_d2b_result_sha256": config["source"]["d2b_result_sha256"],
                "git_commit": code_commit,
                "git_dirty": git_dirty,
                "cache_sha256": cache_sha256,
                "cache_metadata": cache_metadata,
                "support": support,
                "gates": {"all_passed": False, "gates": {"label_support": False}},
                "decision_rule": config["decision_rule"],
                "claim_boundary": config["claim_boundary"],
                "validated_claim": False,
            },
        )

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
        "label_support": support_passed,
        "native_parity": evaluation["parity"]["passed"] and evaluation["parity"]["mismatched_images"] == 0,
        "non_degenerate": diagnostic["selected_count"] >= int(required["min_selected_count"])
        and float(required["min_action_image_rate"]) <= diagnostic["action_image_rate"] <= float(required["max_action_image_rate"]),
        "detector_delta": delta75 > float(required["min_ap75_delta_vs_identity_exclusive"])
        and delta50 >= float(required["min_ap50_delta_vs_identity"]),
        "control_delta": all(
            learned["ap75"] > evaluation["metrics"][arm]["ap75"]
            for arm in ("feature_shuffle", "utility_shuffle")
        ),
    }
    all_passed = all(gates.values())
    return _write_result(
        args.run_dir / "eval_metrics.json",
        {
            "completed": True,
            "scientific_status": "fine_action_signal_detected" if all_passed else "fixed_grid_action_branch_frozen",
            "version_id": config["version_id"],
            "experiment_scope": "fine_action_whole_image_delta_u_smoke_32_32",
            "config_sha256": CONFIG_SHA256,
            "base_config_sha256": config["source"]["base_config_sha256"],
            "source_c3b_result_sha256": config["source"]["c3b_result_sha256"],
            "source_d2b_result_sha256": config["source"]["d2b_result_sha256"],
            "git_commit": code_commit,
            "git_dirty": git_dirty,
            "cache_sha256": cache_sha256,
            "cache_metadata": cache_metadata,
            "support": support,
            "candidate_step": float(effective["candidate_pool"]["step"]),
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
        },
    )


def main() -> int:
    result = run(parse_args())
    print(json.dumps({"status": result["scientific_status"], "summary": result.get("summary"), "gates": result["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
