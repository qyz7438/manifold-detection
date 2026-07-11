"""Run the D2b class-expanded native-NMS topology correctness smoke."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / "spectral_detection_posttrain" / "configs" / "versions" / "det.energy.native_topology.d2b.001.json"
CONFIG_SHA256 = "1b07a3f3862cc49945b0a972a8ed96298c1924ced23638f0d09b7debb7836f92"
C3_SCRIPT = ROOT / "scripts" / "train_nwpu_c3_global_delta_u.py"
EXPECTED_POLICY = {"architecture": "native_action_topology"}


def _load_c3():
    spec = importlib.util.spec_from_file_location("d2b_c3_source", C3_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load C3 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c3 = _load_c3()


def load_config() -> dict[str, Any]:
    if c3.m1.sha256_file(CONFIG) != CONFIG_SHA256:
        raise ValueError("canonical D2b config SHA256 mismatch")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("version_id") != "det.energy.native_topology.d2b.001":
        raise ValueError("D2b version mismatch")
    if config["policy_override"].get("architecture") != EXPECTED_POLICY["architecture"]:
        raise ValueError("D2b native-topology architecture drift")
    if config.get("arms") != ["local_full", "topology_shuffle", "utility_shuffle"]:
        raise ValueError("D2b control arms drift")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "nwpu_d2b_native_topology_smoke_s42")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args()


def _max_abs_error(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.numel() == 0:
        return 0.0
    return float((left.detach().cpu().float() - right.detach().cpu().float()).abs().max().item())


@torch.no_grad()
def build_native_topology_records(
    locked_records: Sequence[dict[str, Any]],
    detector: torch.nn.Module,
    train_loader: Any,
    config: dict[str, Any],
    device: torch.device,
    *,
    max_cache_alignment_abs_error: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from spectral_detection_posttrain.methods.energy_transport.native_contract import build_native_c1_deltas
    from spectral_detection_posttrain.methods.energy_transport.native_topology import (
        NATIVE_TOPOLOGY_FEATURE_NAMES,
        native_action_nms_topology,
    )
    from spectral_detection_posttrain.trainers.detection.action_local_transport import extract_proposal_action_batch

    deltas = build_native_c1_deltas(float(config["candidate_pool"]["step"])).to(device)
    output: list[dict[str, Any]] = []
    maximum_error = 0.0
    aligned_images = 0
    for record_index, (images, targets) in enumerate(train_loader):
        if record_index >= len(locked_records):
            raise RuntimeError("detector train split has more images than the locked C3 cache")
        locked = locked_records[record_index]
        image_id = int(targets[0]["image_id"].flatten()[0].item())
        if image_id != int(locked["image_id"]):
            raise RuntimeError("D2b train image order does not match the locked C3 cache")
        images_device = [image.to(device) for image in images]
        batch = extract_proposal_action_batch(detector, images_device, targets=None, box_base="decoded")
        errors = (
            _max_abs_error(batch.class_logits, locked["class_logits"]),
            _max_abs_error(batch.state.scores, locked["scores"]),
            _max_abs_error(batch.state.boxes, locked["boxes"]),
        )
        if not torch.equal(batch.state.labels.detach().cpu(), locked["predicted_labels"].cpu()):
            raise RuntimeError("D2b predicted-label alignment failed")
        row_error = max(errors)
        maximum_error = max(maximum_error, row_error)
        if row_error > float(max_cache_alignment_abs_error):
            raise RuntimeError(f"D2b detector/cache alignment error {row_error:.6g}")
        observable = locked["observable_mask"].to(device)
        decoded_boxes = detector.roi_heads.box_coder.decode(batch.box_regression, batch.proposals).clone()
        class_probabilities = torch.nn.functional.softmax(batch.class_logits, dim=-1)
        native_topology = native_action_nms_topology(
            decoded_boxes,
            class_probabilities,
            batch.state.labels,
            batch.image_sizes[0],
            deltas,
            observable,
            score_threshold=float(config["detector"]["score_threshold"]),
            nms_threshold=float(config["detector"]["nms_threshold"]),
            detections_per_img=int(config["detector"]["detections_per_image"]),
        )
        enriched = {
            key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
            for key, value in locked.items()
        }
        enriched["native_topology"] = native_topology.detach().cpu().float()
        output.append(enriched)
        aligned_images += 1
    if aligned_images != len(locked_records):
        raise RuntimeError("D2b train split ended before the locked C3 cache")
    return output, {
        "aligned_images": aligned_images,
        "max_abs_error": maximum_error,
        "feature_names": list(NATIVE_TOPOLOGY_FEATURE_NAMES),
        "uses_gt": False,
        "copies_locked_delta_u": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config()
    base = c3.load_locked_config()
    if c3.CONFIG_SHA256 != config["source"]["base_config_sha256"]:
        raise RuntimeError("D2b base config hash mismatch")
    source_cache = ROOT / config["source"]["cache"]
    source_d2 = ROOT / config["source"]["d2_proxy_result"]
    if c3.m1.sha256_file(source_cache) != config["source"]["cache_sha256"]:
        raise RuntimeError("D2b global_delta_u_cache.pt hash mismatch")
    if c3.m1.sha256_file(source_d2) != config["source"]["d2_proxy_result_sha256"]:
        raise RuntimeError("D2b proxy-result hash mismatch")
    d2_result = json.loads(source_d2.read_text(encoding="utf-8"))
    if not d2_result.get("completed") or d2_result.get("scientific_status") != "d2_action_topology_frozen_move_d3":
        raise RuntimeError("D2b requires the completed failed D2 proxy diagnostic")
    git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    if args.require_clean_git and git_dirty:
        raise RuntimeError("--require-clean-git requested but repository is dirty")
    payload = torch.load(source_cache, map_location="cpu")
    if payload.get("format") != "c3_global_delta_u_v1" or not isinstance(payload.get("records"), list):
        raise RuntimeError("locked D2b source cache format mismatch")
    locked_records = payload["records"]
    effective = copy.deepcopy(base)
    effective["policy"].update(config["policy_override"])
    effective["training"].update(config["training_override"])
    effective["controls"].update(config["controls_override"])
    effective["arms"] = list(config["arms"])

    from spectral_detection_posttrain.datasets import build_nwpu_vhr10_loaders
    from spectral_detection_posttrain.models import build_detector
    from spectral_detection_posttrain.utils.io import load_checkpoint
    from spectral_detection_posttrain.utils.seed import resolve_device, set_seed

    checkpoint = (args.checkpoint or ROOT / effective["detector"]["checkpoint"]).resolve()
    annotation = (args.annotation or ROOT / "data" / "NWPU_VHR10_coco.json").resolve()
    data_root = (args.data_root or ROOT / "data" / "NWPU VHR-10 dataset").resolve()
    c3.m1._verify_input_hashes(effective, checkpoint, annotation)
    set_seed(int(effective["dataset"]["seed"]))
    device = resolve_device({"device": args.device or ("cuda" if torch.cuda.is_available() else "cpu")})
    detector_config = c3.m1._make_detector_config(effective, data_root, annotation)
    detector = build_detector(detector_config).to(device)
    load_checkpoint(detector, checkpoint, device)
    detector.eval()
    for parameter in detector.parameters():
        parameter.requires_grad_(False)
    train_loader, val_loader = build_nwpu_vhr10_loaders(
        detector_config,
        limit_train=32,
        limit_val=32,
        batch_size=1,
    )
    c3.verify_locked_manifest(effective, c3.m1.split_manifest(train_loader), "train", 32)
    c3.verify_locked_manifest(effective, c3.m1.split_manifest(val_loader), "val", 32)
    records, cache_alignment = build_native_topology_records(
        locked_records,
        detector,
        train_loader,
        effective,
        device,
        max_cache_alignment_abs_error=float(config["gates"]["max_cache_alignment_abs_error"]),
    )
    if len(cache_alignment["feature_names"]) != int(effective["policy"]["topology_dim"]):
        raise RuntimeError("D2b topology feature dimension drift")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    enriched_cache = args.run_dir / "native_topology_cache.pt"
    torch.save(
        {
            "format": "d2b_native_topology_v1",
            "metadata": {
                "config_sha256": CONFIG_SHA256,
                "source_cache_sha256": config["source"]["cache_sha256"],
                **cache_alignment,
            },
            "records": records,
        },
        enriched_cache,
    )
    enriched_cache_sha256 = c3.m1.sha256_file(enriched_cache)
    support = c3.label_support(records, float(effective["utility"]["min_delta_u"]))

    c3.CONFIG_SHA256 = CONFIG_SHA256
    policies: dict[str, torch.nn.Module] = {}
    training: dict[str, Any] = {}
    for arm in ("local_full", "topology_shuffle", "utility_shuffle"):
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
        "cache_alignment": cache_alignment["max_abs_error"] <= float(required["max_cache_alignment_abs_error"]),
        "native_parity": evaluation["parity"]["passed"] and evaluation["parity"]["mismatched_images"] == 0,
        "non_degenerate": diagnostic["selected_count"] >= int(required["min_selected_count"])
        and float(required["min_action_image_rate"]) <= diagnostic["action_image_rate"] <= float(required["max_action_image_rate"]),
        "detector_delta": delta75 > float(required["min_ap75_delta_vs_identity_exclusive"])
        and delta50 >= float(required["min_ap50_delta_vs_identity"]),
        "control_delta": all(
            learned["ap75"] > evaluation["metrics"][arm]["ap75"]
            for arm in ("topology_shuffle", "utility_shuffle")
        ),
    }
    all_passed = all(gates.values())
    result = {
        "completed": True,
        "scientific_status": "larger_train_confirmation_allowed" if all_passed else "d2b_native_topology_frozen_move_d3",
        "version_id": config["version_id"],
        "experiment_scope": "native_topology_correctness_smoke_32_32",
        "config_sha256": CONFIG_SHA256,
        "base_config_sha256": config["source"]["base_config_sha256"],
        "source_cache_sha256": config["source"]["cache_sha256"],
        "source_d2_proxy_sha256": config["source"]["d2_proxy_result_sha256"],
        "native_topology_cache_sha256": enriched_cache_sha256,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": git_dirty,
        "cache_alignment": cache_alignment,
        "support": support,
        "training": training,
        "evaluation": evaluation,
        "summary": {
            "ap50_delta_vs_identity": delta50,
            "ap75_delta_vs_identity": delta75,
            "ap75_delta_vs_controls": {
                arm: float(learned["ap75"] - evaluation["metrics"][arm]["ap75"])
                for arm in ("topology_shuffle", "utility_shuffle")
            },
        },
        "gates": {"all_passed": all_passed, "gates": gates},
        "decision_rule": config["decision_rule"],
        "claim_boundary": config["claim_boundary"],
        "validated_claim": False,
    }
    (args.run_dir / "eval_metrics.json").write_text(json.dumps(c3.m1._json_value(result), indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    result = run(parse_args())
    print(json.dumps({"status": result["scientific_status"], "summary": result["summary"], "gates": result["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
