"""Run the C2 detector-native nine-action, budget-one NWPU smoke experiment."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG_PATH = ROOT / "spectral_detection_posttrain" / "configs" / "versions" / "det.energy.native_budget1.c2.001.json"
CONFIG_SHA256 = "e9502d42891dab2df01560ed3dd13f9afd2076291af9e808f83cb3ef2bdef635"
M1_SCRIPT = ROOT / "scripts" / "train_nwpu_m1_set_policy.py"


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m1 = _load_script("c2_m1_common", M1_SCRIPT)


def load_locked_config() -> dict[str, Any]:
    if m1.sha256_file(CONFIG_PATH) != CONFIG_SHA256:
        raise ValueError("canonical C2 config SHA256 mismatch")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("version_id") != "det.energy.native_budget1.c2.001":
        raise ValueError("C2 version mismatch")
    pool = config.get("candidate_pool", {})
    if (pool.get("builder"), pool.get("candidate_count"), pool.get("action_budget")) != ("native_c1", 9, 1):
        raise ValueError("C2 native action contract drift")
    if config.get("arms") != ["local_full", "feature_shuffle", "utility_shuffle"]:
        raise ValueError("C2 control arms drift")
    return config


def candidate_deltas(config: dict[str, Any]) -> torch.Tensor:
    return m1.candidate_deltas(config)


def build_budget_one_supervision(
    best_indices: torch.Tensor,
    oracle_gain: torch.Tensor,
    observable_mask: torch.Tensor,
    min_iou_gain: float,
) -> dict[str, torch.Tensor]:
    if best_indices.ndim != 1 or oracle_gain.shape != best_indices.shape or observable_mask.shape != best_indices.shape:
        raise ValueError("supervision inputs must be aligned vectors")
    action_targets = torch.where(
        observable_mask.bool() & oracle_gain.gt(float(min_iou_gain)),
        best_indices.long(),
        torch.zeros_like(best_indices, dtype=torch.long),
    )
    move_targets = torch.zeros_like(action_targets)
    eligible = torch.nonzero(action_targets.ne(0), as_tuple=False).flatten()
    if eligible.numel():
        chosen = eligible[oracle_gain[eligible].argmax()]
        move_targets[chosen] = 1
    return {"action_targets": action_targets, "move_targets": move_targets}


def _non_identity_permutation(count: int, seed: int) -> torch.Tensor:
    if count < 2:
        return torch.arange(count)
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(count, generator=generator)
    identity = torch.arange(count)
    return permutation.roll(1) if torch.equal(permutation, identity) else permutation


def make_control_cache(cache: Sequence[dict[str, Any]], arm: str, seed: int) -> list[dict[str, Any]]:
    if arm not in {"local_full", "feature_shuffle", "utility_shuffle"}:
        raise ValueError(f"unknown C2 arm {arm!r}")
    output: list[dict[str, Any]] = []
    for index, source in enumerate(cache):
        record = {key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value) for key, value in source.items()}
        observable = m1._observable_mask(record["class_logits"], record["scores"], 0.05)
        rows = torch.nonzero(observable, as_tuple=False).flatten()
        permutation = _non_identity_permutation(int(rows.numel()), int(seed) + 1009 * index)
        source_rows = rows[permutation]
        if arm == "feature_shuffle":
            record["spatial_features"][rows] = source["spatial_features"][source_rows]
        elif arm == "utility_shuffle":
            record["action_targets"][rows] = source["action_targets"][source_rows]
            record["move_targets"][rows] = source["move_targets"][source_rows]
        output.append(record)
    return output


@torch.no_grad()
def build_train_cache(model: torch.nn.Module, loader: Any, config: dict[str, Any], device: torch.device, path: Path) -> dict[str, Any]:
    from spectral_detection_posttrain.methods.energy_transport.candidate_energy import build_candidate_quality_targets
    from spectral_detection_posttrain.trainers.detection.action_local_transport import (
        _gather_matched_gt,
        extract_proposal_action_batch,
        rematch_roi_action_state,
    )

    records: list[dict[str, Any]] = []
    deltas = candidate_deltas(config).to(device)
    pool = config["candidate_pool"]
    model.eval()
    for images, batch_targets in loader:
        images_device = [image.to(device) for image in images]
        train_target = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch_targets[0].items()}
        batch = extract_proposal_action_batch(model, images_device, targets=None, box_base="decoded")
        _, transformed_targets = model.transform(images_device, [train_target])
        if transformed_targets is None:
            raise RuntimeError("training target transform unexpectedly returned None")
        matched_state = rematch_roi_action_state(batch.state, transformed_targets, match_mode="class_aware")
        matched_boxes, matched_labels = _gather_matched_gt(matched_state, transformed_targets)
        batch = replace(
            batch,
            state=matched_state,
            matched_gt_boxes=matched_boxes,
            matched_gt_labels=matched_labels,
        )
        quality = build_candidate_quality_targets(
            batch.state,
            deltas.to(dtype=batch.state.boxes.dtype),
            matched_gt_boxes=batch.matched_gt_boxes,
            matched_gt_labels=batch.matched_gt_labels,
            image_sizes=batch.image_sizes,
            min_iou_gain=float(pool["min_iou_gain"]),
        )
        observable = m1._observable_mask(batch.state.logits, batch.state.scores, float(pool["min_score"]))
        supervision = build_budget_one_supervision(
            quality.target_indices,
            quality.oracle_gain,
            observable,
            float(pool["min_iou_gain"]),
        )
        records.append(
            m1.build_cache_record(
                image_id=int(train_target["image_id"].flatten()[0].item()),
                spatial_features=batch.spatial_features,
                class_logits=batch.class_logits,
                predicted_labels=batch.state.labels,
                scores=batch.state.scores,
                boxes=batch.state.boxes,
                image_size=batch.image_sizes[0],
                action_targets=supervision["action_targets"],
                move_targets=supervision["move_targets"],
            )
        )
    m1._write_cache(path, records)
    return {"records": len(records), "candidate_filter_uses_gt": False, **m1.summarize_cache_storage(records)}


def cache_support(cache: Sequence[dict[str, Any]]) -> dict[str, int]:
    masks = [m1._observable_mask(record["class_logits"], record["scores"], 0.05) for record in cache]
    action = sum(int(record["action_targets"][mask].ne(0).sum()) for record, mask in zip(cache, masks))
    noop = sum(int(record["action_targets"][mask].eq(0).sum()) for record, mask in zip(cache, masks))
    selected = sum(int(record["move_targets"][mask].sum()) for record, mask in zip(cache, masks))
    return {"noop": noop, "action": action, "selected_move": selected}


def verify_c1(config: dict[str, Any]) -> dict[str, Any]:
    source = config["source_c1"]
    path = ROOT / source["result"]
    actual = m1.sha256_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    passed = actual == source["result_sha256"] and payload.get("gates", {}).get("all_passed") is True
    if not passed:
        raise ValueError("locked C1 contract source is invalid")
    return {"path": source["result"], "sha256": actual, "all_gates_passed": True}


def evaluate_gates(config: dict[str, Any], c1: dict[str, Any], support: dict[str, int], evaluations: dict[str, Any]) -> dict[str, Any]:
    full = evaluations["local_full"]
    identity = full["metrics"]["identity"]
    learned = full["metrics"]["learned_set"]
    delta75 = float(learned["ap75"] - identity["ap75"])
    delta50 = float(learned["ap50"] - identity["ap50"])
    parity = full["summary"]["parity"]
    control_deltas = {
        arm: float(learned["ap75"] - evaluations[arm]["metrics"]["learned_set"]["ap75"])
        for arm in ("feature_shuffle", "utility_shuffle")
    }
    gates = {
        "G0_c1_contract": c1["all_gates_passed"],
        "G1_support": support["noop"] >= 1 and support["action"] >= 1,
        "G2_native_parity": parity.get("passed") is True and parity.get("mismatched_images") == 0,
        "G3_detector": delta75 >= config["gates"]["G3_detector"]["min_ap75_delta"] and delta50 >= config["gates"]["G3_detector"]["min_ap50_delta"],
        "G4_controls": all(value >= config["gates"]["G4_controls"]["min_ap75_delta_vs_each"] for value in control_deltas.values()),
    }
    return {"all_passed": all(gates.values()), "gates": gates, "detector_deltas": {"ap50": delta50, "ap75": delta75}, "control_ap75_deltas": control_deltas}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "nwpu_c2_native_budget1_smoke_s42")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-train", type=int, default=32)
    parser.add_argument("--limit-val", type=int, default=32)
    parser.add_argument("--require-clean-git", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_locked_config()
    if args.require_clean_git and subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("--require-clean-git requested but repository is dirty")
    from spectral_detection_posttrain.datasets import build_nwpu_vhr10_loaders
    from spectral_detection_posttrain.models import build_detector
    from spectral_detection_posttrain.utils.io import load_checkpoint
    from spectral_detection_posttrain.utils.seed import resolve_device, set_seed

    c1 = verify_c1(config)
    checkpoint = (args.checkpoint or ROOT / config["detector"]["checkpoint"]).resolve()
    annotation = (args.annotation or ROOT / "data" / "NWPU_VHR10_coco.json").resolve()
    data_root = (args.data_root or ROOT / "data" / "NWPU VHR-10 dataset").resolve()
    hashes = m1._verify_input_hashes(config, checkpoint, annotation)
    set_seed(int(config["dataset"]["seed"]))
    device = resolve_device({"device": args.device or ("cuda" if torch.cuda.is_available() else "cpu")})
    detector_config = m1._make_detector_config(config, data_root, annotation)
    detector = build_detector(detector_config).to(device)
    load_checkpoint(detector, checkpoint, device)
    detector.eval()
    for parameter in detector.parameters():
        parameter.requires_grad_(False)
    train_loader, val_loader = build_nwpu_vhr10_loaders(detector_config, limit_train=args.limit_train, limit_val=args.limit_val, batch_size=1)
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = run_dir / "oracle_cache.pt"
    if args.rebuild_cache or not cache_path.exists():
        cache_metadata = build_train_cache(detector, train_loader, config, device, cache_path)
    else:
        cache_metadata = {"reused": True}
    cache = m1._read_cache(cache_path)
    support = cache_support(cache)
    if support["noop"] < 1 or support["action"] < 1:
        raise RuntimeError(f"C2 no-op/action support gate failed: {support}")

    m1.LOCKED_CONFIG = CONFIG_PATH
    evaluations: dict[str, Any] = {}
    histories: dict[str, Any] = {}
    for arm in config["arms"]:
        set_seed(int(config["dataset"]["seed"]))
        arm_cache = make_control_cache(cache, arm, int(config["dataset"]["seed"]))
        arm_dir = run_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        policy, training = m1.train_policy(arm_cache, config, device, arm_dir)
        evaluation, _ = m1.evaluate_validation(detector, policy, val_loader, config, device)
        evaluations[arm] = evaluation
        histories[arm] = training["history"]
    gates = evaluate_gates(config, c1, support, evaluations)
    result = {
        "completed": True,
        "version_id": config["version_id"],
        "experiment_scope": "smoke" if args.limit_train is not None or args.limit_val is not None else "full",
        "claim_boundary": config["claim_boundary"],
        "config_sha256": CONFIG_SHA256,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "inputs": {"checkpoint_sha256": hashes["checkpoint_sha256"], "annotation_sha256": hashes["annotation_sha256"], "source_c1": c1},
        "cache_metadata": cache_metadata,
        "support": support,
        "histories": histories,
        "evaluations": evaluations,
        "gates": gates,
    }
    output = run_dir / "eval_metrics.json"
    output.write_text(json.dumps(m1._json_value(result), indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    print(json.dumps({"completed": result["completed"], "gates": result["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
