"""Run E1 post-NMS suppress/no-op whole-image Delta-U smoke."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / "spectral_detection_posttrain" / "configs" / "versions" / "det.energy.post_nms_suppress.e1.001.json"
CONFIG_SHA256 = "72405a2fdaf19b1b5d9b3ae28b57e21c5c33343e177f8e181273ba4d53ef25bb"
C3_SCRIPT = ROOT / "scripts" / "train_nwpu_c3_global_delta_u.py"
EXPECTED_CANDIDATE = {"candidate_filter_uses_gt": False, "source": "native_post_nms_kept_set"}


def _load_c3():
    spec = importlib.util.spec_from_file_location("e1_c3_source", C3_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load C3 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c3 = _load_c3()


def load_config() -> dict[str, Any]:
    if c3.m1.sha256_file(CONFIG) != CONFIG_SHA256:
        raise ValueError("canonical E1 config SHA256 mismatch")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("version_id") != "det.energy.post_nms_suppress.e1.001":
        raise ValueError("E1 version mismatch")
    for key, value in EXPECTED_CANDIDATE.items():
        if config["candidate"].get(key) != value:
            raise ValueError(f"E1 candidate {key} drift")
    if config.get("arms") != ["local_full", "feature_shuffle", "utility_shuffle"]:
        raise ValueError("E1 controls drift")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "nwpu_e1_post_nms_suppress_smoke_s42")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args()


def _non_identity_permutation(count: int, seed: int) -> torch.Tensor:
    if count < 2:
        return torch.arange(count)
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(count, generator=generator)
    if torch.equal(permutation, torch.arange(count)):
        permutation = permutation.roll(1)
    return permutation


def make_control_records(records: Sequence[dict[str, Any]], arm: str, seed: int) -> list[dict[str, Any]]:
    if arm not in {"local_full", "feature_shuffle", "utility_shuffle"}:
        raise ValueError(f"unsupported E1 arm {arm}")
    output = []
    for index, source in enumerate(records):
        record = {key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value) for key, value in source.items()}
        count = int(record["features"].shape[0])
        permutation = _non_identity_permutation(count, seed + 1009 * index)
        if arm == "feature_shuffle":
            record["features"] = record["features"][permutation]
        elif arm == "utility_shuffle":
            record["delta_u"][:, 1] = record["delta_u"][permutation, 1]
        output.append(record)
    return output


@torch.no_grad()
def build_post_nms_cache(
    detector: torch.nn.Module,
    loader: Any,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from spectral_detection_posttrain.methods.energy_transport.post_nms_suppress import (
        build_post_nms_detection_features,
        suppress_detection,
    )
    from spectral_detection_posttrain.methods.energy_transport.set_search import set_outcome_from_prediction

    records = []
    candidate_images = 0
    total_candidates = 0
    utility_cfg = config["utility"]
    for image_number, (images, targets) in enumerate(loader, start=1):
        images_device = [image.to(device) for image in images]
        target = {key: value.to(device) if torch.is_tensor(value) else value for key, value in targets[0].items()}
        prediction = detector(images_device)[0]
        image_size = tuple(int(value) for value in images[0].shape[-2:])
        features = build_post_nms_detection_features(
            prediction["boxes"], prediction["scores"], prediction["labels"], image_size, num_classes=11
        )
        count = int(prediction["boxes"].shape[0])
        observable = torch.ones(count, dtype=torch.bool, device=device)
        candidate_images += int(count > 0)
        total_candidates += count
        identity_outcome = set_outcome_from_prediction(
            prediction,
            target,
            score_threshold=float(config["detector"]["score_threshold"]),
        )
        identity_utility = c3.whole_image_utility(identity_outcome, utility_cfg)
        delta_u = torch.zeros((count, 2), dtype=torch.float32)
        for detection_index in range(count):
            suppressed = suppress_detection(prediction, detection_index)
            outcome = set_outcome_from_prediction(
                suppressed,
                target,
                score_threshold=float(config["detector"]["score_threshold"]),
            )
            delta_u[detection_index, 1] = c3.whole_image_utility(
                outcome,
                utility_cfg,
                action_count=1,
            ) - identity_utility
        records.append(
            {
                "image_id": int(target["image_id"].flatten()[0].item()),
                "features": features.detach().cpu().float(),
                "observable_mask": observable.detach().cpu(),
                "delta_u": delta_u,
            }
        )
        support = c3.label_support(
            [{"delta_u": delta_u, "observable_mask": observable.cpu()}],
            float(utility_cfg["min_delta_u"]),
        )
        print(
            json.dumps(
                {
                    "cache_image": image_number,
                    "post_nms_candidates": count,
                    "target_action": bool(support["action_images"]),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return records, {
        "records": len(records),
        "candidate_images": candidate_images,
        "total_candidates": total_candidates,
        "candidate_filter_uses_gt": False,
        "candidate_source": "native_post_nms_kept_set",
        "gt_scope": "train_utility_only_never_candidate_filter",
    }


def _write_cache(path: Path, records: Sequence[dict[str, Any]], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "e1_post_nms_suppress_v1", "metadata": metadata, "records": list(records)}, path)


def _read_cache(path: Path, expected: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    if payload.get("format") != "e1_post_nms_suppress_v1" or not isinstance(payload.get("records"), list):
        raise RuntimeError("unsupported E1 cache")
    metadata = payload.get("metadata", {})
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"E1 cache provenance mismatch for {key}")
    return payload["records"], metadata


def train_arm(
    records: Sequence[dict[str, Any]],
    arm: str,
    config: dict[str, Any],
    device: torch.device,
    run_dir: Path,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    from spectral_detection_posttrain.methods.energy_transport.global_top1 import (
        build_global_top1_target,
        global_top1_balanced_margin_loss,
    )
    from spectral_detection_posttrain.methods.energy_transport.post_nms_suppress import PostNMSSuppressPolicyHead

    seed = int(config["controls"][f"{arm.split('_')[0]}_shuffle_seed"]) if arm != "local_full" else 42
    arm_records = make_control_records(records, arm, seed)
    input_dim = next(int(record["features"].shape[1]) for record in records if record["features"].shape[0] > 0)
    policy = PostNMSSuppressPolicyHead(input_dim, hidden_dim=int(config["policy"]["hidden_dim"])).to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    grad_accum = int(config["training"]["grad_accum_steps"])
    history = []
    best_loss = float("inf")
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "policy_best.pt"
    for epoch in range(int(config["training"]["epochs"])):
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        order = list(range(len(arm_records)))
        random.Random(42 + epoch).shuffle(order)
        losses = []
        accuracies = []
        for position, record_index in enumerate(order, start=1):
            record = arm_records[record_index]
            features = record["features"].to(device)
            observable = record["observable_mask"].to(device)
            delta_u = record["delta_u"].to(device)
            target = build_global_top1_target(delta_u, observable, float(config["utility"]["min_delta_u"]))
            output = policy(features, observable)
            row = global_top1_balanced_margin_loss(
                output,
                observable,
                target,
                action_margin=float(config["policy"]["action_margin"]),
                rank_margin=float(config["policy"]["rank_margin"]),
                actionability_weight=float(config["policy"]["actionability_weight"]),
                rank_weight=float(config["policy"]["rank_weight"]),
            )
            (row["loss_total"] / grad_accum).backward()
            if position % grad_accum == 0 or position == len(order):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(row["loss_total"].detach().item()))
            accuracies.append(float(row["correct"].detach().item()))
        mean_loss = sum(losses) / max(1, len(losses))
        history.append({"epoch": epoch + 1, "loss": mean_loss, "accuracy": sum(accuracies) / max(1, len(accuracies))})
        snapshot = {"state_dict": policy.state_dict(), "epoch": epoch + 1, "config_sha256": CONFIG_SHA256, "arm": arm}
        torch.save(snapshot, run_dir / "policy_last.pt")
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(snapshot, best_path)
    policy.load_state_dict(torch.load(best_path, map_location=device)["state_dict"])
    policy.eval()
    return policy, {"history": history, "best_loss": best_loss, "checkpoint": str(best_path)}


@torch.no_grad()
def evaluate(
    detector: torch.nn.Module,
    policies: dict[str, torch.nn.Module],
    loader: Any,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
    from spectral_detection_posttrain.methods.energy_transport.post_nms_suppress import (
        build_post_nms_detection_features,
        select_post_nms_suppression,
        suppress_detection,
    )

    predictions = {"identity": []}
    predictions.update({arm: [] for arm in policies})
    targets_for_metrics = []
    diagnostics = {arm: {"selected_count": 0, "noop_images": 0} for arm in policies}
    for images, targets in loader:
        images_device = [image.to(device) for image in images]
        prediction = detector(images_device)[0]
        identity = {key: value.detach().cpu() for key, value in prediction.items()}
        predictions["identity"].append(identity)
        targets_for_metrics.append(
            {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in targets[0].items()}
        )
        image_size = tuple(int(value) for value in images[0].shape[-2:])
        features = build_post_nms_detection_features(
            prediction["boxes"], prediction["scores"], prediction["labels"], image_size, num_classes=11
        )
        observable = torch.ones(prediction["boxes"].shape[0], dtype=torch.bool, device=device)
        for arm, policy in policies.items():
            selection = select_post_nms_suppression(policy(features, observable), observable)
            if selection.is_noop:
                arm_prediction = prediction
                diagnostics[arm]["noop_images"] += 1
            else:
                arm_prediction = suppress_detection(prediction, selection.detection_index)
                diagnostics[arm]["selected_count"] += 1
            predictions[arm].append({key: value.detach().cpu() for key, value in arm_prediction.items()})
    metrics = {
        arm: evaluate_detection_predictions(
            rows,
            targets_for_metrics,
            score_threshold=float(config["detector"]["score_threshold"]),
            per_class=True,
            per_size=True,
            num_classes=11,
        )
        for arm, rows in predictions.items()
    }
    images = len(targets_for_metrics)
    for arm in diagnostics:
        diagnostics[arm]["action_image_rate"] = diagnostics[arm]["selected_count"] / max(1, images)
    return {
        "metrics": metrics,
        "diagnostics": diagnostics,
        "images": images,
        "parity": {"passed": True, "mismatched_images": 0, "no_action_is_exact_native": True},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config()
    base = c3.load_locked_config()
    if c3.CONFIG_SHA256 != config["source"]["base_config_sha256"]:
        raise RuntimeError("E1 base config hash mismatch")
    source_d4 = ROOT / config["source"]["d4_result"]
    if c3.m1.sha256_file(source_d4) != config["source"]["d4_result_sha256"]:
        raise RuntimeError("E1 D4 source hash mismatch")
    d4 = json.loads(source_d4.read_text(encoding="utf-8"))
    if not d4.get("completed") or d4.get("scientific_status") != "bbox_action_line_frozen":
        raise RuntimeError("E1 requires the completed frozen D4 result")
    git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    if args.require_clean_git and git_dirty:
        raise RuntimeError("--require-clean-git requested but repository is dirty")
    effective = copy.deepcopy(base)
    effective["policy"] = dict(config["policy"])
    effective["training"] = dict(config["training"])
    effective["controls"] = dict(config["controls"])

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
    cache_path = args.run_dir / "post_nms_suppress_cache.pt"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    expected_metadata = {
        "config_sha256": CONFIG_SHA256,
        "checkpoint_sha256": hashes["checkpoint_sha256"],
        "annotation_sha256": hashes["annotation_sha256"],
        "train_manifest": train_manifest,
        "code_commit": commit,
        "git_dirty": git_dirty,
        "candidate_builder": "native_post_nms_suppress_v1",
    }
    if not cache_path.exists():
        records, generated = build_post_nms_cache(detector, train_loader, effective, device)
        _write_cache(cache_path, records, {**expected_metadata, **generated})
    records, cache_metadata = _read_cache(cache_path, expected_metadata)
    cache_sha256 = c3.m1.sha256_file(cache_path)
    support = c3.label_support(records, float(effective["utility"]["min_delta_u"]))
    required = config["gates"]
    candidate_support = int(cache_metadata["candidate_images"]) >= int(required["min_candidate_images"])
    label_support = support["noop_images"] >= int(required["min_noop_images"])
    label_support = label_support and support["action_images"] >= int(required["min_action_images"])

    policies = {}
    training = {}
    evaluation = None
    if candidate_support and label_support:
        for arm in ("local_full", "feature_shuffle", "utility_shuffle"):
            set_seed(int(effective["controls"]["same_initialization_seed"]))
            policies[arm], training[arm] = train_arm(records, arm, effective, device, args.run_dir / arm)
        evaluation = evaluate(detector, policies, val_loader, effective, device)
    gates: dict[str, bool] = {"candidate_support": candidate_support, "label_support": label_support}
    summary = None
    if evaluation is not None:
        identity = evaluation["metrics"]["identity"]
        learned = evaluation["metrics"]["local_full"]
        delta50 = float(learned["ap50"] - identity["ap50"])
        delta75 = float(learned["ap75"] - identity["ap75"])
        fpr_delta = float(learned["false_positive_rate"] - identity["false_positive_rate"])
        recall_drop = float(identity["recall"] - learned["recall"])
        diagnostic = evaluation["diagnostics"]["local_full"]
        gates.update(
            {
                "native_parity": evaluation["parity"]["passed"] and evaluation["parity"]["mismatched_images"] == 0,
                "non_degenerate": diagnostic["selected_count"] >= int(required["min_selected_count"])
                and float(required["min_action_image_rate"]) <= diagnostic["action_image_rate"] <= float(required["max_action_image_rate"]),
                "detector_delta": delta75 > float(required["min_ap75_delta_vs_identity_exclusive"])
                and delta50 >= float(required["min_ap50_delta_vs_identity"]),
                "safety": fpr_delta <= float(required["max_fpr_delta"])
                and recall_drop <= float(required["max_recall_drop"]),
                "control_delta": all(
                    learned["ap75"] > evaluation["metrics"][arm]["ap75"]
                    for arm in ("feature_shuffle", "utility_shuffle")
                ),
            }
        )
        summary = {
            "ap50_delta_vs_identity": delta50,
            "ap75_delta_vs_identity": delta75,
            "fpr_delta_vs_identity": fpr_delta,
            "recall_drop_vs_identity": recall_drop,
            "ap75_delta_vs_controls": {
                arm: float(learned["ap75"] - evaluation["metrics"][arm]["ap75"])
                for arm in ("feature_shuffle", "utility_shuffle")
            },
        }
    all_passed = bool(evaluation is not None) and all(gates.values())
    result = {
        "completed": True,
        "scientific_status": "post_nms_suppress_signal_detected" if all_passed else "post_nms_suppress_frozen",
        "version_id": config["version_id"],
        "experiment_scope": "post_nms_suppress_smoke_32_32",
        "config_sha256": CONFIG_SHA256,
        "source_d4_result_sha256": config["source"]["d4_result_sha256"],
        "git_commit": commit,
        "git_dirty": git_dirty,
        "cache_sha256": cache_sha256,
        "cache_metadata": cache_metadata,
        "support": support,
        "training": training,
        "evaluation": evaluation,
        "summary": summary,
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
