"""Run the final detector-only adaptive-consensus bbox-action smoke."""

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
CONFIG = ROOT / "spectral_detection_posttrain" / "configs" / "versions" / "det.energy.adaptive_consensus.d4.001.json"
CONFIG_SHA256 = "17e9d28770f5fab9fe7fe067d20c559c7c4d11c91251472414ef2ffb0de23642"
C3_SCRIPT = ROOT / "scripts" / "train_nwpu_c3_global_delta_u.py"
EXPECTED_POLICY = {"architecture": "adaptive_consensus"}


def _load_c3():
    spec = importlib.util.spec_from_file_location("d4_c3_source", C3_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load C3 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c3 = _load_c3()


def load_config() -> dict[str, Any]:
    if c3.m1.sha256_file(CONFIG) != CONFIG_SHA256:
        raise ValueError("canonical D4 config SHA256 mismatch")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("version_id") != "det.energy.adaptive_consensus.d4.001":
        raise ValueError("D4 version mismatch")
    if config["policy_override"].get("architecture") != EXPECTED_POLICY["architecture"]:
        raise ValueError("D4 adaptive-consensus architecture drift")
    if config.get("arms") != ["local_full", "feature_shuffle", "utility_shuffle"]:
        raise ValueError("D4 controls drift")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "nwpu_d4_adaptive_consensus_smoke_s42")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def build_adaptive_consensus_cache(
    detector: torch.nn.Module,
    loader: Any,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from spectral_detection_posttrain.methods.energy_transport.adaptive_consensus import (
        proposal_graph_consensus_deltas,
    )
    from spectral_detection_posttrain.methods.energy_transport.set_search import set_outcome_from_prediction
    from spectral_detection_posttrain.trainers.detection.action_local_transport import (
        action_batch_to_predictions,
        extract_proposal_action_batch,
    )

    records: list[dict[str, Any]] = []
    candidate_images = 0
    total_candidates = 0
    consensus = config["adaptive_consensus"]
    utility_cfg = config["utility"]
    for image_number, (images, targets) in enumerate(loader, start=1):
        images_device = [image.to(device) for image in images]
        target = {key: value.to(device) if torch.is_tensor(value) else value for key, value in targets[0].items()}
        batch = extract_proposal_action_batch(detector, images_device, targets=None, box_base="decoded")
        detector_observable = c3.m1._observable_mask(
            batch.state.logits,
            batch.state.scores,
            float(config["candidate_pool"]["min_score"]),
        )
        adaptive_deltas = proposal_graph_consensus_deltas(
            batch.state.boxes,
            batch.state.labels,
            batch.state.scores,
            detector_observable,
            min_peer_iou=float(consensus["min_peer_iou"]),
            max_abs_delta=float(consensus["max_abs_delta"]),
        )
        actionable = detector_observable & adaptive_deltas.abs().amax(dim=1).gt(0.0)
        candidate_images += int(actionable.any().item())
        total_candidates += int(actionable.sum().item())
        identity_prediction = action_batch_to_predictions(
            batch,
            c3._zero_actions(batch.state),
            score_threshold=float(config["detector"]["score_threshold"]),
            nms_threshold=float(config["detector"]["nms_threshold"]),
            detections_per_img=int(config["detector"]["detections_per_image"]),
            native_model=detector,
        )[0]
        identity_outcome = set_outcome_from_prediction(
            identity_prediction,
            target,
            score_threshold=float(config["detector"]["score_threshold"]),
        )
        identity_utility = c3.whole_image_utility(identity_outcome, utility_cfg)
        delta_u = torch.full((batch.state.boxes.shape[0], 2), float("-inf"), dtype=torch.float32)
        delta_u[:, 0] = 0.0
        for proposal_index in torch.nonzero(actionable, as_tuple=False).flatten().tolist():
            actions = c3._zero_actions(batch.state)
            actions.box_delta[proposal_index] = adaptive_deltas[proposal_index]
            prediction = action_batch_to_predictions(
                batch,
                actions,
                score_threshold=float(config["detector"]["score_threshold"]),
                nms_threshold=float(config["detector"]["nms_threshold"]),
                detections_per_img=int(config["detector"]["detections_per_image"]),
                native_model=detector,
            )[0]
            outcome = set_outcome_from_prediction(
                prediction,
                target,
                score_threshold=float(config["detector"]["score_threshold"]),
            )
            energy = float(adaptive_deltas[proposal_index].square().sum().item()) / float(utility_cfg["action_energy_scale"])
            delta_u[proposal_index, 1] = c3.whole_image_utility(
                outcome,
                utility_cfg,
                action_energy=energy,
                action_count=1,
            ) - identity_utility
        record = c3._record(
            image_id=int(target["image_id"].flatten()[0].item()),
            batch=batch,
            observable_mask=actionable,
            delta_u=delta_u,
        )
        record["adaptive_deltas"] = adaptive_deltas.detach().cpu().float()
        records.append(record)
        target_label = c3.label_support([record], float(utility_cfg["min_delta_u"]))
        print(
            json.dumps(
                {
                    "cache_image": image_number,
                    "adaptive_candidates": int(actionable.sum().item()),
                    "target_action": bool(target_label["action_images"]),
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
        "candidate_source": "detector_visible_proposal_graph_consensus",
        "gt_scope": "train_utility_only_never_candidate_filter",
    }


def _write_cache(path: Path, records: Sequence[dict[str, Any]], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "d4_adaptive_consensus_v1", "metadata": metadata, "records": list(records)}, path)


def _read_cache(path: Path, expected: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    if payload.get("format") != "d4_adaptive_consensus_v1" or not isinstance(payload.get("records"), list):
        raise RuntimeError("unsupported D4 cache")
    metadata = payload.get("metadata", {})
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"D4 cache provenance mismatch for {key}")
    return payload["records"], metadata


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config()
    base = c3.load_locked_config()
    if c3.CONFIG_SHA256 != config["source"]["base_config_sha256"]:
        raise RuntimeError("D4 base config hash mismatch")
    source_d3 = ROOT / config["source"]["d3_result"]
    if c3.m1.sha256_file(source_d3) != config["source"]["d3_result_sha256"]:
        raise RuntimeError("D4 D3 source hash mismatch")
    d3 = json.loads(source_d3.read_text(encoding="utf-8"))
    if not d3.get("completed") or d3.get("scientific_status") != "fixed_grid_action_branch_frozen":
        raise RuntimeError("D4 requires the completed frozen D3 result")
    git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    if args.require_clean_git and git_dirty:
        raise RuntimeError("--require-clean-git requested but repository is dirty")
    effective = copy.deepcopy(base)
    effective["policy"].update(config["policy_override"])
    effective["training"].update(config["training_override"])
    effective["arms"] = list(config["arms"])
    effective["adaptive_consensus"] = dict(config["adaptive_consensus"])

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
    cache_path = args.run_dir / "adaptive_consensus_cache.pt"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    expected_metadata = {
        "config_sha256": CONFIG_SHA256,
        "checkpoint_sha256": hashes["checkpoint_sha256"],
        "annotation_sha256": hashes["annotation_sha256"],
        "train_manifest": train_manifest,
        "code_commit": commit,
        "git_dirty": git_dirty,
        "candidate_builder": "detector_graph_consensus_v1",
    }
    if not cache_path.exists():
        records, generated = build_adaptive_consensus_cache(detector, train_loader, effective, device)
        cache_metadata = {**expected_metadata, **generated}
        _write_cache(cache_path, records, cache_metadata)
    records, cache_metadata = _read_cache(cache_path, expected_metadata)
    cache_sha256 = c3.m1.sha256_file(cache_path)
    support = c3.label_support(records, float(effective["utility"]["min_delta_u"]))
    required = config["gates"]
    candidate_support = int(cache_metadata["candidate_images"]) >= int(required["min_candidate_images"])
    label_support = support["noop_images"] >= int(required["min_noop_images"])
    label_support = label_support and support["action_images"] >= int(required["min_action_images"])

    c3.CONFIG_SHA256 = CONFIG_SHA256
    policies: dict[str, torch.nn.Module] = {}
    training: dict[str, Any] = {}
    evaluation = None
    if candidate_support and label_support:
        for arm in ("local_full", "feature_shuffle", "utility_shuffle"):
            set_seed(int(effective["controls"]["same_initialization_seed"]))
            policies[arm], training[arm] = c3.train_arm(records, arm, effective, device, args.run_dir / arm)
        evaluation = c3.evaluate_policies(detector, policies, val_loader, effective, device)

    gates: dict[str, bool] = {"candidate_support": candidate_support, "label_support": label_support}
    summary = None
    if evaluation is not None:
        identity = evaluation["metrics"]["identity"]
        learned = evaluation["metrics"]["local_full"]
        delta50 = float(learned["ap50"] - identity["ap50"])
        delta75 = float(learned["ap75"] - identity["ap75"])
        diagnostic = evaluation["diagnostics"]["local_full"]
        gates.update(
            {
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
        )
        summary = {
            "ap50_delta_vs_identity": delta50,
            "ap75_delta_vs_identity": delta75,
            "ap75_delta_vs_controls": {
                arm: float(learned["ap75"] - evaluation["metrics"][arm]["ap75"])
                for arm in ("feature_shuffle", "utility_shuffle")
            },
        }
    all_passed = bool(evaluation is not None) and all(gates.values())
    result = {
        "completed": True,
        "scientific_status": "adaptive_consensus_signal_detected" if all_passed else "bbox_action_line_frozen",
        "version_id": config["version_id"],
        "experiment_scope": "adaptive_consensus_bbox_action_smoke_32_32",
        "config_sha256": CONFIG_SHA256,
        "source_d3_result_sha256": config["source"]["d3_result_sha256"],
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
