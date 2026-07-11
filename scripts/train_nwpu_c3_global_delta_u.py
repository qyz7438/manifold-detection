"""Train C3 from native-postprocessing whole-image Delta-U top-1/no-op labels."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG_PATH = ROOT / "spectral_detection_posttrain" / "configs" / "versions" / "det.energy.global_delta_u.c3.001.json"
CONFIG_SHA256 = "f191933f6299efb3b22af241fbd77b9685f6bb9127b5ea09196418c4352c70a0"
C2_SCRIPT = ROOT / "scripts" / "train_nwpu_c2_native_budget1.py"


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c2 = _load_script("c3_c2_common", C2_SCRIPT)
m1 = c2.m1


def load_locked_config() -> dict[str, Any]:
    if m1.sha256_file(CONFIG_PATH) != CONFIG_SHA256:
        raise ValueError("canonical C3 config SHA256 mismatch")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("version_id") != "det.energy.global_delta_u.c3.001":
        raise ValueError("C3 version mismatch")
    pool = config.get("candidate_pool", {})
    if (pool.get("source"), pool.get("builder"), pool.get("candidate_count"), pool.get("max_actions_per_image")) != (
        "detector_visible_proposals_only",
        "native_c1",
        9,
        1,
    ):
        raise ValueError("C3 candidate contract drift")
    if config.get("arms") != ["local_full", "feature_shuffle", "utility_shuffle"]:
        raise ValueError("C3 control arms drift")
    return config


def enumerate_singleton_delta_u(
    proposal_count: int,
    candidate_count: int,
    observable_mask: torch.Tensor,
    identity_utility: float,
    evaluate_utility: Callable[[int, int], float],
) -> torch.Tensor:
    if proposal_count < 0 or candidate_count < 2:
        raise ValueError("proposal_count must be non-negative and candidate_count must be at least two")
    if observable_mask.shape != (proposal_count,):
        raise ValueError("observable_mask must have shape (proposal_count,)")
    delta_u = torch.full(
        (proposal_count, candidate_count),
        float("-inf"),
        dtype=torch.float32,
        device=observable_mask.device,
    )
    delta_u[:, 0] = 0.0
    for proposal_index in torch.nonzero(observable_mask.bool(), as_tuple=False).flatten().tolist():
        for candidate_index in range(1, candidate_count):
            utility = float(evaluate_utility(int(proposal_index), int(candidate_index)))
            delta_u[proposal_index, candidate_index] = utility - float(identity_utility)
    return delta_u


def whole_image_utility(
    outcome: Any,
    utility_config: dict[str, Any],
    *,
    action_energy: float = 0.0,
    action_count: int = 0,
) -> float:
    return (
        float(utility_config["tp75_weight"]) * float(outcome.tp75)
        + float(utility_config["fp75_weight"]) * float(outcome.fp75)
        + float(utility_config["fp50_weight"]) * float(outcome.fp50)
        + float(utility_config["action_energy_weight"]) * float(action_energy)
        + float(utility_config["action_count_weight"]) * int(action_count)
    )


def _non_identity_permutation(count: int, seed: int) -> torch.Tensor:
    if count < 2:
        return torch.arange(count)
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(count, generator=generator)
    identity = torch.arange(count)
    return permutation.roll(1) if torch.equal(permutation, identity) else permutation


def make_control_records(records: Sequence[dict[str, Any]], arm: str, seed: int) -> list[dict[str, Any]]:
    if arm not in {"local_full", "feature_shuffle", "utility_shuffle"}:
        raise ValueError(f"unsupported C3 arm {arm!r}")
    output: list[dict[str, Any]] = []
    for index, source in enumerate(records):
        record = {key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value) for key, value in source.items()}
        rows = torch.nonzero(record["observable_mask"].bool(), as_tuple=False).flatten()
        if arm == "feature_shuffle":
            permutation = _non_identity_permutation(int(rows.numel()), int(seed) + 1009 * index)
            record["spatial_features"][rows] = source["spatial_features"][rows[permutation]]
        elif arm == "utility_shuffle":
            values = source["delta_u"][rows, 1:].reshape(-1)
            permutation = _non_identity_permutation(int(values.numel()), int(seed) + 1009 * index)
            record["delta_u"][rows, 1:] = values[permutation].reshape(rows.numel(), -1)
        output.append(record)
    return output


def _zero_actions(state: Any):
    from spectral_detection_posttrain.methods.energy_transport import ROITransportActions

    return ROITransportActions(
        feature_delta=state.features.new_zeros(state.features.shape),
        score_delta=state.scores.new_zeros(state.scores.shape),
        box_delta=state.boxes.new_zeros(state.boxes.shape),
        keep_logit=state.scores.new_zeros(state.scores.shape),
    )


def _record(
    *,
    image_id: int,
    batch: Any,
    observable_mask: torch.Tensor,
    delta_u: torch.Tensor,
) -> dict[str, Any]:
    record = {
        "image_id": int(image_id),
        "spatial_features": batch.spatial_features.detach().cpu().to(torch.float16),
        "class_logits": batch.class_logits.detach().cpu().to(torch.float32),
        "predicted_labels": batch.state.labels.detach().cpu().long(),
        "scores": batch.state.scores.detach().cpu().float(),
        "boxes": batch.state.boxes.detach().cpu().float(),
        "image_size": tuple(int(value) for value in batch.image_sizes[0]),
        "observable_mask": observable_mask.detach().cpu().bool(),
        "delta_u": delta_u.detach().cpu().float(),
    }
    count = int(record["boxes"].shape[0])
    if record["delta_u"].shape[0] != count or record["observable_mask"].shape != (count,):
        raise ValueError("C3 cache row alignment failed")
    return record


def _write_cache(path: Path, records: Sequence[dict[str, Any]], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "c3_global_delta_u_v1", "metadata": dict(metadata), "records": list(records)}, path)


def _read_cache(path: Path, expected_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    if payload.get("format") != "c3_global_delta_u_v1" or not isinstance(payload.get("records"), list):
        raise ValueError("unsupported C3 cache")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("C3 cache provenance is missing")
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"C3 cache provenance mismatch for {key}")
    return payload["records"]


@torch.no_grad()
def build_train_cache(
    model: torch.nn.Module,
    loader: Any,
    config: dict[str, Any],
    device: torch.device,
    path: Path,
    cache_provenance: dict[str, Any],
) -> dict[str, Any]:
    from spectral_detection_posttrain.methods.energy_transport import ROITransportActions
    from spectral_detection_posttrain.methods.energy_transport.global_top1 import build_global_top1_target
    from spectral_detection_posttrain.methods.energy_transport.native_contract import build_native_c1_deltas
    from spectral_detection_posttrain.methods.energy_transport.set_search import set_outcome_from_prediction
    from spectral_detection_posttrain.trainers.detection.action_local_transport import (
        action_batch_to_predictions,
        extract_proposal_action_batch,
    )

    records: list[dict[str, Any]] = []
    deltas = build_native_c1_deltas(float(config["candidate_pool"]["step"])).to(device)
    utility_cfg = config["utility"]
    for image_number, (images, targets) in enumerate(loader, start=1):
        images_device = [image.to(device) for image in images]
        target = {key: value.to(device) if torch.is_tensor(value) else value for key, value in targets[0].items()}
        batch = extract_proposal_action_batch(model, images_device, targets=None, box_base="decoded")
        observable = m1._observable_mask(
            batch.state.logits,
            batch.state.scores,
            float(config["candidate_pool"]["min_score"]),
        )
        identity_prediction = action_batch_to_predictions(
            batch,
            _zero_actions(batch.state),
            score_threshold=float(config["detector"]["score_threshold"]),
            nms_threshold=float(config["detector"]["nms_threshold"]),
            detections_per_img=int(config["detector"]["detections_per_image"]),
            native_model=model,
        )[0]
        identity_outcome = set_outcome_from_prediction(
            identity_prediction,
            target,
            score_threshold=float(config["detector"]["score_threshold"]),
        )

        def evaluate_utility(proposal_index: int, candidate_index: int) -> float:
            box_delta = batch.state.boxes.new_zeros(batch.state.boxes.shape)
            box_delta[proposal_index] = deltas[candidate_index].to(dtype=box_delta.dtype)
            actions = ROITransportActions(
                feature_delta=batch.state.features.new_zeros(batch.state.features.shape),
                score_delta=batch.state.scores.new_zeros(batch.state.scores.shape),
                box_delta=box_delta,
                keep_logit=batch.state.scores.new_zeros(batch.state.scores.shape),
            )
            prediction = action_batch_to_predictions(
                batch,
                actions,
                score_threshold=float(config["detector"]["score_threshold"]),
                nms_threshold=float(config["detector"]["nms_threshold"]),
                detections_per_img=int(config["detector"]["detections_per_image"]),
                native_model=model,
            )[0]
            outcome = set_outcome_from_prediction(
                prediction,
                target,
                score_threshold=float(config["detector"]["score_threshold"]),
            )
            energy = float(deltas[candidate_index].square().sum().item()) / float(utility_cfg["action_energy_scale"])
            return whole_image_utility(
                outcome,
                utility_cfg,
                action_energy=energy,
                action_count=1,
            )

        delta_u = enumerate_singleton_delta_u(
            proposal_count=int(batch.state.boxes.shape[0]),
            candidate_count=int(deltas.shape[0]),
            observable_mask=observable,
            identity_utility=whole_image_utility(identity_outcome, utility_cfg),
            evaluate_utility=evaluate_utility,
        )
        target_label = build_global_top1_target(delta_u, observable, float(utility_cfg["min_delta_u"]))
        records.append(
            _record(
                image_id=int(target["image_id"].flatten()[0].item()),
                batch=batch,
                observable_mask=observable,
                delta_u=delta_u,
            )
        )
        print(
            json.dumps(
                {
                    "cache_image": image_number,
                    "observable": int(observable.sum().item()),
                    "target_noop": target_label.is_noop,
                    "target_delta_u": target_label.delta_u,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    metadata = {
        **cache_provenance,
        "records": len(records),
        "candidate_filter_uses_gt": False,
        "candidate_source": "detector_visible_proposals_only",
        "gt_scope": "train_utility_only_never_candidate_filter",
    }
    _write_cache(path, records, metadata)
    return metadata


def _record_to_device(record: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in record.items()
    }


def label_support(records: Sequence[dict[str, Any]], min_delta_u: float) -> dict[str, int]:
    from spectral_detection_posttrain.methods.energy_transport.global_top1 import build_global_top1_target

    targets = [
        build_global_top1_target(record["delta_u"], record["observable_mask"], min_delta_u)
        for record in records
    ]
    action = sum(not target.is_noop for target in targets)
    return {"images": len(targets), "action_images": int(action), "noop_images": int(len(targets) - action)}


def train_arm(
    records: Sequence[dict[str, Any]],
    arm: str,
    config: dict[str, Any],
    device: torch.device,
    run_dir: Path,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    from spectral_detection_posttrain.methods.energy_transport.global_top1 import (
        GlobalTop1PolicyHead,
        SetContextGlobalTop1PolicyHead,
        build_global_top1_target,
        global_top1_balanced_margin_loss,
        global_top1_loss,
    )
    from spectral_detection_posttrain.methods.energy_transport.native_contract import build_native_c1_deltas

    seed = int(config["controls"][f"{arm.split('_')[0]}_shuffle_seed"]) if arm != "local_full" else int(config["dataset"]["seed"])
    arm_records = make_control_records(records, arm, seed)
    deltas = build_native_c1_deltas(float(config["candidate_pool"]["step"])).to(device)
    policy_class = (
        SetContextGlobalTop1PolicyHead
        if config["policy"].get("architecture") == "set_context"
        else GlobalTop1PolicyHead
    )
    policy = policy_class(
        in_channels=int(records[0]["spatial_features"].shape[1]),
        num_classes=11,
        candidate_deltas=deltas,
        hidden_dim=int(config["policy"]["hidden_dim"]),
        spatial_size=int(config["policy"]["spatial_size"]),
        energy_weight=float(config["policy"]["energy_weight"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_path = run_dir / "policy_best.pt"
    grad_accum = int(config["training"]["grad_accum_steps"])
    for epoch in range(int(config["training"]["epochs"])):
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        accuracies: list[float] = []
        predicted_noop: list[float] = []
        order = list(range(len(arm_records)))
        random.Random(int(config["dataset"]["seed"]) + epoch).shuffle(order)
        for position, record_index in enumerate(order, start=1):
            record = _record_to_device(arm_records[record_index], device)
            target = build_global_top1_target(
                record["delta_u"],
                record["observable_mask"],
                float(config["utility"]["min_delta_u"]),
            )
            output = policy(
                record["spatial_features"].float(),
                record["class_logits"],
                record["predicted_labels"],
                record["scores"],
                record["boxes"],
                record["image_size"],
                record["observable_mask"],
            )
            if config["policy"].get("loss_type", "flat_cross_entropy") == "balanced_margin":
                row = global_top1_balanced_margin_loss(
                    output,
                    record["observable_mask"],
                    target,
                    action_margin=float(config["policy"]["action_margin"]),
                    rank_margin=float(config["policy"]["rank_margin"]),
                    actionability_weight=float(config["policy"]["actionability_weight"]),
                    rank_weight=float(config["policy"]["rank_weight"]),
                )
            else:
                row = global_top1_loss(output, record["observable_mask"], target)
            (row["loss_total"] / grad_accum).backward()
            if position % grad_accum == 0 or position == len(order):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(row["loss_total"].detach().item()))
            accuracies.append(float(row["accuracy"].item()))
            predicted_noop.append(float(row["predicted_noop"].item()))
        summary = {
            "epoch": epoch + 1,
            "loss": sum(losses) / len(losses),
            "accuracy": sum(accuracies) / len(accuracies),
            "predicted_noop_rate": sum(predicted_noop) / len(predicted_noop),
        }
        history.append(summary)
        checkpoint = {"state_dict": policy.state_dict(), "epoch": epoch + 1, "config_sha256": CONFIG_SHA256, "arm": arm}
        torch.save(checkpoint, run_dir / "policy_last.pt")
        if summary["loss"] < best_loss:
            best_loss = summary["loss"]
            torch.save(checkpoint, best_path)
    policy.load_state_dict(torch.load(best_path, map_location=device)["state_dict"])
    return policy, {
        "history": history,
        "best_loss": best_loss,
        "checkpoint": str(best_path),
        "label_support": label_support(arm_records, float(config["utility"]["min_delta_u"])),
    }


@torch.no_grad()
def evaluate_policies(
    model: torch.nn.Module,
    policies: dict[str, torch.nn.Module],
    loader: Any,
    config: dict[str, Any],
    device: torch.device,
    *,
    allow_noop: bool = True,
) -> dict[str, Any]:
    from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
    from spectral_detection_posttrain.methods.energy_transport.global_top1 import select_global_top1_action
    from spectral_detection_posttrain.methods.energy_transport.native_contract import build_native_c1_deltas
    from spectral_detection_posttrain.trainers.detection.action_local_transport import (
        action_batch_to_predictions,
        extract_proposal_action_batch,
    )

    m0 = m1._load_m0()
    deltas = build_native_c1_deltas(float(config["candidate_pool"]["step"])).to(device)
    predictions: dict[str, list[dict[str, torch.Tensor]]] = {"identity": []}
    predictions.update({arm: [] for arm in policies})
    targets_for_metrics: list[dict[str, Any]] = []
    diagnostics = {
        arm: {"selected_count": 0, "noop_images": 0, "candidate_indices": {str(index): 0 for index in range(1, 9)}}
        for arm in policies
    }
    parity = {"passed": True, "mismatched_images": 0, "max_box_abs_error": 0.0, "max_score_abs_error": 0.0}
    for images, targets in loader:
        images_device = [image.to(device) for image in images]
        target = {key: value.to(device) if torch.is_tensor(value) else value for key, value in targets[0].items()}
        batch = extract_proposal_action_batch(model, images_device, targets=None, box_base="decoded")
        native_prediction = {key: value.detach().cpu() for key, value in model(images_device)[0].items()}
        identity_prediction = action_batch_to_predictions(
            batch,
            _zero_actions(batch.state),
            score_threshold=float(config["detector"]["score_threshold"]),
            nms_threshold=float(config["detector"]["nms_threshold"]),
            detections_per_img=int(config["detector"]["detections_per_image"]),
            native_model=model,
        )[0]
        box_error, score_error, _, same = m0._compare_predictions(native_prediction, identity_prediction)
        parity["passed"] = parity["passed"] and same
        parity["mismatched_images"] += 0 if same else 1
        parity["max_box_abs_error"] = max(parity["max_box_abs_error"], float(box_error))
        parity["max_score_abs_error"] = max(parity["max_score_abs_error"], float(score_error))
        predictions["identity"].append(identity_prediction)
        targets_for_metrics.append({key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in target.items()})
        observable = m1._observable_mask(batch.state.logits, batch.state.scores, float(config["candidate_pool"]["min_score"]))
        for arm, policy in policies.items():
            output = policy(
                batch.spatial_features.float(),
                batch.class_logits,
                batch.state.labels,
                batch.state.scores,
                batch.state.boxes,
                batch.image_sizes[0],
                observable,
            )
            selection = select_global_top1_action(output, deltas, observable, allow_noop=allow_noop)
            actions = _zero_actions(batch.state)
            actions.box_delta.copy_(selection.box_delta)
            prediction = action_batch_to_predictions(
                batch,
                actions,
                score_threshold=float(config["detector"]["score_threshold"]),
                nms_threshold=float(config["detector"]["nms_threshold"]),
                detections_per_img=int(config["detector"]["detections_per_image"]),
                native_model=model,
            )[0]
            predictions[arm].append(prediction)
            if selection.is_noop:
                diagnostics[arm]["noop_images"] += 1
            else:
                diagnostics[arm]["selected_count"] += 1
                diagnostics[arm]["candidate_indices"][str(selection.candidate_index)] += 1
    metrics = {
        mode: evaluate_detection_predictions(
            mode_predictions,
            targets_for_metrics,
            score_threshold=float(config["detector"]["score_threshold"]),
            per_class=True,
            per_size=True,
            num_classes=11,
        )
        for mode, mode_predictions in predictions.items()
    }
    image_count = len(targets_for_metrics)
    for arm in diagnostics:
        diagnostics[arm]["action_image_rate"] = diagnostics[arm]["selected_count"] / max(1, image_count)
    return {"metrics": metrics, "diagnostics": diagnostics, "parity": parity, "images": image_count}


def evaluate_gates(
    config: dict[str, Any],
    c1: dict[str, Any],
    support: dict[str, int],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    identity = evaluation["metrics"]["identity"]
    learned = evaluation["metrics"]["local_full"]
    delta = {
        "ap50": float(learned["ap50"] - identity["ap50"]),
        "ap75": float(learned["ap75"] - identity["ap75"]),
        "false_positive_rate": float(learned["false_positive_rate"] - identity["false_positive_rate"]),
        "num_predictions_relative": float((learned["num_predictions"] - identity["num_predictions"]) / max(1, identity["num_predictions"])),
    }
    controls = {
        arm: float(learned["ap75"] - evaluation["metrics"][arm]["ap75"])
        for arm in ("feature_shuffle", "utility_shuffle")
    }
    parity = evaluation["parity"]
    diagnostic = evaluation["diagnostics"]["local_full"]
    cfg = config["gates"]
    gates = {
        "G0_c1_contract": c1["all_gates_passed"],
        "G1_label_support": support["noop_images"] >= cfg["G1_label_support"]["min_noop_images"] and support["action_images"] >= cfg["G1_label_support"]["min_action_images"],
        "G2_native_parity": parity["passed"] and parity["mismatched_images"] <= cfg["G2_native_parity"]["max_mismatched_images"] and parity["max_box_abs_error"] <= cfg["G2_native_parity"]["max_box_abs_error"] and parity["max_score_abs_error"] <= cfg["G2_native_parity"]["max_score_abs_error"],
        "G3_non_degenerate": diagnostic["selected_count"] >= cfg["G3_non_degenerate"]["min_selected_count"] and cfg["G3_non_degenerate"]["min_action_image_rate"] <= diagnostic["action_image_rate"] <= cfg["G3_non_degenerate"]["max_action_image_rate"],
        "G4_detector": delta["ap75"] >= cfg["G4_detector"]["min_ap75_delta"] and delta["ap50"] >= cfg["G4_detector"]["min_ap50_delta"] and delta["false_positive_rate"] <= cfg["G4_detector"]["max_fpr_delta"] and delta["num_predictions_relative"] <= cfg["G4_detector"]["max_prediction_relative_delta"],
        "G5_controls": all(value >= cfg["G5_controls"]["min_ap75_delta_vs_each"] for value in controls.values()),
    }
    return {"all_passed": all(gates.values()), "gates": gates, "detector_deltas": delta, "control_ap75_deltas": controls}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "nwpu_c3_global_delta_u_smoke_s42")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args(argv)


def verify_locked_manifest(
    config: dict[str, Any],
    manifest: dict[str, Any],
    split: str,
    limit: int | None,
) -> None:
    dataset = config["dataset"]
    if split not in {"train", "val"}:
        raise ValueError("split must be train or val")
    if limit is None:
        expected = {
            "count": int(dataset["train_images" if split == "train" else "validation_images"]),
            "image_ids_sha256": dataset["train_image_ids_sha256" if split == "train" else "val_image_ids_sha256"],
        }
    elif limit == 32:
        expected = {
            "count": int(dataset["smoke_train_images" if split == "train" else "smoke_validation_images"]),
            "image_ids_sha256": dataset[
                "smoke_train_image_ids_sha256" if split == "train" else "smoke_val_image_ids_sha256"
            ],
        }
    else:
        raise ValueError("C3 only permits locked full splits or the locked 32/32 smoke split")
    if manifest != expected:
        raise ValueError(f"locked C3 {split} manifest mismatch: expected {expected}, got {manifest}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_locked_config()
    if args.require_clean_git and subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("--require-clean-git requested but repository is dirty")
    from spectral_detection_posttrain.datasets import build_nwpu_vhr10_loaders
    from spectral_detection_posttrain.models import build_detector
    from spectral_detection_posttrain.utils.io import load_checkpoint
    from spectral_detection_posttrain.utils.seed import resolve_device, set_seed

    c1 = c2.verify_c1(config)
    checkpoint = (args.checkpoint or ROOT / config["detector"]["checkpoint"]).resolve()
    annotation = (args.annotation or ROOT / "data" / "NWPU_VHR10_coco.json").resolve()
    data_root = (args.data_root or ROOT / "data" / "NWPU VHR-10 dataset").resolve()
    hashes = m1._verify_input_hashes(config, checkpoint, annotation)
    set_seed(int(config["dataset"]["seed"]))
    device = resolve_device({"device": args.device or ("cuda" if torch.cuda.is_available() else "cpu")})
    detector_config = m1._make_detector_config(config, data_root, annotation)
    model = build_detector(detector_config).to(device)
    load_checkpoint(model, checkpoint, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    train_loader, val_loader = build_nwpu_vhr10_loaders(
        detector_config,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        batch_size=1,
    )
    train_manifest = m1.split_manifest(train_loader)
    val_manifest = m1.split_manifest(val_loader)
    verify_locked_manifest(config, train_manifest, "train", args.limit_train)
    verify_locked_manifest(config, val_manifest, "val", args.limit_val)
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = run_dir / "global_delta_u_cache.pt"
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    expected_cache_provenance = {
        "config_sha256": CONFIG_SHA256,
        "checkpoint_sha256": hashes["checkpoint_sha256"],
        "annotation_sha256": hashes["annotation_sha256"],
        "train_manifest": train_manifest,
        "code_commit": code_commit,
        "cache_schema_version": int(config["cache"]["schema_version"]),
        "candidate_builder": "native_c1_global_delta_u_v1",
    }
    if args.rebuild_cache or not cache_path.exists():
        cache_metadata = build_train_cache(
            model,
            train_loader,
            config,
            device,
            cache_path,
            expected_cache_provenance,
        )
    else:
        cache_metadata = {**expected_cache_provenance, "reused": True}
    records = _read_cache(cache_path, expected_cache_provenance)
    support = label_support(records, float(config["utility"]["min_delta_u"]))
    support_gate = config["gates"]["G1_label_support"]
    if support["noop_images"] < support_gate["min_noop_images"] or support["action_images"] < support_gate["min_action_images"]:
        result = {
            "completed": True,
            "scientific_status": "smoke_label_support_failed" if args.limit_train is not None else "full_label_support_failed",
            "version_id": config["version_id"],
            "experiment_scope": "smoke",
            "config_sha256": CONFIG_SHA256,
            "cache_metadata": cache_metadata,
            "support": support,
            "gates": {"all_passed": False, "gates": {"G1_label_support": False}},
            "claim_boundary": config["claim_boundary"],
        }
        (run_dir / "eval_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result

    policies: dict[str, torch.nn.Module] = {}
    training: dict[str, Any] = {}
    for arm in config["arms"]:
        set_seed(int(config["controls"]["same_initialization_seed"]))
        policies[arm], training[arm] = train_arm(records, arm, config, device, run_dir / arm)
    evaluation = evaluate_policies(model, policies, val_loader, config, device)
    gates = evaluate_gates(config, c1, support, evaluation)
    result = {
        "completed": True,
        "scientific_status": (
            "smoke_signal_detected" if gates["all_passed"] else "smoke_no_signal"
        ) if args.limit_train is not None or args.limit_val is not None else (
            "full_confirmation_signal" if gates["all_passed"] else "full_no_signal"
        ),
        "version_id": config["version_id"],
        "experiment_scope": "smoke" if args.limit_train is not None or args.limit_val is not None else "full",
        "config_sha256": CONFIG_SHA256,
        "git_commit": code_commit,
        "validated_claim": False,
        "inputs": {"checkpoint_sha256": hashes["checkpoint_sha256"], "annotation_sha256": hashes["annotation_sha256"], "source_c1": c1, "train_manifest": train_manifest, "validation_manifest": val_manifest},
        "cache_metadata": cache_metadata,
        "support": support,
        "training": training,
        "evaluation": evaluation,
        "gates": gates,
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "eval_metrics.json").write_text(json.dumps(m1._json_value(result), indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps({"status": result["scientific_status"], "support": result.get("support"), "gates": result["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
