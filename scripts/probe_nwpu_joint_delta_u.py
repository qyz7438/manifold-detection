"""Image-group-heldout NWPU probe for unified native-utility action scoring.

This is deliberately an oracle-pool-conditioned discriminability test.  GT is
used only to construct exact train-split utility labels; validation is untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
LOCKED_CONFIG = (
    ROOT
    / "spectral_detection_posttrain"
    / "configs"
    / "versions"
    / "det.energy.joint_probe.001.json"
)
LOCKED_CONFIG_SHA256 = "66bbf3c4637c5beaeec6ad5313f4a6f02ccba7da04646532b456c28e43ac4210"
TRACE_CACHE_SCHEMA_VERSION = 1
TRACE_RECORD_KEYS = {
    "image_id",
    "proposal_count",
    "action_ids",
    "proposal_indices",
    "candidate_indices",
    "box_deltas",
    "action_energies",
    "identity_utility",
    "singleton_delta_u",
}


def build_trace_record(
    *,
    image_id: int,
    proposal_count: int,
    candidate_pool: Sequence[dict[str, Any]],
    evaluation_trace: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if proposal_count < 0:
        raise ValueError("proposal_count must be non-negative")
    utility_by_actions = {
        tuple(str(action_id) for action_id in item["action_ids"]): float(item["utility"])
        for item in evaluation_trace
    }
    identity_utility = utility_by_actions.get(())
    if identity_utility is None:
        raise ValueError("evaluation trace is missing identity utility")

    proposal_indices: list[int] = []
    candidate_indices: list[int] = []
    box_deltas: list[list[float]] = []
    action_energies: list[float] = []
    singleton_delta_u: list[float] = []
    action_ids: list[str] = []
    for item in candidate_pool:
        action_id = str(item["action_id"])
        proposal_index = int(item["proposal_index"])
        if not 0 <= proposal_index < proposal_count:
            raise ValueError("candidate proposal index is outside the proposal set")
        singleton = utility_by_actions.get((action_id,))
        if singleton is None:
            raise ValueError(f"evaluation trace is missing singleton action {action_id}")
        proposal_indices.append(proposal_index)
        candidate_indices.append(int(item["candidate_index"]))
        box_deltas.append([float(value) for value in item["box_delta"]])
        action_energies.append(float(item["action_energy"]))
        singleton_delta_u.append(singleton - identity_utility)
        action_ids.append(action_id)

    return {
        "image_id": int(image_id),
        "proposal_count": int(proposal_count),
        "action_ids": tuple(action_ids),
        "proposal_indices": torch.tensor(proposal_indices, dtype=torch.long),
        "candidate_indices": torch.tensor(candidate_indices, dtype=torch.long),
        "box_deltas": torch.tensor(box_deltas, dtype=torch.float32).reshape(-1, 4),
        "action_energies": torch.tensor(action_energies, dtype=torch.float32),
        "identity_utility": float(identity_utility),
        "singleton_delta_u": torch.tensor(singleton_delta_u, dtype=torch.float32),
    }


def sparse_target_matrix(
    record: dict[str, Any], *, proposal_count: int, candidate_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if proposal_count < 0 or candidate_count <= 0:
        raise ValueError("proposal_count and candidate_count must be valid")
    proposal_indices = torch.as_tensor(record["proposal_indices"], dtype=torch.long)
    candidate_indices = torch.as_tensor(record["candidate_indices"], dtype=torch.long)
    values = torch.as_tensor(record["singleton_delta_u"], dtype=torch.float32)
    if not (proposal_indices.shape == candidate_indices.shape == values.shape):
        raise ValueError("sparse target fields must share shape")
    if proposal_indices.numel() and (
        proposal_indices.min() < 0 or proposal_indices.max() >= proposal_count
    ):
        raise ValueError("sparse target proposal index is out of range")
    if candidate_indices.numel() and (
        candidate_indices.min() <= 0 or candidate_indices.max() >= candidate_count
    ):
        raise ValueError("sparse target candidate index must be a non-identity candidate")
    target = torch.zeros((proposal_count, candidate_count), dtype=torch.float32)
    mask = torch.zeros((proposal_count, candidate_count), dtype=torch.bool)
    target[proposal_indices, candidate_indices] = values
    mask[proposal_indices, candidate_indices] = True
    return target, mask


def m1_sparse_candidate_scores(
    action_logits: torch.Tensor,
    move_logits: torch.Tensor,
    record: dict[str, Any],
) -> torch.Tensor:
    proposal_indices = torch.as_tensor(
        record["proposal_indices"], dtype=torch.long, device=action_logits.device
    )
    candidate_indices = torch.as_tensor(
        record["candidate_indices"], dtype=torch.long, device=action_logits.device
    )
    if action_logits.ndim != 2 or move_logits.shape != (action_logits.shape[0],):
        raise ValueError("invalid M1 output shapes")
    margin = (
        action_logits[proposal_indices, candidate_indices]
        - action_logits[proposal_indices, 0]
    )
    return margin + move_logits[proposal_indices]


def probe_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    image_ids: torch.Tensor,
    *,
    action_budget: int,
    target_epsilon: float,
) -> dict[str, float | int]:
    predicted = torch.as_tensor(predicted, dtype=torch.float32).flatten().cpu()
    target = torch.as_tensor(target, dtype=torch.float32).flatten().cpu()
    image_ids = torch.as_tensor(image_ids, dtype=torch.long).flatten().cpu()
    if not (predicted.shape == target.shape == image_ids.shape):
        raise ValueError("predicted, target, and image_ids must share shape")
    if action_budget <= 0 or target_epsilon < 0.0:
        raise ValueError("action_budget must be positive and target_epsilon non-negative")
    if predicted.numel() == 0:
        raise ValueError("probe metrics require at least one candidate")

    positive = target > float(target_epsilon)
    predicted_positive = predicted > 0.0
    true_positive = int((positive & predicted_positive).sum().item())
    predicted_positive_count = int(predicted_positive.sum().item())
    positive_count = int(positive.sum().item())
    pairwise_correct = 0.0
    pairwise_count = 0
    per_image_pairwise: list[float] = []
    regrets: list[float] = []
    selected_true: list[float] = []
    unique_images = torch.unique(image_ids, sorted=True)
    for image_id in unique_images.tolist():
        mask = image_ids.eq(int(image_id))
        image_prediction = predicted[mask]
        image_target = target[mask]
        image_pairwise_correct = 0.0
        image_pairwise_count = 0
        for left in range(image_target.numel()):
            for right in range(left + 1, image_target.numel()):
                target_gap = float((image_target[left] - image_target[right]).item())
                if abs(target_gap) <= float(target_epsilon):
                    continue
                predicted_gap = float((image_prediction[left] - image_prediction[right]).item())
                pairwise_count += 1
                image_pairwise_count += 1
                if predicted_gap == 0.0:
                    pairwise_correct += 0.5
                    image_pairwise_correct += 0.5
                elif (predicted_gap > 0.0) == (target_gap > 0.0):
                    pairwise_correct += 1.0
                    image_pairwise_correct += 1.0
        if image_pairwise_count:
            per_image_pairwise.append(image_pairwise_correct / image_pairwise_count)
        predicted_best = int(image_prediction.argmax().item())
        predicted_best_is_action = float(image_prediction[predicted_best].item()) > 0.0
        selected_target = float(image_target[predicted_best].item()) if predicted_best_is_action else 0.0
        oracle_target = max(0.0, float(image_target.max().item()))
        regrets.append(oracle_target - selected_target)
        eligible = torch.nonzero(image_prediction > 0.0, as_tuple=False).flatten()
        if eligible.numel():
            order = torch.argsort(image_prediction[eligible], descending=True)
            selected = eligible[order[:action_budget]]
            selected_true.append(float(image_target[selected].sum().item()))
        else:
            selected_true.append(0.0)

    return {
        "candidate_count": int(predicted.numel()),
        "image_count": int(unique_images.numel()),
        "mae": float((predicted - target).abs().mean().item()),
        "sign_accuracy": float(predicted_positive.eq(positive).float().mean().item()),
        "pairwise_accuracy": sum(per_image_pairwise) / max(1, len(per_image_pairwise)),
        "pairwise_accuracy_candidate_weighted": pairwise_correct / max(1, pairwise_count),
        "pairwise_count": pairwise_count,
        "oracle_regret_mean": sum(regrets) / max(1, len(regrets)),
        "selected_true_delta_mean": sum(selected_true) / max(1, len(selected_true)),
        "positive_precision": true_positive / max(1, predicted_positive_count),
        "positive_recall": true_positive / max(1, positive_count),
        "positive_prevalence": positive_count / max(1, predicted.numel()),
        "action_rate": predicted_positive_count / max(1, predicted.numel()),
    }


def write_trace_cache(path: str | Path, records: Sequence[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for record in records:
        _validate_trace_record(record)
    torch.save(
        {
            "format": "torch_pt",
            "schema_version": TRACE_CACHE_SCHEMA_VERSION,
            "records": list(records),
        },
        destination,
    )


def read_trace_cache(path: str | Path) -> list[dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format") != "torch_pt":
        raise ValueError("unsupported trace cache format")
    if payload.get("schema_version") != TRACE_CACHE_SCHEMA_VERSION:
        raise ValueError("trace cache schema version mismatch")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("trace cache records must be a list")
    for record in records:
        _validate_trace_record(record)
    return records


def _validate_trace_record(record: dict[str, Any]) -> None:
    if not isinstance(record, dict) or set(record) != TRACE_RECORD_KEYS:
        raise ValueError("trace cache record schema mismatch")
    proposal_count = int(record["proposal_count"])
    if proposal_count < 0:
        raise ValueError("trace proposal_count must be non-negative")
    action_ids = tuple(str(value) for value in record["action_ids"])
    count = len(action_ids)
    if len(set(action_ids)) != count:
        raise ValueError("trace action IDs must be unique")
    proposal_indices = torch.as_tensor(record["proposal_indices"])
    candidate_indices = torch.as_tensor(record["candidate_indices"])
    box_deltas = torch.as_tensor(record["box_deltas"])
    action_energies = torch.as_tensor(record["action_energies"])
    singleton_delta_u = torch.as_tensor(record["singleton_delta_u"])
    if proposal_indices.shape != (count,) or candidate_indices.shape != (count,):
        raise ValueError("trace index tensor shape mismatch")
    if box_deltas.shape != (count, 4):
        raise ValueError("trace box_deltas shape mismatch")
    if action_energies.shape != (count,) or singleton_delta_u.shape != (count,):
        raise ValueError("trace utility tensor shape mismatch")
    if count and (
        proposal_indices.min() < 0
        or proposal_indices.max() >= proposal_count
        or candidate_indices.min() <= 0
    ):
        raise ValueError("trace proposal or candidate index is invalid")
    if not (
        torch.isfinite(box_deltas).all()
        and torch.isfinite(action_energies).all()
        and torch.isfinite(singleton_delta_u).all()
        and math.isfinite(float(record["identity_utility"]))
    ):
        raise ValueError("trace values must be finite")
    if (action_energies < 0).any():
        raise ValueError("trace action energies must be non-negative")


def _joint_api() -> tuple[Any, Any, Any, Any]:
    from spectral_detection_posttrain.methods.energy_transport import (
        JointDeltaULossConfig,
        JointDeltaUProbe,
        build_proposal_set_edges,
        joint_delta_u_loss,
    )

    return JointDeltaUProbe, JointDeltaULossConfig, build_proposal_set_edges, joint_delta_u_loss


def _joint_loss_config(config: dict[str, Any], config_cls: Any) -> Any:
    loss = config["loss"]
    return config_cls(
        smooth_l1_weight=float(loss["regression_weight"]),
        sign_bce_weight=float(loss["sign_weight"]),
        pairwise_ranking_weight=float(loss["ranking_weight"]),
        pairwise_epsilon=float(loss["target_epsilon"]),
    )


def _joint_record_inputs(
    detector_record: dict[str, Any],
    trace_record: dict[str, Any],
    candidate_deltas: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
    *,
    arm: str,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    _, _, edge_builder, _ = _joint_api()
    spatial = torch.as_tensor(
        detector_record["spatial_features"], dtype=torch.float32, device=device
    )
    image_id = int(detector_record["image_id"])
    if arm == "joint_spatial_shuffle" and spatial.shape[0] > 1:
        generator = torch.Generator(device=device).manual_seed(31415 + image_id)
        order = torch.randperm(spatial.shape[0], generator=generator, device=device)
        spatial = spatial[order]
    logits = torch.as_tensor(detector_record["class_logits"], dtype=torch.float32, device=device)
    labels = torch.as_tensor(detector_record["predicted_labels"], dtype=torch.long, device=device)
    scores = torch.as_tensor(detector_record["scores"], dtype=torch.float32, device=device)
    boxes = torch.as_tensor(detector_record["boxes"], dtype=torch.float32, device=device)
    count = int(boxes.shape[0])
    _validate_trace_candidate_grid(trace_record, candidate_deltas)
    image_indices = torch.zeros(count, dtype=torch.long, device=device)
    return _finish_joint_record_inputs(
        spatial,
        logits,
        labels,
        scores,
        boxes,
        image_indices,
        detector_record,
        trace_record,
        candidate_deltas,
        config,
        device,
        arm,
        edge_builder,
    )


def _validate_trace_candidate_grid(
    trace_record: dict[str, Any], candidate_deltas: torch.Tensor
) -> None:
    traced_indices = torch.as_tensor(trace_record["candidate_indices"], dtype=torch.long)
    traced_deltas = torch.as_tensor(trace_record["box_deltas"], dtype=torch.float32)
    if traced_indices.numel() and (
        traced_indices.min() <= 0 or traced_indices.max() >= candidate_deltas.shape[0]
    ):
        raise ValueError("trace candidate index is outside the locked candidate grid")
    expected_deltas = candidate_deltas.detach().cpu().float()[traced_indices]
    if traced_deltas.shape != expected_deltas.shape or not torch.allclose(
        traced_deltas, expected_deltas, atol=1e-7, rtol=0.0
    ):
        raise ValueError("trace action deltas do not match the locked candidate grid")


def _finish_joint_record_inputs(
    spatial: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    boxes: torch.Tensor,
    image_indices: torch.Tensor,
    detector_record: dict[str, Any],
    trace_record: dict[str, Any],
    candidate_deltas: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
    arm: str,
    edge_builder: Any,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    count = int(boxes.shape[0])
    image_sizes = torch.tensor(detector_record["image_size"], dtype=torch.float32, device=device)
    edges = None
    if arm != "joint_no_edges":
        edges = edge_builder(
            boxes,
            labels,
            scores,
            image_indices,
            image_sizes,
            max_neighbors=int(config["model"]["max_neighbors"]),
            same_class_only=bool(config["model"]["same_class_edges"]),
        )
    target, mask = sparse_target_matrix(
        trace_record,
        proposal_count=count,
        candidate_count=int(candidate_deltas.shape[0]),
    )
    inputs = {
        "spatial_features": spatial,
        "class_logits": logits,
        "labels": labels,
        "scores": scores,
        "boxes": boxes,
        "image_indices": image_indices,
        "image_sizes": image_sizes,
        "edges": edges,
    }
    return inputs, target.to(device), mask.to(device)


def train_joint_arm(
    records: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    candidate_deltas: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
    run_dir: str | Path,
    *,
    arm: str,
    num_classes: int = 11,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if arm not in {"joint_full", "joint_no_edges", "joint_spatial_shuffle"}:
        raise ValueError(f"unsupported joint probe arm: {arm}")
    if not records:
        raise ValueError("joint probe training records cannot be empty")
    probe_cls, loss_config_cls, _, loss_fn = _joint_api()
    seed = int(config["dataset"]["seed"])
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    in_channels = int(records[0][0]["spatial_features"].shape[1])
    model = probe_cls(
        in_channels=in_channels,
        num_classes=int(num_classes),
        candidate_deltas=candidate_deltas,
        hidden_dim=int(config["model"]["hidden_dim"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    loss_config = _joint_loss_config(config, loss_config_cls)
    destination = Path(run_dir) / arm
    destination.mkdir(parents=True, exist_ok=True)
    best_path = destination / "policy_best.pt"
    last_path = destination / "policy_last.pt"
    best_loss = float("inf")
    history: list[dict[str, float | int]] = []
    epochs = int(config["training"]["epochs"])
    accumulation = int(config["training"]["grad_accum_steps"])
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        order = list(range(len(records)))
        random.Random(seed + epoch).shuffle(order)
        totals = {"loss": 0.0, "smooth_l1": 0.0, "sign_bce": 0.0, "pairwise_ranking": 0.0}
        valid_total = 0
        pair_total = 0
        for order_index, record_index in enumerate(order):
            detector_record, trace_record = records[record_index]
            inputs, target, mask = _joint_record_inputs(
                detector_record, trace_record, candidate_deltas, config, device, arm=arm
            )
            prediction = model(**inputs)
            group_ids = torch.zeros(prediction.shape[0], dtype=torch.long, device=device)
            monitor = loss_fn(
                prediction,
                target,
                mask=mask,
                group_ids=group_ids,
                config=loss_config,
            )
            (monitor["loss"] / accumulation).backward()
            if (order_index + 1) % accumulation == 0 or order_index + 1 == len(order):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for key in totals:
                totals[key] += float(monitor[key].detach().item())
            valid_total += int(monitor["valid_count"].item())
            pair_total += int(monitor["pair_count"].item())
        row = {
            "epoch": epoch + 1,
            **{key: value / len(records) for key, value in totals.items()},
            "valid_count": valid_total,
            "pair_count": pair_total,
        }
        history.append(row)
        torch.save({"state_dict": model.state_dict(), "epoch": epoch + 1, "arm": arm}, last_path)
        if float(row["loss"]) < best_loss:
            best_loss = float(row["loss"])
            torch.save({"state_dict": model.state_dict(), "epoch": epoch + 1, "arm": arm}, best_path)
    selected = torch.load(best_path, map_location=device)
    model.load_state_dict(selected["state_dict"])
    return model, {
        "history": history,
        "best_training_loss": best_loss,
        "selected_checkpoint": str(best_path),
        "final_checkpoint": str(last_path),
        "selection_rule": "minimum fixed-schedule fit-split training loss; heldout labels are evaluated once",
    }


@torch.no_grad()
def evaluate_joint_arm(
    model: torch.nn.Module,
    records: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    candidate_deltas: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
    *,
    arm: str,
) -> dict[str, float | int]:
    model.eval()
    predicted_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    image_rows: list[torch.Tensor] = []
    for detector_record, trace_record in records:
        inputs, target, _ = _joint_record_inputs(
            detector_record, trace_record, candidate_deltas, config, device, arm=arm
        )
        prediction = model(**inputs)
        proposal_indices = torch.as_tensor(trace_record["proposal_indices"], dtype=torch.long, device=device)
        candidate_indices = torch.as_tensor(trace_record["candidate_indices"], dtype=torch.long, device=device)
        predicted_rows.append(prediction[proposal_indices, candidate_indices].cpu())
        target_rows.append(target[proposal_indices, candidate_indices].cpu())
        image_rows.append(torch.full((proposal_indices.numel(),), int(detector_record["image_id"]), dtype=torch.long))
    return probe_metrics(
        torch.cat(predicted_rows),
        torch.cat(target_rows),
        torch.cat(image_rows),
        action_budget=int(config["candidate_pool"]["action_budget"]),
        target_epsilon=float(config["loss"]["target_epsilon"]),
    )


def evaluate_probe_gates(
    metrics: dict[str, dict[str, float | int]], config: dict[str, Any]
) -> dict[str, Any]:
    gates_config = config["gates"]
    full = metrics["joint_full"]
    no_edges = metrics["joint_no_edges"]
    shuffled = metrics["joint_spatial_shuffle"]
    count_gate = (
        int(full["candidate_count"]) >= int(gates_config["min_heldout_candidates"])
        and int(full["image_count"]) >= int(gates_config["min_heldout_images"])
    )
    gates = {
        "G0_heldout_support": count_gate,
        "G1_absolute_safety": (
            float(full["pairwise_accuracy"]) >= float(gates_config["min_pairwise_accuracy"])
            and float(full["sign_accuracy"]) >= float(gates_config["min_sign_accuracy"])
            and float(full["selected_true_delta_mean"])
            >= float(gates_config["min_selected_true_delta_mean"])
            and float(full["positive_precision"]) - float(full["positive_prevalence"])
            >= float(gates_config["min_positive_precision_lift"])
        ),
        "G2_joint_vs_no_edges": float(full["pairwise_accuracy"])
        - float(no_edges["pairwise_accuracy"])
        >= float(gates_config["min_pairwise_gain_over_no_edges"]),
        "G3_joint_vs_shuffle": float(full["pairwise_accuracy"])
        - float(shuffled["pairwise_accuracy"])
        >= float(gates_config["min_pairwise_gain_over_shuffle"]),
    }
    return {"all_passed": all(gates.values()), "gates": gates}


def load_locked_config(path: str | Path = LOCKED_CONFIG) -> dict[str, Any]:
    config_path = Path(path).resolve()
    if config_path != LOCKED_CONFIG.resolve():
        raise ValueError(f"joint probe requires canonical config path {LOCKED_CONFIG}")
    actual_hash = sha256_file(config_path)
    if actual_hash != LOCKED_CONFIG_SHA256:
        raise ValueError(
            f"canonical joint probe config SHA256 mismatch: expected {LOCKED_CONFIG_SHA256}, got {actual_hash}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("version_id") != "det.energy.joint_probe.001":
        raise ValueError("canonical joint probe version_id mismatch")
    return config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=LOCKED_CONFIG)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--rebuild-trace-cache", action="store_true")
    parser.add_argument("--require-clean-git", action="store_true")
    parser.add_argument("--require-full-train", action="store_true")
    parser.add_argument("--limit-train", type=int)
    return parser.parse_args(argv)


def validate_cli_args(args: argparse.Namespace) -> None:
    if args.require_full_train and args.limit_train is not None:
        raise ValueError("--require-full-train cannot be combined with a limited full train probe")
    if args.limit_train is not None and args.limit_train <= 0:
        raise ValueError("--limit-train must be positive")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_script(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load script module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _manifest(image_ids: Sequence[int]) -> dict[str, Any]:
    normalized = sorted(int(image_id) for image_id in image_ids)
    encoded = json.dumps(normalized, separators=(",", ":")).encode("ascii")
    return {
        "count": len(normalized),
        "image_ids_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _pair_records(
    detector_records: Sequence[dict[str, Any]],
    trace_records: Sequence[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    detector_by_id = {int(record["image_id"]): record for record in detector_records}
    trace_by_id = {int(record["image_id"]): record for record in trace_records}
    if len(detector_by_id) != len(detector_records) or len(trace_by_id) != len(trace_records):
        raise ValueError("source caches must contain unique image IDs")
    if set(detector_by_id) != set(trace_by_id):
        raise ValueError("detector and trace cache image IDs do not match")
    return [
        (detector_by_id[image_id], trace_by_id[image_id])
        for image_id in sorted(detector_by_id)
    ]


@torch.no_grad()
def evaluate_m1_arm(
    policy: torch.nn.Module,
    records: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    m1_module: Any,
    m1_config: dict[str, Any],
    probe_config: dict[str, Any],
    device: torch.device,
) -> dict[str, float | int]:
    policy.eval()
    predicted_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    image_rows: list[torch.Tensor] = []
    for detector_record, trace_record in records:
        spatial, logits, labels, scores, boxes = m1_module._policy_input(detector_record, device)
        output = policy(
            spatial,
            logits,
            labels,
            scores,
            boxes,
            image_size=tuple(detector_record["image_size"]),
            energy_weight=float(m1_config["policy"]["energy_weight"]),
        )
        sparse_scores = m1_sparse_candidate_scores(
            output.action_logits, output.move_logits, trace_record
        )
        target = torch.as_tensor(trace_record["singleton_delta_u"], dtype=torch.float32)
        predicted_rows.append(sparse_scores.detach().cpu())
        target_rows.append(target)
        image_rows.append(torch.full((target.numel(),), int(detector_record["image_id"]), dtype=torch.long))
    return probe_metrics(
        torch.cat(predicted_rows),
        torch.cat(target_rows),
        torch.cat(image_rows),
        action_budget=int(probe_config["candidate_pool"]["action_budget"]),
        target_epsilon=float(probe_config["loss"]["target_epsilon"]),
    )


@torch.no_grad()
def build_trace_cache_records(
    detector: torch.nn.Module,
    loader: Any,
    source_records: Sequence[dict[str, Any]],
    m0_module: Any,
    m0_config: dict[str, Any],
    probe_config: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    source_by_id = {int(record["image_id"]): record for record in source_records}
    records: list[dict[str, Any]] = []
    detector.eval()
    for image_index, (images, batch_targets) in enumerate(loader):
        images_device = [image.to(device) for image in images]
        target = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch_targets[0].items()
        }
        batch = m0_module.extract_proposal_action_batch(
            detector, images_device, [target], box_base="decoded"
        )
        pool = m0_module._candidate_pool_for_batch(batch, m0_config)
        search_record, _, _ = m0_module._run_search(
            batch,
            target,
            pool,
            detector,
            m0_config,
            nms_threshold=float(probe_config["detector"]["nms_threshold"]),
            include_evaluation_trace=True,
        )
        image_id = int(target["image_id"].flatten()[0].item())
        if image_id not in source_by_id:
            raise ValueError(f"trace image {image_id} is absent from source M1 cache")
        source = source_by_id[image_id]
        proposal_count = int(batch.state.boxes.shape[0])
        if proposal_count != int(source["boxes"].shape[0]):
            raise ValueError("trace proposal count does not match source M1 cache")
        if not torch.allclose(
            batch.state.boxes.detach().cpu(),
            torch.as_tensor(source["boxes"]),
            atol=1e-5,
            rtol=0.0,
        ):
            raise ValueError("trace proposal boxes do not match source M1 cache")
        records.append(
            build_trace_record(
                image_id=image_id,
                proposal_count=proposal_count,
                candidate_pool=[item.__dict__ for item in pool],
                evaluation_trace=search_record["evaluation_trace"],
            )
        )
        if (image_index + 1) % 25 == 0:
            print(
                json.dumps(
                    {
                        "stage": "trace_cache",
                        "images": image_index + 1,
                        "candidates": sum(len(record["action_ids"]) for record in records),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return records


def _trace_statistics(records: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    rows = [torch.as_tensor(record["singleton_delta_u"], dtype=torch.float32) for record in records]
    values = torch.cat(rows) if rows else torch.empty(0, dtype=torch.float32)
    return {
        "images": len(records),
        "candidates": int(values.numel()),
        "positive_candidates": int(values.gt(0.0).sum().item()),
        "negative_candidates": int(values.lt(0.0).sum().item()),
        "zero_candidates": int(values.eq(0.0).sum().item()),
        "mean_delta_u": float(values.mean().item()) if values.numel() else 0.0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_cli_args(args)
    config_path = Path(args.config).resolve()
    config = load_locked_config(config_path)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.require_clean_git:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        if status.stdout.strip():
            raise RuntimeError("--require-clean-git requested but the repository is dirty")

    m0_module = _load_script(ROOT / "scripts" / "eval_nwpu_m0_set_oracle.py", "joint_probe_m0")
    m1_module = _load_script(ROOT / "scripts" / "train_nwpu_m1_set_policy.py", "joint_probe_m1")
    m0_config_path = (ROOT / config["source_m0"]["config"]).resolve()
    m1_config_path = (ROOT / config["source_m1"]["config"]).resolve()
    source_cache_path = (ROOT / config["source_m1"]["cache"]).resolve()
    source_metadata_path = (ROOT / config["source_m1"]["cache_metadata"]).resolve()
    source_m0_result = (ROOT / config["source_m0"]["validated_result"]).resolve()
    locked_files = (
        (m0_config_path, config["source_m0"]["config_sha256"], "M0 config"),
        (m1_config_path, config["source_m1"]["config_sha256"], "M1 config"),
        (source_cache_path, config["source_m1"]["cache_sha256"], "M1 cache"),
        (source_m0_result, config["source_m0"]["validated_result_sha256"], "M0 result"),
    )
    for path, expected_hash, label in locked_files:
        if not path.is_file():
            raise FileNotFoundError(f"required {label} is missing: {path}")
        if sha256_file(path) != expected_hash:
            raise ValueError(f"{label} SHA256 mismatch")
    if not source_metadata_path.is_file():
        raise FileNotFoundError("source M1 cache metadata is missing")
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    if int(source_metadata.get("cache_schema_version", -1)) != int(
        config["source_m1"]["cache_schema_version"]
    ):
        raise ValueError("source M1 cache metadata schema mismatch")
    m0_result_payload = json.loads(source_m0_result.read_text(encoding="utf-8"))
    if not (
        m0_result_payload.get("completed") is True
        and m0_result_payload.get("gates", {}).get("all_passed") is True
    ):
        raise ValueError("source M0 result is not a completed all-gates-passed run")

    m0_config = m0_module.load_locked_config(m0_config_path)
    m1_config = m1_module.load_locked_config(m1_config_path)
    if int(m0_config["candidate_pool"]["candidate_count"]) != int(
        config["candidate_pool"]["candidate_count"]
    ):
        raise ValueError("M0 and joint probe candidate counts differ")
    if int(m0_config["candidate_pool"]["max_candidates_per_image"]) != int(
        config["candidate_pool"]["max_candidates_per_image"]
    ):
        raise ValueError("M0 and joint probe candidate pool budgets differ")
    if tuple(float(value) for value in m0_config["candidate_pool"]["step_sizes"]) != tuple(
        float(value) for value in config["candidate_pool"]["step_sizes"]
    ):
        raise ValueError("M0 and joint probe candidate step sizes differ")
    if float(m0_config["candidate_pool"]["action_energy_scale"]) != float(
        config["candidate_pool"]["action_energy_scale"]
    ):
        raise ValueError("M0 and joint probe action energy scales differ")

    from spectral_detection_posttrain.datasets import build_nwpu_vhr10_loaders
    from spectral_detection_posttrain.experiments.metadata import collect_experiment_metadata
    from spectral_detection_posttrain.methods.energy_transport import (
        build_symmetric_box_candidates,
        group_heldout_split,
    )
    from spectral_detection_posttrain.models import build_detector
    from spectral_detection_posttrain.utils.io import load_checkpoint
    from spectral_detection_posttrain.utils.seed import resolve_device, set_seed

    set_seed(int(config["dataset"]["seed"]))
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = resolve_device({"device": args.device or ("cuda" if torch.cuda.is_available() else "cpu")})
    checkpoint = (ROOT / config["detector"]["checkpoint"]).resolve()
    annotation = (ROOT / "data" / "NWPU_VHR10_coco.json").resolve()
    data_root = (ROOT / "data" / "NWPU VHR-10 dataset").resolve()
    m1_module.verify_locked_hashes(m1_config, checkpoint, annotation)
    environment = collect_experiment_metadata(config, config_path=config_path)
    environment.update(
        {
            "python_version": sys.version,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "deterministic_algorithms": True,
            "deterministic_warn_only": True,
        }
    )
    source_records = m1_module._read_cache(source_cache_path)
    loader_config = m1_module._make_detector_config(m1_config, data_root, annotation)
    train_loader, _ = build_nwpu_vhr10_loaders(
        loader_config,
        limit_train=args.limit_train,
        limit_val=1,
        batch_size=1,
    )
    train_manifest = m1_module.split_manifest(train_loader)
    if args.limit_train is None:
        m1_module.verify_train_manifest(train_manifest, m1_config)

    code_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    trace_cache_path = run_dir / "joint_utility_trace_cache.pt"
    trace_metadata_path = run_dir / "joint_utility_trace_cache_metadata.json"
    expected_trace_metadata = {
        "schema_version": TRACE_CACHE_SCHEMA_VERSION,
        "config_sha256": sha256_file(config_path),
        "source_m1_cache_sha256": config["source_m1"]["cache_sha256"],
        "source_m0_config_sha256": config["source_m0"]["config_sha256"],
        "train_manifest": train_manifest,
        "code_commit": code_commit,
    }
    if trace_cache_path.exists() and not args.rebuild_trace_cache:
        if not trace_metadata_path.is_file():
            raise FileNotFoundError("trace cache exists without metadata")
        actual_metadata = json.loads(trace_metadata_path.read_text(encoding="utf-8"))
        for key, expected in expected_trace_metadata.items():
            if actual_metadata.get(key) != expected:
                raise ValueError(f"trace cache metadata mismatch for {key}")
        if sha256_file(trace_cache_path) != actual_metadata.get("trace_cache_sha256"):
            raise ValueError("trace cache SHA256 mismatch")
        trace_records = read_trace_cache(trace_cache_path)
        trace_manifest = _manifest([record["image_id"] for record in trace_records])
        if trace_manifest != actual_metadata.get("trace_manifest"):
            raise ValueError("trace cache image manifest mismatch")
    else:
        detector = build_detector(loader_config).to(device)
        load_checkpoint(detector, checkpoint, device)
        for parameter in detector.parameters():
            parameter.requires_grad_(False)
        trace_records = build_trace_cache_records(
            detector,
            train_loader,
            source_records,
            m0_module,
            m0_config,
            config,
            device,
        )
        write_trace_cache(trace_cache_path, trace_records)
        trace_manifest = _manifest([record["image_id"] for record in trace_records])
        if trace_manifest != train_manifest:
            raise ValueError("generated trace cache does not cover the locked train manifest")
        actual_metadata = {
            **expected_trace_metadata,
            "trace_cache_sha256": sha256_file(trace_cache_path),
            "trace_manifest": trace_manifest,
            **_trace_statistics(trace_records),
        }
        trace_metadata_path.write_text(
            json.dumps(actual_metadata, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        del detector
        if device.type == "cuda":
            torch.cuda.empty_cache()

    trace_ids = {int(record["image_id"]) for record in trace_records}
    if len(trace_ids) != len(trace_records):
        raise ValueError("trace cache contains duplicate image IDs")
    source_records = [record for record in source_records if int(record["image_id"]) in trace_ids]
    paired = _pair_records(source_records, trace_records)
    image_ids = torch.tensor([int(detector_record["image_id"]) for detector_record, _ in paired])
    split = group_heldout_split(
        image_ids,
        heldout_fraction=float(config["probe"]["heldout_fraction"]),
        seed=int(config["probe"]["split_seed"]),
    )
    fit_records = [paired[index] for index in torch.nonzero(split.train_mask, as_tuple=False).flatten().tolist()]
    heldout_records = [paired[index] for index in torch.nonzero(split.heldout_mask, as_tuple=False).flatten().tolist()]
    if not fit_records or not heldout_records:
        raise ValueError("group-heldout probe requires non-empty fit and heldout image sets")
    fit_manifest = _manifest([record[0]["image_id"] for record in fit_records])
    heldout_manifest = _manifest([record[0]["image_id"] for record in heldout_records])

    candidate_deltas = build_symmetric_box_candidates(
        tuple(float(value) for value in config["candidate_pool"]["step_sizes"])
    ).to(device)
    for _, trace_record in paired:
        _validate_trace_candidate_grid(trace_record, candidate_deltas)
    training: dict[str, Any] = {}
    metrics: dict[str, dict[str, float | int]] = {}
    m1_fit_dir = run_dir / "m1_factorized"
    m1_fit_dir.mkdir(parents=True, exist_ok=True)
    m1_training_config = json.loads(json.dumps(m1_config))
    m1_training_config["training"]["epochs"] = int(config["training"]["m1_epochs"])
    m1_training_config["training"]["positive_move_weight"] = float(
        config["training"]["m1_positive_move_weight"]
    )
    m1_training_config["training"]["identity_weight"] = float(
        config["training"]["m1_identity_weight"]
    )
    m1_policy, m1_training = m1_module.train_policy(
        [record[0] for record in fit_records],
        m1_training_config,
        device,
        m1_fit_dir,
    )
    training["m1_factorized"] = m1_training
    metrics["m1_factorized"] = evaluate_m1_arm(
        m1_policy, heldout_records, m1_module, m1_training_config, config, device
    )
    del m1_policy
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for arm in ("joint_full", "joint_no_edges", "joint_spatial_shuffle"):
        model, arm_training = train_joint_arm(
            fit_records,
            candidate_deltas,
            config,
            device,
            run_dir,
            arm=arm,
            num_classes=11,
        )
        training[arm] = arm_training
        metrics[arm] = evaluate_joint_arm(
            model,
            heldout_records,
            candidate_deltas,
            config,
            device,
            arm=arm,
        )
        print(json.dumps({"stage": "arm_complete", "arm": arm, "metrics": metrics[arm]}, sort_keys=True), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    gates = evaluate_probe_gates(metrics, config)
    result = {
        "completed": True,
        "scientific_status": "probe_passed" if gates["all_passed"] else "probe_failed",
        "version_id": config["version_id"],
        "scope": config["probe"]["scope"],
        "uses_validation_examples": False,
        "uses_prior_validation_provenance": True,
        "claim_boundary": config["probe"]["claim_boundary"],
        "config": config,
        "inputs": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": config["detector"]["checkpoint_sha256"],
            "annotation": str(annotation),
            "annotation_sha256": config["dataset"]["annotation_sha256"],
            "source_m1_cache": str(source_cache_path),
            "source_m1_cache_sha256": config["source_m1"]["cache_sha256"],
            "source_m0_result": str(source_m0_result),
            "code_commit": code_commit,
            "train_manifest": train_manifest,
            "fit_manifest": fit_manifest,
            "heldout_manifest": heldout_manifest,
            "fit_heldout_overlap": 0,
            "environment": environment,
        },
        "trace_cache": {
            "path": str(trace_cache_path),
            "metadata": actual_metadata,
            **_trace_statistics(trace_records),
        },
        "training": training,
        "heldout_metrics": metrics,
        "metrics": metrics,
        "gates": gates,
    }
    output = run_dir / "eval_metrics.json"
    output.write_text(
        json.dumps(_json_value(result), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    print(
        json.dumps(
            {
                "output": str(Path(args.run_dir).resolve() / "eval_metrics.json"),
                "completed": result["completed"],
                "gates": result["gates"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
