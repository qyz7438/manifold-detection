"""Train and evaluate the preregistered NWPU M1 set-policy experiment.

The oracle is used only while building a train-split cache.  Validation always
extracts proposals with ``targets=None`` and keeps metric targets in a separate
variable, so GT cannot influence policy actions or selection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
LOCKED_CONFIG = ROOT / "spectral_detection_posttrain" / "configs" / "versions" / "det.energy.set_policy.m1.001.json"
LOCKED_CONFIG_SHA256 = "37238720b1eaf6a0a56ace8644a4f3248be566b301c3e6ad7393764b596daf32"
M0_SCRIPT = ROOT / "scripts" / "eval_nwpu_m0_set_oracle.py"
FORBIDDEN_CACHE_TERMS = {"gt", "ground_truth", "iou", "candidate_quality", "local_value", "matched"}
CACHE_KEYS = {
    "image_id",
    "spatial_features",
    "class_logits",
    "predicted_labels",
    "scores",
    "boxes",
    "image_size",
    "action_targets",
    "move_targets",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def validate_locked_config(config: dict[str, Any]) -> None:
    if config.get("version_id") != "det.energy.set_policy.m1.001":
        raise ValueError("M1 runner requires the locked det.energy.set_policy.m1.001 config")
    if config.get("status") != "preregistered":
        raise ValueError("M1 config must remain preregistered")
    dataset = config.get("dataset", {})
    detector = config.get("detector", {})
    training = config.get("training", {})
    policy = config.get("policy", {})
    pool = config.get("candidate_pool", {})
    if (dataset.get("seed"), dataset.get("data_seed")) != (42, 42):
        raise ValueError("M1 dataset seeds are locked to 42")
    if training.get("epochs") != 4 or training.get("grad_accum_steps") != 8:
        raise ValueError("M1 training schedule is locked")
    if policy.get("hidden_dim") != 96 or training.get("lr") != 1e-3 or training.get("weight_decay") != 1e-4:
        raise ValueError("M1 policy hyperparameters are locked")
    if pool.get("candidate_count") != 73 or pool.get("step_sizes") != [0.05, 0.1, 0.2]:
        raise ValueError("M1 candidate pool is locked")
    if detector.get("score_threshold") != 0.05 or detector.get("nms_threshold") != 0.5:
        raise ValueError("M1 detector postprocessing is locked")
    if tuple(config.get("modes", ())) != ("identity", "learned_set", "energy_zero", "shuffled_spatial"):
        raise ValueError("M1 evaluation modes are locked")
    source_m0 = config.get("source_m0", {})
    if source_m0.get("result_sha256") != "372d872ccaebea8418c2ff4428e56a666d4e4440a77b16acbd127f4999869cda":
        raise ValueError("M1 source M0 result is locked")
    cache = config.get("cache", {})
    if cache.get("format") != "torch_pt" or cache.get("schema_version") != 1:
        raise ValueError("M1 cache format is locked to torch_pt")
    if (cache.get("feature_storage_dtype"), cache.get("policy_input_dtype")) != ("float16", "float32_from_float16"):
        raise ValueError("M1 cache feature precision is locked")


def load_locked_config(path: str | Path = LOCKED_CONFIG) -> dict[str, Any]:
    config_path = Path(path).resolve()
    if config_path != LOCKED_CONFIG.resolve():
        raise ValueError(f"M1 requires the canonical locked config: {LOCKED_CONFIG}")
    actual_hash = sha256_file(config_path)
    if actual_hash != LOCKED_CONFIG_SHA256:
        raise ValueError(f"canonical M1 config SHA256 mismatch: expected {LOCKED_CONFIG_SHA256}, got {actual_hash}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_locked_config(config)
    return config


def manifest_hash(image_ids: Sequence[int]) -> str:
    encoded = json.dumps(sorted(int(value) for value in image_ids), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def split_manifest(loader: Any) -> dict[str, Any]:
    image_ids = getattr(loader.dataset, "img_ids", None)
    if image_ids is None:
        return {"count": len(loader.dataset)}
    return {"count": len(image_ids), "image_ids_sha256": manifest_hash(image_ids)}


def validation_manifest(loader: Any) -> dict[str, Any]:
    """M0-compatible name for the locked validation manifest helper."""
    return split_manifest(loader)


def train_manifest(loader: Any) -> dict[str, Any]:
    """Return the same canonical manifest shape for the train split."""
    return split_manifest(loader)


def verify_split_manifest(manifest: dict[str, Any], config: dict[str, Any], split: str) -> None:
    dataset = config["dataset"]
    if split == "train":
        expected = {"count": int(dataset["train_images"]), "image_ids_sha256": dataset["train_image_ids_sha256"]}
    elif split == "val":
        expected = {"count": int(dataset["validation_images"]), "image_ids_sha256": dataset["val_image_ids_sha256"]}
    else:
        raise ValueError(f"unsupported split {split!r}")
    if manifest != expected:
        raise ValueError(f"{split} split mismatch: expected {expected}, got {manifest}")


def verify_validation_manifest(manifest: dict[str, Any], config: dict[str, Any]) -> None:
    verify_split_manifest(manifest, config, "val")


def verify_train_manifest(manifest: dict[str, Any], config: dict[str, Any]) -> None:
    verify_split_manifest(manifest, config, "train")


def build_cache_record(
    *, image_id: int, spatial_features: Any, class_logits: Any, predicted_labels: Any,
    scores: Any, boxes: Any, image_size: Any, action_targets: Any, move_targets: Any,
) -> dict[str, Any]:
    record = {
        "image_id": int(image_id),
        "spatial_features": torch.as_tensor(spatial_features).detach().to(device="cpu", dtype=torch.float16),
        "class_logits": torch.as_tensor(class_logits).detach().to(device="cpu", dtype=torch.float32),
        "predicted_labels": torch.as_tensor(predicted_labels).detach().to(device="cpu", dtype=torch.long),
        "scores": torch.as_tensor(scores).detach().to(device="cpu", dtype=torch.float32),
        "boxes": torch.as_tensor(boxes).detach().to(device="cpu", dtype=torch.float32),
        "image_size": tuple(int(value) for value in torch.as_tensor(image_size).reshape(-1).tolist()),
        "action_targets": torch.as_tensor(action_targets).detach().to(device="cpu", dtype=torch.long),
        "move_targets": torch.as_tensor(move_targets).detach().to(device="cpu", dtype=torch.long),
    }
    if set(record) != CACHE_KEYS or any(term in key.lower() for key in record for term in FORBIDDEN_CACHE_TERMS):
        raise ValueError("oracle cache contains a forbidden GT/quality field")
    count = int(record["boxes"].shape[0])
    if record["spatial_features"].shape[0] != count or record["class_logits"].shape[0] != count:
        raise ValueError("cache detector tensors must share proposal count")
    for key in ("predicted_labels", "scores", "action_targets", "move_targets"):
        if record[key].shape != (count,):
            raise ValueError(f"cache field {key} must have shape (N,)")
    if len(record["image_size"]) != 2:
        raise ValueError("cache image_size must contain height and width")
    return record


def build_oracle_supervision(pool: Sequence[dict[str, Any]], beam_action_ids: Sequence[str], proposal_count: int) -> dict[str, list[int]]:
    action_targets = [0] * int(proposal_count)
    move_targets = [0] * int(proposal_count)
    selected = {str(action_id) for action_id in beam_action_ids}
    for item in pool:
        proposal_index = int(item["proposal_index"])
        if not 0 <= proposal_index < proposal_count:
            raise ValueError("oracle proposal index is outside the cached proposal set")
        candidate_index = int(item["candidate_index"])
        action_targets[proposal_index] = candidate_index
        if str(item["action_id"]) in selected:
            move_targets[proposal_index] = 1
    return {"action_targets": action_targets, "move_targets": move_targets, "candidate_indices": list(action_targets)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=LOCKED_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "nwpu_m1_set_policy_s42_fulltrain_fullval")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--require-full-validation", action="store_true")
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args(argv)


def validate_cli_args(args: argparse.Namespace) -> None:
    if args.require_full_validation and (args.limit_train is not None or args.limit_val is not None):
        raise ValueError("--require-full-validation cannot be combined with --limit-train or --limit-val")
    for name in ("limit_train", "limit_val"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def experiment_scopes(limit_train: int | None, limit_val: int | None) -> dict[str, str]:
    return {
        "experiment_scope": "full" if limit_train is None and limit_val is None else "smoke",
        "train_scope": "full_train" if limit_train is None else "limited_train",
        "eval_scope": "full_val" if limit_val is None else "smoke_val",
    }


def quantize_policy_spatial_features(spatial: torch.Tensor) -> torch.Tensor:
    """Match validation inputs to the locked fp16 cache precision."""
    return spatial.to(dtype=torch.float16).to(dtype=torch.float32)


def evaluate_m1_gates(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    gates_cfg = config["gates"]
    parity = summary.get("parity", {})
    g0_cfg = gates_cfg["G0_parity"]
    g0 = bool(parity.get("passed", False) and int(parity.get("mismatched_images", 10**9)) <= g0_cfg["max_mismatched_images"] and float(parity.get("max_box_abs_error", float("inf"))) <= g0_cfg["max_box_abs_error"] and float(parity.get("max_score_abs_error", float("inf"))) <= g0_cfg["max_score_abs_error"])
    g1 = bool(summary.get("gt_free_eval", False))
    delta = summary.get("detector_deltas", {})
    g2_cfg = gates_cfg["G2_learned_vs_identity"]
    g2 = bool(float(delta.get("ap75", float("-inf"))) >= g2_cfg["min_ap75_delta"] and float(delta.get("ap50", float("-inf"))) >= g2_cfg["min_ap50_delta"] and float(delta.get("false_positive_rate", float("inf"))) <= g2_cfg["max_fpr_delta"] and float(delta.get("num_predictions_relative", float("inf"))) <= g2_cfg["max_prediction_relative_delta"])
    shuffled = summary.get("learned_vs_shuffled", {})
    g3 = float(shuffled.get("ap75", float("-inf"))) >= gates_cfg["G3_learned_vs_shuffled"]["min_ap75_delta"]
    energy = summary.get("learned_vs_energy_zero", {})
    g4 = float(energy.get("ap75", float("-inf"))) >= gates_cfg["G4_energy"]["min_ap75_delta"]
    gates = {"G0_parity": g0, "G1_gt_free_eval": g1, "G2_learned_vs_identity": g2, "G3_learned_vs_shuffled": g3, "G4_energy": g4}
    structural_passed = all((g0, g1, g2, g3))
    return {
        "all_passed": structural_passed,
        "all_with_energy_passed": structural_passed and g4,
        "gates": gates,
        "note": "G4 reports whether the explicit energy term adds value; it does not redefine structural failure.",
    }


def _load_m0() -> Any:
    spec = importlib.util.spec_from_file_location("m1_reused_m0_set_oracle", M0_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load M0 helper from {M0_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_policy_api() -> tuple[Any, Any, Any, Any]:
    from spectral_detection_posttrain.methods.energy_transport import NMSAwareSetPolicyHead, SetPolicyLossConfig, select_set_policy_actions, set_policy_loss
    return NMSAwareSetPolicyHead, SetPolicyLossConfig, select_set_policy_actions, set_policy_loss


def _make_detector_config(config: dict[str, Any], data_root: Path, annotation: Path) -> dict[str, Any]:
    detector = config["detector"]
    return {
        "seed": config["dataset"]["seed"],
        "data_seed": config["dataset"]["data_seed"],
        "data": {"root": str(data_root), "annotation": str(annotation), "max_size": detector["max_size"], "train_fraction": config["dataset"]["train_fraction"], "num_workers": 0},
        "model": {"name": detector["model_name"], "model_name": detector["model_name"], "pretrained": False, "num_classes": 11, "min_size": detector["min_size"], "max_size": detector["max_size"]},
        "train": {"batch_size": 1},
        "eval": {"batch_size": 1},
    }


def _verify_input_hashes(config: dict[str, Any], checkpoint: Path, annotation: Path) -> dict[str, str]:
    if not checkpoint.is_file() or not annotation.is_file():
        raise FileNotFoundError("locked checkpoint and annotation must exist")
    hashes = {"checkpoint_sha256": sha256_file(checkpoint), "annotation_sha256": sha256_file(annotation)}
    if hashes["checkpoint_sha256"] != config["detector"]["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA256 mismatch")
    if hashes["annotation_sha256"] != config["dataset"]["annotation_sha256"]:
        raise ValueError("annotation SHA256 mismatch")
    return hashes


def verify_locked_hashes(config: dict[str, Any], checkpoint: str | Path, annotation: str | Path) -> dict[str, str]:
    """Verify the immutable detector and annotation inputs named by config."""
    return _verify_input_hashes(config, Path(checkpoint), Path(annotation))


def verify_source_m0(config: dict[str, Any]) -> dict[str, Any]:
    source = config["source_m0"]
    result_path = (ROOT / source["result"]).resolve()
    if not result_path.is_file():
        raise FileNotFoundError(f"required M0 result does not exist: {result_path}")
    actual_hash = sha256_file(result_path)
    if actual_hash != source["result_sha256"]:
        raise ValueError("source M0 result SHA256 mismatch")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("completed") is not True or payload.get("gates", {}).get("all_passed") is not True:
        raise ValueError("source M0 result is not complete with all gates passed")
    return {"path": str(result_path), "sha256": actual_hash, "all_gates_passed": True}


def _observable_mask(logits: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
    if logits.shape[-1] <= 1:
        foreground = torch.ones_like(scores, dtype=torch.bool)
    else:
        probabilities = torch.softmax(logits, dim=-1)
        foreground = probabilities[:, 1:].amax(dim=1) > probabilities[:, 0]
    return (scores >= float(threshold)) & foreground


def _policy_input(record: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, ...]:
    spatial = torch.as_tensor(record["spatial_features"], dtype=torch.float32, device=device)
    logits = torch.as_tensor(record["class_logits"], dtype=torch.float32, device=device)
    labels = torch.as_tensor(record["predicted_labels"], dtype=torch.long, device=device)
    scores = torch.as_tensor(record["scores"], dtype=torch.float32, device=device)
    boxes = torch.as_tensor(record["boxes"], dtype=torch.float32, device=device)
    return spatial, logits, labels, scores, boxes


def _loss_config(config: dict[str, Any], loss_config_cls: Any) -> Any:
    wanted = {
        "action_weight": config["training"]["action_weight"],
        "move_weight": config["training"]["move_weight"],
        "identity_weight": config["training"]["identity_weight"],
        "move_positive_weight": config["training"]["positive_move_weight"],
        "move_threshold": config["candidate_pool"]["move_threshold"],
    }
    accepted = set(inspect.signature(loss_config_cls).parameters)
    return loss_config_cls(**{key: value for key, value in wanted.items() if key in accepted})


def _new_policy(config: dict[str, Any], spatial_channels: int, device: torch.device) -> Any:
    head_cls, _, _, _ = _load_policy_api()
    from spectral_detection_posttrain.methods.energy_transport import build_symmetric_box_candidates
    candidates = build_symmetric_box_candidates(tuple(config["candidate_pool"]["step_sizes"])).to(device)
    head = head_cls(
        in_channels=spatial_channels,
        num_classes=11,
        candidate_deltas=candidates,
        hidden_dim=config["policy"]["hidden_dim"],
        spatial_size=config["policy"]["spatial_size"],
        energy_weight=float(config["policy"]["energy_weight"]),
    ).to(device)
    return head


def _write_cache(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "torch_pt", "records": list(records)}, path)


def _read_cache(path: Path) -> list[dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format") != "torch_pt":
        raise ValueError("unsupported oracle cache payload")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("oracle cache records must be a list")
    for record in records:
        if not isinstance(record, dict) or set(record) != CACHE_KEYS:
            raise ValueError("oracle cache record schema mismatch")
    return records


def summarize_cache_storage(records: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    total_proposals = sum(int(record["spatial_features"].shape[0]) for record in records)
    spatial_storage_bytes = sum(
        int(record["spatial_features"].numel() * record["spatial_features"].element_size())
        for record in records
    )
    return {
        "total_proposals": total_proposals,
        "spatial_storage_bytes": spatial_storage_bytes,
        "mean_proposals_per_image": total_proposals / max(1, len(records)),
    }


def validate_cache_metadata(metadata: dict[str, Any], expected: dict[str, Any]) -> None:
    for key in (
        "checkpoint_sha256",
        "annotation_sha256",
        "config_sha256",
        "train_manifest",
        "source_m0_sha256",
        "code_commit",
        "cache_schema_version",
    ):
        if metadata.get(key) != expected.get(key):
            raise ValueError(f"cache metadata mismatch for {key}")


def aggregate_training_metrics(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    keys = (
        "loss_total",
        "loss_action",
        "loss_move",
        "action_accuracy",
        "move_accuracy",
        "action_positive_accuracy",
    )
    if not rows:
        raise ValueError("training metric rows cannot be empty")
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}


def policy_output_diagnostics(
    output: Any,
    selection: Any,
    *,
    observable_mask: torch.Tensor,
    move_threshold: float,
    action_energy_scale: float,
) -> dict[str, float | int]:
    if action_energy_scale <= 0.0:
        raise ValueError("action_energy_scale must be positive")
    if output.action_logits.ndim != 2 or output.action_logits.shape[1] <= 1:
        raise ValueError("policy diagnostics require identity plus non-identity actions")
    observable = observable_mask.bool()
    best_non_identity = output.action_logits[:, 1:].amax(dim=1)
    action_margin = best_non_identity - output.action_logits[:, 0]
    gate_positive = observable & output.move_logits.gt(float(move_threshold))
    margin_positive = observable & action_margin.gt(0.0)
    selected_energy = selection.box_delta.square().sum() / float(action_energy_scale)
    count = int(observable.sum().item())
    return {
        "observable_count": count,
        "move_gate_positive": int(gate_positive.sum().item()),
        "action_margin_positive": int(margin_positive.sum().item()),
        "eligible_before_budget": int((gate_positive & margin_positive).sum().item()),
        "selected_count": int(selection.selected_mask.sum().item()),
        "selected_energy_sum": float(selected_energy.item()),
        "move_logit_mean": float(output.move_logits[observable].mean().item()) if count else 0.0,
        "action_margin_mean": float(action_margin[observable].mean().item()) if count else 0.0,
    }


@torch.no_grad()
def build_train_cache(model: torch.nn.Module, loader: Any, config: dict[str, Any], device: torch.device, cache_path: Path) -> dict[str, Any]:
    m0 = _load_m0()
    oracle_config = m0.load_locked_config()
    records: list[dict[str, Any]] = []
    model.eval()
    for images, batch_targets in loader:
        images_device = [image.to(device) for image in images]
        target_for_oracle = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch_targets[0].items()}
        batch = m0.extract_proposal_action_batch(model, images_device, [target_for_oracle], box_base="decoded")
        pool = m0._candidate_pool_for_batch(batch, oracle_config)
        search_record, _, _ = m0._run_search(batch, target_for_oracle, pool, model, oracle_config, nms_threshold=float(config["detector"]["nms_threshold"]))
        supervision = build_oracle_supervision([item.__dict__ for item in pool], search_record["selected"]["set_beam"], int(batch.state.boxes.shape[0]))
        image_id = int(target_for_oracle["image_id"].flatten()[0].item())
        records.append(build_cache_record(image_id=image_id, spatial_features=batch.spatial_features, class_logits=batch.class_logits, predicted_labels=batch.state.labels, scores=batch.state.scores, boxes=batch.state.boxes, image_size=batch.image_sizes[0], action_targets=supervision["action_targets"], move_targets=supervision["move_targets"]))
    _write_cache(cache_path, records)
    return {
        "path": str(cache_path),
        "records": len(records),
        "format": "torch_pt",
        "train_only_oracle": True,
        "contains_gt_inputs": False,
        "contains_oracle_labels": True,
        **summarize_cache_storage(records),
    }


def train_policy(cache: Sequence[dict[str, Any]], config: dict[str, Any], device: torch.device, run_dir: Path) -> tuple[Any, dict[str, Any]]:
    if not cache:
        raise ValueError("oracle cache is empty")
    _, loss_config_cls, _, loss_fn = _load_policy_api()
    head = _new_policy(config, len(cache[0]["spatial_features"][0]), device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])
    loss_config = _loss_config(config, loss_config_cls)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_path = run_dir / "policy_best.pt"
    for epoch in range(int(config["training"]["epochs"])):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_rows: list[dict[str, float]] = []
        order = list(range(len(cache)))
        random.Random(int(config["dataset"]["seed"]) + epoch).shuffle(order)
        for order_index, record_index in enumerate(order):
            record = cache[record_index]
            spatial, logits, labels, scores, boxes = _policy_input(record, device)
            output = head(spatial, logits, labels, scores, boxes, image_size=tuple(record["image_size"]), energy_weight=float(config["policy"]["energy_weight"]))
            observable = _observable_mask(logits, scores, config["detector"]["score_threshold"])
            losses = loss_fn(output, action_targets=torch.as_tensor(record["action_targets"], device=device), move_targets=torch.as_tensor(record["move_targets"], dtype=torch.float32, device=device), supervision_mask=observable, config=loss_config)
            (losses["loss_total"] / int(config["training"]["grad_accum_steps"])).backward()
            if (order_index + 1) % int(config["training"]["grad_accum_steps"]) == 0 or order_index + 1 == len(cache):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            epoch_rows.append({key: float(value.detach().item()) for key, value in losses.items()})
        epoch_summary = aggregate_training_metrics(epoch_rows)
        mean_loss = epoch_summary["loss_total"]
        history.append({"epoch": epoch + 1, **epoch_summary})
        torch.save({"state_dict": head.state_dict(), "epoch": epoch + 1, "config_sha256": sha256_file(LOCKED_CONFIG)}, run_dir / "policy_last.pt")
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save({"state_dict": head.state_dict(), "epoch": epoch + 1, "config_sha256": sha256_file(LOCKED_CONFIG)}, best_path)
    head.load_state_dict(torch.load(best_path, map_location=device)["state_dict"])
    return head, {
        "history": history,
        "best_training_loss": best_loss,
        "selected_checkpoint": str(best_path),
        "selection_rule": "minimum fixed-schedule train oracle-distillation loss; validation is evaluated once",
        "final_checkpoint": str(run_dir / "policy_last.pt"),
    }


def _zero_actions(state: Any) -> Any:
    from spectral_detection_posttrain.methods.energy_transport import ROITransportActions
    return ROITransportActions(feature_delta=state.features.new_zeros(state.features.shape), score_delta=state.scores.new_zeros(state.scores.shape), box_delta=state.boxes.new_zeros(state.boxes.shape), keep_logit=state.scores.new_zeros(state.scores.shape))


def _selection_to_actions(selection: Any, state: Any) -> Any:
    from spectral_detection_posttrain.methods.energy_transport import ROITransportActions
    return ROITransportActions(feature_delta=state.features.new_zeros(state.features.shape), score_delta=state.scores.new_zeros(state.scores.shape), box_delta=selection.box_delta, keep_logit=state.scores.new_zeros(state.scores.shape))


@torch.no_grad()
def evaluate_validation(model: torch.nn.Module, policy: torch.nn.Module, loader: Any, config: dict[str, Any], device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
    from spectral_detection_posttrain.trainers.detection.action_local_transport import action_batch_to_predictions, extract_proposal_action_batch
    from spectral_detection_posttrain.methods.energy_transport import build_symmetric_box_candidates, select_set_policy_actions
    m0 = _load_m0()
    candidates = build_symmetric_box_candidates(tuple(config["candidate_pool"]["step_sizes"])).to(device)
    predictions = {mode: [] for mode in config["modes"]}
    targets_for_metrics: list[dict[str, torch.Tensor]] = []
    parity = {"mismatched_images": 0, "max_box_abs_error": 0.0, "max_score_abs_error": 0.0, "passed": True}
    diagnostics = {
        mode: {
            "observable_count": 0,
            "move_gate_positive": 0,
            "action_margin_positive": 0,
            "eligible_before_budget": 0,
            "selected_count": 0,
            "selected_energy_sum": 0.0,
            "move_logit_weighted_sum": 0.0,
            "action_margin_weighted_sum": 0.0,
        }
        for mode in config["modes"]
    }
    model.eval()
    policy.eval()
    for images, batch_targets in loader:
        images_device = [image.to(device) for image in images]
        target_for_metrics = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch_targets[0].items()}
        batch = extract_proposal_action_batch(model, images_device, targets=None, box_base="decoded")
        native_prediction = {key: value.detach().cpu() for key, value in model(images_device)[0].items()}
        identity_prediction = action_batch_to_predictions(batch, _zero_actions(batch.state), score_threshold=float(config["detector"]["score_threshold"]), nms_threshold=float(config["detector"]["nms_threshold"]), detections_per_img=int(config["detector"]["detections_per_image"]), native_model=model)[0]
        box_error, score_error, mismatch, same = m0._compare_predictions(native_prediction, identity_prediction)
        parity["mismatched_images"] += 0 if same else 1
        parity["max_box_abs_error"] = max(parity["max_box_abs_error"], box_error)
        parity["max_score_abs_error"] = max(parity["max_score_abs_error"], score_error)
        parity["passed"] = parity["passed"] and same
        targets_for_metrics.append({key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in target_for_metrics.items()})
        image_id = int(target_for_metrics["image_id"].flatten()[0].item())
        observable = _observable_mask(batch.state.logits, batch.state.scores, config["detector"]["score_threshold"])
        image_indices = batch.state.image_indices
        for mode in config["modes"]:
            if mode == "identity":
                actions = _zero_actions(batch.state)
            else:
                spatial = quantize_policy_spatial_features(batch.spatial_features)
                if mode == "shuffled_spatial" and spatial is not None and spatial.shape[0] > 1:
                    generator = torch.Generator(device=spatial.device).manual_seed(31415 + image_id)
                    spatial = spatial[torch.randperm(spatial.shape[0], generator=generator, device=spatial.device)]
                output = policy(spatial, batch.class_logits, batch.state.labels, batch.state.scores, batch.state.boxes, image_size=batch.image_sizes[0], energy_weight=0.0 if mode == "energy_zero" else float(config["policy"]["energy_weight"]))
                selected = select_set_policy_actions(output, candidates, image_indices=image_indices, observable_mask=observable, max_actions_per_image=int(config["candidate_pool"]["action_budget"]), move_threshold=float(config["candidate_pool"]["move_threshold"]))
                actions = _selection_to_actions(selected, batch.state)
                row_diagnostics = policy_output_diagnostics(
                    output,
                    selected,
                    observable_mask=observable,
                    move_threshold=float(config["candidate_pool"]["move_threshold"]),
                    action_energy_scale=float(config["candidate_pool"]["action_energy_scale"]),
                )
                for key in ("observable_count", "move_gate_positive", "action_margin_positive", "eligible_before_budget", "selected_count"):
                    diagnostics[mode][key] += int(row_diagnostics[key])
                diagnostics[mode]["selected_energy_sum"] += float(row_diagnostics["selected_energy_sum"])
                diagnostics[mode]["move_logit_weighted_sum"] += float(row_diagnostics["move_logit_mean"]) * int(row_diagnostics["observable_count"])
                diagnostics[mode]["action_margin_weighted_sum"] += float(row_diagnostics["action_margin_mean"]) * int(row_diagnostics["observable_count"])
            predictions[mode].append(action_batch_to_predictions(batch, actions, score_threshold=float(config["detector"]["score_threshold"]), nms_threshold=float(config["detector"]["nms_threshold"]), detections_per_img=int(config["detector"]["detections_per_image"]), native_model=model)[0])
    metrics = {mode: evaluate_detection_predictions(predictions[mode], targets_for_metrics, score_threshold=float(config["detector"]["score_threshold"]), per_class=True, per_size=True, num_classes=11) for mode in config["modes"]}
    identity = metrics["identity"]
    learned = metrics["learned_set"]
    shuffled = metrics["shuffled_spatial"]
    energy_zero = metrics["energy_zero"]
    summary = {"parity": parity, "gt_free_eval": True, "detector_deltas": {"ap75": float(learned["ap75"] - identity["ap75"]), "ap50": float(learned["ap50"] - identity["ap50"]), "false_positive_rate": float(learned["false_positive_rate"] - identity["false_positive_rate"]), "num_predictions_relative": float((learned["num_predictions"] - identity["num_predictions"]) / max(1, identity["num_predictions"]))}, "learned_vs_shuffled": {"ap75": float(learned["ap75"] - shuffled["ap75"])}, "learned_vs_energy_zero": {"ap75": float(learned["ap75"] - energy_zero["ap75"])}}
    for mode in diagnostics:
        observable_count = int(diagnostics[mode]["observable_count"])
        diagnostics[mode]["mean_selected_energy_per_image"] = diagnostics[mode]["selected_energy_sum"] / max(1, len(targets_for_metrics))
        diagnostics[mode]["move_logit_mean"] = diagnostics[mode].pop("move_logit_weighted_sum") / max(1, observable_count)
        diagnostics[mode]["action_margin_mean"] = diagnostics[mode].pop("action_margin_weighted_sum") / max(1, observable_count)
    return {"metrics": metrics, "summary": summary, "diagnostics": diagnostics}, {"predictions": predictions, "targets": targets_for_metrics}


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_cli_args(args)
    config = load_locked_config(args.config)
    if args.require_clean_git:
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True)
        if status.stdout.strip():
            raise RuntimeError("--require-clean-git requested but the repository is dirty")
    from spectral_detection_posttrain.datasets import build_nwpu_vhr10_loaders
    from spectral_detection_posttrain.models import build_detector
    from spectral_detection_posttrain.trainers.detection.action_local_transport import extract_proposal_action_batch
    from spectral_detection_posttrain.utils.io import load_checkpoint
    from spectral_detection_posttrain.utils.seed import resolve_device, set_seed
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = (args.checkpoint or ROOT / config["detector"]["checkpoint"]).resolve()
    annotation = (args.annotation or ROOT / "data" / "NWPU_VHR10_coco.json").resolve()
    data_root = (args.data_root or ROOT / "data" / "NWPU VHR-10 dataset").resolve()
    hashes = _verify_input_hashes(config, checkpoint, annotation)
    source_m0 = verify_source_m0(config)
    set_seed(int(config["dataset"]["seed"]))
    device = resolve_device({"device": args.device or ("cuda" if torch.cuda.is_available() else "cpu")})
    detector = build_detector(_make_detector_config(config, data_root, annotation)).to(device)
    load_checkpoint(detector, checkpoint, device)
    detector.eval()
    for parameter in detector.parameters():
        parameter.requires_grad_(False)
    train_loader, val_loader = build_nwpu_vhr10_loaders(_make_detector_config(config, data_root, annotation), limit_train=args.limit_train, limit_val=args.limit_val, batch_size=1)
    train_manifest = split_manifest(train_loader)
    val_manifest = split_manifest(val_loader)
    if args.limit_train is None:
        verify_split_manifest(train_manifest, config, "train")
    if args.limit_val is None:
        verify_split_manifest(val_manifest, config, "val")
    cache_path = run_dir / "oracle_cache.pt"
    cache_metadata_path = run_dir / "oracle_cache_metadata.json"
    code_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    expected_cache_metadata = {
        "checkpoint_sha256": hashes["checkpoint_sha256"],
        "annotation_sha256": hashes["annotation_sha256"],
        "train_manifest": train_manifest,
        "config_sha256": sha256_file(LOCKED_CONFIG),
        "source_m0_sha256": source_m0["sha256"],
        "code_commit": code_commit,
        "cache_schema_version": int(config["cache"]["schema_version"]),
    }
    if cache_path.exists() and not args.rebuild_cache:
        if not cache_metadata_path.is_file():
            raise FileNotFoundError("oracle cache exists without metadata")
        cache_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
        validate_cache_metadata(cache_metadata, expected_cache_metadata)
        cache = _read_cache(cache_path)
    else:
        cache_metadata = build_train_cache(detector, train_loader, config, device, cache_path)
        cache_metadata.update(expected_cache_metadata)
        cache_metadata_path.write_text(json.dumps(cache_metadata, indent=2) + "\n", encoding="utf-8")
        cache = _read_cache(cache_path)
    policy, training = train_policy(cache, config, device, run_dir)
    evaluation, _ = evaluate_validation(detector, policy, val_loader, config, device)
    from spectral_detection_posttrain.utils.git_state import get_git_state
    gates = evaluate_m1_gates(evaluation["summary"], config)
    scopes = experiment_scopes(args.limit_train, args.limit_val)
    result = {
        "completed": True,
        "scientific_status": "structural_gates_passed" if gates["all_passed"] else "structural_gates_failed",
        "version_id": config["version_id"],
        **scopes,
        "config": config,
        "inputs": {
            "checkpoint": str(checkpoint),
            "annotation": str(annotation),
            "checkpoint_sha256": hashes["checkpoint_sha256"],
            "annotation_sha256": hashes["annotation_sha256"],
            "config_sha256": sha256_file(LOCKED_CONFIG),
            "source_m0": source_m0,
            "train_manifest": train_manifest,
            "validation_manifest": val_manifest,
            "git_state": get_git_state(ROOT),
            "frozen_detector": True,
            "policy_only": True,
            "validation_policy_inputs": "detector_only_targets_none",
        },
        "cache_metadata": cache_metadata,
        "history": training["history"],
        "selected": {
            "checkpoint": training["selected_checkpoint"],
            "selection_rule": training["selection_rule"],
            "metrics": evaluation["metrics"]["learned_set"],
        },
        "final_metrics": evaluation["metrics"],
        "metrics": evaluation["metrics"],
        "diagnostics": evaluation["diagnostics"],
        "summary": evaluation["summary"],
        "gates": gates,
        "development_selection": "single preregistered development-validation evaluation after train-loss checkpoint selection",
    }
    output = args.output or run_dir / "eval_metrics.json"
    output.write_text(json.dumps(_json_value(result), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    print(json.dumps({"output": str(args.output or Path(args.run_dir) / "eval_metrics.json"), "completed": result["completed"], "gates": result["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
