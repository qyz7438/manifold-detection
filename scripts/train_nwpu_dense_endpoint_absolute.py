"""Train the locked train-only absolute dense set-energy endpoint probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as torch_f
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG_PATH = (
    ROOT
    / "spectral_detection_posttrain"
    / "configs"
    / "versions"
    / "det.energy.dense_endpoint.absolute.001.json"
)
CONFIG_SHA256 = "74b7e73b2b0ac8d3705d5022eae0a91807b2976e7fda29851a06c6f855cd10f9"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_ids_hash(image_ids: Sequence[int]) -> str:
    payload = json.dumps(sorted(int(value) for value in image_ids), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def load_config() -> dict[str, Any]:
    if sha256_file(CONFIG_PATH) != CONFIG_SHA256:
        raise ValueError("canonical dense absolute endpoint config SHA256 mismatch")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("version_id") != "det.energy.dense_endpoint.absolute.001":
        raise ValueError("dense absolute endpoint version mismatch")
    dataset = config.get("dataset", {})
    if dataset.get("detector_validation_forbidden") is not True:
        raise ValueError("detector validation must remain forbidden")
    if dataset.get("outer_read_once_after_selection") is not True:
        raise ValueError("outer split must remain read-once after selection")
    if config.get("teacher", {}).get("calibration_error_role") != "diagnostic_only_not_in_quality":
        raise ValueError("calibration error must not re-enter reduced teacher quality")
    return config


def _non_identity_permutation(count: int, seed: int) -> torch.Tensor:
    if count < 2:
        return torch.arange(count)
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(count, generator=generator)
    identity = torch.arange(count)
    return permutation.roll(1) if torch.equal(permutation, identity) else permutation


def feature_alignment_shuffle(features: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Break box-to-score/class alignment while preserving column marginals."""
    if features.ndim != 2 or features.shape[1] != 22:
        raise ValueError("post-NMS node features must have shape (N, 22)")
    output = features.clone()
    permutation = _non_identity_permutation(features.shape[0], seed).to(features.device)
    output[:, 1:7] = features[permutation, 1:7]
    return output


def _pairwise_accuracy(prediction: torch.Tensor, target: torch.Tensor) -> float:
    count = int(target.numel())
    if count < 2:
        return float("nan")
    indices = torch.triu_indices(count, count, offset=1)
    true_delta = target[indices[0]] - target[indices[1]]
    predicted_delta = prediction[indices[0]] - prediction[indices[1]]
    valid = true_delta.ne(0)
    if not valid.any():
        return float("nan")
    product = predicted_delta[valid] * true_delta[valid]
    return float((product.gt(0).float() + 0.5 * product.eq(0).float()).mean().item())


def _positive_auroc(prediction: torch.Tensor, target: torch.Tensor) -> float | None:
    positive = target.gt(0)
    negative = ~positive
    if not positive.any() or not negative.any():
        return None
    differences = prediction[positive][:, None] - prediction[negative][None, :]
    return float((differences.gt(0).float() + 0.5 * differences.eq(0).float()).mean().item())


def regression_metrics(
    prediction: torch.Tensor, target: torch.Tensor, *, fit_mean: float
) -> dict[str, float | None | int]:
    estimate = torch.as_tensor(prediction, dtype=torch.float64).flatten().cpu()
    truth = torch.as_tensor(target, dtype=torch.float64).flatten().cpu()
    if estimate.shape != truth.shape or truth.numel() == 0:
        raise ValueError("prediction and target must be aligned and non-empty")
    error = estimate - truth
    centered_prediction = estimate - estimate.mean()
    centered_target = truth - truth.mean()
    denominator = float(
        (centered_prediction.square().sum().sqrt() * centered_target.square().sum().sqrt()).item()
    )
    return {
        "count": int(truth.numel()),
        "mae": float(error.abs().mean().item()),
        "rmse": float(error.square().mean().sqrt().item()),
        "pearson": (
            None
            if denominator == 0.0
            else float((centered_prediction * centered_target).sum().item() / denominator)
        ),
        "pairwise_accuracy": _pairwise_accuracy(estimate, truth),
        "auroc_positive": _positive_auroc(estimate, truth),
        "positive_prevalence": float(truth.gt(0).double().mean().item()),
        "zero_baseline_mae": float(truth.abs().mean().item()),
        "fit_mean_baseline_mae": float((truth - float(fit_mean)).abs().mean().item()),
        "prediction_mean": float(estimate.mean().item()),
        "target_mean": float(truth.mean().item()),
        "target_std": float(truth.std(unbiased=False).item()),
    }


def _verify_sources(config: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for name, path_key, hash_key in (
        ("teacher_stats", "teacher_stats_result", "teacher_stats_sha256"),
        ("confounding", "confounding_result", "confounding_sha256"),
    ):
        path = (ROOT / config["sources"][path_key]).resolve()
        actual = sha256_file(path)
        if actual != config["sources"][hash_key]:
            raise ValueError(f"{name} source SHA256 mismatch")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("completed") is not True:
            raise ValueError(f"{name} source is incomplete")
        output[name] = {"path": str(path), "sha256": actual, "status": payload.get("scientific_status")}
    if output["confounding"]["status"] != config["sources"]["required_confounding_decision"]:
        raise ValueError("confounding source decision mismatch")
    return output


def _teacher_config(config: dict[str, Any]):
    from spectral_detection_posttrain.methods.energy_transport.dense_set_energy import DenseTeacherConfig

    return DenseTeacherConfig(
        **{key: config["teacher"][key] for key in DenseTeacherConfig.__dataclass_fields__}
    )


def _detector_config(config: dict[str, Any], data_root: Path, annotation: Path) -> dict[str, Any]:
    detector = config["detector"]
    return {
        "seed": int(config["training"]["seed"]),
        "data_seed": int(config["training"]["seed"]),
        "data": {
            "root": str(data_root),
            "annotation": str(annotation),
            "max_size": int(detector["max_size"]),
            "train_fraction": 0.7,
            "num_workers": int(config["resource"]["num_workers"]),
        },
        "model": {
            "name": detector["model_name"],
            "model_name": detector["model_name"],
            "pretrained": False,
            "num_classes": int(detector["num_classes"]),
            "min_size": int(detector["min_size"]),
            "max_size": int(detector["max_size"]),
        },
        "train": {"batch_size": 1},
        "eval": {"batch_size": 1},
    }


@torch.inference_mode()
def _build_records(
    detector: torch.nn.Module,
    image_ids: Sequence[int],
    data_root: Path,
    annotation: Path,
    config: dict[str, Any],
    device: torch.device,
    *,
    split_name: str,
) -> list[dict[str, Any]]:
    from spectral_detection_posttrain.datasets.nwpu_vhr10 import NWPUVHR10DetectionDataset
    from spectral_detection_posttrain.datasets.penn_fudan import detection_collate
    from spectral_detection_posttrain.methods.energy_transport.dense_endpoint import (
        build_sparse_pair_features,
        reduced_teacher_values,
    )
    from spectral_detection_posttrain.methods.energy_transport.dense_set_energy import dense_teacher_components
    from spectral_detection_posttrain.methods.energy_transport.post_nms_suppress import (
        build_post_nms_detection_features,
    )

    dataset = NWPUVHR10DetectionDataset(
        data_root,
        annotation,
        image_ids,
        max_size=int(config["detector"]["max_size"]),
    )
    if dataset.img_ids != sorted(int(value) for value in image_ids):
        raise RuntimeError(f"{split_name} dataset IDs do not match manifest")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(config["resource"]["num_workers"]),
        collate_fn=detection_collate,
    )
    teacher_config = _teacher_config(config)
    records = []
    for index, (images, targets) in enumerate(loader, start=1):
        image = images[0].to(device)
        prediction = detector([image])[0]
        image_size = tuple(int(value) for value in image.shape[-2:])
        node_features = build_post_nms_detection_features(
            prediction["boxes"],
            prediction["scores"],
            prediction["labels"],
            image_size,
            num_classes=int(config["detector"]["num_classes"]),
        )
        pair_features = build_sparse_pair_features(
            prediction["boxes"],
            prediction["scores"],
            prediction["labels"],
            image_size,
            min_iou=float(config["model"]["pair_min_iou"]),
        )
        image_id = int(targets[0]["image_id"].flatten()[0].item())
        box_permutation = _non_identity_permutation(
            int(prediction["boxes"].shape[0]),
            int(config["controls"]["topology_shuffle_seed"]) + image_id,
        ).to(device)
        topology_pair_features = build_sparse_pair_features(
            prediction["boxes"][box_permutation],
            prediction["scores"],
            prediction["labels"],
            image_size,
            min_iou=float(config["model"]["pair_min_iou"]),
        )
        target = {
            "boxes": targets[0]["boxes"].to(device),
            "labels": targets[0]["labels"].to(device),
        }
        components = dense_teacher_components(prediction, target, teacher_config)
        records.append(
            {
                "image_id": image_id,
                "node_features": node_features.detach().cpu().to(torch.float16),
                "pair_features": pair_features.detach().cpu().to(torch.float16),
                "topology_pair_features": topology_pair_features.detach().cpu().to(torch.float16),
                "teacher_values": reduced_teacher_values(components).detach().cpu().float(),
                "calibration_error_per_prediction": float(components.calibration_error.item())
                / max(int(components.prediction_count), 1),
                "prediction_count": int(components.prediction_count),
                "ground_truth_count": int(components.ground_truth_count),
                "duplicate_edge_count": int(components.duplicate_edge_count),
            }
        )
        if index % 25 == 0 or index == len(dataset):
            print(
                json.dumps(
                    {"cache_split": split_name, "progress_images": index, "total_images": len(dataset)}
                ),
                flush=True,
            )
    return records


def _write_cache(path: Path, records: Sequence[dict[str, Any]], metadata: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "dense_endpoint_absolute_v1", "metadata": metadata, "records": list(records)}, path)
    return sha256_file(path)


def _fit_feature_stats(records: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
    nodes = torch.cat([record["node_features"].float() for record in records], dim=0)
    pairs = torch.cat([record["pair_features"].float() for record in records], dim=0)
    return {
        "node_mean": nodes.mean(dim=0),
        "node_scale": nodes.std(dim=0, unbiased=False).clamp_min(1e-4),
        "pair_mean": pairs.mean(dim=0),
        "pair_scale": pairs.std(dim=0, unbiased=False).clamp_min(1e-4),
    }


def _features_for_arm(
    record: dict[str, Any],
    arm: str,
    feature_stats: dict[str, torch.Tensor],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    nodes = record["node_features"].float()
    if arm == "feature_alignment_shuffle":
        nodes = feature_alignment_shuffle(
            nodes,
            seed=int(config["controls"]["feature_shuffle_seed"]) + int(record["image_id"]),
        )
    pair_key = "topology_pair_features" if arm == "edge_topology_shuffle" else "pair_features"
    pairs = record[pair_key].float()
    nodes = (nodes - feature_stats["node_mean"]) / feature_stats["node_scale"]
    pairs = (pairs - feature_stats["pair_mean"]) / feature_stats["pair_scale"]
    return nodes.to(device), pairs.to(device)


def _targets(records: Sequence[dict[str, Any]], teacher_stats: Any) -> torch.Tensor:
    values = torch.stack([record["teacher_values"] for record in records])
    return teacher_stats.quality(values).detach().cpu().float()


def _new_model(config: dict[str, Any], device: torch.device):
    from spectral_detection_posttrain.methods.energy_transport.dense_endpoint import DenseSetEnergyEndpoint

    return DenseSetEnergyEndpoint(
        node_dim=int(config["model"]["node_dim"]),
        pair_dim=int(config["model"]["pair_dim"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
    ).to(device)


def _train_arm(
    records: Sequence[dict[str, Any]],
    true_targets: torch.Tensor,
    arm: str,
    weight_decay: float,
    feature_stats: dict[str, torch.Tensor],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    seed = int(config["controls"]["same_initialization_seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = _new_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["lr"]), weight_decay=float(weight_decay)
    )
    train_targets = true_targets.clone()
    if arm == "teacher_shuffle":
        permutation = _non_identity_permutation(
            len(records), int(config["controls"]["teacher_shuffle_seed"])
        )
        train_targets = train_targets[permutation]
    accumulation = int(config["training"]["grad_accum_images"])
    history = []
    model.train()
    for epoch in range(int(config["training"]["epochs"])):
        order = list(range(len(records)))
        random.Random(seed + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for position, record_index in enumerate(order, start=1):
            chunk_start = ((position - 1) // accumulation) * accumulation
            chunk_size = min(accumulation, len(order) - chunk_start)
            nodes, pairs = _features_for_arm(records[record_index], arm, feature_stats, config, device)
            prediction = model(nodes, pairs).quality
            target = train_targets[record_index].to(device)
            loss = torch_f.smooth_l1_loss(
                prediction,
                target,
                beta=float(config["training"]["smooth_l1_beta"]),
            )
            (loss / chunk_size).backward()
            total_loss += float(loss.detach().item())
            if position % accumulation == 0 or position == len(order):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        history.append(total_loss / len(records))
    return model, {
        "weight_decay": float(weight_decay),
        "epochs": int(config["training"]["epochs"]),
        "loss_first": history[0],
        "loss_last": history[-1],
        "loss_min": min(history),
    }


@torch.inference_mode()
def _predict(
    model: torch.nn.Module,
    records: Sequence[dict[str, Any]],
    arm: str,
    feature_stats: dict[str, torch.Tensor],
    config: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    output = []
    for record in records:
        nodes, pairs = _features_for_arm(record, arm, feature_stats, config, device)
        output.append(model(nodes, pairs).quality.detach().cpu())
    return torch.stack(output).float()


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    metadata: dict[str, Any],
    feature_stats: dict[str, torch.Tensor],
    teacher_stats: Any,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "metadata": metadata,
            "feature_stats": feature_stats,
            "teacher_median": teacher_stats.median,
            "teacher_iqr": teacher_stats.iqr,
        },
        path,
    )
    return sha256_file(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / "runs" / "nwpu_dense_endpoint_absolute_s42_nested"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    from spectral_detection_posttrain.methods.energy_transport.dense_endpoint import RobustTeacherStats
    from spectral_detection_posttrain.models import build_detector
    from spectral_detection_posttrain.utils.io import load_checkpoint
    from spectral_detection_posttrain.utils.seed import resolve_device, set_seed

    config = load_config()
    sources = _verify_sources(config)
    manifest_path = (ROOT / config["dataset"]["split_manifest"]).resolve()
    if sha256_file(manifest_path) != config["dataset"]["split_manifest_sha256"]:
        raise ValueError("dense endpoint split manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_contract = {
        "inner_fit": (config["dataset"]["inner_fit_count"], config["dataset"]["inner_fit_sha256"]),
        "inner_tune": (config["dataset"]["inner_tune_count"], config["dataset"]["inner_tune_sha256"]),
        "outer_train_heldout": (config["dataset"]["outer_count"], config["dataset"]["outer_sha256"]),
    }
    split_ids = {}
    for name, (count, expected_hash) in split_contract.items():
        values = [int(value) for value in manifest["splits"][name]["image_ids"]]
        if len(values) != int(count) or image_ids_hash(values) != expected_hash:
            raise ValueError(f"{name} split contract mismatch")
        split_ids[name] = values
    if set(split_ids["outer_train_heldout"]) & (
        set(split_ids["inner_fit"]) | set(split_ids["inner_tune"])
    ):
        raise ValueError("outer split overlaps model-selection splits")

    checkpoint = (args.checkpoint or ROOT / config["detector"]["checkpoint"]).resolve()
    annotation = (args.annotation or ROOT / "data" / "NWPU_VHR10_coco.json").resolve()
    data_root = (args.data_root or ROOT / "data" / "NWPU VHR-10 dataset").resolve()
    if sha256_file(checkpoint) != config["detector"]["checkpoint_sha256"]:
        raise ValueError("detector checkpoint SHA256 mismatch")
    if sha256_file(annotation) != config["dataset"]["annotation_sha256"]:
        raise ValueError("annotation SHA256 mismatch")
    git_dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    )
    if args.require_clean_git and git_dirty:
        raise RuntimeError("--require-clean-git requested but repository is dirty")

    set_seed(int(config["training"]["seed"]))
    device = resolve_device({"device": args.device or ("cuda" if torch.cuda.is_available() else "cpu")})
    detector = build_detector(_detector_config(config, data_root, annotation)).to(device)
    load_checkpoint(detector, checkpoint, device)
    detector.roi_heads.score_thresh = float(config["detector"]["score_threshold"])
    detector.roi_heads.nms_thresh = float(config["detector"]["nms_threshold"])
    detector.roi_heads.detections_per_img = int(config["detector"]["detections_per_image"])
    detector.eval()
    for parameter in detector.parameters():
        parameter.requires_grad_(False)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    fit_records = _build_records(
        detector,
        split_ids["inner_fit"],
        data_root,
        annotation,
        config,
        device,
        split_name="inner_fit",
    )
    tune_records = _build_records(
        detector,
        split_ids["inner_tune"],
        data_root,
        annotation,
        config,
        device,
        split_name="inner_tune",
    )
    fit_tune_cache = args.run_dir / "fit_tune_cache.pt"
    fit_tune_cache_sha = _write_cache(
        fit_tune_cache,
        [*fit_records, *tune_records],
        {
            "config_sha256": CONFIG_SHA256,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_dirty": git_dirty,
            "splits": ["inner_fit", "inner_tune"],
            "outer_read": False,
        },
    )
    teacher_values = torch.stack([record["teacher_values"] for record in fit_records])
    teacher_stats = RobustTeacherStats.fit(teacher_values)
    feature_stats = _fit_feature_stats(fit_records)
    fit_targets = _targets(fit_records, teacher_stats)
    tune_targets = _targets(tune_records, teacher_stats)
    fit_mean = float(fit_targets.mean().item())

    grid = []
    full_models: dict[float, torch.nn.Module] = {}
    for weight_decay in config["training"]["weight_decay_grid"]:
        model, training = _train_arm(
            fit_records,
            fit_targets,
            "local_full",
            float(weight_decay),
            feature_stats,
            config,
            device,
        )
        tune_prediction = _predict(model, tune_records, "local_full", feature_stats, config, device)
        tune_metrics = regression_metrics(tune_prediction, tune_targets, fit_mean=fit_mean)
        grid.append({"weight_decay": float(weight_decay), "training": training, "tune": tune_metrics})
        full_models[float(weight_decay)] = model
    selected_row = min(grid, key=lambda row: (float(row["tune"]["mae"]), float(row["weight_decay"])))
    selected_weight_decay = float(selected_row["weight_decay"])
    models = {"local_full": full_models[selected_weight_decay]}
    training_rows = {"local_full": selected_row["training"]}
    for arm in ("feature_alignment_shuffle", "teacher_shuffle", "edge_topology_shuffle"):
        models[arm], training_rows[arm] = _train_arm(
            fit_records,
            fit_targets,
            arm,
            selected_weight_decay,
            feature_stats,
            config,
            device,
        )

    # This is the first and only outer read, after all hyperparameters and controls are frozen.
    outer_records = _build_records(
        detector,
        split_ids["outer_train_heldout"],
        data_root,
        annotation,
        config,
        device,
        split_name="outer_train_heldout",
    )
    outer_cache = args.run_dir / "outer_read_once_cache.pt"
    outer_cache_sha = _write_cache(
        outer_cache,
        outer_records,
        {
            "config_sha256": CONFIG_SHA256,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_dirty": git_dirty,
            "split": "outer_train_heldout",
            "outer_read_once_after_selection": True,
            "selected_weight_decay": selected_weight_decay,
        },
    )
    outer_targets = _targets(outer_records, teacher_stats)
    evaluation: dict[str, dict[str, Any]] = {}
    checkpoints = {}
    for arm, model in models.items():
        evaluation[arm] = {}
        for split_name, records, targets in (
            ("inner_fit", fit_records, fit_targets),
            ("inner_tune", tune_records, tune_targets),
            ("outer_train_heldout", outer_records, outer_targets),
        ):
            prediction = _predict(model, records, arm, feature_stats, config, device)
            evaluation[arm][split_name] = regression_metrics(prediction, targets, fit_mean=fit_mean)
        checkpoints[arm] = _save_checkpoint(
            args.run_dir / arm / "endpoint.pth",
            model,
            {
                "arm": arm,
                "config_sha256": CONFIG_SHA256,
                "weight_decay": selected_weight_decay,
            },
            feature_stats,
            teacher_stats,
        )

    full_outer = evaluation["local_full"]["outer_train_heldout"]
    full_fit = evaluation["local_full"]["inner_fit"]
    control_arms = ("feature_alignment_shuffle", "teacher_shuffle", "edge_topology_shuffle")
    pairwise_gains = {
        arm: float(full_outer["pairwise_accuracy"])
        - float(evaluation[arm]["outer_train_heldout"]["pairwise_accuracy"])
        for arm in control_arms
    }
    mae_gains = {
        arm: float(evaluation[arm]["outer_train_heldout"]["mae"]) - float(full_outer["mae"])
        for arm in control_arms
    }
    best_constant_mae = min(
        float(full_outer["zero_baseline_mae"]), float(full_outer["fit_mean_baseline_mae"])
    )
    constant_relative_gain = (
        (best_constant_mae - float(full_outer["mae"])) / best_constant_mae
        if best_constant_mae > 0
        else float("-inf")
    )
    gates = {
        "outer_support": int(full_outer["count"]) >= int(config["gates"]["outer_min_images"]),
        "outer_pairwise": float(full_outer["pairwise_accuracy"])
        >= float(config["gates"]["outer_pairwise_accuracy_min"]),
        "outer_auroc": full_outer["auroc_positive"] is not None
        and float(full_outer["auroc_positive"]) >= float(config["gates"]["outer_auroc_min"]),
        "pairwise_controls": all(
            value >= float(config["gates"]["pairwise_gain_over_each_control_min"])
            for value in pairwise_gains.values()
        ),
        "mae_controls": all(
            value >= float(config["gates"]["mae_gain_over_each_control_min"])
            for value in mae_gains.values()
        ),
        "constant_baseline": constant_relative_gain
        >= float(config["gates"]["mae_relative_gain_over_best_constant_min"]),
        "fit_outer_gap": float(full_fit["pairwise_accuracy"])
        - float(full_outer["pairwise_accuracy"])
        <= float(config["gates"]["fit_to_outer_pairwise_gap_max"]),
    }
    all_passed = all(gates.values())
    result = {
        "completed": True,
        "scientific_status": (
            "absolute_endpoint_supported_local_delta_probe_allowed"
            if all_passed
            else "reduced_absolute_endpoint_frozen"
        ),
        "version_id": config["version_id"],
        "experiment_scope": "nwpu_train_only_nested_absolute_endpoint",
        "config_sha256": CONFIG_SHA256,
        "split_manifest_sha256": config["dataset"]["split_manifest_sha256"],
        "checkpoint_sha256": config["detector"]["checkpoint_sha256"],
        "annotation_sha256": config["dataset"]["annotation_sha256"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": git_dirty,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(device),
        "sources": sources,
        "fit_tune_cache_sha256": fit_tune_cache_sha,
        "outer_cache_sha256": outer_cache_sha,
        "outer_read_count": 1,
        "detector_validation_read": False,
        "teacher_stats": {
            "median": teacher_stats.median.tolist(),
            "iqr": teacher_stats.iqr.tolist(),
            "fit_target_mean": fit_mean,
        },
        "feature_stats_scope": "inner_fit_only",
        "weight_decay_grid": grid,
        "selected_weight_decay": selected_weight_decay,
        "training": training_rows,
        "checkpoint_hashes": checkpoints,
        "evaluation": evaluation,
        "summary": {
            "outer_full": full_outer,
            "pairwise_gains_over_controls": pairwise_gains,
            "mae_gains_over_controls": mae_gains,
            "mae_relative_gain_over_best_constant": constant_relative_gain,
            "fit_to_outer_pairwise_gap": float(full_fit["pairwise_accuracy"])
            - float(full_outer["pairwise_accuracy"]),
        },
        "gates": {"all_passed": all_passed, "gates": gates},
        "decision_rule": config["decision_rule"],
        "claim_boundary": config["claim_boundary"],
        "validated_claim": False,
    }
    (args.run_dir / "eval_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    result = run(_parse_args())
    print(
        json.dumps(
            {"status": result["scientific_status"], "summary": result["summary"], "gates": result["gates"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
