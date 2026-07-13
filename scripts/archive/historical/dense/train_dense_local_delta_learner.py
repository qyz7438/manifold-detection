"""Train the minimal train-only shift-invariant local Delta-Q endpoint learner."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    / "det.energy.dense_local_delta_learner.001.json"
)
CONFIG_SHA256 = "6bfee0169539dc32f0e90736f0cb1de7a59db4928e9af22cd79f16461882185b"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_ids_hash(image_ids: Sequence[int]) -> str:
    encoded = json.dumps(sorted(int(value) for value in image_ids), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def load_config() -> dict[str, Any]:
    if sha256_file(CONFIG_PATH) != CONFIG_SHA256:
        raise ValueError("canonical dense local learner config SHA256 mismatch")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("version_id") != "det.energy.dense_local_delta_learner.001":
        raise ValueError("dense local learner version mismatch")
    if config.get("dataset", {}).get("all_other_splits_forbidden") is not True:
        raise ValueError("dense local learner must remain train-only")
    return config


def split_image_ids(
    image_ids: Sequence[int], config: dict[str, Any]
) -> tuple[list[int], list[int]]:
    seed = int(config["training"]["seed"])
    ordered = sorted(
        (int(value) for value in image_ids),
        key=lambda image_id: hashlib.sha256(
            f"dense-local-learner:{seed}:{image_id}".encode("ascii")
        ).digest(),
    )
    fit_count = int(config["dataset"]["fit_count"])
    return sorted(ordered[:fit_count]), sorted(ordered[fit_count:])


def imagewise_pairwise_accuracy(
    prediction: torch.Tensor, target: torch.Tensor, image_ids: torch.Tensor
) -> float:
    prediction = torch.as_tensor(prediction).flatten().cpu()
    target = torch.as_tensor(target).flatten().cpu()
    image_ids = torch.as_tensor(image_ids).flatten().cpu()
    values = []
    for image_id in torch.unique(image_ids, sorted=True):
        rows = torch.nonzero(image_ids.eq(image_id), as_tuple=False).flatten()
        if rows.numel() < 2:
            continue
        indices = torch.triu_indices(rows.numel(), rows.numel(), offset=1)
        true_delta = target[rows[indices[0]]] - target[rows[indices[1]]]
        predicted_delta = prediction[rows[indices[0]]] - prediction[rows[indices[1]]]
        valid = true_delta.ne(0)
        if valid.any():
            product = true_delta[valid] * predicted_delta[valid]
            values.extend((product.gt(0).float() + 0.5 * product.eq(0).float()).tolist())
    return float(torch.tensor(values).mean().item()) if values else float("nan")


def _sign_auroc(prediction: torch.Tensor, target: torch.Tensor) -> float | None:
    positive = target.gt(0)
    negative = ~positive
    if not positive.any() or not negative.any():
        return None
    differences = prediction[positive][:, None] - prediction[negative][None, :]
    return float((differences.gt(0).float() + 0.5 * differences.eq(0).float()).mean().item())


def _metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    image_ids: torch.Tensor,
    identity_mask: torch.Tensor,
) -> dict[str, float | int | None]:
    identity = identity_mask.bool()
    selected = ~identity
    estimate = prediction[selected].double().cpu()
    truth = target[selected].double().cpu()
    ids = image_ids[selected].cpu()
    mae = float((estimate - truth).abs().mean().item())
    zero_mae = float(truth.abs().mean().item())
    return {
        "rows": int(truth.numel()),
        "images": int(torch.unique(ids).numel()),
        "mae": mae,
        "zero_baseline_mae": zero_mae,
        "mae_relative_gain_over_zero": (zero_mae - mae) / zero_mae if zero_mae > 0 else float("-inf"),
        "imagewise_pairwise_accuracy": imagewise_pairwise_accuracy(estimate, truth, ids),
        "sign_auroc": _sign_auroc(estimate, truth),
        "identity_max_abs_prediction": float(prediction[identity].abs().max().item()),
        "target_positive_fraction": float(truth.gt(0).double().mean().item()),
        "prediction_mean": float(estimate.mean().item()),
        "target_mean": float(truth.mean().item()),
    }


def _set_features(
    prediction: dict[str, torch.Tensor],
    image_size: tuple[int, int],
    config: dict[str, Any],
    *,
    strong_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from spectral_detection_posttrain.methods.energy_transport.dense_endpoint import build_sparse_pair_features
    from spectral_detection_posttrain.methods.energy_transport.post_nms_suppress import (
        build_post_nms_detection_features,
    )
    import scripts.train_dense_endpoint_geometry_control as geometry

    nodes = build_post_nms_detection_features(
        prediction["boxes"],
        prediction["scores"],
        prediction["labels"],
        image_size,
        num_classes=int(config["detector"]["num_classes"]),
    )
    boxes = prediction["boxes"]
    if strong_seed is not None:
        nodes = geometry.apply_geometry_control(nodes, seed=strong_seed, config=config)
        permutation = geometry._non_identity_permutation(int(boxes.shape[0]), strong_seed).to(boxes.device)
        boxes = boxes[permutation]
    pairs = build_sparse_pair_features(
        boxes,
        prediction["scores"],
        prediction["labels"],
        image_size,
        min_iou=float(config["model"]["pair_min_iou"]),
    )
    return nodes.detach().cpu().to(torch.float16), pairs.detach().cpu().to(torch.float16)


@torch.inference_mode()
def _build_records(
    detector: torch.nn.Module,
    loader: DataLoader,
    teacher_stats: Any,
    config: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    from spectral_detection_posttrain.methods.energy_transport.dense_endpoint import reduced_teacher_values
    from spectral_detection_posttrain.methods.energy_transport.dense_set_energy import (
        DenseTeacherConfig,
        dense_teacher_components,
    )
    from scripts.analyze_dense_local_delta_stats import apply_set_perturbation

    teacher_config = DenseTeacherConfig(
        **{key: config["teacher"][key] for key in DenseTeacherConfig.__dataclass_fields__}
    )
    records = []
    families = list(config["perturbations"]["families"])
    for image_number, (images, targets) in enumerate(loader, start=1):
        image = images[0].to(device)
        prediction = detector([image])[0]
        target = {
            "boxes": targets[0]["boxes"].to(device),
            "labels": targets[0]["labels"].to(device),
        }
        image_size = tuple(int(value) for value in image.shape[-2:])
        image_id = int(targets[0]["image_id"].flatten()[0].item())
        base_quality = teacher_stats.quality(
            reduced_teacher_values(dense_teacher_components(prediction, target, teacher_config)).cpu()
        ).item()
        base_node, base_pair = _set_features(prediction, image_size, config)
        strong_seed = int(config["controls"]["geometry_shuffle_seed"]) + image_id
        base_strong_node, base_strong_pair = _set_features(
            prediction, image_size, config, strong_seed=strong_seed
        )
        top = torch.argsort(prediction["scores"], descending=True)[: int(config["perturbations"]["top_detections_per_image"])]
        candidates = [("identity_permutation", -1)] + [
            (family, int(index))
            for index in top.tolist()
            for family in families
            if family != "identity_permutation"
        ]
        for candidate_number, (family, detection_index) in enumerate(candidates):
            perturbed = apply_set_perturbation(
                prediction,
                family,
                max(detection_index, 0),
                image_size,
                score_step=float(config["perturbations"]["score_step"]),
                box_step=float(config["perturbations"]["box_step_fraction"]),
            )
            perturbed_quality = teacher_stats.quality(
                reduced_teacher_values(
                    dense_teacher_components(perturbed, target, teacher_config)
                ).cpu()
            ).item()
            perturbed_node, perturbed_pair = _set_features(perturbed, image_size, config)
            perturbed_strong_node, perturbed_strong_pair = _set_features(
                perturbed, image_size, config, strong_seed=strong_seed
            )
            records.append(
                {
                    "image_id": image_id,
                    "family": family,
                    "target": float(perturbed_quality - base_quality),
                    "identity": family == "identity_permutation",
                    "base_node": base_node,
                    "base_pair": base_pair,
                    "perturbed_node": perturbed_node,
                    "perturbed_pair": perturbed_pair,
                    "base_strong_node": base_strong_node,
                    "base_strong_pair": base_strong_pair,
                    "perturbed_strong_node": perturbed_strong_node,
                    "perturbed_strong_pair": perturbed_strong_pair,
                }
            )
        if image_number % 16 == 0 or image_number == len(loader.dataset):
            print(json.dumps({"cache_progress_images": image_number, "total_images": len(loader.dataset)}), flush=True)
    return records


def _fit_feature_stats(records: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
    nodes = torch.cat(
        [tensor.float() for row in records for tensor in (row["base_node"], row["perturbed_node"])],
        dim=0,
    )
    pairs = torch.cat(
        [tensor.float() for row in records for tensor in (row["base_pair"], row["perturbed_pair"]) if tensor.numel()],
        dim=0,
    )
    return {
        "node_mean": nodes.mean(0),
        "node_scale": nodes.std(0, unbiased=False).clamp_min(1e-4),
        "pair_mean": pairs.mean(0),
        "pair_scale": pairs.std(0, unbiased=False).clamp_min(1e-4),
    }


def _features(
    row: dict[str, Any], arm: str, stats: dict[str, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prefix = "strong_" if arm == "strong_geometry_shuffle" else ""
    values = []
    for side in ("base", "perturbed"):
        node = row[f"{side}_{prefix}node"].float()
        pair = row[f"{side}_{prefix}pair"].float()
        node = ((node - stats["node_mean"]) / stats["node_scale"]).to(device)
        pair = ((pair - stats["pair_mean"]) / stats["pair_scale"]).to(device)
        values.extend((node, pair))
    return tuple(values)  # type: ignore[return-value]


def _new_model(config: dict[str, Any], device: torch.device):
    from spectral_detection_posttrain.methods.energy_transport.dense_endpoint import DenseSetEnergyEndpoint

    return DenseSetEnergyEndpoint(22, 5, int(config["model"]["hidden_dim"])).to(device)


def _train(
    records: Sequence[dict[str, Any]],
    arm: str,
    stats: dict[str, torch.Tensor],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, float | int]]:
    seed = int(config["controls"]["same_initialization_seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = _new_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["lr"]), weight_decay=float(config["training"]["weight_decay"])
    )
    targets = torch.tensor([row["target"] for row in records], dtype=torch.float32)
    if arm == "utility_shuffle":
        movable = torch.tensor([not row["identity"] for row in records])
        rows = torch.nonzero(movable, as_tuple=False).flatten()
        permutation = geometry_permutation(len(rows), int(config["controls"]["utility_shuffle_seed"]))
        targets[rows] = targets[rows[permutation]]
    accumulation = int(config["training"]["grad_accum_pairs"])
    losses = []
    for epoch in range(int(config["training"]["epochs"])):
        order = list(range(len(records)))
        random.Random(seed + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for position, index in enumerate(order, start=1):
            chunk_start = ((position - 1) // accumulation) * accumulation
            chunk_size = min(accumulation, len(order) - chunk_start)
            base_node, base_pair, pert_node, pert_pair = _features(records[index], arm, stats, device)
            prediction = model(pert_node, pert_pair).quality - model(base_node, base_pair).quality
            loss = torch_f.smooth_l1_loss(
                prediction, targets[index].to(device), beta=float(config["training"]["smooth_l1_beta"])
            )
            (loss / chunk_size).backward()
            total += float(loss.detach().item())
            if position % accumulation == 0 or position == len(order):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        losses.append(total / len(records))
    return model, {"epochs": int(config["training"]["epochs"]), "loss_first": losses[0], "loss_last": losses[-1], "loss_min": min(losses)}


def geometry_permutation(count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(count, generator=generator)
    return permutation.roll(1) if count > 1 and torch.equal(permutation, torch.arange(count)) else permutation


@torch.inference_mode()
def _predict(model: torch.nn.Module, records: Sequence[dict[str, Any]], arm: str, stats: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    model.eval()
    output = []
    for row in records:
        base_node, base_pair, pert_node, pert_pair = _features(row, arm, stats, device)
        output.append((model(pert_node, pert_pair).quality - model(base_node, base_pair).quality).cpu())
    return torch.stack(output).float()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "nwpu_dense_local_delta_learner_s42_48_16")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    from spectral_detection_posttrain.datasets.nwpu_vhr10 import NWPUVHR10DetectionDataset
    from spectral_detection_posttrain.datasets.penn_fudan import detection_collate
    from spectral_detection_posttrain.methods.energy_transport.dense_endpoint import RobustTeacherStats
    from spectral_detection_posttrain.models import build_detector
    from spectral_detection_posttrain.utils.io import load_checkpoint
    import scripts.train_dense_endpoint_geometry_control as geometry

    config = load_config()
    source_path = ROOT / config["sources"]["delta_stats_result"]
    teacher_checkpoint = ROOT / config["sources"]["teacher_checkpoint"]
    if sha256_file(source_path) != config["sources"]["delta_stats_sha256"]:
        raise ValueError("local Delta-Q stats SHA256 mismatch")
    if sha256_file(teacher_checkpoint) != config["sources"]["teacher_checkpoint_sha256"]:
        raise ValueError("teacher checkpoint SHA256 mismatch")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("scientific_status") != config["sources"]["required_status"]:
        raise ValueError("local Delta-Q support decision mismatch")
    teacher_payload = torch.load(teacher_checkpoint, map_location="cpu")
    teacher_stats = RobustTeacherStats(teacher_payload["teacher_median"].float(), teacher_payload["teacher_iqr"].float())
    nested = json.loads((ROOT / "spectral_detection_posttrain/configs/splits/nwpu_dense_endpoint_s42_nested.json").read_text(encoding="utf-8"))
    inner_fit = [int(value) for value in nested["splits"]["inner_fit"]["image_ids"]]
    _, source_ids = geometry.resplit_image_ids(inner_fit, geometry.load_config())
    if image_ids_hash(source_ids) != config["dataset"]["source_sha256"]:
        raise ValueError("local learner source image hash mismatch")
    fit_ids, tune_ids = split_image_ids(source_ids, config)
    if image_ids_hash(fit_ids) != config["dataset"]["fit_sha256"] or image_ids_hash(tune_ids) != config["dataset"]["tune_sha256"]:
        raise ValueError("local learner split hash mismatch")
    detector_checkpoint = (args.checkpoint or ROOT / config["detector"]["checkpoint"]).resolve()
    annotation = (args.annotation or ROOT / "data/NWPU_VHR10_coco.json").resolve()
    data_root = (args.data_root or ROOT / "data/NWPU VHR-10 dataset").resolve()
    if sha256_file(detector_checkpoint) != config["detector"]["checkpoint_sha256"] or sha256_file(annotation) != config["dataset"]["annotation_sha256"]:
        raise ValueError("local learner detector inputs mismatch")
    git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    if args.require_clean_git and git_dirty:
        raise RuntimeError("--require-clean-git requested but repository is dirty")
    device = torch.device(args.device)
    detector_config = {
        "seed": 42, "data_seed": 42,
        "data": {"root": str(data_root), "annotation": str(annotation), "max_size": 480, "train_fraction": 0.7, "num_workers": 0},
        "model": {"name": config["detector"]["model_name"], "model_name": config["detector"]["model_name"], "pretrained": False, "num_classes": 11, "min_size": 480, "max_size": 480},
        "train": {"batch_size": 1}, "eval": {"batch_size": 1},
    }
    detector = build_detector(detector_config).to(device)
    load_checkpoint(detector, detector_checkpoint, device)
    detector.roi_heads.score_thresh = 0.05
    detector.roi_heads.nms_thresh = 0.5
    detector.roi_heads.detections_per_img = 100
    detector.eval()
    dataset = NWPUVHR10DetectionDataset(data_root, annotation, source_ids, max_size=480)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=detection_collate)
    records = _build_records(detector, loader, teacher_stats, config, device)
    fit_set, tune_set = set(fit_ids), set(tune_ids)
    fit_records = [row for row in records if int(row["image_id"]) in fit_set]
    tune_records = [row for row in records if int(row["image_id"]) in tune_set]
    stats = _fit_feature_stats(fit_records)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.run_dir / "pair_cache.pt"
    torch.save({"format": "dense_local_delta_pairs_v1", "config_sha256": CONFIG_SHA256, "records": records}, cache_path)
    cache_sha = sha256_file(cache_path)
    models, training, evaluation, checkpoints = {}, {}, {}, {}
    for arm in config["arms"]:
        models[arm], training[arm] = _train(fit_records, arm, stats, config, device)
        evaluation[arm] = {}
        for name, split_records in (("fit", fit_records), ("tune", tune_records)):
            prediction = _predict(models[arm], split_records, arm, stats, device)
            target = torch.tensor([row["target"] for row in split_records])
            ids = torch.tensor([row["image_id"] for row in split_records])
            identity = torch.tensor([row["identity"] for row in split_records])
            evaluation[arm][name] = _metrics(prediction, target, ids, identity)
        checkpoint_path = args.run_dir / arm / "endpoint.pth"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": models[arm].state_dict(), "arm": arm, "config_sha256": CONFIG_SHA256, "feature_stats": stats}, checkpoint_path)
        checkpoints[arm] = sha256_file(checkpoint_path)
    full_fit = evaluation["local_full"]["fit"]
    full = evaluation["local_full"]["tune"]
    controls = ("strong_geometry_shuffle", "utility_shuffle")
    pairwise_gains = {arm: float(full["imagewise_pairwise_accuracy"]) - float(evaluation[arm]["tune"]["imagewise_pairwise_accuracy"]) for arm in controls}
    mae_gains = {arm: float(evaluation[arm]["tune"]["mae"]) - float(full["mae"]) for arm in controls}
    fit_gap = float(full_fit["imagewise_pairwise_accuracy"]) - float(full["imagewise_pairwise_accuracy"])
    gates = {
        "tune_support": int(full["rows"]) >= int(config["gates"]["tune_min_rows"]),
        "identity": float(full["identity_max_abs_prediction"]) <= float(config["gates"]["identity_max_abs_prediction"]),
        "pairwise": float(full["imagewise_pairwise_accuracy"]) >= float(config["gates"]["tune_imagewise_pairwise_min"]),
        "sign": full["sign_auroc"] is not None and float(full["sign_auroc"]) >= float(config["gates"]["tune_sign_auroc_min"]),
        "zero_baseline": float(full["mae_relative_gain_over_zero"]) >= float(config["gates"]["mae_relative_gain_over_zero_min"]),
        "pairwise_controls": all(value >= float(config["gates"]["pairwise_gain_over_each_control_min"]) for value in pairwise_gains.values()),
        "mae_controls": all(value >= float(config["gates"]["mae_gain_over_each_control_min"]) for value in mae_gains.values()),
        "fit_tune_gap": fit_gap <= float(config["gates"]["fit_to_tune_pairwise_gap_max"]),
    }
    all_passed = all(gates.values())
    result = {
        "completed": True,
        "scientific_status": "train_only_local_delta_learner_supported" if all_passed else "train_only_local_delta_learner_frozen",
        "version_id": config["version_id"], "experiment_scope": "researcher_adaptive_train_only_48_16_local_delta",
        "config_sha256": CONFIG_SHA256, "source_delta_stats_sha256": config["sources"]["delta_stats_sha256"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "git_dirty": git_dirty,
        "detector_validation_read": False, "native_nms_rerun_after_perturbation": False,
        "pair_cache_sha256": cache_sha, "split": {"fit_images": len(fit_ids), "tune_images": len(tune_ids), "fit_rows": len(fit_records), "tune_rows": len(tune_records)},
        "training": training, "checkpoint_hashes": checkpoints, "evaluation": evaluation,
        "summary": {"full_tune": full, "pairwise_gains_over_controls": pairwise_gains, "mae_gains_over_controls": mae_gains, "fit_to_tune_pairwise_gap": fit_gap},
        "gates": {"all_passed": all_passed, "gates": gates}, "decision_rule": config["decision_rule"], "claim_boundary": config["claim_boundary"], "validated_claim": False,
    }
    (args.run_dir / "eval_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    result = run(_parse_args())
    print(json.dumps({"status": result["scientific_status"], "summary": result["summary"], "gates": result["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
