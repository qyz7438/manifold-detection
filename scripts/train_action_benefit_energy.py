"""Train a pairwise energy gate that chooses a learned box action or identity."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectral_detection_posttrain.datasets import build_detection_loaders
from spectral_detection_posttrain.eval.action_transport_diagnostics import (
    box_only_actions,
    concatenate_transition_batches,
    proposal_transition_tensors,
    summarize_proposal_transitions,
)
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.methods.energy_transport import (
    ActionBenefitEnergyHead,
    ActionLocalTransportHead,
    BenefitEnergyLossConfig,
    action_benefit_energy_loss,
    apply_action_benefit_gate,
    build_action_benefit_targets,
)
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.trainers.detection.action_local_transport import (
    action_batch_to_predictions,
    extract_proposal_action_batch,
)
from spectral_detection_posttrain.utils.checkpoint_hash import sha256_file
from spectral_detection_posttrain.utils.git_state import get_git_state
from spectral_detection_posttrain.utils.io import ensure_run_dir, load_checkpoint, save_checkpoint, save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--action-run-dir", required=True)
    parser.add_argument("--model-name", default="fasterrcnn_mobilenet_v3_large_320_fpn")
    parser.add_argument("--nwpu-root", default="./data/NWPU VHR-10 dataset")
    parser.add_argument("--nwpu-annotation", default="./data/NWPU_VHR10_coco.json")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-positive-gain", type=float, default=0.005)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--regression-weight", type=float, default=1.0)
    parser.add_argument("--boundary-boost", type=float, default=2.0)
    parser.add_argument("--foreground-boost", type=float, default=1.0)
    parser.add_argument("--gate-threshold", type=float, default=0.0)
    parser.add_argument("--eval-gate-thresholds", default="-0.01,-0.005,0,0.0025,0.005,0.01")
    parser.add_argument("--min-score", type=float, default=0.05)
    parser.add_argument("--require-foreground-dominant", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-actions-per-image", type=int, default=32)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--detections-per-img", type=int, default=100)
    parser.add_argument("--shuffle-seed", type=int, default=2718)
    parser.add_argument("--require-clean-git", action="store_true", default=False)
    return parser.parse_args()


def parse_thresholds(value: str, primary: float) -> list[float]:
    thresholds = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not thresholds:
        raise ValueError("eval gate thresholds cannot be empty")
    if not any(math.isclose(item, primary, abs_tol=1e-12) for item in thresholds):
        thresholds.append(float(primary))
        thresholds.sort()
    return thresholds


def _threshold_name(value: float) -> str:
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return f"gate_t{text}"


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "seed": int(args.seed),
        "data_seed": int(args.data_seed),
        "device": "cuda" if args.device == "auto" else args.device,
        "data": {
            "dataset": "nwpu",
            "root": args.nwpu_root,
            "annotation": args.nwpu_annotation,
            "train_fraction": 0.7,
            "max_size": 480,
            "num_workers": int(args.num_workers),
        },
        "model": {
            "name": args.model_name,
            "model_name": args.model_name,
            "pretrained": False,
            "num_classes": 11,
            "min_size": 480,
            "max_size": 480,
        },
        "train": {
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
        },
        "matching": {"iou_threshold": 0.5, "score_threshold": args.score_threshold},
        "eval": {"batch_size": int(args.batch_size), "high_conf_threshold": 0.7},
    }


def load_frozen_action_head(
    run_dir: Path,
    *,
    device: torch.device,
    detector_hash: str,
) -> tuple[ActionLocalTransportHead, dict[str, Any]]:
    metrics_path = run_dir / "eval_metrics.json"
    checkpoint = run_dir / "action_head_best_ap75.pth"
    if not metrics_path.exists() or not checkpoint.exists():
        raise FileNotFoundError(f"missing action metrics/checkpoint under {run_dir}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    source_hash = (payload.get("source_checkpoint") or {}).get("sha256")
    if source_hash != detector_hash:
        raise ValueError("action head was not trained from the requested detector checkpoint")
    config = payload["action_config"]
    required = {"feature_dim", "max_score_delta", "max_box_delta"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"action_config missing required fields: {missing}")
    head = ActionLocalTransportHead(
        feature_dim=int(config["feature_dim"]),
        hidden_dim=config.get("hidden_dim"),
        max_score_delta=float(config["max_score_delta"]),
        max_box_delta=float(config["max_box_delta"]),
        residual_scale=0.0,
    ).to(device)
    load_checkpoint(head, checkpoint, device)
    for parameter in head.parameters():
        parameter.requires_grad = False
    head.eval()
    return head, {
        "run_dir": str(run_dir.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "best_epoch": payload.get("best_epoch"),
        "best_ap75": payload.get("best_ap75"),
        "action_config": config,
    }


def _to_device(targets: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    return [
        {key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()}
        for target in targets
    ]


def train_one_epoch(
    model: torch.nn.Module,
    action_head: ActionLocalTransportHead,
    energy_head: ActionBenefitEnergyHead,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_config: BenefitEnergyLossConfig,
    *,
    epoch: int,
) -> dict[str, float]:
    model.eval()
    action_head.eval()
    energy_head.train()
    totals: dict[str, float] = {}
    batches = 0
    for images, raw_targets in tqdm(loader, desc=f"benefit energy epoch {epoch}"):
        images = [image.to(device) for image in images]
        batch = extract_proposal_action_batch(
            model,
            images,
            _to_device(raw_targets, device),
            match_mode="class_agnostic",
            box_base="decoded",
        )
        if batch.state.logits is None:
            raise RuntimeError("benefit energy requires detector class logits")
        with torch.no_grad():
            actions = box_only_actions(action_head(batch.state.features), scale=1.0)
            targets = build_action_benefit_targets(
                batch.state,
                actions,
                matched_gt_boxes=batch.matched_gt_boxes,
                matched_gt_labels=batch.matched_gt_labels,
                image_sizes=batch.image_sizes,
            )
        predicted_gain = energy_head.energy_gap(
            batch.state.features,
            batch.state.logits,
            batch.state.labels,
            batch.state.scores,
            actions.box_delta,
        )
        losses = action_benefit_energy_loss(
            predicted_gain,
            true_gain=targets.true_gain,
            base_iou=targets.base_iou,
            class_correct=targets.class_correct,
            foreground_dominant=targets.foreground_dominant,
            scores=batch.state.scores,
            config=loss_config,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["loss_total"].backward()
        optimizer.step()
        batches += 1
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().item())
    return {key: value / max(1, batches) for key, value in totals.items()}


@torch.no_grad()
def evaluate_energy_gate(
    model: torch.nn.Module,
    action_head: ActionLocalTransportHead,
    energy_head: ActionBenefitEnergyHead,
    loader,
    device: torch.device,
    *,
    thresholds: list[float],
    primary_threshold: float,
    min_positive_gain: float,
    min_score: float,
    require_foreground_dominant: bool,
    max_actions_per_image: int | None,
    score_threshold: float,
    nms_threshold: float,
    detections_per_img: int,
    shuffle_seed: int,
    desc: str,
) -> dict[str, Any]:
    model.eval()
    action_head.eval()
    energy_head.eval()
    predictions: dict[str, list[dict[str, torch.Tensor]]] = {
        "identity": [],
        "ungated": [],
        **{_threshold_name(value): [] for value in thresholds},
        "shuffled_gain_primary": [],
    }
    targets_out: list[dict[str, torch.Tensor]] = []
    predicted_gains: list[torch.Tensor] = []
    true_gains: list[torch.Tensor] = []
    positive_labels: list[torch.Tensor] = []
    primary_masks: list[torch.Tensor] = []
    shuffled_masks: list[torch.Tensor] = []
    transition_batches: list[dict[str, torch.Tensor]] = []
    generator = torch.Generator(device="cpu").manual_seed(int(shuffle_seed))

    for images, raw_targets in tqdm(loader, desc=desc):
        images_device = [image.to(device) for image in images]
        batch = extract_proposal_action_batch(
            model,
            images_device,
            _to_device(raw_targets, device),
            match_mode="class_agnostic",
            box_base="decoded",
        )
        if batch.state.logits is None:
            raise RuntimeError("benefit energy requires detector class logits")
        actions = box_only_actions(action_head(batch.state.features), scale=1.0)
        targets = build_action_benefit_targets(
            batch.state,
            actions,
            matched_gt_boxes=batch.matched_gt_boxes,
            matched_gt_labels=batch.matched_gt_labels,
            image_sizes=batch.image_sizes,
        )
        predicted_gain = energy_head.energy_gap(
            batch.state.features,
            batch.state.logits,
            batch.state.labels,
            batch.state.scores,
            actions.box_delta,
        )
        identity = box_only_actions(actions, scale=0.0)
        predictions["identity"].extend(
            _predictions_from_actions(
                batch,
                identity,
                model,
                score_threshold,
                nms_threshold,
                detections_per_img,
            )
        )
        predictions["ungated"].extend(
            _predictions_from_actions(
                batch,
                actions,
                model,
                score_threshold,
                nms_threshold,
                detections_per_img,
            )
        )
        primary_actions = None
        primary_mask = None
        for threshold in thresholds:
            gated, mask = apply_action_benefit_gate(
                batch.state,
                actions,
                predicted_gain,
                min_predicted_gain=float(threshold),
                min_score=float(min_score),
                require_foreground_dominant=bool(require_foreground_dominant),
                max_actions_per_image=max_actions_per_image,
            )
            predictions[_threshold_name(threshold)].extend(
                _predictions_from_actions(
                    batch,
                    gated,
                    model,
                    score_threshold,
                    nms_threshold,
                    detections_per_img,
                )
            )
            if math.isclose(threshold, primary_threshold, abs_tol=1e-12):
                primary_actions = gated
                primary_mask = mask
        assert primary_actions is not None and primary_mask is not None

        shuffled_gain = _permute_values_within_images(
            predicted_gain,
            batch.state.image_indices,
            generator=generator,
        )
        shuffled_actions, shuffled_mask = apply_action_benefit_gate(
            batch.state,
            actions,
            shuffled_gain,
            min_predicted_gain=float(primary_threshold),
            min_score=float(min_score),
            require_foreground_dominant=bool(require_foreground_dominant),
            max_actions_per_image=max_actions_per_image,
        )
        predictions["shuffled_gain_primary"].extend(
            _predictions_from_actions(
                batch,
                shuffled_actions,
                model,
                score_threshold,
                nms_threshold,
                detections_per_img,
            )
        )
        transition_batches.append(
            proposal_transition_tensors(
                batch.state,
                matched_gt_boxes=batch.matched_gt_boxes,
                matched_gt_labels=batch.matched_gt_labels,
                box_delta=primary_actions.box_delta,
                image_sizes=batch.image_sizes,
            )
        )
        predicted_gains.append(predicted_gain.detach().cpu())
        true_gains.append(targets.true_gain.detach().cpu())
        positive_labels.append(
            (
                targets.class_correct
                & (targets.true_gain > float(min_positive_gain))
            ).detach().cpu()
        )
        primary_masks.append(primary_mask.detach().cpu())
        shuffled_masks.append(shuffled_mask.detach().cpu())
        targets_out.extend(
            [
                {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in target.items()}
                for target in raw_targets
            ]
        )

    metric_kwargs = {
        "iou_threshold": 0.5,
        "score_threshold": float(score_threshold),
        "high_conf_threshold": 0.7,
        "per_class": True,
        "num_classes": 11,
        "per_size": True,
    }
    mode_metrics = {
        mode: evaluate_detection_predictions(outputs, targets_out, **metric_kwargs)
        for mode, outputs in predictions.items()
    }
    predicted = torch.cat(predicted_gains)
    true = torch.cat(true_gains)
    labels = torch.cat(positive_labels).bool()
    primary = torch.cat(primary_masks).bool()
    shuffled = torch.cat(shuffled_masks).bool()
    transitions = concatenate_transition_batches(transition_batches)
    return {
        "modes": mode_metrics,
        "benefit_prediction": _benefit_prediction_summary(predicted, true, labels),
        "primary_acceptance": _acceptance_summary(primary, true, labels),
        "shuffled_acceptance": _acceptance_summary(shuffled, true, labels),
        "primary_transitions": summarize_proposal_transitions(transitions, threshold=0.75),
    }


def _predictions_from_actions(
    batch,
    actions,
    model,
    score_threshold: float,
    nms_threshold: float,
    detections_per_img: int,
):
    return action_batch_to_predictions(
        batch,
        actions,
        gate_actions=False,
        score_threshold=float(score_threshold),
        nms_threshold=float(nms_threshold),
        detections_per_img=int(detections_per_img),
        native_model=model,
    )


def _permute_values_within_images(
    values: torch.Tensor,
    image_indices: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    output = values.clone()
    for image_idx in torch.unique(image_indices.detach().cpu(), sorted=True).tolist():
        rows = torch.nonzero(image_indices == int(image_idx), as_tuple=False).flatten()
        if rows.numel() <= 1:
            continue
        order = torch.randperm(int(rows.numel()), generator=generator, device="cpu").to(rows.device)
        output[rows] = values[rows[order]]
    return output


def _benefit_prediction_summary(
    predicted: torch.Tensor,
    true: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float | int | None]:
    return {
        "count": int(predicted.numel()),
        "positive_count": int(labels.sum().item()),
        "roc_auc": _binary_roc_auc(labels, predicted),
        "average_precision": _binary_average_precision(labels, predicted),
        "gain_pearson": _pearson(predicted, true),
        "predicted_gain_mean": float(predicted.mean().item()),
        "predicted_gain_std": float(predicted.std(unbiased=False).item()),
        "true_gain_mean": float(true.mean().item()),
    }


def _acceptance_summary(
    mask: torch.Tensor,
    true_gain: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float | int | None]:
    count = int(mask.sum().item())
    return {
        "accepted": count,
        "accepted_rate": count / max(1, int(mask.numel())),
        "accepted_positive_rate": float(labels[mask].float().mean().item()) if count else None,
        "accepted_true_gain_mean": float(true_gain[mask].mean().item()) if count else None,
    }


def _binary_average_precision(labels: torch.Tensor, scores: torch.Tensor) -> float | None:
    labels = labels.bool()
    positives = int(labels.sum().item())
    if positives == 0:
        return None
    order = torch.argsort(scores, descending=True)
    ranked = labels[order].float()
    precision = ranked.cumsum(0) / torch.arange(1, ranked.numel() + 1, dtype=torch.float32)
    return float((precision * ranked).sum().item() / positives)


def _binary_roc_auc(labels: torch.Tensor, scores: torch.Tensor) -> float | None:
    labels = labels.bool()
    positives = int(labels.sum().item())
    negatives = int((~labels).sum().item())
    if positives == 0 or negatives == 0:
        return None
    order = torch.argsort(scores, descending=False)
    sorted_scores = scores[order]
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    ends = counts.cumsum(0).to(dtype=torch.float32)
    starts = ends - counts.to(dtype=torch.float32)
    average_ranks = 0.5 * ((starts + 1.0) + ends)
    sorted_ranks = torch.repeat_interleave(average_ranks, counts)
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    positive_rank_sum = ranks[labels].sum()
    auc = (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / float(positives * negatives)
    return float(auc.item())


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float | None:
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    denominator = x.norm() * y.norm()
    if denominator.item() <= 1e-12:
        return None
    return float((x * y).sum().item() / denominator.item())


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    git_state = get_git_state()
    if args.require_clean_git and git_state["dirty"]:
        raise RuntimeError(f"Git working tree is dirty at {git_state['commit']}")
    thresholds = parse_thresholds(args.eval_gate_thresholds, float(args.gate_threshold))
    config = build_config(args)
    device = resolve_device(config)
    detector_path = Path(args.detector_checkpoint).resolve()
    detector_hash = sha256_file(detector_path)
    train_loader, val_loader = build_detection_loaders(
        config,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        batch_size=int(args.batch_size),
    )
    model = build_detector(config).to(device)
    load_checkpoint(model, detector_path, device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    action_head, action_metadata = load_frozen_action_head(
        Path(args.action_run_dir),
        device=device,
        detector_hash=detector_hash,
    )
    feature_dim = int(action_metadata["action_config"]["feature_dim"])
    energy_head = ActionBenefitEnergyHead(
        feature_dim=feature_dim,
        num_classes=11,
        hidden_dim=int(args.hidden_dim),
    ).to(device)
    loss_config = BenefitEnergyLossConfig(
        min_positive_gain=float(args.min_positive_gain),
        temperature=float(args.temperature),
        ranking_weight=float(args.ranking_weight),
        regression_weight=float(args.regression_weight),
        boundary_boost=float(args.boundary_boost),
        foreground_boost=float(args.foreground_boost),
    )
    optimizer = torch.optim.AdamW(
        energy_head.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    eval_kwargs = {
        "thresholds": thresholds,
        "primary_threshold": float(args.gate_threshold),
        "min_positive_gain": float(args.min_positive_gain),
        "min_score": float(args.min_score),
        "require_foreground_dominant": bool(args.require_foreground_dominant),
        "max_actions_per_image": int(args.max_actions_per_image),
        "score_threshold": float(args.score_threshold),
        "nms_threshold": float(args.nms_threshold),
        "detections_per_img": int(args.detections_per_img),
        "shuffle_seed": int(args.shuffle_seed),
    }
    initial = evaluate_energy_gate(
        model,
        action_head,
        energy_head,
        val_loader,
        device,
        desc="initial benefit energy eval",
        **eval_kwargs,
    )
    history: list[dict[str, Any]] = []
    best_ap75 = -1.0
    best_epoch = None
    run_dir = ensure_run_dir(args.run_name)
    primary_mode = _threshold_name(float(args.gate_threshold))
    for epoch in range(1, int(args.epochs) + 1):
        losses = train_one_epoch(
            model,
            action_head,
            energy_head,
            train_loader,
            optimizer,
            device,
            loss_config,
            epoch=epoch,
        )
        evaluation = evaluate_energy_gate(
            model,
            action_head,
            energy_head,
            val_loader,
            device,
            desc=f"benefit energy eval {epoch}",
            **eval_kwargs,
        )
        row = {"epoch": epoch, "train": losses, "evaluation": evaluation}
        history.append(row)
        ap75 = float(evaluation["modes"][primary_mode]["ap75"])
        print(
            f"epoch {epoch}: loss={losses['loss_total']:.4f} "
            f"gate_AP75={ap75:.4f} identity_AP75={evaluation['modes']['identity']['ap75']:.4f}"
        )
        save_checkpoint(energy_head, run_dir / "benefit_energy_last.pth", {"epoch": epoch})
        if ap75 > best_ap75:
            best_ap75 = ap75
            best_epoch = epoch
            save_checkpoint(
                energy_head,
                run_dir / "benefit_energy_best_ap75.pth",
                {"epoch": epoch, "ap75": ap75},
            )

    result = {
        "run_name": args.run_name,
        "completed": True,
        "config": {**config, "git_state": git_state, "args": vars(args)},
        "detector_checkpoint": {
            "path": str(detector_path),
            "sha256": detector_hash,
        },
        "action_source": action_metadata,
        "loss_config": loss_config.__dict__,
        "primary_mode": primary_mode,
        "initial": initial,
        "history": history,
        "best_epoch": best_epoch,
        "best_ap75": best_ap75,
    }
    save_json(result, run_dir / "eval_metrics.json")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
