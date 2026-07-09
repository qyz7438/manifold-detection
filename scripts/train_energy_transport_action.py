from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectral_detection_posttrain.datasets import build_detection_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.methods.energy_transport import (
    ActionLocalTransportHead,
    capture_high_water_mark_module,
    load_high_water_mark_module,
    should_update_high_water_mark,
)
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.trainers.detection.action_local_transport import (
    SupervisedActionLossConfig,
    action_batch_to_predictions,
    extract_proposal_action_batch,
    supervised_action_transport_loss,
)
from spectral_detection_posttrain.utils.git_state import get_git_state
from spectral_detection_posttrain.utils.io import ensure_run_dir, load_checkpoint, save_checkpoint, save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an action-local ROI transport head with high-water teacher anchoring."
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dataset", default="nwpu", choices=["nwpu", "penn_fudan", "voc"])
    parser.add_argument("--checkpoint", default=None, help="Frozen detector checkpoint")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model-name", default="fasterrcnn_mobilenet_v3_large_320_fpn",
                        choices=["fasterrcnn_mobilenet_v3_large_320_fpn", "fasterrcnn_resnet50_fpn"])
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-size", type=int, default=None)
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument("--voc-root", default="./data")
    parser.add_argument("--nwpu-root", default="./data/NWPU VHR-10 dataset")
    parser.add_argument("--nwpu-annotation", default="./data/NWPU_VHR10_coco.json")

    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--max-score-delta", type=float, default=0.05)
    parser.add_argument("--max-box-delta", type=float, default=0.20)
    parser.add_argument("--residual-scale", type=float, default=0.0)

    parser.add_argument("--target-iou", type=float, default=0.75)
    parser.add_argument("--positive-iou-floor", type=float, default=0.50)
    parser.add_argument("--positive-iou-ceiling", type=float, default=0.75)
    parser.add_argument("--boundary-band", type=float, default=0.10)
    parser.add_argument("--min-boundary-weight", type=float, default=0.05)
    parser.add_argument("--box-weight", type=float, default=1.0)
    parser.add_argument("--high-iou-preserve-weight", type=float, default=0.20)
    parser.add_argument("--energy-weight", type=float, default=0.01)
    parser.add_argument("--gate-actions", action="store_true", default=False)
    parser.add_argument("--gate-loss-weight", type=float, default=0.0)
    parser.add_argument("--hwm-weight", type=float, default=0.0)
    parser.add_argument("--hwm-epsilon", type=float, default=0.05)
    parser.add_argument("--hwm-box-weight", type=float, default=1.0)
    parser.add_argument("--hwm-keep-weight", type=float, default=0.0)
    parser.add_argument("--hwm-min-delta", type=float, default=0.0)
    parser.add_argument("--match-mode", default="class_agnostic",
                        choices=["class_aware", "class_agnostic"],
                        help="GT matching used for local box-action supervision")
    parser.add_argument("--box-base", default="decoded", choices=["decoded", "proposal"],
                        help="Base boxes corrected by action head")

    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--action-score-threshold", type=float, default=0.001)
    parser.add_argument("--nms-threshold", type=float, default=0.50)
    parser.add_argument("--detections-per-img", type=int, default=300)
    parser.add_argument("--per-class", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--per-size", action="store_true", default=False)
    parser.add_argument("--skip-default-eval", action="store_true", default=False)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    min_size = args.min_size
    max_size = args.max_size
    if args.dataset == "nwpu":
        min_size = 480 if min_size is None else min_size
        max_size = 480 if max_size is None else max_size
        num_classes = 11
        data = {
            "dataset": "nwpu",
            "root": args.nwpu_root,
            "annotation": args.nwpu_annotation,
            "train_fraction": 0.7,
            "max_size": max_size,
            "num_workers": args.num_workers,
        }
    elif args.dataset == "voc":
        min_size = 480 if min_size is None else min_size
        max_size = 480 if max_size is None else max_size
        classes = ["person", "car", "dog"]
        num_classes = len(classes) + 1
        data = {
            "dataset": "voc",
            "root": args.voc_root,
            "download": False,
            "classes": classes,
            "max_size": max_size,
            "num_workers": args.num_workers,
            "train_years": [{"year": "2007", "image_set": "trainval"}],
            "val_years": [{"year": "2007", "image_set": "test"}],
        }
    else:
        min_size = 320 if min_size is None else min_size
        max_size = 320 if max_size is None else max_size
        num_classes = 2
        data = {
            "dataset": "penn_fudan",
            "root": "./data",
            "download": True,
            "max_size": max_size,
            "train_fraction": 0.8,
            "num_workers": args.num_workers,
        }

    return {
        "seed": args.seed,
        "device": "cuda" if args.device == "auto" else args.device,
        "data": data,
        "model": {
            "name": args.model_name,
            "model_name": args.model_name,
            "pretrained": bool(args.pretrained),
            "num_classes": num_classes,
            "min_size": min_size,
            "max_size": max_size,
        },
        "train": {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "matching": {"iou_threshold": 0.5, "score_threshold": args.score_threshold},
        "eval": {"batch_size": args.batch_size, "high_conf_threshold": 0.7},
    }


def freeze_detector(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    model.eval()


def infer_action_feature_dim(model: torch.nn.Module) -> int:
    predictor = model.roi_heads.box_predictor
    cls_score = getattr(predictor, "cls_score", None)
    if cls_score is None or not hasattr(cls_score, "in_features"):
        raise ValueError("Cannot infer action feature dim from roi_heads.box_predictor")
    return int(cls_score.in_features)


@torch.no_grad()
def evaluate_detector(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    metric_kwargs: dict[str, Any],
    *,
    desc: str,
) -> dict[str, Any]:
    model.eval()
    predictions: list[dict[str, torch.Tensor]] = []
    targets_out: list[dict[str, torch.Tensor]] = []
    for images, targets in tqdm(loader, desc=desc):
        outputs = model([image.to(device) for image in images])
        predictions.extend(
            [{key: value.detach().cpu() for key, value in output.items()} for output in outputs]
        )
        targets_out.extend(
            [
                {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in target.items()}
                for target in targets
            ]
        )
    return evaluate_detection_predictions(predictions, targets_out, **metric_kwargs)


@torch.no_grad()
def evaluate_action_head(
    model: torch.nn.Module,
    action_head: ActionLocalTransportHead,
    loader,
    device: torch.device,
    loss_config: SupervisedActionLossConfig,
    metric_kwargs: dict[str, Any],
    *,
    match_mode: str,
    box_base: str,
    action_score_threshold: float,
    nms_threshold: float,
    detections_per_img: int,
    desc: str,
) -> dict[str, Any]:
    model.eval()
    action_head.eval()
    predictions: list[dict[str, torch.Tensor]] = []
    targets_out: list[dict[str, torch.Tensor]] = []
    for images, targets in tqdm(loader, desc=desc):
        images = [img.to(device) for img in images]
        targets_device = _to_device(targets, device)
        batch = extract_proposal_action_batch(
            model,
            images,
            targets_device,
            match_mode=match_mode,
            box_base=box_base,
        )
        actions = action_head(batch.state.features)
        predictions.extend(
            action_batch_to_predictions(
                batch,
                actions,
                gate_actions=loss_config.gate_actions,
                score_threshold=action_score_threshold,
                nms_threshold=nms_threshold,
                detections_per_img=detections_per_img,
            )
        )
        targets_out.extend(
            [
                {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in target.items()}
                for target in targets
            ]
        )
    return evaluate_detection_predictions(predictions, targets_out, **metric_kwargs)


def train_one_epoch(
    model: torch.nn.Module,
    action_head: ActionLocalTransportHead,
    teacher_head: ActionLocalTransportHead | None,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_config: SupervisedActionLossConfig,
    match_mode: str,
    box_base: str,
    *,
    epoch: int,
) -> dict[str, float]:
    model.eval()
    action_head.train()
    if teacher_head is not None:
        teacher_head.eval()

    totals: dict[str, float] = {}
    seen_batches = 0
    for images, targets in tqdm(loader, desc=f"action epoch {epoch}"):
        images = [img.to(device) for img in images]
        targets_device = _to_device(targets, device)
        batch = extract_proposal_action_batch(
            model,
            images,
            targets_device,
            match_mode=match_mode,
            box_base=box_base,
        )
        actions = action_head(batch.state.features)
        teacher_actions = None
        if teacher_head is not None and loss_config.hwm_weight > 0.0:
            with torch.no_grad():
                teacher_actions = teacher_head(batch.state.features)
        loss_dict = supervised_action_transport_loss(
            batch,
            actions,
            config=loss_config,
            teacher_actions=teacher_actions,
        )
        loss = loss_dict["loss_total"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        seen_batches += 1
        for key, value in loss_dict.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().item())

    return {key: value / max(1, seen_batches) for key, value in totals.items()}


def _to_device(targets: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    return [
        {key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()}
        for target in targets
    ]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    config = build_config(args)
    device = resolve_device(config)
    run_dir = ensure_run_dir(args.run_name)
    config["git_state"] = get_git_state()

    train_loader, val_loader = build_detection_loaders(
        config,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        batch_size=args.batch_size,
    )
    model = build_detector(config).to(device)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, device)
    freeze_detector(model)

    feature_dim = infer_action_feature_dim(model)
    action_head = ActionLocalTransportHead(
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        max_score_delta=args.max_score_delta,
        max_box_delta=args.max_box_delta,
        residual_scale=args.residual_scale,
    ).to(device)
    teacher_head: ActionLocalTransportHead | None = None

    loss_config = SupervisedActionLossConfig(
        target_iou=args.target_iou,
        positive_iou_floor=args.positive_iou_floor,
        positive_iou_ceiling=args.positive_iou_ceiling,
        boundary_band=args.boundary_band,
        min_boundary_weight=args.min_boundary_weight,
        max_box_delta=args.max_box_delta,
        box_weight=args.box_weight,
        high_iou_preserve_weight=args.high_iou_preserve_weight,
        energy_weight=args.energy_weight,
        gate_loss_weight=args.gate_loss_weight,
        gate_actions=args.gate_actions,
        hwm_weight=args.hwm_weight,
        hwm_epsilon=args.hwm_epsilon,
        hwm_box_weight=args.hwm_box_weight,
        hwm_keep_weight=args.hwm_keep_weight,
    )
    optimizer = torch.optim.AdamW(
        action_head.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    metric_kwargs = {
        "iou_threshold": 0.5,
        "score_threshold": float(args.score_threshold),
        "high_conf_threshold": 0.7,
        "per_class": bool(args.per_class),
        "num_classes": int(config["model"]["num_classes"]),
        "per_size": bool(args.per_size),
    }

    history: list[dict[str, Any]] = []
    best_snapshot = None
    best_ap75 = -1.0

    default_metrics = None
    if not args.skip_default_eval:
        default_metrics = evaluate_detector(
            model,
            val_loader,
            device,
            metric_kwargs,
            desc="default detector eval",
        )
        print(
            f"default: AP50={default_metrics['ap50']:.4f} "
            f"AP75={default_metrics['ap75']:.4f}"
        )

    initial_metrics = evaluate_action_head(
        model,
        action_head,
        val_loader,
        device,
        loss_config,
        metric_kwargs,
        match_mode=str(args.match_mode),
        box_base=str(args.box_base),
        action_score_threshold=float(args.action_score_threshold),
        nms_threshold=float(args.nms_threshold),
        detections_per_img=int(args.detections_per_img),
        desc="initial action eval",
    )
    print(
        f"initial: AP50={initial_metrics['ap50']:.4f} "
        f"AP75={initial_metrics['ap75']:.4f}"
    )

    for epoch in range(1, int(args.epochs) + 1):
        train_loss = train_one_epoch(
            model,
            action_head,
            teacher_head,
            train_loader,
            optimizer,
            device,
            loss_config,
            match_mode=str(args.match_mode),
            box_base=str(args.box_base),
            epoch=epoch,
        )
        metrics = evaluate_action_head(
            model,
            action_head,
            val_loader,
            device,
            loss_config,
            metric_kwargs,
            match_mode=str(args.match_mode),
            box_base=str(args.box_base),
            action_score_threshold=float(args.action_score_threshold),
            nms_threshold=float(args.nms_threshold),
            detections_per_img=int(args.detections_per_img),
            desc=f"eval epoch {epoch}",
        )
        row = {"epoch": epoch, **train_loss, **{f"val_{k}": v for k, v in metrics.items() if _is_scalar(v)}}
        history.append(row)
        print(
            f"epoch {epoch}: loss={train_loss.get('loss_total', 0.0):.4f} "
            f"AP50={metrics['ap50']:.4f} AP75={metrics['ap75']:.4f}"
        )

        save_checkpoint(action_head, run_dir / "action_head_last.pth", {"epoch": epoch})
        ap75 = float(metrics["ap75"])
        if should_update_high_water_mark(ap75, best_snapshot, min_delta=float(args.hwm_min_delta)):
            best_ap75 = ap75
            best_snapshot = capture_high_water_mark_module(
                action_head,
                metric_value=ap75,
                epoch=epoch,
                metadata={"metric": "ap75", "ap50": float(metrics["ap50"])},
            )
            save_checkpoint(
                action_head,
                run_dir / "action_head_best_ap75.pth",
                {"epoch": epoch, "ap75": ap75, "ap50": float(metrics["ap50"])},
            )
            teacher_head = copy.deepcopy(action_head).to(device)
            load_high_water_mark_module(teacher_head, best_snapshot)
            for param in teacher_head.parameters():
                param.requires_grad = False

    final_metrics = history[-1] if history else {}
    result = {
        "run_name": args.run_name,
        "config": config,
        "action_config": {
            "feature_dim": feature_dim,
            "hidden_dim": args.hidden_dim,
            "max_score_delta": args.max_score_delta,
            "max_box_delta": args.max_box_delta,
            "match_mode": args.match_mode,
            "box_base": args.box_base,
            "loss": loss_config.__dict__,
            "action_score_threshold": args.action_score_threshold,
            "nms_threshold": args.nms_threshold,
            "detections_per_img": args.detections_per_img,
        },
        "initial_metrics": initial_metrics,
        "default_metrics": default_metrics,
        "best_ap75": best_ap75,
        "best_epoch": best_snapshot.epoch if best_snapshot is not None else None,
        "history": history,
        "final_row": final_metrics,
    }
    save_json(result, run_dir / "eval_metrics.json")
    print(result)


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float)) or value is None


if __name__ == "__main__":
    main()
