r"""Train a bottleneck box head as a structural replacement for RoI feature path.

Loads a pretrained Faster R-CNN baseline, replaces the standard TwoMLPHead
with BottleneckTwoMLPHead, freezes backbone/RPN, and fine-tunes the new
box_head + box_predictor for a few epochs.

Example:
    python train_bottleneck_box_head.py \
        --config spectral_detection_posttrain/configs/manifold_nwpu.yaml \
        --baseline runs/round2100_nwpu_baseline/checkpoint_best.pth \
        --run-name nwpu_bottleneck_head_c64_5ep \
        --bottleneck-channels 64 \
        --epochs 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from spectral_detection_posttrain.core.models.bottleneck_box_head import BottleneckTwoMLPHead
from spectral_detection_posttrain.core.models.build_detector import (
    build_detector,
    freeze_backbone,
    freeze_rpn,
)
from spectral_detection_posttrain.datasets import build_detection_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.utils.config import load_config
from spectral_detection_posttrain.utils.io import append_jsonl, save_checkpoint, save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--bottleneck-channels", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _to_device_targets(targets: list[dict], device: torch.device) -> list[dict]:
    return [{k: v.to(device) if torch.is_tensor(v) else v for k, v in target.items()} for target in targets]


def _set_model_train_for_detection_loss(model: nn.Module) -> None:
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


@torch.no_grad()
def eval_model(model: nn.Module, val_loader, device: torch.device, config: dict) -> dict[str, float]:
    model.eval()
    all_predictions = []
    all_targets = []
    for images, targets in val_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        all_predictions.extend(outputs)
        all_targets.extend([{k: v.to(device) if torch.is_tensor(v) else v for k, v in tgt.items()} for tgt in targets])
    metrics_50 = evaluate_detection_predictions(
        all_predictions, all_targets, num_classes=int(config["model"]["num_classes"]), iou_threshold=0.5
    )
    metrics_75 = evaluate_detection_predictions(
        all_predictions, all_targets, num_classes=int(config["model"]["num_classes"]), iou_threshold=0.75
    )
    return {"ap50": float(metrics_50["ap50"]), "ap75": float(metrics_75["ap50"])}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    device = torch.device(args.device) if args.device else resolve_device(config)

    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_detection_loaders(
        config,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        batch_size=int(config["posttrain"].get("batch_size", 1)),
    )

    # Build baseline model and load weights.
    model = build_detector(config)
    checkpoint = torch.load(args.baseline, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)

    # Replace box_head with bottleneck version.
    original_head = model.roi_heads.box_head
    grid_size = 7  # Standard MultiScaleRoIAlign output size.
    if isinstance(original_head, nn.Sequential):
        # Some wrapped heads; unwrap the actual TwoMLPHead.
        original_head = original_head[-1]
    bottleneck_head = BottleneckTwoMLPHead(
        in_channels=256,
        bottleneck_channels=args.bottleneck_channels,
        representation_size=1024,
        grid_size=grid_size,
    ).to(device)
    model.roi_heads.box_head = bottleneck_head

    # Freeze backbone and RPN; keep box_head and box_predictor trainable.
    freeze_backbone(model)
    freeze_rpn(model)

    optimizer = torch.optim.AdamW(
        list(bottleneck_head.parameters()) + list(model.roi_heads.box_predictor.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Initial eval before any updates.
    model.eval()
    initial_metrics = eval_model(model, val_loader, device, config)
    print(f"Initial after head replacement: AP50={initial_metrics['ap50']:.4f}, AP75={initial_metrics['ap75']:.4f}")

    best_ap50 = initial_metrics["ap50"]
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        _set_model_train_for_detection_loss(model)
        total_loss = 0.0
        total_seen = 0

        progress = tqdm(train_loader, desc=f"{args.run_name} epoch {epoch}/{args.epochs}")
        for images, targets in progress:
            images = [img.to(device) for img in images]
            targets = _to_device_targets(targets, device)

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = len(images)
            total_loss += float(loss.item()) * batch_size
            total_seen += batch_size
            progress.set_postfix(loss=total_loss / max(1, total_seen))

        row = {
            "epoch": epoch,
            "loss": total_loss / max(1, total_seen),
        }

        if args.eval_every > 0 and epoch % args.eval_every == 0:
            val_metrics = eval_model(model, val_loader, device, config)
            row["val_ap50"] = val_metrics["ap50"]
            row["val_ap75"] = val_metrics["ap75"]
            print(f"Epoch {epoch}: val AP50 = {val_metrics['ap50']:.4f}, AP75 = {val_metrics['ap75']:.4f}")
            if val_metrics["ap50"] > best_ap50:
                best_ap50 = val_metrics["ap50"]
                best_epoch = epoch
                save_checkpoint(
                    model,
                    run_dir / "checkpoint_best.pth",
                    {"epoch": epoch, **val_metrics, **row},
                )

        append_jsonl(row, run_dir / "metrics_train.jsonl")
        save_checkpoint(model, run_dir / "checkpoint_last.pth", {"epoch": epoch, **row})

    result = {
        "run_name": args.run_name,
        "bottleneck_channels": args.bottleneck_channels,
        "epochs": args.epochs,
        "lr": args.lr,
        "initial_val_ap50": initial_metrics["ap50"],
        "initial_val_ap75": initial_metrics["ap75"],
        "best_val_ap50": best_ap50,
        "best_epoch": best_epoch,
    }
    save_json(result, run_dir / "bottleneck_result.json")
    print(result)

if __name__ == "__main__":
    main()
