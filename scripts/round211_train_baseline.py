"""Train baseline Faster R-CNN on VOC detection subset for Round 2.11."""
from __future__ import annotations

import argparse
import yaml

import torch
from tqdm import tqdm

from spectral_detection_posttrain.datasets.voc_detection import build_voc_detection_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import ensure_run_dir, save_checkpoint, save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


def _to_device(targets: list[dict], device: torch.device) -> list[dict]:
    return [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]


def _eval_model(model, val_loader, device, config, run_dir):
    model.eval()
    predictions, targets_list = [], []
    for images, batch_targets in val_loader:
        outputs = model([img.to(device) for img in images])
        predictions.extend([{k: v.detach().cpu() for k, v in output.items()} for output in outputs])
        targets_list.extend([{k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in t.items()} for t in batch_targets])
    eval_cfg = config.get("eval", {})
    metrics = evaluate_detection_predictions(
        predictions, targets_list,
        iou_threshold=float(eval_cfg.get("score_threshold", 0.05)),
        score_threshold=float(eval_cfg.get("score_threshold", 0.05)),
        high_conf_threshold=float(eval_cfg.get("high_conf_threshold", 0.7)),
    )
    save_json(metrics, run_dir / "eval_metrics.json")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["seed"] = args.seed

    set_seed(args.seed)
    device = resolve_device(config)
    run_dir = ensure_run_dir(args.run_name)
    train_loader, val_loader = build_voc_detection_loaders(
        config, limit_train=args.limit_train, limit_val=args.limit_val)

    model = build_detector(config).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    train_cfg = config.get("train", {})
    optimizer = torch.optim.SGD(
        trainable_params,
        lr=float(train_cfg.get("lr", 0.003)),
        momentum=float(train_cfg.get("momentum", 0.9)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0005)),
    )

    history = []
    total_epochs = int(args.epochs)
    for epoch in range(1, total_epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for images, targets in tqdm(train_loader, desc=f"{args.run_name} epoch {epoch}"):
            images = [img.to(device) for img in images]
            targets = _to_device(targets, device)
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(images)
            total_seen += len(images)
        avg_loss = total_loss / max(1, total_seen)
        history.append({"epoch": epoch, "train_loss": avg_loss})
        save_checkpoint(model, run_dir / "checkpoint_last.pth", {"epoch": epoch})

    metrics = _eval_model(model, val_loader, device, config, run_dir)
    metrics["run_name"] = args.run_name
    metrics["history"] = history
    save_json(metrics, run_dir / "eval_metrics.json")
    print(f"AP50={metrics.get('AP50', 'N/A')}  AP75={metrics.get('AP75', 'N/A')}")


if __name__ == "__main__":
    main()
