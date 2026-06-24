from __future__ import annotations

import argparse

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from mfvpt.datasets import build_cifar100_loaders
from mfvpt.losses import compute_posttrain_loss
from mfvpt.models import build_model
from mfvpt.train_baseline import evaluate_clean
from mfvpt.utils.config import load_config, override_epochs, save_config
from mfvpt.utils.io import append_jsonl, ensure_run_dir, load_checkpoint, save_checkpoint
from mfvpt.utils.seed import set_seed
from mfvpt.utils.train import make_optimizer, resolve_device, set_trainable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MFVPT post-training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    override_epochs(config, args.epochs, "posttrain")
    set_seed(int(config.get("seed", 42)))
    run_dir = ensure_run_dir(args.run_name)
    save_config(config, run_dir / "config.yaml")

    batch_size = int(config["posttrain"].get("batch_size", config["train"].get("batch_size", 32)))
    train_loader, val_loader = build_cifar100_loaders(
        config,
        train_mode="mfvpt_posttrain",
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        batch_size=batch_size,
    )
    device = resolve_device(config)
    config["model"]["pretrained"] = False
    model = build_model(config).to(device)
    load_checkpoint(model, args.baseline, device)
    set_trainable(model, str(config["posttrain"].get("trainable", "full")))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(
        trainable_params,
        lr=float(config["posttrain"].get("lr", 1e-5)),
        weight_decay=float(config["train"].get("weight_decay", 0.05)),
    )
    amp_enabled = bool(config["train"].get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)
    epochs = int(config["posttrain"].get("epochs", 1))
    metrics_path = run_dir / "metrics_train.jsonl"

    for epoch in range(1, epochs + 1):
        model.train()
        totals = {
            "loss_total": 0.0,
            "loss_ce": 0.0,
            "loss_view_consistency": 0.0,
            "loss_confidence": 0.0,
        }
        total_seen = 0
        progress = tqdm(train_loader, desc=f"{args.run_name} epoch {epoch}/{epochs}")
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                losses = compute_posttrain_loss(model, images, labels, config)
                loss = losses["loss_total"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size_seen = labels.size(0)
            total_seen += batch_size_seen
            for key in totals:
                totals[key] += losses[key].item() * batch_size_seen
            progress.set_postfix(loss=totals["loss_total"] / max(1, total_seen))

        val_clean_acc = evaluate_clean(model, val_loader, device)
        row = {key: value / max(1, total_seen) for key, value in totals.items()}
        row.update({"epoch": epoch, "val_clean_acc": val_clean_acc})
        append_jsonl(row, metrics_path)
        save_checkpoint(model, run_dir / "checkpoint_last.pth", {"epoch": epoch, "run_name": args.run_name})
        print(row)


if __name__ == "__main__":
    main()
