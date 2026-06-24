from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from mfvpt.datasets import build_cifar100_loaders
from mfvpt.models import build_model, normalize_for_imagenet
from mfvpt.transforms.standard_aug import apply_fourier_training_aug
from mfvpt.utils.config import load_config, override_epochs, save_config
from mfvpt.utils.io import append_jsonl, ensure_run_dir, save_checkpoint
from mfvpt.utils.metrics import accuracy
from mfvpt.utils.seed import set_seed
from mfvpt.utils.train import make_optimizer, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CIFAR-100 ViT baselines.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def evaluate_clean(model: torch.nn.Module, loader, device: torch.device) -> float:
    model.eval()
    logits_all = []
    labels_all = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits_all.append(model(normalize_for_imagenet(images)).detach().cpu())
        labels_all.append(labels.detach().cpu())
    return accuracy(torch.cat(logits_all), torch.cat(labels_all))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    override_epochs(config, args.epochs, "train")
    set_seed(int(config.get("seed", 42)))
    run_dir = ensure_run_dir(args.run_name)
    save_config(config, run_dir / "config.yaml")

    mode = str(config["train"].get("mode", "baseline"))
    batch_size = int(config["train"].get("batch_size", 32))
    train_loader, val_loader = build_cifar100_loaders(
        config,
        train_mode=mode,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        batch_size=batch_size,
    )
    device = resolve_device(config)
    model = build_model(config).to(device)
    optimizer = make_optimizer(
        model.parameters(),
        lr=float(config["train"].get("lr", 3e-4)),
        weight_decay=float(config["train"].get("weight_decay", 0.05)),
    )
    amp_enabled = bool(config["train"].get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)
    epochs = int(config["train"].get("epochs", 1))
    metrics_path = run_dir / "metrics_train.jsonl"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        progress = tqdm(train_loader, desc=f"{args.run_name} epoch {epoch}/{epochs}")
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if mode == "fourier_aug":
                images = apply_fourier_training_aug(images, config)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                logits = model(normalize_for_imagenet(images))
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size_seen = labels.size(0)
            total_loss += loss.item() * batch_size_seen
            total_seen += batch_size_seen
            progress.set_postfix(loss=total_loss / max(1, total_seen))

        val_clean_acc = evaluate_clean(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, total_seen),
            "val_clean_acc": val_clean_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        append_jsonl(row, metrics_path)
        save_checkpoint(model, run_dir / "checkpoint_last.pth", {"epoch": epoch, "run_name": args.run_name})
        print(row)


if __name__ == "__main__":
    main()
