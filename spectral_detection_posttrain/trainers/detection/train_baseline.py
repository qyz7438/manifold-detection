from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.experiments.canonical_runner import (
    build_experiment_model,
    checkpoint_metadata,
    prepare_experiment_from_config,
)
from spectral_detection_posttrain.utils.config import load_config, override_epochs
from spectral_detection_posttrain.utils.io import append_jsonl, save_checkpoint
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline detector on Penn-Fudan.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def _to_device_targets(targets: list[dict], device: torch.device) -> list[dict]:
    return [{k: v.to(device) if torch.is_tensor(v) else v for k, v in target.items()} for target in targets]


def train_one_epoch(model, loader, optimizer, device, run_name: str, epoch: int, epochs: int) -> dict:
    model.train()
    total_loss = 0.0
    total_seen = 0
    progress = tqdm(loader, desc=f"{run_name} epoch {epoch}/{epochs}")
    for images, targets in progress:
        images = [image.to(device) for image in images]
        targets = _to_device_targets(targets, device)
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        batch_size = len(images)
        total_seen += batch_size
        total_loss += float(loss.item()) * batch_size
        progress.set_postfix(loss=total_loss / max(1, total_seen))
    return {"train_loss": total_loss / max(1, total_seen)}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    override_epochs(config, "train", args.epochs)
    set_seed(int(config.get("seed", 42)))
    context = prepare_experiment_from_config(config, args.config, args.run_name, phase="train")
    config = context.config
    run_dir = context.run_dir

    train_loader, _ = build_penn_fudan_loaders(
        config,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        batch_size=int(config["train"].get("batch_size", 2)),
    )
    device = resolve_device(config)
    model = build_experiment_model(context, device=device)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["train"].get("lr", 0.005)),
        momentum=float(config["train"].get("momentum", 0.9)),
        weight_decay=float(config["train"].get("weight_decay", 0.0005)),
    )
    epochs = int(config["train"].get("epochs", 1))
    for epoch in range(1, epochs + 1):
        row = train_one_epoch(model, train_loader, optimizer, device, args.run_name, epoch, epochs)
        row["epoch"] = epoch
        append_jsonl(row, run_dir / "metrics_train.jsonl")
        save_checkpoint(model, run_dir / "checkpoint_last.pth", checkpoint_metadata(context, {"epoch": epoch}))
        print(row)


if __name__ == "__main__":
    main()
