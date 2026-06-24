from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.core.models import build_detector, freeze_backbone
from spectral_detection_posttrain.signals.fft.spectral_reward import compute_prediction_rewards
from spectral_detection_posttrain.trainers.detection.train_baseline import _to_device_targets
from spectral_detection_posttrain.utils.config import load_config, override_epochs, save_config
from spectral_detection_posttrain.utils.io import append_jsonl, ensure_run_dir, load_checkpoint, save_checkpoint
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reward-weighted detector post-training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def _batch_image_reward(model, images: list[torch.Tensor], targets: list[dict], device: torch.device, config: dict) -> float:
    model.eval()
    predictions = model([image.to(device) for image in images])
    rewards = []
    for image, prediction, target in zip(images, predictions, targets):
        reward_info = compute_prediction_rewards(
            image.cpu(),
            {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in prediction.items()},
            target,
            iou_threshold=float(config["matching"].get("iou_threshold", 0.5)),
            score_threshold=float(config["matching"].get("score_threshold", 0.05)),
        )
        values = reward_info["tp_r_amp"]
        if values:
            rewards.append(sum(values) / len(values))
        else:
            rewards.append(0.0)
    model.train()
    return float(sum(rewards) / max(1, len(rewards)))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    override_epochs(config, "posttrain", args.epochs)
    set_seed(int(config.get("seed", 42)))
    run_dir = ensure_run_dir(args.run_name)
    save_config(config, run_dir / "config.yaml")

    train_loader, _ = build_penn_fudan_loaders(
        config,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        batch_size=int(config["posttrain"].get("batch_size", 2)),
    )
    device = resolve_device(config)
    model_cfg = dict(config)
    model_cfg["model"] = dict(config["model"])
    model_cfg["model"]["pretrained"] = False
    model = build_detector(model_cfg).to(device)
    load_checkpoint(model, args.baseline, device)
    if bool(config["posttrain"].get("freeze_backbone", True)):
        freeze_backbone(model)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["posttrain"].get("lr", 0.0005)),
        momentum=float(config["train"].get("momentum", 0.9)),
        weight_decay=float(config["train"].get("weight_decay", 0.0005)),
    )
    reward_lambda = float(config["posttrain"].get("reward_lambda", 1.0))
    epochs = int(config["posttrain"].get("epochs", 1))

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_det_loss = 0.0
        total_reward = 0.0
        total_weight = 0.0
        total_seen = 0
        progress = tqdm(train_loader, desc=f"{args.run_name} epoch {epoch}/{epochs}")
        for images, targets in progress:
            image_reward = _batch_image_reward(model, images, targets, device, config)
            image_weight = 1.0 + reward_lambda * (1.0 - image_reward)
            device_images = [image.to(device) for image in images]
            device_targets = _to_device_targets(targets, device)
            loss_dict = model(device_images, device_targets)
            det_loss = sum(loss_dict.values())
            loss = det_loss * image_weight
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = len(images)
            total_seen += batch_size
            total_loss += float(loss.item()) * batch_size
            total_det_loss += float(det_loss.item()) * batch_size
            total_reward += image_reward * batch_size
            total_weight += image_weight * batch_size
            progress.set_postfix(loss=total_loss / max(1, total_seen), reward=total_reward / max(1, total_seen))

        row = {
            "epoch": epoch,
            "loss_post": total_loss / max(1, total_seen),
            "loss_det": total_det_loss / max(1, total_seen),
            "image_reward": total_reward / max(1, total_seen),
            "image_weight": total_weight / max(1, total_seen),
        }
        append_jsonl(row, run_dir / "metrics_train.jsonl")
        save_checkpoint(model, run_dir / "checkpoint_last.pth", {"epoch": epoch, "run_name": args.run_name})
        print(row)


if __name__ == "__main__":
    main()
