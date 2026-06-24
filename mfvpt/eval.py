from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from mfvpt.datasets import build_cifar100_loaders
from mfvpt.models import build_model, normalize_for_imagenet
from mfvpt.transforms.fourier import high_freq_perturb, low_pass_filter
from mfvpt.transforms.patch import add_patch
from mfvpt.utils.config import load_config, save_config
from mfvpt.utils.io import ensure_run_dir, load_checkpoint, save_json
from mfvpt.utils.metrics import (
    accuracy,
    expected_calibration_error,
    high_confidence_error_rate,
    prediction_consistency,
)
from mfvpt.utils.reporting import write_report
from mfvpt.utils.seed import set_seed
from mfvpt.utils.train import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate robustness and calibration metrics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--limit-val", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    run_dir = ensure_run_dir(args.run_name)
    save_config(config, run_dir / "config.yaml")

    _, val_loader = build_cifar100_loaders(
        config,
        train_mode="baseline",
        limit_train=1,
        limit_val=args.limit_val,
        batch_size=int(config["train"].get("batch_size", 32)),
    )
    device = resolve_device(config)
    config["model"]["pretrained"] = False
    model = build_model(config).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()
    perturb_cfg = config["perturb"]

    logits_clean, logits_low, logits_high, logits_patch, labels_all = [], [], [], [], []
    for images, labels in tqdm(val_loader, desc=args.run_name):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        low = low_pass_filter(images, ratio=float(perturb_cfg.get("low_ratio", 0.25)))
        high = high_freq_perturb(
            images,
            strength=float(perturb_cfg.get("high_strength", 0.10)),
            ratio=float(perturb_cfg.get("high_ratio", 0.25)),
        )
        patch = add_patch(
            images,
            patch_type=str(perturb_cfg.get("patch_type", "random")),
            patch_size=int(perturb_cfg.get("patch_size", 32)),
        )
        logits_clean.append(model(normalize_for_imagenet(images)).cpu())
        logits_low.append(model(normalize_for_imagenet(low)).cpu())
        logits_high.append(model(normalize_for_imagenet(high)).cpu())
        logits_patch.append(model(normalize_for_imagenet(patch)).cpu())
        labels_all.append(labels.cpu())

    clean = torch.cat(logits_clean)
    low = torch.cat(logits_low)
    high = torch.cat(logits_high)
    patch = torch.cat(logits_patch)
    labels = torch.cat(labels_all)
    threshold = float(config["eval"].get("hce_threshold", 0.90))
    bins = int(config["eval"].get("ece_bins", 15))
    metrics = {
        "clean_acc": accuracy(clean, labels),
        "low_acc": accuracy(low, labels),
        "high_acc": accuracy(high, labels),
        "patch_acc": accuracy(patch, labels),
        "cons_low": prediction_consistency(clean, low),
        "cons_high": prediction_consistency(clean, high),
        "cons_patch": prediction_consistency(clean, patch),
        "hce_clean": high_confidence_error_rate(clean, labels, threshold),
        "hce_low": high_confidence_error_rate(low, labels, threshold),
        "hce_high": high_confidence_error_rate(high, labels, threshold),
        "hce_patch": high_confidence_error_rate(patch, labels, threshold),
        "ece_clean": expected_calibration_error(clean, labels, bins),
        "ece_low": expected_calibration_error(low, labels, bins),
        "ece_high": expected_calibration_error(high, labels, bins),
        "ece_patch": expected_calibration_error(patch, labels, bins),
    }
    save_json(metrics, run_dir / "eval_metrics.json")
    print(metrics)
    write_report("runs")


if __name__ == "__main__":
    main()
