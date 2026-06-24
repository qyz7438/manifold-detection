from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("runs/.matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from mfvpt.datasets import build_cifar100_loaders
from mfvpt.models import build_model, normalize_for_imagenet
from mfvpt.transforms.fourier import high_freq_perturb, low_pass_filter
from mfvpt.transforms.patch import add_patch
from mfvpt.utils.config import load_config, save_config
from mfvpt.utils.io import ensure_run_dir, load_checkpoint
from mfvpt.utils.seed import set_seed
from mfvpt.utils.train import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize perturbations and prediction differences.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--ours", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--limit-val", type=int, default=16)
    return parser.parse_args()


def _top1(model, images):
    logits = model(normalize_for_imagenet(images))
    prob = torch.softmax(logits, dim=1)
    conf, pred = prob.max(dim=1)
    return pred, conf


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
        batch_size=min(16, int(config["train"].get("batch_size", 16))),
    )
    images, labels = next(iter(val_loader))
    device = resolve_device(config)
    images = images.to(device)
    labels = labels.to(device)
    perturb_cfg = config["perturb"]
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

    views = [("original", images), ("low-pass", low), ("high-frequency", high), ("patch", patch)]
    rows = min(8, images.size(0))
    fig, axes = plt.subplots(rows, len(views), figsize=(len(views) * 3, rows * 3))
    if rows == 1:
        axes = axes[None, :]
    for row in range(rows):
        for col, (name, tensor) in enumerate(views):
            axes[row, col].imshow(tensor[row].detach().cpu().permute(1, 2, 0).clamp(0, 1))
            axes[row, col].set_axis_off()
            if row == 0:
                axes[row, col].set_title(name)
    fig.tight_layout()
    fig.savefig(run_dir / "perturbation_grid.png", dpi=160)
    plt.close(fig)

    config["model"]["pretrained"] = False
    baseline = build_model(config).to(device).eval()
    ours = build_model(config).to(device).eval()
    load_checkpoint(baseline, args.baseline, device)
    load_checkpoint(ours, args.ours, device)

    lines = [
        "| Sample | View | Label | Baseline Top-1 | Baseline Conf | Posttrain Top-1 | Posttrain Conf |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for name, tensor in views:
        pred_b, conf_b = _top1(baseline, tensor)
        pred_o, conf_o = _top1(ours, tensor)
        for idx in range(min(rows, tensor.size(0))):
            lines.append(
                f"| {idx} | {name} | {labels[idx].item()} | {pred_b[idx].item()} | "
                f"{conf_b[idx].item():.4f} | {pred_o[idx].item()} | {conf_o[idx].item():.4f} |"
            )
    (run_dir / "prediction_compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {run_dir / 'perturbation_grid.png'} and {run_dir / 'prediction_compare.md'}")


if __name__ == "__main__":
    main()
