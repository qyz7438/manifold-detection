from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("runs/.matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.core.models import build_detector
from spectral_detection_posttrain.signals.fft.spectral_reward import auc_tp_vs_fp, compute_prediction_rewards
from spectral_detection_posttrain.utils.config import load_config, save_config
from spectral_detection_posttrain.utils.io import ensure_run_dir, load_checkpoint, save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate R_amp discrimination for TP vs FP detections.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--limit-val", type=int, default=None)
    return parser.parse_args()


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    run_dir = ensure_run_dir(args.run_name)
    save_config(config, run_dir / "config.yaml")

    _, val_loader = build_penn_fudan_loaders(
        config,
        limit_train=1,
        limit_val=args.limit_val,
        batch_size=int(config["eval"].get("batch_size", 2)),
    )
    device = resolve_device(config)
    model_cfg = dict(config)
    model_cfg["model"] = dict(config["model"])
    model_cfg["model"]["pretrained"] = False
    model = build_detector(model_cfg).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    tp_values: list[float] = []
    fp_values: list[float] = []
    for images, targets in tqdm(val_loader, desc=args.run_name):
        outputs = model([image.to(device) for image in images])
        for image, output, target in zip(images, outputs, targets):
            reward_info = compute_prediction_rewards(
                image.cpu(),
                {k: v.detach().cpu() for k, v in output.items()},
                {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in target.items()},
                iou_threshold=float(config["matching"].get("iou_threshold", 0.5)),
                score_threshold=float(config["matching"].get("score_threshold", 0.05)),
            )
            tp_values.extend(reward_info["tp_r_amp"])
            fp_values.extend(reward_info["fp_r_amp"])

    metrics = {
        "mean_r_amp_tp": _mean(tp_values),
        "mean_r_amp_fp": _mean(fp_values),
        "auc_tp_vs_fp": auc_tp_vs_fp(tp_values, fp_values),
        "num_tp": len(tp_values),
        "num_fp": len(fp_values),
    }
    save_json(metrics, run_dir / "spectral_reward_metrics.json")

    fig, ax = plt.subplots(figsize=(6, 4))
    if tp_values:
        ax.hist(tp_values, bins=10, alpha=0.6, label="TP R_amp")
    if fp_values:
        ax.hist(fp_values, bins=10, alpha=0.6, label="FP R_amp")
    ax.set_xlabel("R_amp")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "r_amp_distribution.png", dpi=160)
    plt.close(fig)
    print(metrics)


if __name__ == "__main__":
    main()
