from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("runs/.matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.core.matching.pred_gt_matcher import match_predictions_to_gt
from spectral_detection_posttrain.core.models import build_detector
from spectral_detection_posttrain.signals.fft.fft_features import compute_fft_amplitude
from spectral_detection_posttrain.signals.fft.roi_crop import crop_and_resize_roi
from spectral_detection_posttrain.utils.config import load_config, save_config
from spectral_detection_posttrain.utils.io import ensure_run_dir, load_checkpoint
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize matched prediction/GT ROIs and FFT amplitudes.")
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

    _, val_loader = build_penn_fudan_loaders(
        config,
        limit_train=1,
        limit_val=args.limit_val,
        batch_size=1,
    )
    device = resolve_device(config)
    model_cfg = dict(config)
    model_cfg["model"] = dict(config["model"])
    model_cfg["model"]["pretrained"] = False
    model = build_detector(model_cfg).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    for images, targets in val_loader:
        image = images[0]
        target = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in targets[0].items()}
        output = model([image.to(device)])[0]
        prediction = {k: v.detach().cpu() for k, v in output.items()}
        matched = match_predictions_to_gt(
            prediction,
            target,
            iou_threshold=float(config["matching"].get("iou_threshold", 0.5)),
            score_threshold=float(config["matching"].get("score_threshold", 0.05)),
        )
        if not matched["matches"]:
            (run_dir / "no_match.txt").write_text("No matched prediction/GT pair was found for visualization.\n", encoding="utf-8")
            print(f"Wrote {run_dir / 'no_match.txt'}")
            return
        pair = matched["matches"][0]
        pred_roi = crop_and_resize_roi(image, prediction["boxes"][pair["pred_index"]])
        gt_roi = crop_and_resize_roi(image, target["boxes"][pair["gt_index"]])
        pred_amp = compute_fft_amplitude(pred_roi)
        gt_amp = compute_fft_amplitude(gt_roi)

        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        axes[0].imshow(image.permute(1, 2, 0).clamp(0, 1))
        axes[0].set_title("image")
        axes[1].imshow(pred_roi.permute(1, 2, 0).clamp(0, 1))
        axes[1].set_title("pred ROI")
        axes[2].imshow(gt_roi.permute(1, 2, 0).clamp(0, 1))
        axes[2].set_title("GT ROI")
        axes[3].imshow(pred_amp, cmap="magma")
        axes[3].set_title("pred FFT amp")
        axes[4].imshow(gt_amp, cmap="magma")
        axes[4].set_title("GT FFT amp")
        for ax in axes:
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(run_dir / "roi_fft_comparison.png", dpi=160)
        plt.close(fig)
        print(f"Wrote {run_dir / 'roi_fft_comparison.png'}")
        return

    (run_dir / "no_data.txt").write_text("No validation samples available.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
