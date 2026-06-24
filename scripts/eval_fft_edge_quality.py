"""Evaluate FFT-reconstruction edge quality as a reward signal for object detection.

Pipeline:
  proposal crop -> FFT -> spectral processing -> iFFT -> Sobel edge -> scalar score
  Compare edge score ranking vs IoU ranking on TP/FP pairs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.spectral.roi_crop import crop_and_resize_roi
from spectral_detection_posttrain.utils.config import load_config
from spectral_detection_posttrain.utils.io import ensure_run_dir, save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed
from spectral_detection_posttrain.matching.pred_gt_matcher import match_predictions_to_gt


def sobel_edge_strength(crops: torch.Tensor) -> torch.Tensor:
    """Compute mean Sobel edge magnitude per crop.

    Args:
        crops: (M, C, H, W) tensor.

    Returns:
        (M,) scalar edge strengths.
    """
    gray = crops.mean(dim=1, keepdim=True)  # (M, 1, H, W)
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=crops.dtype, device=crops.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=crops.dtype, device=crops.device,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8).squeeze(1)  # (M, H, W)
    return mag.flatten(1).mean(dim=1)  # (M,)


def reconstruct_phase_only(crop: torch.Tensor) -> torch.Tensor:
    """Phase-only reconstruction: set |FFT|=1, keep phase."""
    fft = torch.fft.rfft2(crop, dim=(-2, -1))  # (C, H, W//2+1)
    phase = torch.angle(fft)
    fft_phase = torch.exp(1j * phase)
    return torch.fft.irfft2(fft_phase, s=crop.shape[-2:], dim=(-2, -1))


def reconstruct_hf_boost(crop: torch.Tensor, boost: float = 2.0) -> torch.Tensor:
    """High-frequency emphasis: linear radial boost."""
    _, h, w = crop.shape
    fft = torch.fft.rfft2(crop, dim=(-2, -1))
    amp = torch.abs(fft)
    phase = torch.angle(fft)
    # radial frequency grid
    fh = torch.fft.fftfreq(h, device=crop.device)
    fw = torch.fft.rfftfreq(w, device=crop.device)
    yy, xx = torch.meshgrid(fh, fw, indexing="ij")
    r = torch.sqrt(xx ** 2 + yy ** 2)
    r_max = r.max().clamp_min(1e-6)
    # linear boost from 1 at DC to `boost` at Nyquist
    weight = 1.0 + (boost - 1.0) * (r / r_max)
    fft_boosted = amp * weight * torch.exp(1j * phase)
    return torch.fft.irfft2(fft_boosted, s=(h, w), dim=(-2, -1))


def compute_edge_scores(crops: torch.Tensor, mode: str) -> torch.Tensor:
    """Compute edge scores for a batch of crops under a reconstruction mode."""
    if mode == "raw":
        return sobel_edge_strength(crops)
    elif mode == "phase_only":
        recons = torch.stack([reconstruct_phase_only(c) for c in crops])
        return sobel_edge_strength(recons)
    elif mode == "hf_boost":
        recons = torch.stack([reconstruct_hf_boost(c) for c in crops])
        return sobel_edge_strength(recons)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def pair_agreement(edge_scores: np.ndarray, ious: np.ndarray) -> float:
    """Fraction of pairs where edge and IoU order agree."""
    n = len(edge_scores)
    if n < 2:
        return 0.0
    concordant = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            edge_order = int(edge_scores[i] > edge_scores[j]) - int(edge_scores[i] < edge_scores[j])
            iou_order = int(ious[i] > ious[j]) - int(ious[i] < ious[j])
            if edge_order == iou_order:
                concordant += 1
    return concordant / total if total > 0 else 0.0


def uncertain_agreement(edge_scores: np.ndarray, ious: np.ndarray, confs: np.ndarray) -> float:
    """Pair agreement restricted to proposals with confidence in [0.1, 0.5]."""
    mask = (confs >= 0.1) & (confs <= 0.5)
    if mask.sum() < 2:
        return 0.0
    return pair_agreement(edge_scores[mask], ious[mask])


def build_proposal_iou_and_conf(
    image: torch.Tensor,
    output: dict,
    target: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (boxes, ious, confs, matched_gt_boxes) for all proposals above score threshold."""
    pred_boxes = output.get("boxes", torch.empty((0, 4), device=device))
    pred_scores = output.get("scores", torch.ones((len(pred_boxes),), device=device))
    gt_boxes = target.get("boxes", torch.empty((0, 4), device=device))

    # keep all proposals with score > 0.05 (standard RPN threshold)
    keep = pred_scores > 0.05
    boxes = pred_boxes[keep]
    confs = pred_scores[keep]

    if len(boxes) == 0 or len(gt_boxes) == 0:
        return boxes, torch.zeros(len(boxes), device=device), confs, gt_boxes

    # IoU with best-matching GT per proposal
    iou_matrix = torchvision.ops.box_iou(boxes, gt_boxes.to(device))
    ious = iou_matrix.max(dim=1).values

    # matched GT box per proposal (for cropping GT region)
    matched_gt_idx = iou_matrix.argmax(dim=1)
    matched_gt_boxes = gt_boxes.to(device)[matched_gt_idx]

    return boxes, ious, confs, matched_gt_boxes


def evaluate_mode(
    model: torch.nn.Module,
    val_loader: Any,
    device: torch.device,
    mode: str,
    limit_val: int | None = None,
) -> dict:
    """Evaluate one reconstruction mode on the val set."""
    model.eval()
    all_pair_agrs: list[float] = []
    all_uncertain_agrs: list[float] = []
    num_images = 0

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc=f"Eval {mode}"):
            images_d = [img.to(device) for img in images]
            outputs = model(images_d)

            for img, output, target in zip(images, outputs, targets):
                num_images += 1
                if limit_val and num_images > limit_val:
                    break

                boxes, ious, confs, matched_gt = build_proposal_iou_and_conf(
                    img, output, target, device,
                )
                if len(boxes) < 2:
                    continue

                # Crop proposals and matched GT boxes
                crops = torch.stack([
                    crop_and_resize_roi(img.cpu(), b, size=64) for b in boxes.cpu()
                ]).to(device)

                edge_scores = compute_edge_scores(crops, mode).cpu().numpy()
                ious_np = ious.cpu().numpy()
                confs_np = confs.cpu().numpy()

                all_pair_agrs.append(pair_agreement(edge_scores, ious_np))
                all_uncertain_agrs.append(uncertain_agreement(edge_scores, ious_np, confs_np))

            if limit_val and num_images > limit_val:
                break

    return {
        "pair_agreement_mean": float(np.mean(all_pair_agrs)) if all_pair_agrs else 0.0,
        "pair_agreement_std": float(np.std(all_pair_agrs)) if all_pair_agrs else 0.0,
        "uncertain_agreement_mean": float(np.mean(all_uncertain_agrs)) if all_uncertain_agrs else 0.0,
        "uncertain_agreement_std": float(np.std(all_uncertain_agrs)) if all_uncertain_agrs else 0.0,
        "num_images": num_images,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FFT-reconstruction edge quality vs IoU.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", default="runs/mvp_pf_baseline/checkpoint_last.pth")
    parser.add_argument("--run-name", default="edge_quality_eval")
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device({})
    run_dir = ensure_run_dir(args.run_name)

    # Load config and model
    config = load_config(args.config) if Path(args.config).exists() else {
        "data": {"root": "./data", "max_size": 320, "train_fraction": 0.8, "num_workers": 0},
        "train": {"batch_size": 2},
        "model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn", "pretrained": False, "num_classes": 2, "min_size": 320, "max_size": 320},
        "eval": {"batch_size": 2},
        "matching": {"iou_threshold": 0.5, "score_threshold": 0.05},
    }
    _, val_loader = build_penn_fudan_loaders(config, limit_train=1, limit_val=args.limit_val)
    model = build_detector(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)

    results: dict[str, dict] = {}
    for mode in ["raw", "phase_only", "hf_boost"]:
        results[mode] = evaluate_mode(model, val_loader, device, mode, limit_val=args.limit_val)
        print(f"\n{mode}:")
        for k, v in results[mode].items():
            print(f"  {k}: {v}")

    save_json(results, run_dir / "edge_quality_metrics.json")
    print(f"\nSaved to {run_dir / 'edge_quality_metrics.json'}")


if __name__ == "__main__":
    import torchvision
    main()
