"""Common utilities for Plan 2.x runner scripts.

This module collects the helper functions that are copy-pasted across dozens of
scripts/round2*_runner.py files.  Functions are intentionally thin wrappers
around the existing spectral_detection_posttrain APIs so that legacy runners can
be migrated incrementally without behaviour changes.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.core.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def setup_seed(seed: int) -> None:
    """Deterministic seed including cudnn flags."""
    set_seed(seed)


# ---------------------------------------------------------------------------
# Model / data / eval builders used by almost every runner
# ---------------------------------------------------------------------------


def build_mobv3_detector(num_classes: int = 2, pretrained: bool = True, afm_type: str = "none",
                         afm_residual_mode: str = "current") -> torch.nn.Module:
    """Build the canonical Faster R-CNN MobileNetV3-Large-320-FPN used in Plan 2.x."""
    is_afm = afm_type != "none"
    cfg = {
        "model": {
            "name": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "pretrained": pretrained and not is_afm,
            "num_classes": num_classes,
            "min_size": 320,
            "max_size": 320,
            "afm_channels": 256 if is_afm else 0,
            "afm_type": afm_type,
            "afm_residual_mode": afm_residual_mode,
        }
    }
    return build_detector(cfg)


def build_penn_fudan_loaders_320(batch_size: int = 2, train_fraction: float = 0.8,
                                 num_workers: int = 0) -> tuple:
    """Canonical PF loaders for Plan 2.x (max_size=320)."""
    return build_penn_fudan_loaders({
        "data": {"root": "./data", "max_size": 320, "train_fraction": train_fraction,
                 "num_workers": num_workers},
        "train": {"batch_size": batch_size},
    })


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, val_loader, device: torch.device,
                   iou_threshold: float = 0.5, score_threshold: float = 0.05) -> dict:
    """Run inference on val_loader and return detection metrics (CPU tensors)."""
    model.eval()
    predictions, targets = [], []
    for images, batch_targets in val_loader:
        outputs = model([img.to(device) for img in images])
        predictions.extend([{k: v.cpu() for k, v in out.items()} for out in outputs])
        targets.extend([{k: v.cpu() for k, v in t.items()} for t in batch_targets])
    return evaluate_detection_predictions(
        predictions, targets,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )


# ---------------------------------------------------------------------------
# Parameter freeze / optimizer helpers
# ---------------------------------------------------------------------------


def unfreeze_rlvr(model: torch.nn.Module, backbone: bool = False, fpn: bool = True,
                  rpn: bool = True, box_head: bool = True, box_predictor: bool = True,
                  afm: bool = False) -> None:
    """Set requires_grad for the standard RLVR/AFM fine-tuning scheme.

    Defaults match the most common runner pattern: freeze backbone.body, unfreeze
    FPN/RPN/box_head/box_predictor, and put BatchNorm in eval mode.
    """
    for param in model.backbone.body.parameters():
        param.requires_grad = backbone
    if hasattr(model.backbone, "fpn"):
        for param in model.backbone.fpn.parameters():
            param.requires_grad = fpn
    for param in model.rpn.parameters():
        param.requires_grad = rpn
    for param in model.roi_heads.box_head.parameters():
        param.requires_grad = box_head
    for param in model.roi_heads.box_predictor.parameters():
        param.requires_grad = box_predictor

    if afm:
        for name, param in model.named_parameters():
            if any(k in name.lower() for k in ("afm", "gate", "mag", "phase")):
                param.requires_grad = True

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def build_sgd_optimizer(model: torch.nn.Module, head_lr: float = 0.001,
                        body_lr: float = 0.0001, extra_modules: Sequence | None = None,
                        momentum: float = 0.9, weight_decay: float = 0.0005) -> torch.optim.SGD:
    """Build SGD with separate lr groups for body (FPN+RPN) and head (box)."""
    body_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "box_head" in name or "box_predictor" in name:
            head_params.append(param)
        else:
            body_params.append(param)

    extra = []
    if extra_modules:
        for module in extra_modules:
            if module is not None:
                extra.extend([p for p in module.parameters() if p.requires_grad])

    return torch.optim.SGD(
        [
            {"params": body_params, "lr": body_lr},
            {"params": head_params, "lr": head_lr},
            {"params": extra, "lr": head_lr},
        ],
        lr=head_lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )


# ---------------------------------------------------------------------------
# Box decoding / GRPO sampling helpers
# ---------------------------------------------------------------------------


def decode_boxes(proposals: torch.Tensor, deltas: torch.Tensor,
                 stds: tuple[float, float, float, float] = (10.0, 10.0, 5.0, 5.0)) -> torch.Tensor:
    """Faster R-CNN BoxCoder decode used across Plan 2.x runners.

    Args:
        proposals: (N, 4) boxes in [x1, y1, x2, y2].
        deltas: (N, 4) encoded deltas (or raw network outputs divided by stds).
        stds: BoxCoder standard deviations.

    Returns:
        (N, 4) decoded boxes [x1, y1, x2, y2], clamped to >= 0.
    """
    dx, dy, dw, dh = deltas[:, 0] / stds[0], deltas[:, 1] / stds[1], deltas[:, 2] / stds[2], deltas[:, 3] / stds[3]
    widths = proposals[:, 2] - proposals[:, 0]
    heights = proposals[:, 3] - proposals[:, 1]
    ctr_x = proposals[:, 0] + 0.5 * widths
    ctr_y = proposals[:, 1] + 0.5 * heights

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    return torch.stack([
        pred_ctr_x - 0.5 * pred_w,
        pred_ctr_y - 0.5 * pred_h,
        pred_ctr_x + 0.5 * pred_w,
        pred_ctr_y + 0.5 * pred_h,
    ], dim=1).clamp(min=0)


def gaussian_log_prob(deltas: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Log probability of Gaussian-sampled deltas (shape: (N, G, 4))."""
    errors = (deltas - mu.unsqueeze(1)) / sigma.unsqueeze(1)
    return -0.5 * (errors.pow(2) + 2 * torch.log(sigma.unsqueeze(1)) + math.log(2 * math.pi)).sum(dim=-1)


def grpo_advantage(reward: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-row (per-proposal) advantage normalization used in GRPO variants."""
    mean = reward.mean(dim=1, keepdim=True)
    std = reward.std(dim=1, keepdim=True).clamp_min(eps)
    return (reward - mean) / std


def compute_loc_reward(iou_img: torch.Tensor) -> torch.Tensor:
    """Step reward used by Loc/Select splits and several GRPO runners."""
    reward = torch.zeros_like(iou_img)
    reward[iou_img >= 0.75] = 1.0
    reward[(iou_img >= 0.5) & (iou_img < 0.75)] = 0.3
    reward[iou_img < 0.5] = -0.5
    return reward


# ---------------------------------------------------------------------------
# Spectral / FFT helpers
# ---------------------------------------------------------------------------


def extract_perchan_fft(x: torch.Tensor, bands: tuple[float, float] = (0.3, 0.7)) -> torch.Tensor:
    """Extract per-channel FFT band features used by verifier variants.

    Args:
        x: (N, C, H, W) spatial tensor.
        bands: Radial frequency band thresholds.

    Returns:
        (N, C*6) tensor with low/mid/high amplitude and phase summaries.
    """
    lo_thr, hi_thr = bands
    fft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
    amp = torch.abs(fft)
    pha = torch.angle(fft)

    freq_h = torch.fft.fftfreq(x.shape[-2], device=x.device)
    freq_w = torch.fft.rfftfreq(x.shape[-1], device=x.device)
    grid_y, grid_x = torch.meshgrid(freq_h, freq_w, indexing="ij")
    radius = torch.sqrt(grid_x ** 2 + grid_y ** 2)
    radius = radius / radius.max().clamp_min(1e-6)

    lo_mask = (radius <= lo_thr).float()
    mid_mask = ((radius > lo_thr) & (radius <= hi_thr)).float()
    hi_mask = (radius > hi_thr).float()

    def _band_summary(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (tensor * mask).flatten(2).sum(2)

    return torch.cat([
        _band_summary(amp, lo_mask), _band_summary(amp, mid_mask), _band_summary(amp, hi_mask),
        _band_summary(pha, lo_mask), _band_summary(pha, mid_mask), _band_summary(pha, hi_mask),
    ], dim=1)


def fft_energy(crops: torch.Tensor) -> torch.Tensor:
    """Total log-spectral energy of a batch of cropped images."""
    if crops.numel() == 0:
        return torch.zeros(crops.shape[0], device=crops.device)
    fft = torch.fft.rfft2(crops, dim=(-2, -1), norm="ortho")
    return torch.log1p(torch.abs(fft).pow(2).mean(dim=(-3, -2, -1)))


# ---------------------------------------------------------------------------
# Image-space action helpers used by pixel-patch runners
# ---------------------------------------------------------------------------


def crop_image_batch(raw_images: list[torch.Tensor], boxes: torch.Tensor,
                     target_size: tuple[int, int] = (32, 32)) -> torch.Tensor:
    """Crop fixed-size patches from raw images for Plan 2.85/2.90 style actions.

    Args:
        raw_images: list of (C, H, W) tensors.
        boxes: (N, 4) boxes in image coordinates [x1, y1, x2, y2].
        target_size: (H, W) to interpolate each crop to.

    Returns:
        (N, C, H, W) cropped and resized tensor.
    """
    crops = []
    for box in boxes:
        img = raw_images[int(box[0].item())]
        x1, y1, x2, y2 = box[1:].long().tolist()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[-1], x2), min(img.shape[-2], y2)
        if x2 <= x1 or y2 <= y1:
            crop = torch.zeros(img.shape[0], 1, 1, device=img.device)
        else:
            crop = img[:, y1:y2, x1:x2]
        crop = F.interpolate(crop.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False)
        crops.append(crop.squeeze(0))
    return torch.stack(crops) if crops else torch.empty(0, device=boxes.device)


def apply_action_indices(base_boxes: torch.Tensor, action_indices: torch.Tensor,
                         grid: torch.Tensor | None = None) -> torch.Tensor:
    """Apply a discrete set of coordinate shifts to base boxes.

    The concrete shift table is the one used in round288/round289/round290.
    """
    if grid is None:
        # 9 actions: (dx, dy, dw, dh) in normalized coordinates
        grid = torch.tensor([
            [0.0, 0.0, 0.0, 0.0],
            [-0.05, 0.0, 0.0, 0.0], [0.05, 0.0, 0.0, 0.0],
            [0.0, -0.05, 0.0, 0.0], [0.0, 0.05, 0.0, 0.0],
            [0.0, 0.0, -0.05, 0.0], [0.0, 0.0, 0.05, 0.0],
            [0.0, 0.0, 0.0, -0.05], [0.0, 0.0, 0.0, 0.05],
        ], device=base_boxes.device)
    boxes = base_boxes.clone()
    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
    cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    shifts = grid[action_indices]
    cx = cx + shifts[:, 0] * w
    cy = cy + shifts[:, 1] * h
    w = w * (1.0 + shifts[:, 2])
    h = h * (1.0 + shifts[:, 3])
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=1).clamp(min=0)
