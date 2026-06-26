"""Geometry-based spatial rewards for segmentation masks."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mask_iou_reward(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.bool()
    target = target.bool()
    inter = (pred & target).sum().float()
    union = (pred | target).sum().float()
    return (inter / union.clamp_min(eps)).clamp(0.0, 1.0)


def dice_reward(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.bool()
    target = target.bool()
    inter = (pred & target).sum().float()
    denom = pred.sum().float() + target.sum().float()
    return ((2.0 * inter) / denom.clamp_min(eps)).clamp(0.0, 1.0)


def _boundary(mask: torch.Tensor) -> torch.Tensor:
    x = mask.float().view(1, 1, *mask.shape)
    eroded = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
    return (x - eroded).squeeze(0).squeeze(0) > 0


def _dilate(mask: torch.Tensor, tolerance: int) -> torch.Tensor:
    x = mask.float().view(1, 1, *mask.shape)
    size = 2 * tolerance + 1
    return F.max_pool2d(x, kernel_size=size, stride=1, padding=tolerance).squeeze(0).squeeze(0) > 0


def boundary_reward(pred: torch.Tensor, target: torch.Tensor, tolerance: int = 2, eps: float = 1e-6) -> torch.Tensor:
    pred_b = _boundary(pred.bool())
    target_b = _boundary(target.bool())
    if not pred_b.any() and not target_b.any():
        return torch.tensor(1.0, dtype=torch.float32, device=pred.device)
    precision = (pred_b & _dilate(target_b, tolerance)).sum().float() / pred_b.sum().float().clamp_min(eps)
    recall = (target_b & _dilate(pred_b, tolerance)).sum().float() / target_b.sum().float().clamp_min(eps)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(eps)
    return f1.clamp(0.0, 1.0)


def connected_component_reward(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.bool()
    target = target.bool()
    if not target.any():
        return torch.tensor(1.0, dtype=torch.float32, device=pred.device) if not pred.any() else torch.tensor(0.0, dtype=torch.float32, device=pred.device)
    overlap = (pred & target).sum().float() / target.sum().float().clamp_min(eps)
    extra = (pred & ~target).sum().float() / pred.sum().float().clamp_min(eps)
    return (overlap * (1.0 - extra)).clamp(0.0, 1.0)


def centroid_distance_reward(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.bool()
    target = target.bool()
    if not pred.any() or not target.any():
        return torch.tensor(0.0, dtype=torch.float32, device=pred.device)

    def _centroid(m: torch.Tensor) -> torch.Tensor:
        coords = torch.nonzero(m, as_tuple=False).float()
        return coords.mean(dim=0) if coords.numel() else m.new_zeros((2,))

    pred_c = _centroid(pred)
    target_c = _centroid(target)
    pred_y, pred_x = torch.nonzero(pred, as_tuple=True)
    size = torch.sqrt(((pred_x.float().max() - pred_x.float().min() + 1) ** 2 + (pred_y.float().max() - pred_y.float().min() + 1) ** 2).clamp_min(eps))
    dist = torch.sqrt(((pred_c - target_c) ** 2).sum())
    return torch.exp(-2.0 * dist / size.clamp_min(eps)).clamp(0.0, 1.0)
