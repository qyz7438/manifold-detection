"""Interpretable pixel-level signal functions for segmentation.

Each function takes an ``(C, H, W)`` image and an ``(H, W)`` boolean predicted
mask.  Where the original detection-side code used bounding boxes, these
versions use mask-aware cropping (either the masked image itself or the
smallest enclosing crop).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


EPS = 1.0e-8


def _clamp_box(box: torch.Tensor, height: int, width: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, int(math.floor(float(box[0])))))
    y1 = max(0, min(height - 1, int(math.floor(float(box[1])))))
    x2 = max(x1 + 1, min(width, int(math.ceil(float(box[2])))))
    y2 = max(y1 + 1, min(height, int(math.ceil(float(box[3])))))
    return x1, y1, x2, y2


def _crop_box(image: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    _, height, width = image.shape
    x1, y1, x2, y2 = _clamp_box(box, height, width)
    return image[:, y1:y2, x1:x2]


def _mask_crop(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    coords = torch.nonzero(mask, as_tuple=False)
    if coords.numel() == 0:
        return image
    y1 = int(coords[:, 0].min().item())
    x1 = int(coords[:, 1].min().item())
    y2 = int(coords[:, 0].max().item()) + 1
    x2 = int(coords[:, 1].max().item()) + 1
    return image[:, y1:y2, x1:x2]


def sobel_map(gray: torch.Tensor) -> torch.Tensor:
    gray4 = gray.reshape(1, 1, gray.shape[-2], gray.shape[-1]).float()
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=gray4.dtype,
        device=gray4.device,
    ).reshape(1, 1, 3, 3) / 8.0
    kernel_y = kernel_x.transpose(-1, -2)
    gx = F.conv2d(gray4, kernel_x, padding=1)
    gy = F.conv2d(gray4, kernel_y, padding=1)
    return torch.sqrt(gx.square() + gy.square() + EPS).squeeze(0).squeeze(0)


def robust_normalize_map(values: torch.Tensor) -> torch.Tensor:
    flat = values.flatten()
    if flat.numel() < 4:
        return values.clamp_min(0.0)
    lo = torch.quantile(flat, 0.05)
    hi = torch.quantile(flat, 0.95)
    if float(hi - lo) < EPS:
        return torch.zeros_like(values)
    return ((values - lo) / (hi - lo)).clamp(0.0, 1.0)


def phase_only_edge_map(gray: torch.Tensor) -> torch.Tensor:
    spectrum = torch.fft.fft2(gray.float())
    phase_only = torch.exp(1j * torch.angle(spectrum))
    recon = torch.fft.ifft2(phase_only).real
    recon = recon - recon.mean()
    std = recon.std()
    if float(std) > EPS:
        recon = recon / std
    return sobel_map(recon)


def boundary_and_interior_means(feature_map: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    crop = _mask_crop(feature_map.unsqueeze(0), mask).squeeze(0)
    if crop.numel() == 0:
        return crop.new_tensor(0.0), crop.new_tensor(0.0)
    crop_h, crop_w = crop.shape
    boundary_width = max(1, min(4, int(round(0.08 * min(crop_h, crop_w)))))
    bm = torch.zeros_like(crop, dtype=torch.bool)
    bm[:boundary_width, :] = True
    bm[-boundary_width:, :] = True
    bm[:, :boundary_width] = True
    bm[:, -boundary_width:] = True
    boundary = crop[bm]
    interior = crop[~bm]
    if interior.numel() == 0:
        interior = crop.reshape(-1)
    return boundary.mean(), interior.mean()


def ring_mean(feature_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    height, width = feature_map.shape
    coords = torch.nonzero(mask, as_tuple=False)
    if coords.numel() == 0:
        return feature_map.mean()
    y1 = int(coords[:, 0].min().item())
    x1 = int(coords[:, 1].min().item())
    y2 = int(coords[:, 0].max().item()) + 1
    x2 = int(coords[:, 1].max().item()) + 1
    box_w = x2 - x1
    box_h = y2 - y1
    pad = max(4, int(round(0.20 * max(box_w, box_h))))
    rx1 = max(0, x1 - pad)
    ry1 = max(0, y1 - pad)
    rx2 = min(width, x2 + pad)
    ry2 = min(height, y2 + pad)
    expanded = feature_map[ry1:ry2, rx1:rx2]
    if expanded.numel() == 0:
        return feature_map.mean()
    inner_x1 = x1 - rx1
    inner_y1 = y1 - ry1
    inner_x2 = x2 - rx1
    inner_y2 = y2 - ry1
    ring_mask = torch.ones_like(expanded, dtype=torch.bool)
    ring_mask[inner_y1:inner_y2, inner_x1:inner_x2] = False
    ring = expanded[ring_mask]
    if ring.numel() == 0:
        return expanded.mean()
    return ring.mean()


def crop_mean(feature_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    crop = _mask_crop(feature_map.unsqueeze(0), mask).squeeze(0)
    return crop.mean() if crop.numel() else crop.new_tensor(0.0)


def centroid_consistency(saliency_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    crop = _mask_crop(saliency_map.unsqueeze(0), mask).squeeze(0)
    if crop.numel() == 0:
        return crop.new_tensor(0.0)
    weight = crop.float().clamp_min(0.0)
    total = weight.sum()
    if total <= EPS:
        return crop.new_tensor(0.0)
    crop_h, crop_w = crop.shape
    yy, xx = torch.meshgrid(
        torch.arange(crop_h, device=crop.device, dtype=torch.float32),
        torch.arange(crop_w, device=crop.device, dtype=torch.float32),
        indexing="ij",
    )
    cx = (weight * xx).sum() / total
    cy = (weight * yy).sum() / total
    center_x = (crop_w - 1) / 2.0
    center_y = (crop_h - 1) / 2.0
    norm_x = max(1.0, crop_w / 2.0)
    norm_y = max(1.0, crop_h / 2.0)
    dist = math.sqrt(((cx - center_x) / norm_x) ** 2 + ((cy - center_y) / norm_y) ** 2)
    return torch.tensor(max(0.0, 1.0 - min(1.0, dist / math.sqrt(2.0))), dtype=torch.float32, device=crop.device)


def build_multiscale_edge_maps(gray: torch.Tensor) -> list[tuple[float, torch.Tensor]]:
    height, width = gray.shape
    maps: list[tuple[float, torch.Tensor]] = []
    for scale in (0.5, 1.0, 2.0):
        if scale == 1.0:
            scaled_gray = gray
        else:
            new_h = max(4, int(round(height * scale)))
            new_w = max(4, int(round(width * scale)))
            scaled_gray = F.interpolate(
                gray.reshape(1, 1, height, width),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            ).reshape(new_h, new_w)
        maps.append((float(scale), robust_normalize_map(sobel_map(scaled_gray))))
    return maps


def multiscale_saliency_score(edge_maps: list[tuple[float, torch.Tensor]], mask: torch.Tensor) -> torch.Tensor:
    values = []
    for scale, scaled_edge in edge_maps:
        scaled_mask = _scale_mask(mask, 1.0 / scale)
        values.append(crop_mean(scaled_edge, scaled_mask))
    return torch.stack(values).min()


def _scale_mask(mask: torch.Tensor, inv_scale: float) -> torch.Tensor:
    if abs(inv_scale - 1.0) < 1e-6:
        return mask
    h, w = mask.shape
    new_h = max(4, int(round(h / inv_scale)))
    new_w = max(4, int(round(w / inv_scale)))
    scaled = F.interpolate(mask.float().reshape(1, 1, h, w), size=(new_h, new_w), mode="nearest").reshape(new_h, new_w)
    return scaled > 0.5


def boundary_phase_coherence(image: torch.Tensor, pred_mask: torch.Tensor) -> torch.Tensor:
    gray = image.mean(dim=0)
    phase_edge = robust_normalize_map(phase_only_edge_map(gray))
    phase_boundary, phase_interior = boundary_and_interior_means(phase_edge, pred_mask)
    return phase_boundary / (phase_interior + EPS)


def interior_exterior_texture_contrast(image: torch.Tensor, pred_mask: torch.Tensor) -> torch.Tensor:
    gray = image.mean(dim=0)
    edge = robust_normalize_map(sobel_map(gray))
    inside_edge = crop_mean(edge, pred_mask)
    outside_edge = ring_mean(edge, pred_mask)
    return (inside_edge - outside_edge).abs() / (inside_edge + outside_edge + EPS)


def multi_scale_saliency_consistency(image: torch.Tensor, pred_mask: torch.Tensor) -> torch.Tensor:
    gray = image.mean(dim=0)
    maps = build_multiscale_edge_maps(gray)
    return multiscale_saliency_score(maps, pred_mask)


def score_edge_alignment(image: torch.Tensor, pred_mask: torch.Tensor, confidence: torch.Tensor | None = None) -> torch.Tensor:
    gray = image.mean(dim=0)
    edge = robust_normalize_map(sobel_map(gray))
    edge_boundary, edge_interior = boundary_and_interior_means(edge, pred_mask)
    boundary_ratio = edge_boundary / (edge_interior + EPS)
    cls_prob = 0.5 if confidence is None else float(confidence.item())
    return boundary_ratio * (1.0 - cls_prob)


def activation_centroid_consistency(image: torch.Tensor, pred_mask: torch.Tensor) -> torch.Tensor:
    gray = image.mean(dim=0)
    edge = robust_normalize_map(sobel_map(gray))
    phase_edge = robust_normalize_map(phase_only_edge_map(gray))
    saliency = robust_normalize_map(edge + phase_edge)
    return centroid_consistency(saliency, pred_mask)


def aspect_ratio_plausibility(pred_mask: torch.Tensor) -> torch.Tensor:
    coords = torch.nonzero(pred_mask, as_tuple=False)
    if coords.numel() < 2:
        return torch.tensor(0.0, dtype=torch.float32, device=pred_mask.device)
    ys = coords[:, 0].float()
    xs = coords[:, 1].float()
    height = (ys.max() - ys.min() + 1).clamp_min(1)
    width = (xs.max() - xs.min() + 1).clamp_min(1)
    ratio = width / height
    return torch.exp(-0.5 * (ratio.log() ** 2)).clamp(0.0, 1.0)


def nms_survivor_density(pred_mask: torch.Tensor, neighbor_masks: list[torch.Tensor] | None = None) -> torch.Tensor:
    if neighbor_masks is None:
        return torch.tensor(0.0, dtype=torch.float32, device=pred_mask.device)
    iou_values = []
    for other in neighbor_masks:
        inter = (pred_mask & other).sum().float()
        union = (pred_mask | other).sum().float()
        iou = (inter / union.clamp_min(EPS)) if union > 0 else inter.new_tensor(0.0)
        iou_values.append(iou)
    if not iou_values:
        return torch.tensor(0.0, dtype=torch.float32, device=pred_mask.device)
    support = (torch.stack(iou_values) >= 0.5).sum().float() - 1.0
    return (support / max(1.0, math.sqrt(len(iou_values)))).clamp(0.0, 1.0)
