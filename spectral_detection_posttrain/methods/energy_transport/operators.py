"""Operators that apply bounded detector transport actions."""

from __future__ import annotations

import torch


def clip_boxes_to_image(boxes: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    """Clamp ``xyxy`` boxes to ``(height, width)`` and repair corner order."""
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes must have shape (B, 4), got {boxes.shape}")
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError("image_size must contain positive height and width")

    clipped = boxes.clone()
    clipped[:, 0::2] = clipped[:, 0::2].clamp(0.0, float(width))
    clipped[:, 1::2] = clipped[:, 1::2].clamp(0.0, float(height))
    x1 = torch.minimum(clipped[:, 0], clipped[:, 2])
    y1 = torch.minimum(clipped[:, 1], clipped[:, 3])
    x2 = torch.maximum(clipped[:, 0], clipped[:, 2])
    y2 = torch.maximum(clipped[:, 1], clipped[:, 3])
    return torch.stack((x1, y1, x2, y2), dim=-1)


def apply_box_delta(
    boxes: torch.Tensor,
    deltas: torch.Tensor,
    *,
    image_size: tuple[int, int] | None = None,
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Decode standard center-size box deltas and optionally clip to image."""
    if boxes.shape != deltas.shape or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes and deltas must share shape (B, 4)")
    wx, wy, ww, wh = weights
    if min(weights) <= 0.0:
        raise ValueError("weights must be positive")

    widths = (boxes[:, 2] - boxes[:, 0]).clamp_min(1e-6)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(1e-6)
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0] / wx
    dy = deltas[:, 1] / wy
    dw = (deltas[:, 2] / ww).clamp(max=4.135)
    dh = (deltas[:, 3] / wh).clamp(max=4.135)

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    pred_boxes = torch.stack(
        (
            pred_ctr_x - 0.5 * pred_w,
            pred_ctr_y - 0.5 * pred_h,
            pred_ctr_x + 0.5 * pred_w,
            pred_ctr_y + 0.5 * pred_h,
        ),
        dim=-1,
    )
    if image_size is not None:
        pred_boxes = clip_boxes_to_image(pred_boxes, image_size)
    return pred_boxes
