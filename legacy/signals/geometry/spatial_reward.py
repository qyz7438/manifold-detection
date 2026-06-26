from __future__ import annotations

import torch

from spectral_detection_posttrain.core.matching.box_iou import box_iou


def iou_reward(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return torch.zeros((len(pred_boxes),), dtype=torch.float32, device=pred_boxes.device)
    return box_iou(pred_boxes, gt_boxes).max(dim=1).values.clamp(0.0, 1.0)


def center_size_reward(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return torch.zeros((len(pred_boxes),), dtype=torch.float32, device=pred_boxes.device)
    best = box_iou(pred_boxes, gt_boxes).argmax(dim=1)
    matched = gt_boxes[best]
    pred_ctr = torch.stack(
        [
            (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2,
            (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2,
        ],
        dim=1,
    )
    gt_ctr = torch.stack(
        [
            (matched[:, 0] + matched[:, 2]) / 2,
            (matched[:, 1] + matched[:, 3]) / 2,
        ],
        dim=1,
    )
    gt_size = torch.stack(
        [
            (matched[:, 2] - matched[:, 0]).clamp_min(1),
            (matched[:, 3] - matched[:, 1]).clamp_min(1),
        ],
        dim=1,
    )
    error = ((pred_ctr - gt_ctr).abs() / gt_size).mean(dim=1)
    return (1.0 - error).clamp(0.0, 1.0)
