"""Light-weight evaluation helpers for remote-sensing detection tasks.

These utilities operate on axis-aligned bounding boxes (the common case for
NWPU VHR-10 and VisDrone) and provide average-precision computation as well
as small/medium/large split analysis.  They are intentionally dependency-free
beyond PyTorch so that they can be used inside training loops.
"""

from __future__ import annotations

import torch


def _box_area(boxes: torch.Tensor) -> torch.Tensor:
    """Compute the area of axis-aligned boxes in ``(x1, y1, x2, y2)`` format."""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two sets of axis-aligned boxes.

    Args:
        boxes1: tensor of shape ``(N, 4)``.
        boxes2: tensor of shape ``(M, 4)``.

    Returns:
        IoU matrix of shape ``(N, M)``.
    """
    area1 = _box_area(boxes1)
    area2 = _box_area(boxes2)

    inter_top_left = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    inter_bottom_right = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter_wh = (inter_bottom_right - inter_top_left).clamp(min=0.0)
    inter = inter_wh[:, :, 0] * inter_wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def compute_ap(recall: torch.Tensor, precision: torch.Tensor) -> float:
    """Compute average precision via 11-point interpolation.

    Args:
        recall: 1-D tensor of recall values in ``[0, 1]``.
        precision: 1-D tensor of precision values in ``[0, 1]``.

    Returns:
        Average precision scalar.
    """
    if recall.numel() == 0 or precision.numel() == 0:
        return 0.0

    recall, order = torch.sort(recall)
    precision = precision[order]

    ap = 0.0
    for t in torch.linspace(0.0, 1.0, 11):
        mask = recall >= t
        if mask.any():
            ap += precision[mask].max().item()
    return ap / 11.0


def _match_predictions(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float,
) -> tuple[int, int, int]:
    """Match a sorted list of predictions to ground-truth boxes.

    Args:
        pred_boxes: predicted boxes of shape ``(N, 4)``.
        pred_scores: confidence scores of shape ``(N,)``.
        gt_boxes: ground-truth boxes of shape ``(M, 4)``.
        iou_threshold: minimum IoU for a true positive.

    Returns:
        ``(true_positives, false_positives, num_gt)``.
    """
    if pred_boxes.numel() == 0:
        return 0, 0, gt_boxes.shape[0]
    if gt_boxes.numel() == 0:
        return 0, pred_boxes.shape[0], 0

    order = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[order]

    ious = _box_iou(pred_boxes, gt_boxes)
    gt_matched = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)

    tp = 0
    fp = 0
    for i in range(pred_boxes.shape[0]):
        max_iou, gt_idx = ious[i].max(dim=0)
        if max_iou >= iou_threshold and not gt_matched[gt_idx]:
            tp += 1
            gt_matched[gt_idx] = True
        else:
            fp += 1

    return tp, fp, gt_boxes.shape[0]


def evaluate_remote_sensing_ap(
    predictions: dict[str, list[dict[str, torch.Tensor]]],
    ground_truths: dict[str, list[dict[str, torch.Tensor]]],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute average precision for a single IoU threshold.

    Both ``predictions`` and ``ground_truths`` are dictionaries keyed by image
    identifier.  Each value is a list of detection dictionaries with keys
    ``"bbox"`` (tensor ``(4,)`` in ``(x1, y1, x2, y2)`` format) and
    ``"score"`` (scalar tensor).

    Args:
        predictions: predicted detections per image.
        ground_truths: ground-truth detections per image.
        iou_threshold: IoU threshold used to count true positives.

    Returns:
        Dictionary containing ``AP``, ``precision``, ``recall`` and
        ``F1`` at the chosen threshold.
    """
    all_scores: list[torch.Tensor] = []
    all_tps: list[torch.Tensor] = []
    all_fps: list[torch.Tensor] = []
    total_gt = 0

    for image_id in ground_truths:
        gt_boxes = torch.stack(
            [ann["bbox"] for ann in ground_truths[image_id]]
        ) if ground_truths[image_id] else torch.zeros((0, 4))
        preds = predictions.get(image_id, [])
        pred_boxes = (
            torch.stack([p["bbox"] for p in preds])
            if preds
            else torch.zeros((0, 4))
        )
        pred_scores = (
            torch.stack([p["score"] for p in preds])
            if preds
            else torch.zeros((0,))
        )

        tp, fp, num_gt = _match_predictions(
            pred_boxes, pred_scores, gt_boxes, iou_threshold
        )
        total_gt += num_gt

        if preds:
            all_scores.append(pred_scores)
            all_tps.append(torch.full_like(pred_scores, tp, dtype=torch.float32))
            all_fps.append(torch.full_like(pred_scores, fp, dtype=torch.float32))

    if not all_scores:
        return {"AP": 0.0, "precision": 0.0, "recall": 0.0, "F1": 0.0}

    scores = torch.cat(all_scores)
    tps = torch.cat(all_tps)
    fps = torch.cat(all_fps)

    order = torch.argsort(scores, descending=True)
    tps = tps[order]
    fps = fps[order]

    tp_cumsum = tps.cumsum(dim=0)
    fp_cumsum = fps.cumsum(dim=0)

    eps = 1e-8
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + eps)
    recall = tp_cumsum / (total_gt + eps)

    ap = compute_ap(recall, precision)

    # Micro-averaged precision/recall/F1 across the whole dataset.
    total_tp = int(tp_cumsum[-1].item())
    total_fp = int(fp_cumsum[-1].item())
    micro_precision = total_tp / (total_tp + total_fp + eps)
    micro_recall = total_tp / (total_gt + eps)
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall + eps)
    )

    return {
        "AP": ap,
        "precision": micro_precision,
        "recall": micro_recall,
        "F1": micro_f1,
    }
