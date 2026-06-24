from __future__ import annotations

import torch

from .box_iou import box_iou


def match_predictions_to_gt(
    prediction: dict,
    target: dict,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
) -> dict:
    pred_boxes = prediction.get("boxes", torch.empty((0, 4))).detach().cpu()
    pred_labels = prediction.get("labels", torch.empty((0,), dtype=torch.long)).detach().cpu()
    pred_scores = prediction.get("scores", torch.ones((len(pred_boxes),))).detach().cpu()
    gt_boxes = target.get("boxes", torch.empty((0, 4))).detach().cpu()
    gt_labels = target.get("labels", torch.empty((0,), dtype=torch.long)).detach().cpu()

    keep = pred_scores >= score_threshold
    original_indices = torch.arange(len(pred_boxes))[keep]
    pred_boxes = pred_boxes[keep]
    pred_labels = pred_labels[keep]
    pred_scores = pred_scores[keep]
    order = torch.argsort(pred_scores, descending=True)

    ious = box_iou(pred_boxes, gt_boxes)
    used_gt: set[int] = set()
    matches = []
    unmatched_predictions = []

    for pred_pos in order.tolist():
        same_class = gt_labels == pred_labels[pred_pos]
        if same_class.any():
            candidate_ious = ious[pred_pos].clone()
            candidate_ious[~same_class] = -1
            for gt_idx in used_gt:
                candidate_ious[gt_idx] = -1
            best_iou, best_gt = candidate_ious.max(dim=0)
            if best_iou.item() >= iou_threshold:
                used_gt.add(int(best_gt.item()))
                matches.append(
                    {
                        "pred_index": int(original_indices[pred_pos].item()),
                        "local_pred_index": int(pred_pos),
                        "gt_index": int(best_gt.item()),
                        "iou": float(best_iou.item()),
                        "score": float(pred_scores[pred_pos].item()),
                        "label": int(pred_labels[pred_pos].item()),
                    }
                )
                continue
        unmatched_predictions.append(int(original_indices[pred_pos].item()))

    unmatched_gt = [idx for idx in range(len(gt_boxes)) if idx not in used_gt]
    return {"matches": matches, "unmatched_predictions": unmatched_predictions, "unmatched_gt": unmatched_gt}
