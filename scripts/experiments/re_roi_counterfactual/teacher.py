"""Teacher structured result and utility for the re-ROI protocol."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch
from torchvision.ops import box_iou


@dataclass(frozen=True)
class TeacherResult:
    """Structured native-output change for one action."""

    native_changed: bool
    delta_tp75: int
    delta_fp75: int
    delta_fp50: int
    delta_duplicate: int
    delta_score_margin: float
    delta_localization_quality: float
    action_energy: float

    def to_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


# Locked scalarization from the preregistered protocol.
Q_TEACHER_WEIGHTS = {
    "delta_tp75": 1.00,
    "delta_fp75": -0.25,
    "delta_fp50": -0.10,
    "delta_duplicate": -0.50,
    "delta_score_margin": 0.10,
    "delta_localization_quality": 0.10,
    "action_energy": -1.00,
}


def _match_predictions_to_gt(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score-order predictions and greedily assign each GT at most once."""
    num_preds = boxes.shape[0]
    if scores.shape[0] != num_preds or labels.shape[0] != num_preds:
        raise ValueError("prediction boxes, scores, and labels must align")
    matched_gt = torch.full((num_preds,), -1, dtype=torch.long, device=boxes.device)
    matched_iou = torch.zeros((num_preds,), dtype=boxes.dtype, device=boxes.device)
    if num_preds == 0 or gt_boxes.numel() == 0:
        return matched_gt, matched_iou
    ious = box_iou(boxes, gt_boxes)
    used_gt = torch.zeros(gt_boxes.shape[0], dtype=torch.bool, device=boxes.device)
    order = torch.argsort(scores, descending=True, stable=True)
    for prediction_index in order.tolist():
        eligible = (labels[prediction_index] == gt_labels) & ~used_gt
        candidate_ious = torch.where(
            eligible,
            ious[prediction_index],
            torch.full_like(ious[prediction_index], -1.0),
        )
        best_iou, best_gt = candidate_ious.max(dim=0)
        if float(best_iou.item()) >= iou_threshold:
            used_gt[best_gt] = True
            matched_gt[prediction_index] = best_gt
            matched_iou[prediction_index] = best_iou
    return matched_gt, matched_iou


def _count_tp_fp(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float,
    score_threshold: float,
) -> tuple[int, int]:
    """Count TP and FP at one IoU threshold after applying score threshold."""
    if boxes.numel() == 0:
        return 0, 0
    keep = scores >= score_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    if boxes.numel() == 0:
        return 0, 0
    matched_gt, _ = _match_predictions_to_gt(
        boxes, scores, labels, gt_boxes, gt_labels, iou_threshold
    )
    tp = int((matched_gt >= 0).sum().item())
    fp = int(boxes.shape[0]) - tp
    return tp, fp


def _count_duplicates(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.50,
) -> int:
    """Count same-class overlapping pairs above IoU threshold."""
    if boxes.shape[0] < 2:
        return 0
    overlaps = box_iou(boxes, boxes)
    upper = torch.triu(torch.ones_like(overlaps, dtype=torch.bool), diagonal=1)
    same_class = labels[:, None] == labels[None, :]
    duplicates = upper & same_class & overlaps.ge(iou_threshold)
    return int(duplicates.sum().item())


def _localization_quality(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
) -> float:
    """Mean matched IoU for predictions above score 0.05, or 0 if none."""
    if boxes.numel() == 0 or gt_boxes.numel() == 0:
        return 0.0
    keep = scores >= 0.05
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    if boxes.numel() == 0:
        return 0.0
    matched_gt, matched_iou = _match_predictions_to_gt(
        boxes, scores, labels, gt_boxes, gt_labels, 0.50
    )
    valid = matched_iou > 0.0
    if not valid.any():
        return 0.0
    return float(matched_iou[valid].mean().item())


def _score_margin(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
) -> float:
    """Mean score margin for matched predictions above IoU 0.5."""
    if boxes.numel() == 0 or gt_boxes.numel() == 0:
        return 0.0
    matched_gt, matched_iou = _match_predictions_to_gt(
        boxes, scores, labels, gt_boxes, gt_labels, 0.50
    )
    valid = matched_iou > 0.0
    if not valid.any():
        return 0.0
    return float(scores[valid].mean().item())


def compute_teacher_result(
    baseline: dict[str, torch.Tensor],
    counterfactual: dict[str, torch.Tensor],
    gt: dict[str, torch.Tensor],
    action_energy: float,
) -> TeacherResult:
    """Compute Y_teacher for a single action on one image."""
    base_boxes = baseline["boxes"]
    base_scores = baseline["scores"]
    base_labels = baseline["labels"]
    cf_boxes = counterfactual["boxes"]
    cf_scores = counterfactual["scores"]
    cf_labels = counterfactual["labels"]
    gt_boxes = gt["boxes"]
    gt_labels = gt["labels"]

    base_tp75, base_fp75 = _count_tp_fp(base_boxes, base_scores, base_labels, gt_boxes, gt_labels, 0.75, 0.05)
    cf_tp75, cf_fp75 = _count_tp_fp(cf_boxes, cf_scores, cf_labels, gt_boxes, gt_labels, 0.75, 0.05)
    base_tp50, base_fp50 = _count_tp_fp(base_boxes, base_scores, base_labels, gt_boxes, gt_labels, 0.50, 0.05)
    cf_tp50, cf_fp50 = _count_tp_fp(cf_boxes, cf_scores, cf_labels, gt_boxes, gt_labels, 0.50, 0.05)

    base_dup = _count_duplicates(base_boxes, base_scores, base_labels)
    cf_dup = _count_duplicates(cf_boxes, cf_scores, cf_labels)

    base_loc = _localization_quality(base_boxes, base_scores, base_labels, gt_boxes, gt_labels)
    cf_loc = _localization_quality(cf_boxes, cf_scores, cf_labels, gt_boxes, gt_labels)

    base_margin = _score_margin(base_boxes, base_scores, base_labels, gt_boxes, gt_labels)
    cf_margin = _score_margin(cf_boxes, cf_scores, cf_labels, gt_boxes, gt_labels)

    native_changed = not (
        base_boxes.shape == cf_boxes.shape
        and torch.allclose(base_boxes, cf_boxes, atol=1e-5)
        and torch.allclose(base_scores, cf_scores, atol=1e-5)
        and base_labels.shape == cf_labels.shape
        and torch.equal(base_labels, cf_labels)
    )

    return TeacherResult(
        native_changed=native_changed,
        delta_tp75=cf_tp75 - base_tp75,
        delta_fp75=cf_fp75 - base_fp75,
        delta_fp50=cf_fp50 - base_fp50,
        delta_duplicate=cf_dup - base_dup,
        delta_score_margin=cf_margin - base_margin,
        delta_localization_quality=cf_loc - base_loc,
        action_energy=action_energy,
    )


def compute_q_teacher(result: TeacherResult) -> float:
    """Apply the locked scalarization g(Y_teacher)."""
    values = result.to_dict()
    total = 0.0
    for key, weight in Q_TEACHER_WEIGHTS.items():
        total += weight * float(values[key])
    return total


def standardize_q_teacher(
    values: torch.Tensor,
    stats: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Robust-standardize Q_teacher using inner-fit median and IQR."""
    median = stats["median"]
    iqr = stats["iqr"].clamp_min(1e-8)
    return (values - median) / iqr


def fit_q_teacher_stats(values: torch.Tensor) -> dict[str, torch.Tensor]:
    """Fit robust statistics on inner-fit Q_teacher values."""
    if values.numel() == 0:
        raise ValueError("cannot fit Q_teacher stats on empty values")
    quantiles = torch.quantile(
        values.float(),
        torch.tensor([0.25, 0.5, 0.75], dtype=values.dtype),
        dim=0,
    )
    return {"median": quantiles[1], "iqr": quantiles[2] - quantiles[0]}
