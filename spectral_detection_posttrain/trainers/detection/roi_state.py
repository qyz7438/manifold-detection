"""Proposal-aligned ROI state construction for energy-guided transport."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F

from spectral_detection_posttrain.core.matching.box_iou import box_iou
from spectral_detection_posttrain.methods.energy_transport import ROIActionState


def build_roi_action_state_from_predictions(
    predictions: list[dict[str, torch.Tensor]],
    roi_features: torch.Tensor,
    targets: list[dict[str, torch.Tensor]] | None = None,
    *,
    logits: torch.Tensor | None = None,
    verifier_scores: torch.Tensor | None = None,
) -> ROIActionState:
    """Build a proposal-aligned :class:`ROIActionState`.

    ``predictions`` should follow the usual detector output convention with
    per-image ``boxes``, ``scores``, and ``labels``. ``roi_features`` is the
    concatenation of the ROI features in the same image/proposal order.
    """
    if roi_features.ndim != 2:
        raise ValueError(f"roi_features must have shape (N, D), got {roi_features.shape}")
    if targets is not None and len(targets) != len(predictions):
        raise ValueError("targets must have the same length as predictions")

    device = roi_features.device
    dtype = roi_features.dtype
    boxes_list: list[torch.Tensor] = []
    scores_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    image_indices_list: list[torch.Tensor] = []
    proposal_indices_list: list[torch.Tensor] = []
    matched_gt_list: list[torch.Tensor] = []
    iou_list: list[torch.Tensor] = []

    for image_idx, prediction in enumerate(predictions):
        boxes = prediction.get("boxes")
        scores = prediction.get("scores")
        labels = prediction.get("labels")
        if boxes is None or scores is None or labels is None:
            raise ValueError("each prediction must contain boxes, scores, and labels")
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(f"prediction boxes must have shape (N, 4), got {boxes.shape}")
        num_props = boxes.shape[0]
        if scores.shape != (num_props,):
            raise ValueError("prediction scores must have shape (N,)")
        if labels.shape != (num_props,):
            raise ValueError("prediction labels must have shape (N,)")

        boxes = boxes.to(device=device, dtype=dtype)
        scores = scores.to(device=device, dtype=dtype)
        labels = labels.to(device=device, dtype=torch.long)
        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)
        image_indices_list.append(
            torch.full((num_props,), image_idx, dtype=torch.long, device=device)
        )
        proposal_indices_list.append(
            torch.arange(num_props, dtype=torch.long, device=device)
        )

        if targets is not None:
            matched_gt, matched_iou = _best_class_aware_matches(
                boxes, labels, targets[image_idx], dtype=dtype, device=device
            )
            matched_gt_list.append(matched_gt)
            iou_list.append(matched_iou)

    boxes_cat = _cat_or_empty(boxes_list, (0, 4), dtype=dtype, device=device)
    total = boxes_cat.shape[0]
    if roi_features.shape[0] != total:
        raise ValueError(
            f"roi_features has {roi_features.shape[0]} rows but predictions contain {total} boxes"
        )

    scores_cat = _cat_or_empty(scores_list, (0,), dtype=dtype, device=device)
    labels_cat = _cat_or_empty(labels_list, (0,), dtype=torch.long, device=device)
    image_indices = _cat_or_empty(
        image_indices_list, (0,), dtype=torch.long, device=device
    )
    proposal_indices = _cat_or_empty(
        proposal_indices_list, (0,), dtype=torch.long, device=device
    )

    matched_gt_indices = None
    ious = None
    if targets is not None:
        matched_gt_indices = _cat_or_empty(
            matched_gt_list, (0,), dtype=torch.long, device=device
        )
        ious = _cat_or_empty(iou_list, (0,), dtype=dtype, device=device)

    return ROIActionState(
        features=roi_features,
        boxes=boxes_cat,
        scores=scores_cat,
        labels=labels_cat,
        image_indices=image_indices,
        proposal_indices=proposal_indices,
        logits=logits,
        matched_gt_indices=matched_gt_indices,
        ious=ious,
        verifier_scores=verifier_scores,
    )


def extract_proposal_roi_action_state(
    model: torch.nn.Module,
    images: list[torch.Tensor],
    targets: list[dict[str, torch.Tensor]] | None = None,
) -> ROIActionState:
    """Extract RPN proposal ROI features and classifier-derived scores.

    This function deliberately returns proposal boxes, not decoded final boxes.
    Box-delta actions should be applied later through
    :func:`energy_transport.apply_box_delta` so action outcomes remain explicit.
    """
    transformed, transformed_targets = model.transform(images, targets)
    features = model.backbone(transformed.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])

    proposals, _ = model.rpn(transformed, features, transformed_targets)
    roi_features = model.roi_heads.box_roi_pool(
        features, proposals, transformed.image_sizes
    )
    roi_features = model.roi_heads.box_head(roi_features)
    logits, _ = model.roi_heads.box_predictor(roi_features)
    scores, labels = _foreground_scores_and_labels(logits)

    predictions: list[dict[str, torch.Tensor]] = []
    offset = 0
    for proposal in proposals:
        count = int(proposal.shape[0])
        predictions.append(
            {
                "boxes": proposal,
                "scores": scores[offset : offset + count],
                "labels": labels[offset : offset + count],
            }
        )
        offset += count

    return build_roi_action_state_from_predictions(
        predictions,
        roi_features,
        transformed_targets,
        logits=logits,
    )


def _foreground_scores_and_labels(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape (N, C), got {logits.shape}")
    probs = F.softmax(logits, dim=-1)
    if probs.shape[1] <= 1:
        scores, labels = probs.max(dim=1)
        return scores, labels.long()
    fg_scores, fg_labels = probs[:, 1:].max(dim=1)
    return fg_scores, (fg_labels + 1).long()


def _best_class_aware_matches(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    target: dict[str, torch.Tensor],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_props = boxes.shape[0]
    matched_gt = torch.full((num_props,), -1, dtype=torch.long, device=device)
    matched_iou = torch.zeros((num_props,), dtype=dtype, device=device)
    if num_props == 0:
        return matched_gt, matched_iou

    gt_boxes = target.get("boxes", torch.empty(0, 4)).to(device=device, dtype=dtype)
    gt_labels = target.get("labels", torch.empty(0, dtype=torch.long)).to(
        device=device, dtype=torch.long
    )
    if gt_boxes.numel() == 0:
        return matched_gt, matched_iou

    ious = box_iou(boxes, gt_boxes)
    same_class = labels[:, None] == gt_labels[None, :]
    masked_ious = torch.where(same_class, ious, torch.full_like(ious, -1.0))
    best_iou, best_gt = masked_ious.max(dim=1)
    valid = best_iou > 0.0
    matched_gt[valid] = best_gt[valid]
    matched_iou[valid] = best_iou[valid]
    return matched_gt, matched_iou


def _cat_or_empty(
    tensors: list[torch.Tensor],
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if tensors:
        return torch.cat(tensors, dim=0)
    return torch.empty(shape, dtype=dtype, device=device)
