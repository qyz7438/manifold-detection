"""Build the re-ROI counterfactual evidence cache for one image."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch
from torchvision.ops import boxes as box_ops

from scripts.experiments.re_roi_counterfactual.action_family import ActionSpec
from scripts.experiments.re_roi_counterfactual.teacher import compute_q_teacher, compute_teacher_result


@dataclass(frozen=True)
class CacheConfig:
    """Configuration for re-ROI cache generation."""

    score_threshold: float = 0.05
    nms_threshold: float = 0.50
    detections_per_img: int = 100
    top_k_candidates: int = 3
    min_box_size: float = 1e-2


def _native_postprocess(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    image_shape: tuple[int, int],
    config: CacheConfig,
) -> dict[str, torch.Tensor]:
    """Apply torchvision-equivalent postprocessing to class-expanded predictions."""
    boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
    keep = torch.where(scores > float(config.score_threshold))[0]
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    keep = box_ops.remove_small_boxes(boxes, min_size=float(config.min_box_size))
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    keep = box_ops.batched_nms(
        boxes,
        scores,
        labels,
        float(config.nms_threshold),
    )[: int(config.detections_per_img)]
    return {
        "boxes": boxes[keep],
        "scores": scores[keep],
        "labels": labels[keep],
    }


def _decode_and_expand_predictions(
    class_logits: torch.Tensor,
    box_regression: torch.Tensor,
    proposals: list[torch.Tensor],
    image_sizes: list[tuple[int, int]],
    box_coder,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Decode per-proposal class-specific boxes and expand for class-aware NMS."""
    boxes_per_image = [proposal.shape[0] for proposal in proposals]
    pred_boxes = box_coder.decode(box_regression, proposals)
    pred_scores = torch.softmax(class_logits, dim=-1)
    num_classes = class_logits.shape[-1]

    all_boxes: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    boxes_split = pred_boxes.split(boxes_per_image, 0)
    scores_split = pred_scores.split(boxes_per_image, 0)

    for boxes, scores, image_shape in zip(boxes_split, scores_split, image_sizes):
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
        labels_per_image = torch.arange(num_classes, device=boxes.device)
        labels_per_image = labels_per_image.view(1, -1).expand_as(scores)

        # Drop background class 0.
        boxes = boxes[:, 1:].reshape(-1, 4)
        scores = scores[:, 1:].reshape(-1)
        labels_per_image = labels_per_image[:, 1:].reshape(-1)

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels_per_image)

    return all_boxes, all_scores, all_labels


def _extract_native_detections(
    model: torch.nn.Module,
    images: list[torch.Tensor],
    config: CacheConfig,
) -> tuple[
    list[dict[str, torch.Tensor]],
    OrderedDict[str, torch.Tensor],
    list[torch.Tensor],
    list[tuple[int, int]],
]:
    """Run detector forward and return native detections plus cached FPN features.

    Returns:
        predictions: per-image dicts with boxes, scores, labels after native postprocess.
        fpn_features: cached FPN features.
        proposals: detector proposals per image.
    """
    model.eval()
    transformed, _ = model.transform(images, None)
    fpn_features = model.backbone(transformed.tensors)
    if isinstance(fpn_features, torch.Tensor):
        fpn_features = OrderedDict([("0", fpn_features)])

    proposals, _ = model.rpn(transformed, fpn_features, None)
    roi_features = model.roi_heads.box_roi_pool(fpn_features, proposals, transformed.image_sizes)
    roi_features = model.roi_heads.box_head(roi_features)
    class_logits, box_regression = model.roi_heads.box_predictor(roi_features)

    all_boxes, all_scores, all_labels = _decode_and_expand_predictions(
        class_logits,
        box_regression,
        proposals,
        transformed.image_sizes,
        model.roi_heads.box_coder,
    )

    predictions: list[dict[str, torch.Tensor]] = []
    for boxes, scores, labels, image_size in zip(all_boxes, all_scores, all_labels, transformed.image_sizes):
        predictions.append(_native_postprocess(boxes, scores, labels, image_size, config))

    return predictions, fpn_features, proposals, list(transformed.image_sizes)


def _resize_target_to_image_size(
    target: dict[str, torch.Tensor],
    source_size: tuple[int, int],
    destination_size: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """Copy a target and scale its boxes into detector-transform coordinates."""
    source_height, source_width = (int(value) for value in source_size)
    destination_height, destination_width = (int(value) for value in destination_size)
    if min(source_height, source_width, destination_height, destination_width) <= 0:
        raise ValueError("source and destination image sizes must be positive")
    if "boxes" not in target or not torch.is_tensor(target["boxes"]):
        raise ValueError("target must contain tensor boxes")

    resized = dict(target)
    boxes = target["boxes"].clone()
    boxes[:, 0::2] *= destination_width / source_width
    boxes[:, 1::2] *= destination_height / source_height
    resized["boxes"] = boxes
    return resized


def _roi_features_for_boxes(
    model: torch.nn.Module,
    fpn_features: OrderedDict[str, torch.Tensor],
    boxes: torch.Tensor,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Run ROIAlign + box_head on a set of boxes against cached FPN features."""
    features = model.roi_heads.box_roi_pool(fpn_features, [boxes], [image_size])
    features = model.roi_heads.box_head(features)
    return features


def _apply_action_to_detection(
    detection: dict[str, torch.Tensor],
    candidate_index: int,
    action: ActionSpec,
) -> dict[str, torch.Tensor]:
    """Apply one action to a single candidate, return the full counterfactual set."""
    boxes = detection["boxes"].clone()
    scores = detection["scores"].clone()
    labels = detection["labels"].clone()

    if action.family == "drop":
        keep = torch.ones(boxes.shape[0], dtype=torch.bool, device=boxes.device)
        keep[candidate_index] = False
        return {
            "boxes": boxes[keep],
            "scores": scores[keep],
            "labels": labels[keep],
        }

    new_boxes, new_scores = action.apply(
        boxes[candidate_index : candidate_index + 1],
        scores[candidate_index : candidate_index + 1],
        labels[candidate_index : candidate_index + 1],
    )
    boxes[candidate_index] = new_boxes[0]
    scores[candidate_index] = new_scores[0]
    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
    }


def build_image_cache_record(
    model: torch.nn.Module,
    image: torch.Tensor,
    target: dict[str, torch.Tensor],
    image_id: int,
    actions: tuple[ActionSpec, ...],
    config: CacheConfig | None = None,
) -> dict[str, Any]:
    """Build the full re-ROI cache record for one image.

    Returns a dict with native detections, pre-action ROI features, and one
    counterfactual record per (candidate, action) pair.
    """
    config = config or CacheConfig()
    predictions, fpn_features, proposals, transformed_sizes = _extract_native_detections(
        model, [image], config
    )
    prediction = predictions[0]
    proposal = proposals[0]
    source_image_size = tuple(int(value) for value in image.shape[-2:])
    image_size = tuple(int(value) for value in transformed_sizes[0])
    transformed_target = _resize_target_to_image_size(
        target, source_image_size, image_size
    )

    top_k = min(config.top_k_candidates, prediction["boxes"].shape[0])
    acted_indices = list(range(top_k))

    # Pre-action ROI features are extracted on the native detection boxes, not
    # on the proposal set, so that h_pre is aligned with the acted candidate.
    h_pre_all = (
        _roi_features_for_boxes(model, fpn_features, prediction["boxes"], image_size)
        .detach()
        .cpu()
        .half()
    )

    action_records: list[dict[str, Any]] = []
    for candidate_index in acted_indices:
        h_pre = h_pre_all[candidate_index]
        for action in actions:
            counterfactual = _apply_action_to_detection(prediction, candidate_index, action)
            cf_post = _native_postprocess(
                counterfactual["boxes"],
                counterfactual["scores"],
                counterfactual["labels"],
                image_size,
                config,
            )
            if action.family == "drop":
                h_post = torch.zeros_like(h_pre)
                h_post_present = False
            else:
                # Re-extract ROI evidence for the acted box before set-level
                # postprocessing can reorder detections.
                acted_box = counterfactual["boxes"][candidate_index : candidate_index + 1]
                h_post = (
                    _roi_features_for_boxes(model, fpn_features, acted_box, image_size)[0]
                    .detach()
                    .cpu()
                    .half()
                )
                h_post_present = True

            teacher = compute_teacher_result(
                baseline=prediction,
                counterfactual=cf_post,
                gt=transformed_target,
                action_energy=action.energy,
            )
            action_records.append(
                {
                    "candidate_index": candidate_index,
                    "family": action.family,
                    "h_pre": h_pre,
                    "h_post": h_post,
                    "h_post_present": h_post_present,
                    "teacher": teacher.to_dict(),
                    "q_teacher": compute_q_teacher(teacher),
                }
            )

    return {
        "image_id": image_id,
        "image_size": image_size,
        "source_image_size": source_image_size,
        "baseline": {
            "boxes": prediction["boxes"].detach().cpu(),
            "scores": prediction["scores"].detach().cpu(),
            "labels": prediction["labels"].detach().cpu(),
            "proposal_boxes": proposal.detach().cpu(),
            "roi_features": h_pre_all,
        },
        "fpn_keys": list(fpn_features.keys()),
        "actions": action_records,
    }
