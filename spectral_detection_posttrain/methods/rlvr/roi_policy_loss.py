from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F


def resize_boxes_to_image(boxes: torch.Tensor, original_size: tuple[int, int], new_size: tuple[int, int]) -> torch.Tensor:
    ratio_h = float(new_size[0]) / float(original_size[0])
    ratio_w = float(new_size[1]) / float(original_size[1])
    ratios = boxes.new_tensor([ratio_w, ratio_h, ratio_w, ratio_h])
    return boxes * ratios


def weighted_fastrcnn_policy_loss(
    class_logits: torch.Tensor,
    box_regression: torch.Tensor,
    labels: torch.Tensor,
    regression_targets: torch.Tensor,
    weights: torch.Tensor,
    box_loss_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    if labels.numel() == 0:
        zero = class_logits.sum() * 0.0 + box_regression.sum() * 0.0
        return {"loss_roi_policy_cls": zero, "loss_roi_policy_box": zero}

    weights = weights.to(class_logits.device).float().clamp_min(0.0)
    normalizer = weights.sum().clamp_min(1.0)
    cls_loss = F.cross_entropy(class_logits, labels.to(class_logits.device), reduction="none")
    cls_loss = (cls_loss * weights).sum() / normalizer

    pos_inds = torch.where(labels > 0)[0]
    if pos_inds.numel() == 0:
        box_loss = box_regression.sum() * 0.0
    else:
        labels_pos = labels[pos_inds].to(box_regression.device)
        box_regression_4d = box_regression.reshape(box_regression.shape[0], -1, 4)
        target = regression_targets[pos_inds].to(box_regression.device)
        raw_box_loss = F.smooth_l1_loss(
            box_regression_4d[pos_inds, labels_pos],
            target,
            beta=1.0 / 9.0,
            reduction="none",
        ).sum(dim=1)
        box_loss = (raw_box_loss * weights[pos_inds].to(box_regression.device)).sum() / normalizer

    return {"loss_roi_policy_cls": cls_loss, "loss_roi_policy_box": box_loss * float(box_loss_weight)}


def extract_roi_head_outputs_for_boxes(model, images: list[torch.Tensor], boxes: list[torch.Tensor]):
    original_sizes = [tuple(img.shape[-2:]) for img in images]
    transformed, _ = model.transform(images, None)
    features = model.backbone(transformed.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    scaled_boxes = [
        resize_boxes_to_image(b.to(transformed.tensors.device), original, new)
        for b, original, new in zip(boxes, original_sizes, transformed.image_sizes)
    ]
    box_features = model.roi_heads.box_roi_pool(features, scaled_boxes, transformed.image_sizes)
    box_features = model.roi_heads.box_head(box_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)
    return class_logits, box_regression, scaled_boxes, transformed.image_sizes


def signed_roi_policy_loss(
    class_logits: torch.Tensor,
    action_labels: torch.Tensor,
    advantages: torch.Tensor,
    max_abs_advantage: float = 3.0,
) -> torch.Tensor:
    if action_labels.numel() == 0:
        return class_logits.sum() * 0.0
    action_labels = action_labels.to(class_logits.device).long()
    advantages = advantages.to(class_logits.device).float().clamp(
        min=-float(max_abs_advantage),
        max=float(max_abs_advantage),
    )
    log_probs = F.log_softmax(class_logits, dim=1)
    selected = log_probs[torch.arange(action_labels.numel(), device=class_logits.device), action_labels]
    return -(advantages.detach() * selected).mean()


def baseline_kl_loss(current_logits: torch.Tensor, baseline_logits: torch.Tensor) -> torch.Tensor:
    if current_logits.numel() == 0:
        return current_logits.sum() * 0.0
    log_current = F.log_softmax(current_logits, dim=1)
    baseline_prob = F.softmax(baseline_logits.to(current_logits.device), dim=1)
    return F.kl_div(log_current, baseline_prob, reduction="batchmean")


@torch.no_grad()
def roi_logit_max_abs_diff(current_logits: torch.Tensor, baseline_logits: torch.Tensor) -> float:
    if current_logits.numel() == 0:
        return 0.0
    return float((current_logits - baseline_logits.to(current_logits.device)).abs().max().item())
