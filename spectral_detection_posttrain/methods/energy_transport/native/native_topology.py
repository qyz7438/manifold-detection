"""Detector-only topology features from torchvision-equivalent ROI postprocessing."""

from __future__ import annotations

import torch
from torchvision.ops import batched_nms, box_iou, clip_boxes_to_image, remove_small_boxes

from spectral_detection_posttrain.methods.energy_transport.operators import apply_box_delta


NATIVE_TOPOLOGY_FEATURE_NAMES = (
    "acted_candidate_kept",
    "identity_kept",
    "kept_delta",
    "max_iou_higher_score_same_class",
    "nms_margin",
    "max_iou_delta",
    "normalized_kept_rank",
    "kept_rank_delta",
)


def native_action_nms_topology(
    decoded_boxes: torch.Tensor,
    class_probabilities: torch.Tensor,
    action_labels: torch.Tensor,
    image_size: tuple[int, int] | torch.Tensor,
    candidate_deltas: torch.Tensor,
    observable_mask: torch.Tensor,
    *,
    score_threshold: float,
    nms_threshold: float,
    detections_per_img: int,
) -> torch.Tensor:
    """Return class-expanded NMS topology for each observable proposal/action.

    Each candidate changes only ``decoded_boxes[proposal, action_label]``;
    all other detector outputs remain fixed.  Postprocessing follows the
    torchvision ROI-head order exactly: clip, discard background, threshold,
    remove small boxes, class-aware NMS, then the per-image top-k cap.
    """
    _validate_inputs(
        decoded_boxes,
        class_probabilities,
        action_labels,
        image_size,
        candidate_deltas,
        observable_mask,
        score_threshold,
        nms_threshold,
        detections_per_img,
    )
    count = decoded_boxes.shape[0]
    candidates = candidate_deltas.shape[0]
    topology = decoded_boxes.new_zeros((count, candidates, len(NATIVE_TOPOLOGY_FEATURE_NAMES)))
    if count == 0 or not observable_mask.any():
        return topology

    labels = action_labels.to(device=decoded_boxes.device, dtype=torch.long)
    baseline = _postprocess_class_expanded(
        decoded_boxes,
        class_probabilities,
        image_size,
        score_threshold,
        nms_threshold,
        detections_per_img,
    )
    baseline_features = _target_features(
        baseline,
        decoded_boxes,
        class_probabilities,
        labels,
        image_size,
        nms_threshold,
    )

    for proposal in torch.nonzero(observable_mask, as_tuple=False).flatten().tolist():
        label = int(labels[proposal].item())
        for candidate in range(candidates):
            if candidate == 0:
                topology[proposal, candidate] = _compose_features(
                    baseline_features,
                    baseline_features,
                    proposal,
                    nms_threshold,
                )
                continue
            moved = decoded_boxes.clone()
            moved[proposal, label] = apply_box_delta(
                moved[proposal, label].unsqueeze(0),
                candidate_deltas[candidate].to(dtype=moved.dtype).unsqueeze(0),
            ).squeeze(0)
            current = _postprocess_class_expanded(
                moved,
                class_probabilities,
                image_size,
                score_threshold,
                nms_threshold,
                detections_per_img,
            )
            current_features = _target_features(
                current,
                moved,
                class_probabilities,
                labels,
                image_size,
                nms_threshold,
                proposal_indices=torch.tensor([proposal], device=decoded_boxes.device),
            )
            topology[proposal, candidate] = _compose_features(
                current_features,
                baseline_features,
                proposal,
                nms_threshold,
            )
    return topology


def _validate_inputs(
    decoded_boxes: torch.Tensor,
    class_probabilities: torch.Tensor,
    action_labels: torch.Tensor,
    image_size: tuple[int, int] | torch.Tensor,
    candidate_deltas: torch.Tensor,
    observable_mask: torch.Tensor,
    score_threshold: float,
    nms_threshold: float,
    detections_per_img: int,
) -> None:
    if decoded_boxes.ndim != 3 or decoded_boxes.shape[-1] != 4:
        raise ValueError("decoded_boxes must have shape (N, C, 4)")
    count, classes, _ = decoded_boxes.shape
    if class_probabilities.shape != (count, classes):
        raise ValueError("class_probabilities must have shape (N, C)")
    if classes < 2:
        raise ValueError("class probabilities must include background and foreground classes")
    if action_labels.shape != (count,):
        raise ValueError("action_labels must have shape (N,)")
    if observable_mask.shape != (count,):
        raise ValueError("observable_mask must have shape (N,)")
    if candidate_deltas.ndim != 2 or candidate_deltas.shape[1] != 4 or candidate_deltas.shape[0] == 0:
        raise ValueError("candidate_deltas must have shape (K, 4) with K > 0")
    if not torch.equal(candidate_deltas[0], torch.zeros_like(candidate_deltas[0])):
        raise ValueError("candidate_deltas[0] must be the exact zero identity action")
    if not 0.0 <= float(score_threshold) <= 1.0:
        raise ValueError("score_threshold must be in [0, 1]")
    if not 0.0 <= float(nms_threshold) <= 1.0:
        raise ValueError("nms_threshold must be in [0, 1]")
    if int(detections_per_img) <= 0:
        raise ValueError("detections_per_img must be positive")
    if not action_labels.dtype.is_floating_point:
        labels = action_labels
    else:
        if not torch.equal(action_labels, action_labels.round()):
            raise ValueError("action_labels must contain integer class indices")
        labels = action_labels.long()
    if (labels <= 0).any() or (labels >= classes).any():
        raise ValueError("action_labels must be foreground class indices")
    size = torch.as_tensor(image_size).flatten()
    if size.numel() != 2 or (size <= 0).any():
        raise ValueError("image_size must contain positive height and width")
    devices = {tensor.device for tensor in (decoded_boxes, class_probabilities, action_labels, candidate_deltas, observable_mask)}
    if len(devices) != 1:
        raise ValueError("all tensor inputs must be on the same device")


def _postprocess_class_expanded(
    decoded_boxes: torch.Tensor,
    class_probabilities: torch.Tensor,
    image_size: tuple[int, int] | torch.Tensor,
    score_threshold: float,
    nms_threshold: float,
    detections_per_img: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the relevant torchvision GeneralizedRCNN postprocessing sequence."""
    count, classes = class_probabilities.shape
    boxes = clip_boxes_to_image(decoded_boxes, image_size)
    proposal_ids = torch.arange(count, device=boxes.device).view(-1, 1).expand(count, classes)
    labels = torch.arange(classes, device=boxes.device).view(1, -1).expand(count, classes)
    boxes = boxes[:, 1:].reshape(-1, 4)
    scores = class_probabilities[:, 1:].reshape(-1)
    labels = labels[:, 1:].reshape(-1)
    proposal_ids = proposal_ids[:, 1:].reshape(-1)
    keep = torch.where(scores > float(score_threshold))[0]
    boxes, scores, labels, proposal_ids = (boxes[keep], scores[keep], labels[keep], proposal_ids[keep])
    keep = remove_small_boxes(boxes, min_size=1e-2)
    boxes, scores, labels, proposal_ids = (boxes[keep], scores[keep], labels[keep], proposal_ids[keep])
    keep = batched_nms(boxes, scores, labels, float(nms_threshold))[: int(detections_per_img)]
    return boxes[keep], scores[keep], labels[keep], proposal_ids[keep]


def _target_features(
    detections: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    decoded_boxes: torch.Tensor,
    class_probabilities: torch.Tensor,
    action_labels: torch.Tensor,
    image_size: tuple[int, int] | torch.Tensor,
    nms_threshold: float,
    proposal_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    boxes, scores, labels, proposal_ids = detections
    count = decoded_boxes.shape[0]
    if proposal_indices is None:
        proposal_indices = torch.arange(count, device=decoded_boxes.device)
    kept = decoded_boxes.new_zeros(count)
    max_iou = decoded_boxes.new_zeros(count)
    normalized_rank = decoded_boxes.new_ones(count)
    for proposal in proposal_indices.tolist():
        label = action_labels[proposal]
        target_box = clip_boxes_to_image(decoded_boxes[proposal, label].unsqueeze(0), image_size)[0]
        target_score = class_probabilities[proposal, label]
        target = proposal_ids.eq(proposal) & labels.eq(label)
        if target.any():
            rank = int(torch.nonzero(target, as_tuple=False)[0].item())
            kept[proposal] = 1.0
            normalized_rank[proposal] = rank / max(int(boxes.shape[0]) - 1, 1)
        higher_same = labels.eq(label) & scores.gt(target_score)
        if higher_same.any():
            max_iou[proposal] = box_iou(target_box.unsqueeze(0), boxes[higher_same]).amax()
    return kept, max_iou, normalized_rank


def _compose_features(
    current: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    baseline: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    proposal: int,
    nms_threshold: float,
) -> torch.Tensor:
    kept, max_iou, rank = current
    base_kept, base_max_iou, base_rank = baseline
    return torch.stack(
        (
            kept[proposal],
            base_kept[proposal],
            kept[proposal] - base_kept[proposal],
            max_iou[proposal],
            max_iou.new_tensor(float(nms_threshold)) - max_iou[proposal],
            max_iou[proposal] - base_max_iou[proposal],
            rank[proposal],
            rank[proposal] - base_rank[proposal],
        )
    )
