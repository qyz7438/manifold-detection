"""Detector-only graph-consensus actions for proposal-manifold transport."""

from __future__ import annotations

import torch

from spectral_detection_posttrain.core.matching.box_iou import box_iou


def proposal_graph_consensus_deltas(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    observable_mask: torch.Tensor,
    *,
    min_peer_iou: float = 0.30,
    max_abs_delta: float = 0.05,
) -> torch.Tensor:
    """Move each observable box toward stronger overlapping same-class peers."""
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    count = boxes.shape[0]
    if labels.shape != (count,) or scores.shape != (count,) or observable_mask.shape != (count,):
        raise ValueError("labels, scores, and observable_mask must have shape (N,)")
    if not 0.0 < float(min_peer_iou) < 1.0:
        raise ValueError("min_peer_iou must be in (0, 1)")
    if not 0.0 < float(max_abs_delta) <= 1.0:
        raise ValueError("max_abs_delta must be in (0, 1]")
    devices = {tensor.device for tensor in (boxes, labels, scores, observable_mask)}
    if len(devices) != 1:
        raise ValueError("all inputs must share a device")
    output = boxes.new_zeros((count, 4))
    if count == 0:
        return output
    ious = box_iou(boxes, boxes)
    indices = torch.arange(count, device=boxes.device)
    visible = observable_mask.bool()
    for row in torch.nonzero(visible, as_tuple=False).flatten().tolist():
        peers = (
            visible
            & labels.eq(labels[row])
            & scores.gt(scores[row])
            & ious[row].ge(float(min_peer_iou))
            & indices.ne(row)
        )
        if not peers.any():
            continue
        weights = (scores[peers] * ious[row, peers]).clamp_min(1e-8)
        target = (boxes[peers] * weights[:, None]).sum(dim=0) / weights.sum()
        raw = _encode_box_delta(boxes[row : row + 1], target[None, :])[0]
        scale = (float(max_abs_delta) / raw.abs().amax().clamp_min(float(max_abs_delta))).clamp(max=1.0)
        output[row] = raw * scale
    return output


def _encode_box_delta(boxes: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    widths = (boxes[:, 2] - boxes[:, 0]).clamp_min(1e-6)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(1e-6)
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
    target_widths = (targets[:, 2] - targets[:, 0]).clamp_min(1e-6)
    target_heights = (targets[:, 3] - targets[:, 1]).clamp_min(1e-6)
    target_ctr_x = targets[:, 0] + 0.5 * target_widths
    target_ctr_y = targets[:, 1] + 0.5 * target_heights
    return torch.stack(
        (
            (target_ctr_x - ctr_x) / widths,
            (target_ctr_y - ctr_y) / heights,
            torch.log(target_widths / widths),
            torch.log(target_heights / heights),
        ),
        dim=1,
    )
