from __future__ import annotations

from copy import deepcopy

import torch

ROLLOUT_CONFIGS: list[dict] = [
    {"score_threshold": 0.01, "nms_threshold": 0.3},
    {"score_threshold": 0.05, "nms_threshold": 0.5},
    {"score_threshold": 0.3, "nms_threshold": 0.5},
]


@torch.no_grad()
def generate_rollouts(
    model: torch.nn.Module,
    images: list[torch.Tensor],
    configs: list[dict] | None = None,
) -> list[list[dict]]:
    """Generate K groups of detection results per image by varying inference thresholds.

    Returns: list of K lists, each inner list has len(images) prediction dicts.
        rollout[k][i] = predictions for image i under config k.
    """
    if configs is None:
        configs = ROLLOUT_CONFIGS

    original_score_thresh = model.roi_heads.score_thresh
    original_nms_thresh = model.roi_heads.nms_thresh
    was_training = model.training
    model.eval()

    device = next(model.parameters()).device
    device_images = [image.to(device) for image in images]

    all_rollouts: list[list[dict]] = []
    for cfg in configs:
        model.roi_heads.score_thresh = cfg["score_threshold"]
        model.roi_heads.nms_thresh = cfg["nms_threshold"]
        outputs = model(device_images)
        predictions = [{k: v.detach().cpu() for k, v in output.items()} for output in outputs]
        all_rollouts.append(predictions)

    model.roi_heads.score_thresh = original_score_thresh
    model.roi_heads.nms_thresh = original_nms_thresh
    if was_training:
        model.train()

    return all_rollouts


def rollout_diversity_check(rollouts: list[list[dict]]) -> dict:
    """Diagnostic: measure how different the rollouts are from each other."""
    if len(rollouts) < 2:
        return {"unique": True, "box_count_range": (0, 0)}

    counts = []
    for group in rollouts:
        total_boxes = sum(len(pred["boxes"]) for pred in group)
        counts.append(total_boxes)

    min_c, max_c = min(counts), max(counts)
    return {
        "box_counts": counts,
        "box_count_range": (min_c, max_c),
        "max_divergence_pct": 0.0 if max_c == 0 else (max_c - min_c) / max_c * 100,
        "has_diversity": max_c != min_c,
    }
