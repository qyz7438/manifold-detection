from __future__ import annotations

import torch

from spectral_detection_posttrain.core.matching.box_iou import box_iou
from spectral_detection_posttrain.core.matching.pred_gt_matcher import match_predictions_to_gt
from spectral_detection_posttrain.core.models.spectral_quality_head import (
    SpectralQualityHead,
    build_quality_features,
)
from spectral_detection_posttrain.signals.fft.fft_features import compute_amplitude_profile
from spectral_detection_posttrain.signals.fft.roi_crop import crop_and_resize_roi


def compute_r_amp_stats_from_loader(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    config: dict,
) -> dict:
    """Precompute global mean/std of R_amp from training set."""
    matching_cfg = config["matching"]
    iou_threshold = float(matching_cfg.get("iou_threshold", 0.5))
    score_threshold = float(matching_cfg.get("score_threshold", 0.05))
    amp_bins = int(config.get("quality_head", {}).get("amp_bins", 32))

    model.eval()
    all_r_amp: list[float] = []

    for images, targets in loader:
        device_images = [img.to(device) for img in images]
        outputs = model(device_images)
        for image, prediction, target in zip(images, outputs, targets):
            pred_cpu = {k: v.detach().cpu() for k, v in prediction.items()}
            tgt_cpu = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in target.items()}
            matched = match_predictions_to_gt(
                pred_cpu, tgt_cpu, iou_threshold=iou_threshold, score_threshold=score_threshold
            )
            gt_boxes = tgt_cpu.get("boxes", torch.empty((0, 4)))
            if len(gt_boxes) == 0:
                continue
            gt_amp_cache: dict[int, torch.Tensor] = {}
            for match in matched["matches"]:
                pred_idx = match["pred_index"]
                gt_idx = match["gt_index"]
                pred_box = pred_cpu["boxes"][pred_idx]
                pred_roi = crop_and_resize_roi(image.cpu(), pred_box)
                pred_profile = compute_amplitude_profile(pred_roi, num_bins=amp_bins)
                if gt_idx not in gt_amp_cache:
                    gt_roi = crop_and_resize_roi(image.cpu(), gt_boxes[gt_idx])
                    gt_amp_cache[gt_idx] = compute_amplitude_profile(gt_roi, num_bins=amp_bins)
                cosine = torch.nn.functional.cosine_similarity(
                    pred_profile, gt_amp_cache[gt_idx], dim=0
                ).clamp(-1.0, 1.0)
                r_amp = float(torch.exp(-(1.0 - cosine)).item())
                all_r_amp.append(r_amp)

    if not all_r_amp:
        return {"p05": 0.0, "p95": 1.0, "min": 0.0, "max": 1.0, "count": 0}

    values = torch.tensor(all_r_amp, dtype=torch.float32)
    k5 = max(1, int(len(values) * 0.05))
    k95 = max(1, int(len(values) * 0.95))
    sorted_vals, _ = values.sort()
    return {
        "p05": float(sorted_vals[k5].item()),
        "p95": float(sorted_vals[min(k95, len(values) - 1)].item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "count": len(all_r_amp),
    }


def compute_per_box_ramp(
    image: torch.Tensor,
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    best_gt_indices: torch.Tensor,
    amp_bins: int = 32,
) -> torch.Tensor:
    """Compute R_amp for each pred box (only valid for matched boxes)."""
    r_amp_values: list[float] = []
    gt_amp_cache: dict[int, torch.Tensor] = {}

    for pred_box, gt_idx in zip(pred_boxes, best_gt_indices):
        pred_roi = crop_and_resize_roi(image, pred_box)
        pred_profile = compute_amplitude_profile(pred_roi, num_bins=amp_bins)
        idx = int(gt_idx.item())
        if idx >= 0 and len(gt_boxes) > 0:
            if idx not in gt_amp_cache:
                gt_roi = crop_and_resize_roi(image, gt_boxes[idx])
                gt_amp_cache[idx] = compute_amplitude_profile(gt_roi, num_bins=amp_bins)
            cosine = torch.nn.functional.cosine_similarity(
                pred_profile, gt_amp_cache[idx], dim=0
            ).clamp(-1.0, 1.0)
            r_amp_values.append(float(torch.exp(-(1.0 - cosine)).item()))
        else:
            r_amp_values.append(0.0)

    return torch.tensor(r_amp_values, dtype=torch.float32)


def normalize_ramp(r_amp: torch.Tensor, stats: dict) -> torch.Tensor:
    """Percentile min-max normalize R_amp to [0, 1] using precomputed p05/p95."""
    if "mean" in stats and "std" in stats and "p05" not in stats and "p95" not in stats:
        mean = float(stats.get("mean", 0.0))
        std = max(float(stats.get("std", 1.0)), 1e-6)
        return (r_amp - mean) / std
    p05 = float(stats.get("p05", 0.0))
    p95 = float(stats.get("p95", 1.0))
    return ((r_amp - p05) / max(p95 - p05, 1e-6)).clamp(0.0, 1.0)


def compute_per_box_structure(
    image: torch.Tensor,
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    best_gt_indices: torch.Tensor,
) -> torch.Tensor:
    from spectral_detection_posttrain.signals.fft.fft_features import compute_structure_similarity

    values: list[float] = []
    gt_roi_cache: dict[int, torch.Tensor] = {}
    for pred_box, gt_idx in zip(pred_boxes, best_gt_indices):
        idx = int(gt_idx.item())
        if idx < 0 or len(gt_boxes) == 0:
            values.append(0.0)
            continue
        pred_roi = crop_and_resize_roi(image, pred_box)
        if idx not in gt_roi_cache:
            gt_roi_cache[idx] = crop_and_resize_roi(image, gt_boxes[idx])
        score = compute_structure_similarity(pred_roi, gt_roi_cache[idx])
        values.append(float(score.item()))
    return torch.tensor(values, dtype=torch.float32)


@torch.no_grad()
def compute_per_box_qspec(
    model: torch.nn.Module,
    quality_head: SpectralQualityHead,
    image: torch.Tensor,
    pred_boxes: torch.Tensor,
    device: torch.device,
    feature_mode: str = "roi_amp_structure",
    amp_bins: int = 32,
) -> torch.Tensor:
    """Compute q_spec for each pred box using frozen quality head."""
    if len(pred_boxes) == 0:
        return torch.empty((0,), dtype=torch.float32)

    roi_features = _extract_roi_features(model, image, pred_boxes, device)

    amp_profiles: list[torch.Tensor] = []
    for pred_box in pred_boxes:
        pred_roi = crop_and_resize_roi(image, pred_box)
        amp_profiles.append(compute_amplitude_profile(pred_roi, num_bins=amp_bins))
    amp_tensor = torch.stack(amp_profiles).float()

    structure_dim = 8
    sample = {
        "roi_features": roi_features.float(),
        "amp_profiles": amp_tensor,
        "structure_features": torch.zeros((len(pred_boxes), structure_dim), dtype=torch.float32),
    }
    features = build_quality_features(sample, feature_mode).to(device)
    q_spec = quality_head.predict_quality(features).detach().cpu()
    return q_spec


@torch.no_grad()
def _extract_roi_features(
    model: torch.nn.Module,
    image: torch.Tensor,
    boxes: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Extract ROI box features from detector's box_head (must be frozen)."""
    if len(boxes) == 0:
        return torch.empty((0, 0), dtype=torch.float32)

    original_size = tuple(image.shape[-2:])
    transformed_images, _ = model.transform([image.to(device)], None)
    features = model.backbone(transformed_images.tensors)
    if isinstance(features, torch.Tensor):
        features = {"0": features}

    orig_h, orig_w = original_size
    new_h, new_w = transformed_images.image_sizes[0]
    scaled = boxes.clone().to(device)
    scaled[:, [0, 2]] *= new_w / float(orig_w)
    scaled[:, [1, 3]] *= new_h / float(orig_h)

    pooled = model.roi_heads.box_roi_pool(features, [scaled], transformed_images.image_sizes)
    roi_features = model.roi_heads.box_head(pooled)
    return roi_features.detach().cpu()


def compute_group_reward(
    per_box_rewards: torch.Tensor,
    is_tp: torch.Tensor,
    scores: torch.Tensor,
    total_gt: int,
    matched_gt_indices: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.3,
    high_conf_threshold: float = 0.7,
) -> float:
    """Aggregate per-box rewards into group-level scalar with FP/FN penalty.

    Returns max(0, mean_TP_reward - alpha * high_conf_FP_rate - beta * miss_rate)
    """
    tp_mask = is_tp.bool()
    if tp_mask.any():
        mean_tp = float(per_box_rewards[tp_mask].mean().item())
    else:
        mean_tp = 0.0

    high_conf_mask = scores >= high_conf_threshold
    high_conf_fp = (~tp_mask & high_conf_mask).sum().item()
    high_conf_total = max(1, high_conf_mask.sum().item())
    fp_rate = high_conf_fp / high_conf_total

    matched_gt_count = len(set(idx.item() for idx in matched_gt_indices if idx >= 0))
    miss_rate = 1.0 - (matched_gt_count / max(1, total_gt))

    reward = mean_tp - alpha * fp_rate - beta * miss_rate
    return max(0.0, reward)
