from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectral_detection_posttrain.core.matching.box_iou import box_iou
from spectral_detection_posttrain.signals.fft.raw_ifft_verifier import (
    LEGACY_IFFT_FEATURE_NAMES,
    fit_train_effect_scorer,
)


DEFAULT_CACHE = Path("runs/round2150_raw_ifft_legacy_full_ap75/candidate_raw_ifft_features.npz")
DEFAULT_OUT_DIR = Path("runs/round2221_interpretable_reward_signal_diagnostics")
DATA_ROOT = Path("data/NWPU VHR-10 dataset")
COCO_JSON = Path("data/NWPU_VHR10_coco.json")
MAX_SIZE = 480
EPS = 1.0e-8

RAW_IFFT_REFERENCE_FEATURES = [
    "fft_edge_truncation@64",
    "phase_edge@64",
    "phase_abs_high@11",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline diagnostics for interpretable non-network reward signals on cached NWPU proposals."
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--coco-json", type=Path, default=COCO_JSON)
    parser.add_argument("--max-size", type=int, default=MAX_SIZE)
    parser.add_argument("--target-precision", type=float, default=0.7)
    parser.add_argument("--limit-images", type=int, default=0, help="Smoke mode: keep only the first N images per split.")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def load_coco(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_image_tensor(data_root: Path, image_info: dict[str, Any], max_size: int, device: torch.device) -> torch.Tensor:
    image_path = data_root / "positive image set" / image_info["file_name"]
    if not image_path.exists():
        image_path = data_root / "negative image set" / image_info["file_name"]
    image = Image.open(str(image_path)).convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).to(device=device)
    _, height, width = tensor.shape
    if max(height, width) > int(max_size):
        scale = float(max_size) / float(max(height, width))
        new_h = max(1, int(height * scale))
        new_w = max(1, int(width * scale))
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return tensor.clamp(0.0, 1.0)


def sobel_map(gray: torch.Tensor) -> torch.Tensor:
    gray4 = gray.reshape(1, 1, gray.shape[-2], gray.shape[-1]).float()
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=gray4.dtype,
        device=gray4.device,
    ).reshape(1, 1, 3, 3) / 8.0
    kernel_y = kernel_x.transpose(-1, -2)
    gx = F.conv2d(gray4, kernel_x, padding=1)
    gy = F.conv2d(gray4, kernel_y, padding=1)
    return torch.sqrt(gx.square() + gy.square() + 1.0e-12).squeeze(0).squeeze(0)


def robust_normalize_map(values: torch.Tensor) -> torch.Tensor:
    flat = values.flatten()
    if flat.numel() < 4:
        return values.clamp_min(0.0)
    lo = torch.quantile(flat, 0.05)
    hi = torch.quantile(flat, 0.95)
    if float(hi - lo) < 1.0e-8:
        return torch.zeros_like(values)
    return ((values - lo) / (hi - lo)).clamp(0.0, 1.0)


def phase_only_edge_map(gray: torch.Tensor) -> torch.Tensor:
    spectrum = torch.fft.fft2(gray.float())
    phase_only = torch.exp(1j * torch.angle(spectrum))
    recon = torch.fft.ifft2(phase_only).real
    recon = recon - recon.mean()
    std = recon.std()
    if float(std) > 1.0e-8:
        recon = recon / std
    return sobel_map(recon)


def clamp_box(box: np.ndarray, height: int, width: int) -> tuple[int, int, int, int]:
    x1 = int(math.floor(float(box[0])))
    y1 = int(math.floor(float(box[1])))
    x2 = int(math.ceil(float(box[2])))
    y2 = int(math.ceil(float(box[3])))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def crop_mean(feature_map: torch.Tensor, box: np.ndarray) -> float:
    height, width = feature_map.shape
    x1, y1, x2, y2 = clamp_box(box, height, width)
    crop = feature_map[y1:y2, x1:x2]
    if crop.numel() == 0:
        return 0.0
    return float(crop.mean().item())


def boundary_and_interior_means(feature_map: torch.Tensor, box: np.ndarray) -> tuple[float, float]:
    height, width = feature_map.shape
    x1, y1, x2, y2 = clamp_box(box, height, width)
    crop = feature_map[y1:y2, x1:x2]
    if crop.numel() == 0:
        return 0.0, 0.0
    crop_h, crop_w = crop.shape
    boundary_width = max(1, min(4, int(round(0.08 * min(crop_h, crop_w)))))
    mask = torch.zeros_like(crop, dtype=torch.bool)
    mask[:boundary_width, :] = True
    mask[-boundary_width:, :] = True
    mask[:, :boundary_width] = True
    mask[:, -boundary_width:] = True
    boundary = crop[mask]
    interior = crop[~mask]
    if interior.numel() == 0:
        interior = crop.reshape(-1)
    return float(boundary.mean().item()), float(interior.mean().item())


def ring_mean(feature_map: torch.Tensor, box: np.ndarray) -> float:
    height, width = feature_map.shape
    x1, y1, x2, y2 = clamp_box(box, height, width)
    box_w = x2 - x1
    box_h = y2 - y1
    pad = max(4, int(round(0.20 * max(box_w, box_h))))
    rx1 = max(0, x1 - pad)
    ry1 = max(0, y1 - pad)
    rx2 = min(width, x2 + pad)
    ry2 = min(height, y2 + pad)
    expanded = feature_map[ry1:ry2, rx1:rx2]
    if expanded.numel() == 0:
        return 0.0
    mask = torch.ones_like(expanded, dtype=torch.bool)
    inner_x1 = x1 - rx1
    inner_y1 = y1 - ry1
    inner_x2 = x2 - rx1
    inner_y2 = y2 - ry1
    mask[inner_y1:inner_y2, inner_x1:inner_x2] = False
    ring = expanded[mask]
    if ring.numel() == 0:
        return float(expanded.mean().item())
    return float(ring.mean().item())


def centroid_consistency(saliency_map: torch.Tensor, box: np.ndarray) -> float:
    height, width = saliency_map.shape
    x1, y1, x2, y2 = clamp_box(box, height, width)
    crop = saliency_map[y1:y2, x1:x2].float().clamp_min(0.0)
    if crop.numel() == 0:
        return 0.0
    weight = crop - crop.min()
    total = float(weight.sum().item())
    crop_h, crop_w = crop.shape
    if total <= 1.0e-8:
        return 0.0
    yy, xx = torch.meshgrid(
        torch.arange(crop_h, device=crop.device, dtype=torch.float32),
        torch.arange(crop_w, device=crop.device, dtype=torch.float32),
        indexing="ij",
    )
    cx = float((weight * xx).sum().item() / total)
    cy = float((weight * yy).sum().item() / total)
    center_x = (crop_w - 1) / 2.0
    center_y = (crop_h - 1) / 2.0
    norm_x = max(1.0, crop_w / 2.0)
    norm_y = max(1.0, crop_h / 2.0)
    dist = math.sqrt(((cx - center_x) / norm_x) ** 2 + ((cy - center_y) / norm_y) ** 2)
    return float(max(0.0, 1.0 - min(1.0, dist / math.sqrt(2.0))))


def build_multiscale_edge_maps(gray: torch.Tensor) -> list[tuple[float, torch.Tensor]]:
    height, width = gray.shape
    maps: list[tuple[float, torch.Tensor]] = []
    for scale in (0.5, 1.0, 2.0):
        if scale == 1.0:
            scaled_gray = gray
        else:
            new_h = max(4, int(round(height * scale)))
            new_w = max(4, int(round(width * scale)))
            scaled_gray = F.interpolate(
                gray.reshape(1, 1, height, width),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            ).reshape(new_h, new_w)
        maps.append((float(scale), robust_normalize_map(sobel_map(scaled_gray))))
    return maps


def multiscale_saliency_score_from_maps(edge_maps: list[tuple[float, torch.Tensor]], box: np.ndarray) -> float:
    values = []
    for scale, scaled_edge in edge_maps:
        scaled_box = np.asarray(box, dtype=np.float64) * float(scale)
        values.append(crop_mean(scaled_edge, scaled_box))
    values_np = np.asarray(values, dtype=np.float64)
    return float(values_np.min())


def compute_gt_aspect_stats(coco: dict[str, Any], train_image_ids: set[int]) -> dict[int, tuple[float, float]]:
    class_values: dict[int, list[float]] = defaultdict(list)
    all_values: list[float] = []
    for ann in coco["annotations"]:
        if int(ann["image_id"]) not in train_image_ids:
            continue
        _, _, width, height = ann["bbox"]
        if float(width) <= 0.0 or float(height) <= 0.0:
            continue
        value = math.log(float(width) / float(height))
        class_values[int(ann["category_id"])].append(value)
        all_values.append(value)
    global_mean = float(np.mean(all_values)) if all_values else 0.0
    global_std = max(0.25, float(np.std(all_values)) if all_values else 1.0)
    stats: dict[int, tuple[float, float]] = {}
    for class_id in range(1, 11):
        values = class_values.get(class_id, [])
        if len(values) < 3:
            stats[class_id] = (global_mean, global_std)
        else:
            stats[class_id] = (float(np.mean(values)), max(0.25, float(np.std(values))))
    return stats


def aspect_plausibility(boxes: np.ndarray, class_ids: np.ndarray, stats: dict[int, tuple[float, float]]) -> np.ndarray:
    output = np.zeros((boxes.shape[0],), dtype=np.float64)
    for index, (box, class_id) in enumerate(zip(boxes, class_ids)):
        width = max(1.0, float(box[2] - box[0]))
        height = max(1.0, float(box[3] - box[1]))
        mean, std = stats.get(int(class_id), (0.0, 1.0))
        z = (math.log(width / height) - mean) / max(0.25, std)
        output[index] = math.exp(-0.5 * z * z)
    return output


def nms_support_density(boxes: np.ndarray, class_ids: np.ndarray, image_ids: np.ndarray) -> np.ndarray:
    output = np.zeros((boxes.shape[0],), dtype=np.float64)
    by_group: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (image_id, class_id) in enumerate(zip(image_ids, class_ids)):
        by_group[(int(image_id), int(class_id))].append(index)
    for indices in by_group.values():
        if len(indices) <= 1:
            continue
        group_boxes = torch.as_tensor(boxes[indices], dtype=torch.float32)
        ious = box_iou(group_boxes, group_boxes).cpu().numpy()
        support = ((ious >= 0.5).sum(axis=1) - 1).astype(np.float64)
        output[np.asarray(indices, dtype=np.int64)] = support / max(1.0, math.sqrt(len(indices)))
    return output


def subset_indices_for_limit(image_ids: np.ndarray, limit_images: int) -> np.ndarray:
    if int(limit_images) <= 0:
        return np.arange(image_ids.shape[0])
    keep_images = set(np.unique(image_ids)[: int(limit_images)].tolist())
    return np.flatnonzero(np.asarray([int(image_id) in keep_images for image_id in image_ids], dtype=bool))


def extract_split_arrays(data: np.lib.npyio.NpzFile, split: str, limit_images: int) -> dict[str, np.ndarray]:
    image_ids = np.asarray(data[f"{split}_image_ids"])
    indices = subset_indices_for_limit(image_ids, limit_images)
    keys = [
        "labels",
        "best_iou",
        "label_probs",
        "rollout_scores",
        "class_ids",
        "image_ids",
        "proposal_boxes",
    ]
    return {key: np.asarray(data[f"{split}_{key}"])[indices] for key in keys}


def compute_image_signals(
    *,
    arrays: dict[str, np.ndarray],
    coco_infos: dict[int, dict[str, Any]],
    data_root: Path,
    max_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    count = arrays["proposal_boxes"].shape[0]
    boundary_phase = np.zeros((count,), dtype=np.float64)
    texture_contrast = np.zeros((count,), dtype=np.float64)
    multiscale_saliency = np.zeros((count,), dtype=np.float64)
    score_edge_alignment = np.zeros((count,), dtype=np.float64)
    activation_centroid = np.zeros((count,), dtype=np.float64)

    by_image: dict[int, list[int]] = defaultdict(list)
    for index, image_id in enumerate(arrays["image_ids"]):
        by_image[int(image_id)].append(index)

    for image_id, indices in sorted(by_image.items()):
        image = load_image_tensor(data_root, coco_infos[int(image_id)], max_size, device)
        gray = image.mean(dim=0)
        edge = robust_normalize_map(sobel_map(gray))
        phase_edge = robust_normalize_map(phase_only_edge_map(gray))
        saliency = robust_normalize_map(edge + phase_edge)
        multiscale_edge_maps = build_multiscale_edge_maps(gray)
        for index in indices:
            box = arrays["proposal_boxes"][index]
            phase_boundary, phase_interior = boundary_and_interior_means(phase_edge, box)
            edge_boundary, edge_interior = boundary_and_interior_means(edge, box)
            inside_edge = crop_mean(edge, box)
            outside_edge = ring_mean(edge, box)
            boundary_phase[index] = phase_boundary / (phase_interior + EPS)
            texture_contrast[index] = abs(inside_edge - outside_edge) / (inside_edge + outside_edge + EPS)
            multiscale_saliency[index] = multiscale_saliency_score_from_maps(multiscale_edge_maps, box)
            score_edge_alignment[index] = (edge_boundary / (edge_interior + EPS)) * (1.0 - float(arrays["label_probs"][index]))
            activation_centroid[index] = centroid_consistency(saliency, box)

    return {
        "boundary_phase_coherence": boundary_phase,
        "interior_exterior_texture_contrast": texture_contrast,
        "multi_scale_saliency_consistency": multiscale_saliency,
        "score_edge_alignment": score_edge_alignment,
        "activation_centroid_consistency": activation_centroid,
    }


def load_legacy_feature_matrix(data: np.lib.npyio.NpzFile, split: str, specs: list[str], indices: np.ndarray) -> np.ndarray:
    columns = []
    for spec in specs:
        name, crop_text = spec.split("@", maxsplit=1)
        crop = int(crop_text)
        key = f"{split}_legacy_ifft_{crop}"
        columns.append(np.asarray(data[key])[indices, LEGACY_IFFT_FEATURE_NAMES.index(name)])
    return np.stack(columns, axis=1).astype(np.float64)


def rankdata_simple(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.shape[0], dtype=np.float64)
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return None
    x = x[mask] - x[mask].mean()
    y = y[mask] - y[mask].mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom < 1.0e-12:
        return None
    return float(np.dot(x, y) / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    return pearson(rankdata_simple(np.asarray(x, dtype=np.float64)), rankdata_simple(np.asarray(y, dtype=np.float64)))


def orient_by_train(train_scores: np.ndarray, val_scores: np.ndarray, train_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    train_scores = np.asarray(train_scores, dtype=np.float64)
    val_scores = np.asarray(val_scores, dtype=np.float64)
    labels = np.asarray(train_labels, dtype=bool)
    if int(labels.sum()) == 0 or int((~labels).sum()) == 0:
        return train_scores, val_scores, 1
    pos_mean = float(np.nanmean(train_scores[labels]))
    neg_mean = float(np.nanmean(train_scores[~labels]))
    if pos_mean < neg_mean:
        return -train_scores, -val_scores, -1
    return train_scores, val_scores, 1


def ranking_metrics(scores: np.ndarray, labels: np.ndarray, precision_targets: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9)) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    positive_count = int(labels.sum())
    negative_count = int((~labels).sum())
    output: dict[str, Any] = {
        "count": int(labels.shape[0]),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "prevalence": float(positive_count / max(1, labels.shape[0])),
        "auc": 0.0,
        "average_precision": 0.0,
    }
    unique_scores = np.unique(scores[np.isfinite(scores)])
    if positive_count > 0 and negative_count > 0 and unique_scores.shape[0] > 1:
        output["auc"] = float(roc_auc_score(labels.astype(np.int32), scores))
        output["average_precision"] = float(average_precision_score(labels.astype(np.int32), scores))

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    tp = np.cumsum(sorted_labels.astype(np.int64))
    rank = np.arange(1, sorted_labels.shape[0] + 1)
    precision = tp / rank
    recall = tp / max(1, positive_count)
    for target in precision_targets:
        valid = np.flatnonzero(precision >= float(target))
        key = f"recall_at_precision_{target:g}"
        if valid.shape[0] == 0:
            output[key] = 0.0
            output[f"{key}_selected"] = 0
            output[f"{key}_precision"] = 0.0
            output[f"{key}_threshold"] = None
            continue
        best_recall = recall[valid].max()
        best_candidates = valid[recall[valid] == best_recall]
        best = int(best_candidates[-1])
        output[key] = float(recall[best])
        output[f"{key}_selected"] = int(rank[best])
        output[f"{key}_precision"] = float(precision[best])
        output[f"{key}_threshold"] = float(sorted_scores[best])
    return output


def calibrate_threshold(scores: np.ndarray, labels: np.ndarray, target_precision: float) -> dict[str, Any]:
    metrics = ranking_metrics(scores, labels, precision_targets=(float(target_precision),))
    key = f"recall_at_precision_{target_precision:g}"
    return {
        "target_precision": float(target_precision),
        "threshold": metrics.get(f"{key}_threshold"),
        "train_recall": metrics.get(key, 0.0),
        "train_precision": metrics.get(f"{key}_precision", 0.0),
        "train_selected": metrics.get(f"{key}_selected", 0),
    }


def apply_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float | None) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=bool)
    if threshold is None:
        return {"selected": 0, "tp": 0, "fp": 0, "precision": 0.0, "recall": 0.0}
    selected = np.asarray(scores, dtype=np.float64) >= float(threshold)
    tp = int((selected & labels).sum())
    fp = int((selected & (~labels)).sum())
    selected_count = int(selected.sum())
    return {
        "selected": selected_count,
        "tp": tp,
        "fp": fp,
        "precision": float(tp / selected_count) if selected_count else 0.0,
        "recall": float(tp / max(1, int(labels.sum()))),
    }


def evaluate_signal(
    name: str,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    train_arrays: dict[str, np.ndarray],
    val_arrays: dict[str, np.ndarray],
    target_precision: float,
) -> dict[str, Any]:
    train_labels = train_arrays["labels"].astype(bool)
    val_labels = val_arrays["labels"].astype(bool)
    train_scores, val_scores, direction = orient_by_train(train_scores, val_scores, train_labels)
    calibration = calibrate_threshold(train_scores, train_labels, target_precision)
    fixed_val = apply_threshold(val_scores, val_labels, calibration["threshold"])
    return {
        "signal": name,
        "direction": direction,
        "train": {
            "ranking": ranking_metrics(train_scores, train_labels),
            "spearman_iou": spearman(train_scores, train_arrays["best_iou"]),
            "positive_mean": float(np.nanmean(train_scores[train_labels])) if int(train_labels.sum()) else None,
            "negative_mean": float(np.nanmean(train_scores[~train_labels])) if int((~train_labels).sum()) else None,
        },
        "val": {
            "ranking": ranking_metrics(val_scores, val_labels),
            "spearman_iou": spearman(val_scores, val_arrays["best_iou"]),
            "positive_mean": float(np.nanmean(val_scores[val_labels])) if int(val_labels.sum()) else None,
            "negative_mean": float(np.nanmean(val_scores[~val_labels])) if int((~val_labels).sum()) else None,
        },
        "train_calibration": calibration,
        "fixed_threshold_val": fixed_val,
    }


def effect_sum_fusion(
    train_matrix: np.ndarray,
    val_matrix: np.ndarray,
    train_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(train_matrix)
    train_z = scaler.transform(train_matrix)
    val_z = scaler.transform(val_matrix)
    labels = train_labels.astype(bool)
    weights = train_z[labels].mean(axis=0) - train_z[~labels].mean(axis=0)
    return train_z @ weights, val_z @ weights


def logistic_fusion(
    train_matrix: np.ndarray,
    val_matrix: np.ndarray,
    train_labels: np.ndarray,
    *,
    c_value: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(train_matrix)
    train_z = scaler.transform(train_matrix)
    val_z = scaler.transform(val_matrix)
    model = LogisticRegression(
        class_weight="balanced",
        C=float(c_value),
        solver="liblinear",
        random_state=42,
        max_iter=1000,
    ).fit(train_z, train_labels.astype(np.int32))
    return model.decision_function(train_z), model.decision_function(val_z)


def load_all_legacy_feature_matrix(data: np.lib.npyio.NpzFile, split: str, indices: np.ndarray) -> tuple[np.ndarray, list[str]]:
    columns = []
    names = []
    for crop in (7, 11, 15, 21, 64):
        key = f"{split}_legacy_ifft_{crop}"
        matrix = np.asarray(data[key], dtype=np.float64)[indices]
        for feature_index, feature_name in enumerate(LEGACY_IFFT_FEATURE_NAMES):
            columns.append(matrix[:, feature_index])
            names.append(f"{feature_name}@{crop}")
    return np.stack(columns, axis=1), names


def write_leaderboard(rows: list[dict[str, Any]], out_path: Path) -> None:
    fields = [
        "signal",
        "direction",
        "train_auc",
        "train_ap",
        "train_r_at_p70",
        "train_spearman_iou",
        "val_auc",
        "val_ap",
        "val_r_at_p70_oracle",
        "val_spearman_iou",
        "fixed_val_selected",
        "fixed_val_tp",
        "fixed_val_fp",
        "fixed_val_precision",
        "fixed_val_recall",
        "train_threshold",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            train_ranking = row["train"]["ranking"]
            val_ranking = row["val"]["ranking"]
            fixed = row["fixed_threshold_val"]
            writer.writerow(
                {
                    "signal": row["signal"],
                    "direction": row["direction"],
                    "train_auc": train_ranking["auc"],
                    "train_ap": train_ranking["average_precision"],
                    "train_r_at_p70": train_ranking["recall_at_precision_0.7"],
                    "train_spearman_iou": row["train"]["spearman_iou"],
                    "val_auc": val_ranking["auc"],
                    "val_ap": val_ranking["average_precision"],
                    "val_r_at_p70_oracle": val_ranking["recall_at_precision_0.7"],
                    "val_spearman_iou": row["val"]["spearman_iou"],
                    "fixed_val_selected": fixed["selected"],
                    "fixed_val_tp": fixed["tp"],
                    "fixed_val_fp": fixed["fp"],
                    "fixed_val_precision": fixed["precision"],
                    "fixed_val_recall": fixed["recall"],
                    "train_threshold": row["train_calibration"]["threshold"],
                }
            )


def flatten_eval_row(row: dict[str, Any], *, group: str | None = None, method: str | None = None, n_features: int | None = None) -> dict[str, Any]:
    train_ranking = row["train"]["ranking"]
    val_ranking = row["val"]["ranking"]
    fixed = row["fixed_threshold_val"]
    return {
        "group": group if group is not None else row["signal"],
        "method": method if method is not None else "single",
        "n_features": n_features if n_features is not None else 1,
        "signal": row["signal"],
        "direction": row["direction"],
        "train_auc": train_ranking["auc"],
        "train_ap": train_ranking["average_precision"],
        "train_r_at_p70": train_ranking["recall_at_precision_0.7"],
        "train_spearman_iou": row["train"]["spearman_iou"],
        "val_auc": val_ranking["auc"],
        "val_ap": val_ranking["average_precision"],
        "val_r_at_p70_oracle": val_ranking["recall_at_precision_0.7"],
        "val_spearman_iou": row["val"]["spearman_iou"],
        "fixed_val_selected": fixed["selected"],
        "fixed_val_tp": fixed["tp"],
        "fixed_val_fp": fixed["fp"],
        "fixed_val_precision": fixed["precision"],
        "fixed_val_recall": fixed["recall"],
        "train_threshold": row["train_calibration"]["threshold"],
    }


def write_fusion_sweep(rows: list[dict[str, Any]], out_path: Path) -> None:
    fields = [
        "group",
        "method",
        "n_features",
        "signal",
        "direction",
        "train_auc",
        "train_ap",
        "train_r_at_p70",
        "train_spearman_iou",
        "val_auc",
        "val_ap",
        "val_r_at_p70_oracle",
        "val_spearman_iou",
        "fixed_val_selected",
        "fixed_val_tp",
        "fixed_val_fp",
        "fixed_val_precision",
        "fixed_val_recall",
        "train_threshold",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_fusion_sweep(
    *,
    train_signals: dict[str, np.ndarray],
    val_signals: dict[str, np.ndarray],
    train_arrays: dict[str, np.ndarray],
    val_arrays: dict[str, np.ndarray],
    data: np.lib.npyio.NpzFile,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    target_precision: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    new_top3 = [
        "score_edge_alignment",
        "boundary_phase_coherence",
        "interior_exterior_texture_contrast",
    ]
    new_all7 = [
        "boundary_phase_coherence",
        "interior_exterior_texture_contrast",
        "aspect_ratio_plausibility",
        "multi_scale_saliency_consistency",
        "score_edge_alignment",
        "nms_survivor_density",
        "activation_centroid_consistency",
    ]
    raw3_train = load_legacy_feature_matrix(data, "train", RAW_IFFT_REFERENCE_FEATURES, train_indices)
    raw3_val = load_legacy_feature_matrix(data, "val", RAW_IFFT_REFERENCE_FEATURES, val_indices)
    legacy_train, legacy_names = load_all_legacy_feature_matrix(data, "train", train_indices)
    legacy_val, _ = load_all_legacy_feature_matrix(data, "val", val_indices)

    def matrix_from_signals(names: list[str], split_signals: dict[str, np.ndarray]) -> np.ndarray:
        return np.stack([split_signals[name] for name in names], axis=1).astype(np.float64)

    groups: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {
        "new_top3": (
            matrix_from_signals(new_top3, train_signals),
            matrix_from_signals(new_top3, val_signals),
            new_top3,
        ),
        "all_new7": (
            matrix_from_signals(new_all7, train_signals),
            matrix_from_signals(new_all7, val_signals),
            new_all7,
        ),
        "raw_ifft_recipe_score": (
            matrix_from_signals(["reference_raw_ifft_recipe"], train_signals),
            matrix_from_signals(["reference_raw_ifft_recipe"], val_signals),
            ["reference_raw_ifft_recipe"],
        ),
        "raw_ifft_individual3": (
            raw3_train,
            raw3_val,
            RAW_IFFT_REFERENCE_FEATURES,
        ),
        "legacy_ifft_full115": (
            legacy_train,
            legacy_val,
            legacy_names,
        ),
        "new_top3_plus_raw_ifft_recipe": (
            np.concatenate([matrix_from_signals(new_top3, train_signals), matrix_from_signals(["reference_raw_ifft_recipe"], train_signals)], axis=1),
            np.concatenate([matrix_from_signals(new_top3, val_signals), matrix_from_signals(["reference_raw_ifft_recipe"], val_signals)], axis=1),
            [*new_top3, "reference_raw_ifft_recipe"],
        ),
        "new_top3_plus_raw_ifft_individual3": (
            np.concatenate([matrix_from_signals(new_top3, train_signals), raw3_train], axis=1),
            np.concatenate([matrix_from_signals(new_top3, val_signals), raw3_val], axis=1),
            [*new_top3, *RAW_IFFT_REFERENCE_FEATURES],
        ),
        "new_top3_plus_legacy_ifft_full115": (
            np.concatenate([matrix_from_signals(new_top3, train_signals), legacy_train], axis=1),
            np.concatenate([matrix_from_signals(new_top3, val_signals), legacy_val], axis=1),
            [*new_top3, *legacy_names],
        ),
        "all_new7_plus_raw_ifft_recipe": (
            np.concatenate([matrix_from_signals(new_all7, train_signals), matrix_from_signals(["reference_raw_ifft_recipe"], train_signals)], axis=1),
            np.concatenate([matrix_from_signals(new_all7, val_signals), matrix_from_signals(["reference_raw_ifft_recipe"], val_signals)], axis=1),
            [*new_all7, "reference_raw_ifft_recipe"],
        ),
        "all_new7_plus_legacy_ifft_full115": (
            np.concatenate([matrix_from_signals(new_all7, train_signals), legacy_train], axis=1),
            np.concatenate([matrix_from_signals(new_all7, val_signals), legacy_val], axis=1),
            [*new_all7, *legacy_names],
        ),
    }

    sweep_rows: list[dict[str, Any]] = []
    detailed: dict[str, Any] = {}
    labels = train_arrays["labels"].astype(bool)
    for group_name, (train_matrix, val_matrix, feature_names) in groups.items():
        methods: list[tuple[str, np.ndarray, np.ndarray]] = []
        effect_train, effect_val = effect_sum_fusion(train_matrix, val_matrix, labels)
        methods.append(("effect_sum", effect_train, effect_val))
        for c_value in (0.05, 0.25, 1.0):
            log_train, log_val = logistic_fusion(train_matrix, val_matrix, labels, c_value=c_value)
            methods.append((f"logistic_c{c_value:g}", log_train, log_val))
        detailed[group_name] = {"features": feature_names}
        for method_name, train_scores, val_scores in methods:
            row = evaluate_signal(
                f"{group_name}:{method_name}",
                train_scores,
                val_scores,
                train_arrays,
                val_arrays,
                target_precision,
            )
            detailed[group_name][method_name] = row
            sweep_rows.append(flatten_eval_row(row, group=group_name, method=method_name, n_features=len(feature_names)))

    sweep_rows.sort(
        key=lambda row: (
            float(row["fixed_val_precision"] >= target_precision),
            float(row["fixed_val_recall"]),
            float(row["fixed_val_precision"]),
            float(row["val_ap"]),
        ),
        reverse=True,
    )
    return sweep_rows, detailed



def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    data = np.load(args.cache)
    train_indices = subset_indices_for_limit(np.asarray(data["train_image_ids"]), args.limit_images)
    val_indices = subset_indices_for_limit(np.asarray(data["val_image_ids"]), args.limit_images)
    train_arrays = extract_split_arrays(data, "train", args.limit_images)
    val_arrays = extract_split_arrays(data, "val", args.limit_images)

    coco = load_coco(args.coco_json)
    image_infos = {int(info["id"]): info for info in coco["images"]}
    train_image_ids = set(int(value) for value in np.unique(train_arrays["image_ids"]))
    aspect_stats = compute_gt_aspect_stats(coco, train_image_ids)

    print(
        f"Loaded cache: train={train_arrays['labels'].shape[0]} val={val_arrays['labels'].shape[0]} "
        f"train_pos={int(train_arrays['labels'].sum())} val_pos={int(val_arrays['labels'].sum())}"
    )
    print("Computing image-based interpretable signals...")
    train_signals = compute_image_signals(
        arrays=train_arrays,
        coco_infos=image_infos,
        data_root=args.data_root,
        max_size=args.max_size,
        device=device,
    )
    val_signals = compute_image_signals(
        arrays=val_arrays,
        coco_infos=image_infos,
        data_root=args.data_root,
        max_size=args.max_size,
        device=device,
    )

    train_signals["aspect_ratio_plausibility"] = aspect_plausibility(
        train_arrays["proposal_boxes"], train_arrays["class_ids"], aspect_stats
    )
    val_signals["aspect_ratio_plausibility"] = aspect_plausibility(
        val_arrays["proposal_boxes"], val_arrays["class_ids"], aspect_stats
    )
    train_signals["nms_survivor_density"] = nms_support_density(
        train_arrays["proposal_boxes"], train_arrays["class_ids"], train_arrays["image_ids"]
    )
    val_signals["nms_survivor_density"] = nms_support_density(
        val_arrays["proposal_boxes"], val_arrays["class_ids"], val_arrays["image_ids"]
    )

    train_raw = load_legacy_feature_matrix(data, "train", RAW_IFFT_REFERENCE_FEATURES, train_indices)
    val_raw = load_legacy_feature_matrix(data, "val", RAW_IFFT_REFERENCE_FEATURES, val_indices)
    raw_scorer = fit_train_effect_scorer(train_raw, train_arrays["labels"].astype(bool), method="train_effect_sum")
    train_signals["reference_raw_ifft_recipe"] = raw_scorer.score(train_raw)
    val_signals["reference_raw_ifft_recipe"] = raw_scorer.score(val_raw)

    fusion_features = [
        "score_edge_alignment",
        "boundary_phase_coherence",
        "interior_exterior_texture_contrast",
        "reference_raw_ifft_recipe",
    ]
    train_fusion_matrix = np.stack([train_signals[name] for name in fusion_features], axis=1)
    val_fusion_matrix = np.stack([val_signals[name] for name in fusion_features], axis=1)
    train_effect_fusion, val_effect_fusion = effect_sum_fusion(
        train_fusion_matrix,
        val_fusion_matrix,
        train_arrays["labels"].astype(bool),
    )
    train_logistic_fusion, val_logistic_fusion = logistic_fusion(
        train_fusion_matrix,
        val_fusion_matrix,
        train_arrays["labels"].astype(bool),
    )
    train_signals["fusion_interpretable_effect_sum"] = train_effect_fusion
    val_signals["fusion_interpretable_effect_sum"] = val_effect_fusion
    train_signals["fusion_interpretable_logistic"] = train_logistic_fusion
    val_signals["fusion_interpretable_logistic"] = val_logistic_fusion

    signal_order = [
        "boundary_phase_coherence",
        "interior_exterior_texture_contrast",
        "aspect_ratio_plausibility",
        "multi_scale_saliency_consistency",
        "score_edge_alignment",
        "nms_survivor_density",
        "activation_centroid_consistency",
        "reference_raw_ifft_recipe",
        "fusion_interpretable_effect_sum",
        "fusion_interpretable_logistic",
    ]
    rows = [
        evaluate_signal(
            name,
            train_signals[name],
            val_signals[name],
            train_arrays,
            val_arrays,
            args.target_precision,
        )
        for name in signal_order
    ]
    rows.sort(
        key=lambda row: (
            float(row["fixed_threshold_val"]["precision"] >= args.target_precision),
            float(row["fixed_threshold_val"]["recall"]),
            float(row["val"]["ranking"]["recall_at_precision_0.7"]),
            float(row["val"]["ranking"]["average_precision"]),
        ),
        reverse=True,
    )
    fusion_sweep_rows, fusion_sweep_detail = run_fusion_sweep(
        train_signals=train_signals,
        val_signals=val_signals,
        train_arrays=train_arrays,
        val_arrays=val_arrays,
        data=data,
        train_indices=train_indices,
        val_indices=val_indices,
        target_precision=args.target_precision,
    )

    report = {
        "cache": str(args.cache),
        "target_precision": float(args.target_precision),
        "limit_images": int(args.limit_images),
        "train_count": int(train_arrays["labels"].shape[0]),
        "val_count": int(val_arrays["labels"].shape[0]),
        "train_positive": int(train_arrays["labels"].sum()),
        "val_positive": int(val_arrays["labels"].sum()),
        "raw_ifft_reference_features": RAW_IFFT_REFERENCE_FEATURES,
        "fusion_features": fusion_features,
        "notes": {
            "activation_centroid_consistency": (
                "Non-network proxy for class activation consistency: saliency centroid should align with the box center."
            ),
            "score_edge_alignment": (
                "Rescue-oriented score: boundary edge ratio weighted by one minus the cached class probability."
            ),
        },
        "leaderboard": rows,
        "fusion_sweep": fusion_sweep_detail,
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    write_leaderboard(rows, args.out_dir / "leaderboard.csv")
    write_fusion_sweep(fusion_sweep_rows, args.out_dir / "fusion_sweep.csv")

    print(f"Wrote {args.out_dir / 'report.json'}")
    print("Top signals:")
    for row in rows[:8]:
        fixed = row["fixed_threshold_val"]
        val_rank = row["val"]["ranking"]
        print(
            f"{row['signal']}: fixed P={fixed['precision']:.3f} R={fixed['recall']:.3f} "
            f"sel={fixed['selected']} valAP={val_rank['average_precision']:.3f} "
            f"valR@P0.7={val_rank['recall_at_precision_0.7']:.3f}"
        )
    print("Top fusion sweep:")
    for row in fusion_sweep_rows[:10]:
        print(
            f"{row['group']} {row['method']}: fixed P={float(row['fixed_val_precision']):.3f} "
            f"R={float(row['fixed_val_recall']):.3f} sel={int(row['fixed_val_selected'])} "
            f"valAP={float(row['val_ap']):.3f} valR@P0.7={float(row['val_r_at_p70_oracle']):.3f}"
        )


if __name__ == "__main__":
    main()
