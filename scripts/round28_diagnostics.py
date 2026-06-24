"""Run threshold curves + localization diagnostics for one trained group."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.matching.box_iou import box_iou
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import load_checkpoint, save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed

THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]


def _config(afm_type: str, residual_mode: str) -> dict:
    return {
        "seed": 42, "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data": {"root": "./data", "download": True, "max_size": 320, "train_fraction": 0.8, "num_workers": 0},
        "model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn", "pretrained": True,
                  "num_classes": 2, "min_size": 320, "max_size": 320,
                  "afm_channels": 256 if afm_type != "none" else 0,
                  "afm_type": afm_type, "afm_residual_mode": residual_mode},
        "train": {"batch_size": 2},
        "matching": {"iou_threshold": 0.5, "score_threshold": 0.05},
        "eval": {"batch_size": 2, "high_conf_threshold": 0.7},
    }


def _localization_stats(predictions: list[dict], targets: list[dict], score_threshold: float) -> dict:
    matched_ious: list[float] = []
    center_errors: list[float] = []
    size_errors: list[float] = []
    duplicates = 0
    for prediction, target in zip(predictions, targets):
        boxes = prediction.get("boxes", torch.empty((0, 4)))
        scores = prediction.get("scores", torch.empty((0,)))
        keep = scores >= score_threshold
        boxes = boxes[keep]
        gt_boxes = target.get("boxes", torch.empty((0, 4)))
        if len(boxes) == 0 or len(gt_boxes) == 0:
            continue
        ious = box_iou(boxes, gt_boxes)
        best_iou, best_gt = ious.max(dim=1)
        gt_match_counts: dict[int, int] = {}
        for pred_idx, iou in enumerate(best_iou.tolist()):
            if iou < 0.5:
                continue
            gt_idx = int(best_gt[pred_idx].item())
            gt_match_counts[gt_idx] = gt_match_counts.get(gt_idx, 0) + 1
            pred_box = boxes[pred_idx]
            gt_box = gt_boxes[gt_idx]
            pred_center = torch.stack([(pred_box[0] + pred_box[2]) / 2, (pred_box[1] + pred_box[3]) / 2])
            gt_center = torch.stack([(gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2])
            pred_size = torch.stack([(pred_box[2] - pred_box[0]).clamp_min(1), (pred_box[3] - pred_box[1]).clamp_min(1)])
            gt_size = torch.stack([(gt_box[2] - gt_box[0]).clamp_min(1), (gt_box[3] - gt_box[1]).clamp_min(1)])
            center_errors.append(float(torch.norm(pred_center - gt_center).item()))
            size_errors.append(float((pred_size - gt_size).abs().mean().item()))
            matched_ious.append(float(iou))
        duplicates += sum(max(0, count - 1) for count in gt_match_counts.values())

    def mean(values: list[float]) -> float:
        return float(sum(values) / max(1, len(values)))

    return {"matched_iou_mean": mean(matched_ious), "matched_iou_count": len(matched_ious),
            "center_error_mean": mean(center_errors), "size_error_mean": mean(size_errors),
            "duplicate_predictions": duplicates}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--afm-type", required=True, choices=["none", "old", "identity"])
    parser.add_argument("--afm-residual-mode", default="current", choices=["current", "delta", "norm_delta"])
    args = parser.parse_args()

    set_seed(42)
    config = _config(args.afm_type, args.afm_residual_mode)
    device = resolve_device(config)
    model = build_detector(config).to(device)
    load_checkpoint(model, Path("runs") / args.run_name / "checkpoint_last.pth", device)
    model.eval()
    _, val_loader = build_penn_fudan_loaders(config)
    predictions, targets = [], []
    with torch.no_grad():
        for images, batch_targets in val_loader:
            outputs = model([img.to(device) for img in images])
            predictions.extend([{k: v.detach().cpu() for k, v in output.items()} for output in outputs])
            targets.extend([{k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in t.items()} for t in batch_targets])

    threshold_curve = {}
    for threshold in THRESHOLDS:
        metrics = evaluate_detection_predictions(predictions, targets, score_threshold=threshold)
        metrics.update(_localization_stats(predictions, targets, score_threshold=threshold))
        threshold_curve[str(threshold)] = metrics
    save_json(threshold_curve, Path("runs") / args.run_name / "round28_diagnostics.json")
    print(json.dumps(threshold_curve, indent=2))


if __name__ == "__main__":
    main()
