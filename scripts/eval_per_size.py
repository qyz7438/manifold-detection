"""Evaluate a trained checkpoint with per-object-size AP breakdown.

Uses COCO-style area buckets on the validation set:
- small  : area < 32^2
- medium : 32^2 <= area < 96^2
- large  : area >= 96^2

Buckets are applied to both ground-truth boxes and predicted boxes, which gives a
reasonable proxy for size-specific performance without requiring a full COCO API.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from spectral_detection_posttrain.core.models.build_detector import build_detector
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.utils.io import load_checkpoint, save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


SIZE_BUCKETS = {
    "small": (0.0, 32.0 * 32.0),
    "medium": (32.0 * 32.0, 96.0 * 96.0),
    "large": (96.0 * 96.0, float("inf")),
}


def _box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)


def _filter_by_area(
    predictions: list[dict],
    targets: list[dict],
    lo: float,
    hi: float,
) -> tuple[list[dict], list[dict]]:
    """Return predictions/targets filtered by box area."""
    pred_out, target_out = [], []
    for pred, target in zip(predictions, targets):
        pred_boxes = pred.get("boxes", torch.empty((0, 4)))
        pred_mask = (_box_area(pred_boxes) >= lo) & (_box_area(pred_boxes) < hi)
        pred_out.append({
            "boxes": pred_boxes[pred_mask],
            "scores": pred["scores"][pred_mask] if "scores" in pred else torch.empty((0,)),
            "labels": pred["labels"][pred_mask] if "labels" in pred else torch.empty((0,), dtype=torch.long),
        })

        target_boxes = target.get("boxes", torch.empty((0, 4)))
        target_mask = (_box_area(target_boxes) >= lo) & (_box_area(target_boxes) < hi)
        target_out.append({
            "boxes": target_boxes[target_mask],
            "labels": target["labels"][target_mask] if "labels" in target else torch.empty((0,), dtype=torch.long),
        })
    return pred_out, target_out


def _load_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found in {run_dir}; train with the latest script first.")
    with open(config_path) as f:
        return json.load(f)


def _build_val_loader(config: dict, limit_val: int | None = None):
    dataset = config["data"].get("dataset", "penn_fudan")
    if dataset == "voc":
        from spectral_detection_posttrain.datasets.voc_detection import build_voc_detection_loaders
        _, val_loader = build_voc_detection_loaders(config, limit_val=limit_val)
    elif dataset == "nwpu":
        from spectral_detection_posttrain.datasets.nwpu_vhr10 import build_nwpu_vhr10_loaders
        _, val_loader = build_nwpu_vhr10_loaders(config, limit_val=limit_val)
    else:
        from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
        _, val_loader = build_penn_fudan_loaders(config, limit_val=limit_val)
    return val_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", default="checkpoint_last.pth")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_dir = Path("runs") / args.run_name
    config = _load_config(run_dir)

    if args.device:
        config["device"] = args.device
    if args.batch_size:
        config["eval"]["batch_size"] = args.batch_size

    set_seed(config["seed"])
    device = resolve_device(config)

    model = build_detector(config).to(device)
    ckpt_path = run_dir / args.checkpoint
    load_checkpoint(model, ckpt_path, device)
    model.eval()

    val_loader = _build_val_loader(config, limit_val=args.limit_val)

    predictions, targets_list = [], []
    with torch.no_grad():
        for images, batch_targets in val_loader:
            outputs = model([img.to(device) for img in images])
            predictions.extend([{k: v.detach().cpu() for k, v in output.items()} for output in outputs])
            targets_list.extend([{
                k: v.detach().cpu() if torch.is_tensor(v) else v
                for k, v in t.items()
            } for t in batch_targets])

    iou_threshold = float(config["matching"]["iou_threshold"])
    score_threshold = float(config["matching"]["score_threshold"])
    high_conf_threshold = float(config["eval"]["high_conf_threshold"])

    results = {
        "run_name": args.run_name,
        "checkpoint": str(ckpt_path),
        "device": str(device),
        "overall": evaluate_detection_predictions(
            predictions, targets_list,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            high_conf_threshold=high_conf_threshold,
        ),
        "per_size": {},
    }

    for name, (lo, hi) in SIZE_BUCKETS.items():
        size_pred, size_targets = _filter_by_area(predictions, targets_list, lo, hi)
        results["per_size"][name] = evaluate_detection_predictions(
            size_pred, size_targets,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            high_conf_threshold=high_conf_threshold,
        )
        results["per_size"][name]["lo"] = lo
        results["per_size"][name]["hi"] = hi

    output_path = Path(args.output) if args.output else run_dir / "eval_per_size.json"
    save_json(results, output_path)
    print(json.dumps({k: v for k, v in results.items() if k != "overall" and k != "per_size"}, indent=2))
    print("overall", {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in results["overall"].items() if k in ("ap50", "ap75", "ece")})
    for name, metrics in results["per_size"].items():
        print(f"{name:8s}", {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items() if k in ("ap50", "ap75", "ece", "num_gt", "num_predictions")})


if __name__ == "__main__":
    main()
