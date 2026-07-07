from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectral_detection_posttrain.core.matching.box_iou import box_iou
from spectral_detection_posttrain.datasets import build_detection_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.eval.rescue_oracle import matched_gt_indices
from spectral_detection_posttrain.experiments.canonical_runner import (
    build_experiment_model,
    prepare_experiment_from_config,
)
from spectral_detection_posttrain.methods.energy_transport import (
    ActionSearchConfig,
    apply_score_action_to_prediction,
    select_min_energy_score_actions,
)
from spectral_detection_posttrain.utils.config import load_config
from spectral_detection_posttrain.utils.io import save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate oracle-bounded low-energy score actions on detector rollouts."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--target-iou", type=float, default=0.75)
    parser.add_argument("--low-quality-iou", type=float, default=0.30)
    parser.add_argument("--max-score-delta", type=float, default=0.20)
    parser.add_argument("--threshold-margin", type=float, default=0.01)
    parser.add_argument("--positive-target-score", type=float, default=None)
    parser.add_argument("--rescue-budget", type=int, default=1)
    parser.add_argument("--demote-low-quality", action="store_true", default=False)
    parser.add_argument("--demote-target-score", type=float, default=None)
    parser.add_argument("--oracle-relabel", action="store_true", default=False)
    parser.add_argument("--only-unmatched-gt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rollout-score-threshold", type=float, default=0.001)
    parser.add_argument("--rollout-nms-threshold", type=float, default=None)
    parser.add_argument("--detections-per-img", type=int, default=300)
    parser.add_argument("--per-class", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--per-size", action="store_true", default=False)
    return parser.parse_args()


@contextmanager
def temporary_roi_postprocess(
    model: torch.nn.Module,
    *,
    score_threshold: float | None = None,
    nms_threshold: float | None = None,
    detections_per_img: int | None = None,
):
    roi_heads = getattr(model, "roi_heads", None)
    if roi_heads is None:
        yield
        return

    updates = {
        "score_thresh": score_threshold,
        "nms_thresh": nms_threshold,
        "detections_per_img": detections_per_img,
    }
    old_values: dict[str, Any] = {}
    for name, value in updates.items():
        if value is None or not hasattr(roi_heads, name):
            continue
        old_values[name] = getattr(roi_heads, name)
        setattr(roi_heads, name, value)
    try:
        yield
    finally:
        for name, value in old_values.items():
            setattr(roi_heads, name, value)


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    *,
    desc: str,
) -> tuple[list[dict], list[dict]]:
    model.eval()
    predictions: list[dict] = []
    targets_out: list[dict] = []
    for images, targets in tqdm(val_loader, desc=desc):
        outputs = model([image.to(device) for image in images])
        predictions.extend(
            [{key: value.detach().cpu() for key, value in output.items()} for output in outputs]
        )
        targets_out.extend(
            [
                {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in target.items()}
                for target in targets
            ]
        )
    return predictions, targets_out


def best_iou_and_gt(
    prediction: dict,
    target: dict,
    *,
    class_aware: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    boxes = prediction.get("boxes", torch.empty((0, 4))).float()
    labels = prediction.get("labels", torch.empty((0,), dtype=torch.long)).long()
    gt_boxes = target.get("boxes", torch.empty((0, 4))).float()
    gt_labels = target.get("labels", torch.empty((0,), dtype=torch.long)).long()
    num_preds = int(boxes.shape[0])
    if num_preds == 0:
        empty_float = torch.empty((0,), dtype=torch.float32)
        empty_long = torch.empty((0,), dtype=torch.long)
        return empty_float, empty_long, empty_long.clone()
    if gt_boxes.numel() == 0:
        return (
            torch.zeros((num_preds,), dtype=torch.float32),
            torch.full((num_preds,), -1, dtype=torch.long),
            torch.zeros((num_preds,), dtype=torch.long),
        )

    ious = box_iou(boxes, gt_boxes)
    if class_aware:
        valid = labels[:, None] == gt_labels[None, :]
    else:
        valid = gt_labels[None, :] > 0
    masked = ious.masked_fill(~valid, -1.0)
    best_iou, best_gt = masked.max(dim=1)
    invalid = best_iou < 0.0
    best_iou = best_iou.clamp(min=0.0)
    best_gt = best_gt.long()
    best_gt[invalid] = -1
    best_labels = torch.zeros((num_preds,), dtype=torch.long)
    valid_gt = best_gt >= 0
    if valid_gt.any():
        best_labels[valid_gt] = gt_labels[best_gt[valid_gt]]
    return best_iou.float(), best_gt, best_labels


def build_unmatched_candidate_mask(
    prediction: dict,
    target: dict,
    gt_indices: torch.Tensor,
    *,
    target_iou: float,
    score_threshold: float,
    only_unmatched_gt: bool,
) -> torch.Tensor:
    candidate = gt_indices >= 0
    if not only_unmatched_gt or not candidate.any():
        return candidate

    matched = matched_gt_indices(
        prediction,
        target,
        iou_threshold=float(target_iou),
        score_threshold=float(score_threshold),
    )
    if not matched:
        return candidate
    matched_tensor = torch.tensor(sorted(matched), dtype=torch.long)
    already_matched = (gt_indices.unsqueeze(1) == matched_tensor.unsqueeze(0)).any(dim=1)
    return candidate & (~already_matched)


def apply_action_search(
    predictions: list[dict],
    targets: list[dict],
    *,
    config: ActionSearchConfig,
    oracle_relabel: bool,
    only_unmatched_gt: bool,
) -> tuple[list[dict], dict[str, float | int]]:
    adjusted: list[dict] = []
    totals: dict[str, float | int] = {
        "num_images": len(predictions),
        "num_candidates": 0,
        "num_rescue_candidates": 0,
        "num_rescued": 0,
        "num_demoted": 0,
        "num_threshold_crossings": 0,
        "num_low_quality_crossings": 0,
        "num_nonzero_actions": 0,
        "sum_abs_score_delta": 0.0,
        "total_action_energy": 0.0,
    }

    for image_idx, (prediction, target) in enumerate(zip(predictions, targets)):
        class_aware = not oracle_relabel
        ious, gt_indices, oracle_labels = best_iou_and_gt(
            prediction,
            target,
            class_aware=class_aware,
        )
        scores = prediction.get("scores", torch.empty((0,), dtype=torch.float32)).float()
        image_indices = torch.full((scores.numel(),), image_idx, dtype=torch.long)
        candidate_mask = build_unmatched_candidate_mask(
            prediction,
            target,
            gt_indices,
            target_iou=float(config.target_iou),
            score_threshold=float(config.score_threshold),
            only_unmatched_gt=only_unmatched_gt,
        )
        low_quality_mask = ious <= float(config.low_quality_iou)
        result = select_min_energy_score_actions(
            scores,
            ious,
            image_indices,
            gt_indices=gt_indices,
            rescue_candidate_mask=candidate_mask,
            low_quality_mask=low_quality_mask,
            config=config,
        )
        adjusted.append(
            apply_score_action_to_prediction(
                prediction,
                result.score_delta,
                labels=oracle_labels if oracle_relabel else None,
                relabel_mask=result.rescue_mask if oracle_relabel else None,
            )
        )

        for key in (
            "num_candidates",
            "num_rescue_candidates",
            "num_rescued",
            "num_demoted",
            "num_threshold_crossings",
            "num_low_quality_crossings",
        ):
            totals[key] = int(totals[key]) + int(result.summary[key])
        totals["num_nonzero_actions"] = int(totals["num_nonzero_actions"]) + int(
            (result.score_delta != 0).sum().item()
        )
        totals["sum_abs_score_delta"] = float(totals["sum_abs_score_delta"]) + float(
            result.score_delta.abs().sum().item()
        )
        totals["total_action_energy"] = float(totals["total_action_energy"]) + float(
            result.score_delta.square().sum().item()
        )

    num_candidates = max(1, int(totals["num_candidates"]))
    totals["mean_abs_score_delta"] = float(totals["sum_abs_score_delta"]) / num_candidates
    totals["mean_action_energy"] = float(totals["total_action_energy"]) / num_candidates
    return adjusted, totals


def metric_delta(after: dict, before: dict) -> dict[str, float]:
    keys = [
        "ap50",
        "ap75",
        "precision",
        "recall",
        "false_positive_rate",
        "ece",
        "num_predictions",
    ]
    delta = {}
    for key in keys:
        if key in after and key in before and after[key] is not None and before[key] is not None:
            delta[key] = float(after[key]) - float(before[key])
    return delta


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config["device"] = args.device
    set_seed(int(config.get("seed", 42)))

    context = prepare_experiment_from_config(
        config,
        args.config,
        args.run_name,
        phase="energy_action_search",
        checkpoint_path=args.checkpoint,
        runs_root=args.runs_root,
    )
    config = context.config
    batch_size = args.batch_size or int(config.get("eval", {}).get("batch_size", 1))
    _, val_loader = build_detection_loaders(
        config,
        limit_train=None,
        limit_val=args.limit_val,
        batch_size=batch_size,
    )
    device = resolve_device(config)
    model = build_experiment_model(context, checkpoint_path=args.checkpoint, device=device, pretrained=False)

    score_threshold = (
        float(args.score_threshold)
        if args.score_threshold is not None
        else float(config.get("matching", {}).get("score_threshold", 0.05))
    )
    num_classes = int(config["model"]["num_classes"])
    metric_kwargs = {
        "iou_threshold": float(config.get("matching", {}).get("iou_threshold", 0.5)),
        "score_threshold": score_threshold,
        "high_conf_threshold": float(config.get("eval", {}).get("high_conf_threshold", 0.7)),
        "per_class": bool(args.per_class),
        "num_classes": num_classes,
        "per_size": bool(args.per_size),
    }

    default_predictions, default_targets = collect_predictions(
        model,
        val_loader,
        device,
        desc="default",
    )
    default_metrics = evaluate_detection_predictions(default_predictions, default_targets, **metric_kwargs)

    with temporary_roi_postprocess(
        model,
        score_threshold=float(args.rollout_score_threshold),
        nms_threshold=args.rollout_nms_threshold,
        detections_per_img=int(args.detections_per_img),
    ):
        rollout_predictions, rollout_targets = collect_predictions(
            model,
            val_loader,
            device,
            desc="rollout",
        )
    rollout_raw_metrics = evaluate_detection_predictions(rollout_predictions, rollout_targets, **metric_kwargs)

    search_config = ActionSearchConfig(
        score_threshold=score_threshold,
        target_iou=float(args.target_iou),
        low_quality_iou=float(args.low_quality_iou),
        max_score_delta=float(args.max_score_delta),
        threshold_margin=float(args.threshold_margin),
        max_rescues_per_image=int(args.rescue_budget),
        positive_target_score=args.positive_target_score,
        demote_low_quality=bool(args.demote_low_quality),
        demote_target_score=args.demote_target_score,
    )
    action_predictions, action_summary = apply_action_search(
        rollout_predictions,
        rollout_targets,
        config=search_config,
        oracle_relabel=bool(args.oracle_relabel),
        only_unmatched_gt=bool(args.only_unmatched_gt),
    )
    action_metrics = evaluate_detection_predictions(action_predictions, rollout_targets, **metric_kwargs)

    result = {
        "config": {
            "score_threshold": score_threshold,
            "target_iou": float(args.target_iou),
            "low_quality_iou": float(args.low_quality_iou),
            "max_score_delta": float(args.max_score_delta),
            "threshold_margin": float(args.threshold_margin),
            "positive_target_score": args.positive_target_score,
            "rescue_budget": int(args.rescue_budget),
            "demote_low_quality": bool(args.demote_low_quality),
            "demote_target_score": args.demote_target_score,
            "oracle_relabel": bool(args.oracle_relabel),
            "only_unmatched_gt": bool(args.only_unmatched_gt),
            "rollout_score_threshold": float(args.rollout_score_threshold),
            "rollout_nms_threshold": args.rollout_nms_threshold,
            "detections_per_img": int(args.detections_per_img),
            "limit_val": args.limit_val,
        },
        "default_metrics": default_metrics,
        "rollout_raw_metrics": rollout_raw_metrics,
        "action_metrics": action_metrics,
        "action_summary": action_summary,
        "delta_action_vs_rollout": metric_delta(action_metrics, rollout_raw_metrics),
        "delta_action_vs_default": metric_delta(action_metrics, default_metrics),
        "delta_rollout_vs_default": metric_delta(rollout_raw_metrics, default_metrics),
    }
    output_path = context.run_dir / "energy_action_search_result.json"
    save_json(result, output_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved result to {output_path}")


if __name__ == "__main__":
    main()
