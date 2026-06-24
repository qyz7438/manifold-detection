from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectral_detection_posttrain.matching.box_iou import box_iou
from spectral_detection_posttrain.matching.pred_gt_matcher import match_predictions_to_gt
from spectral_detection_posttrain.models.bbox_adapter import install_residual_bbox_adapter
from spectral_detection_posttrain.rlvr.action_verifier import decode_box_actions
from spectral_detection_posttrain.rlvr.confidence_rescue import match_boxes_to_target_boxes
from spectral_detection_posttrain.rlvr.roi_policy_loss import extract_roi_head_outputs_for_boxes, resize_boxes_to_image

from scripts.round2129_nwpu_posttrain_smoke import (
    NUM_CLASSES,
    build_chain_rescue_candidate_mask,
    build_loaders,
    build_nwpu_model,
    class_box_deltas,
    configure_detector_rollout,
    matched_label_probabilities,
    proposal_iou_for_scores,
    select_rpn_proposals_for_images,
)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def candidate_fate_masks(
    candidate_boxes: torch.Tensor,
    candidate_labels: torch.Tensor,
    candidate_scores: torch.Tensor,
    target: dict,
    final_prediction: dict,
    *,
    score_threshold: float,
    candidate_to_final_iou: float,
    tp_iou_threshold: float,
) -> dict[str, float]:
    candidate_boxes = candidate_boxes.detach().cpu().float()
    candidate_labels = candidate_labels.detach().cpu().long()
    candidate_scores = candidate_scores.detach().cpu().float()
    final_boxes = final_prediction.get("boxes", torch.empty((0, 4))).detach().cpu().float()
    final_labels = final_prediction.get("labels", torch.empty((0,), dtype=torch.long)).detach().cpu().long()
    final_scores = final_prediction.get("scores", torch.empty((0,))).detach().cpu().float()
    candidate_count = int(candidate_boxes.shape[0])
    score_ge = candidate_scores >= float(score_threshold)
    entered = torch.zeros((candidate_count,), dtype=torch.bool)
    if final_boxes.numel() > 0:
        final_keep = final_scores >= float(score_threshold)
        if final_keep.any():
            ious = box_iou(candidate_boxes, final_boxes[final_keep])
            same_label = candidate_labels.unsqueeze(1) == final_labels[final_keep].unsqueeze(0)
            candidate_final_iou = torch.where(same_label, ious, torch.full_like(ious, -1.0))
            entered = candidate_final_iou.max(dim=1).values >= float(candidate_to_final_iou)

    matched = match_predictions_to_gt(
        final_prediction,
        target,
        iou_threshold=float(tp_iou_threshold),
        score_threshold=float(score_threshold),
    )
    tp_final_indices = torch.tensor(
        [int(item["pred_index"]) for item in matched["matches"]],
        dtype=torch.long,
    )
    ap75_tp = torch.zeros((candidate_count,), dtype=torch.bool)
    if tp_final_indices.numel() > 0:
        tp_boxes = final_boxes[tp_final_indices]
        tp_labels = final_labels[tp_final_indices]
        ious = box_iou(candidate_boxes, tp_boxes)
        same_label = candidate_labels.unsqueeze(1) == tp_labels.unsqueeze(0)
        candidate_tp_iou = torch.where(same_label, ious, torch.full_like(ious, -1.0))
        ap75_tp = candidate_tp_iou.max(dim=1).values >= float(candidate_to_final_iou)
    return {"score_ge": score_ge, "entered": entered, "ap75_tp": ap75_tp}


def summarize_candidate_fate(
    candidate_boxes: torch.Tensor,
    candidate_labels: torch.Tensor,
    candidate_scores: torch.Tensor,
    target: dict,
    final_prediction: dict,
    *,
    score_threshold: float,
    candidate_to_final_iou: float,
    tp_iou_threshold: float,
) -> dict[str, float]:
    candidate_count = int(candidate_boxes.shape[0])
    if candidate_count == 0:
        return {
            "candidate_count": 0,
            "score_ge_threshold_count": 0,
            "entered_final_count": 0,
            "ap75_tp_count": 0,
            "entered_but_not_tp_count": 0,
            "score_ge_threshold_but_not_entered_count": 0,
            "entered_final_rate": 0.0,
            "ap75_tp_rate": 0.0,
        }
    masks = candidate_fate_masks(
        candidate_boxes,
        candidate_labels,
        candidate_scores,
        target,
        final_prediction,
        score_threshold=score_threshold,
        candidate_to_final_iou=candidate_to_final_iou,
        tp_iou_threshold=tp_iou_threshold,
    )
    score_ge = masks["score_ge"]
    entered = masks["entered"]
    ap75_tp = masks["ap75_tp"]

    entered_but_not_tp = entered & (~ap75_tp)
    score_ge_but_not_entered = score_ge & (~entered)
    return {
        "candidate_count": candidate_count,
        "score_ge_threshold_count": int(score_ge.sum().item()),
        "entered_final_count": int(entered.sum().item()),
        "ap75_tp_count": int(ap75_tp.sum().item()),
        "entered_but_not_tp_count": int(entered_but_not_tp.sum().item()),
        "score_ge_threshold_but_not_entered_count": int(score_ge_but_not_entered.sum().item()),
        "entered_final_rate": float(entered.float().mean().item()),
        "ap75_tp_rate": float(ap75_tp.float().mean().item()),
    }


def classify_candidate_suppressors(
    candidate_boxes: torch.Tensor,
    candidate_labels: torch.Tensor,
    candidate_scores: torch.Tensor,
    candidate_gt_indices: torch.Tensor,
    target: dict,
    final_prediction: dict,
    *,
    score_threshold: float,
    candidate_to_final_iou: float,
    tp_iou_threshold: float,
    nms_iou_threshold: float,
) -> dict[str, float]:
    candidate_boxes = candidate_boxes.detach().cpu().float()
    candidate_labels = candidate_labels.detach().cpu().long()
    candidate_scores = candidate_scores.detach().cpu().float()
    candidate_gt_indices = candidate_gt_indices.detach().cpu().long()
    final_boxes = final_prediction.get("boxes", torch.empty((0, 4))).detach().cpu().float()
    final_labels = final_prediction.get("labels", torch.empty((0,), dtype=torch.long)).detach().cpu().long()
    final_scores = final_prediction.get("scores", torch.empty((0,))).detach().cpu().float()
    gt_boxes = target.get("boxes", torch.empty((0, 4))).detach().cpu().float()
    gt_labels = target.get("labels", torch.empty((0,), dtype=torch.long)).detach().cpu().long()

    masks = candidate_fate_masks(
        candidate_boxes,
        candidate_labels,
        candidate_scores,
        target,
        final_prediction,
        score_threshold=score_threshold,
        candidate_to_final_iou=candidate_to_final_iou,
        tp_iou_threshold=tp_iou_threshold,
    )
    blocked = masks["score_ge"] & (~masks["entered"])
    report = {
        "blocked_candidate_count": int(blocked.sum().item()),
        "same_gt_worse_duplicate_count": 0,
        "same_gt_better_duplicate_count": 0,
        "same_class_nms_overlap_count": 0,
        "class_mismatch_overlap_count": 0,
        "not_decoded_close_enough_count": 0,
        "no_same_class_overlap_count": 0,
        "mean_candidate_gt_iou": 0.0,
        "mean_suppressor_gt_iou": 0.0,
        "mean_suppressor_score": 0.0,
    }
    if not blocked.any():
        return report

    candidate_gt_iou = torch.zeros((candidate_boxes.shape[0],), dtype=torch.float32)
    if gt_boxes.numel() > 0:
        ious_to_gt = box_iou(candidate_boxes, gt_boxes)
        candidate_gt_iou = ious_to_gt.max(dim=1).values
    report["not_decoded_close_enough_count"] = int(
        (blocked & (candidate_gt_iou < float(tp_iou_threshold))).sum().item()
    )

    suppressor_gt_ious = []
    suppressor_scores = []
    for idx in torch.nonzero(blocked, as_tuple=False).flatten().tolist():
        if final_boxes.numel() == 0:
            report["no_same_class_overlap_count"] += 1
            continue
        overlaps = box_iou(candidate_boxes[idx].unsqueeze(0), final_boxes).squeeze(0)
        same_class = final_labels == candidate_labels[idx]
        same_class_overlaps = torch.where(same_class, overlaps, torch.full_like(overlaps, -1.0))
        best_same_iou, best_same_idx = same_class_overlaps.max(dim=0)
        any_class_best_iou, any_class_best_idx = overlaps.max(dim=0)
        if best_same_iou.item() < float(nms_iou_threshold):
            if any_class_best_iou.item() >= float(nms_iou_threshold):
                report["class_mismatch_overlap_count"] += 1
            else:
                report["no_same_class_overlap_count"] += 1
            continue

        report["same_class_nms_overlap_count"] += 1
        suppressor_idx = int(best_same_idx.item())
        suppressor_scores.append(float(final_scores[suppressor_idx].item()))
        gt_idx = int(candidate_gt_indices[idx].item())
        suppressor_gt_iou = 0.0
        if 0 <= gt_idx < int(gt_boxes.shape[0]):
            suppressor_gt_iou = float(
                box_iou(final_boxes[suppressor_idx].unsqueeze(0), gt_boxes[gt_idx].unsqueeze(0)).item()
            )
            suppressor_gt_ious.append(suppressor_gt_iou)
            same_gt = int(final_labels[suppressor_idx].item()) == int(gt_labels[gt_idx].item())
        else:
            same_gt = False
        if same_gt and suppressor_gt_iou < float(candidate_gt_iou[idx].item()):
            report["same_gt_worse_duplicate_count"] += 1
        elif same_gt:
            report["same_gt_better_duplicate_count"] += 1

    blocked_iou = candidate_gt_iou[blocked]
    report["mean_candidate_gt_iou"] = float(blocked_iou.mean().item()) if blocked_iou.numel() else 0.0
    report["mean_suppressor_gt_iou"] = (
        float(sum(suppressor_gt_ious) / len(suppressor_gt_ious)) if suppressor_gt_ious else 0.0
    )
    report["mean_suppressor_score"] = float(sum(suppressor_scores) / len(suppressor_scores)) if suppressor_scores else 0.0
    return report


def load_round_args(run_dir: Path) -> SimpleNamespace:
    defaults = {
        "limit_train": 100000,
        "limit_val": 100000,
        "batch_size": 1,
        "max_proposals": 300,
        "rollout_score_threshold": 0.001,
        "rollout_detections_per_img": 300,
        "score_threshold": 0.05,
        "rescue_mode": True,
        "rescue_low_conf_max": 0.5,
        "rescue_high_iou_min": 0.75,
        "chain_topk_per_gt": 1,
        "cls_adapter_scale": 0.25,
    }
    config_path = run_dir / "round_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    defaults.update(config)
    defaults["run_name"] = config.get("run_name", run_dir.name)
    defaults["rescue_mode"] = True
    defaults["proposal_source"] = "rpn"
    defaults["rescue_positive_filter"] = "ap75_misses"
    defaults["limit_train"] = 100000
    defaults["limit_val"] = int(defaults.get("limit_val", 100000))
    defaults["batch_size"] = 1
    defaults["max_proposals"] = int(defaults.get("max_proposals", 300))
    defaults["rollout_score_threshold"] = float(defaults.get("rollout_score_threshold", 0.001))
    defaults["rollout_detections_per_img"] = int(defaults.get("rollout_detections_per_img", 300))
    defaults["score_threshold"] = float(defaults.get("score_threshold", 0.05))
    defaults["rescue_low_conf_max"] = float(defaults.get("rescue_low_conf_max", 0.5))
    defaults["rescue_high_iou_min"] = float(defaults.get("rescue_high_iou_min", 0.75))
    defaults["chain_topk_per_gt"] = int(defaults.get("chain_topk_per_gt", 1))
    return SimpleNamespace(**defaults)


def load_posttrain_model(run_dir: Path, device: torch.device, args: SimpleNamespace, checkpoint_name: str):
    model = build_nwpu_model(device)
    configure_detector_rollout(
        model,
        score_threshold=float(args.rollout_score_threshold),
        detections_per_img=int(args.rollout_detections_per_img),
    )
    install_residual_bbox_adapter(
        model,
        hidden_dim=128,
        scale=1.0,
        enable_cls_adapter=True,
        cls_scale=float(args.cls_adapter_scale),
    )
    checkpoint = torch.load(run_dir / checkpoint_name, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def model_candidate_outputs(
    model,
    image: torch.Tensor,
    proposals: torch.Tensor,
    labels: torch.Tensor,
    target_boxes: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if proposals.numel() == 0:
        return {
            "scores": torch.empty((0,), dtype=torch.float32),
            "decoded_boxes": torch.empty((0, 4), dtype=torch.float32),
            "iou": torch.empty((0,), dtype=torch.float32),
        }
    class_logits, box_regression, scaled_boxes, transformed_sizes = extract_roi_head_outputs_for_boxes(
        model,
        [image.to(device)],
        [proposals],
    )
    scores = matched_label_probabilities(class_logits, labels.to(device))
    decoded = decode_box_actions(
        scaled_boxes[0],
        class_box_deltas(box_regression, labels.to(device), NUM_CLASSES).unsqueeze(1),
        tuple(transformed_sizes[0]),
    ).squeeze(1)
    decoded_original = resize_boxes_to_image(
        decoded,
        tuple(transformed_sizes[0]),
        tuple(image.shape[-2:]),
    )
    scaled_target_boxes = resize_boxes_to_image(
        target_boxes.to(device),
        tuple(image.shape[-2:]),
        tuple(transformed_sizes[0]),
    )
    iou = proposal_iou_for_scores(decoded, scaled_target_boxes)
    return {
        "scores": scores.detach().cpu(),
        "decoded_boxes": decoded_original.detach().cpu(),
        "iou": iou.detach().cpu(),
    }


def add_prefixed_totals(totals: dict[str, float], prefix: str, report: dict[str, float]) -> None:
    for key, value in report.items():
        if isinstance(value, (int, float)):
            totals[f"{prefix}_{key}"] = totals.get(f"{prefix}_{key}", 0.0) + float(value)


def trace_shared_candidates(
    *,
    baseline_model,
    models: dict[str, object],
    val_loader,
    args: SimpleNamespace,
    device: torch.device,
    score_threshold: float,
    candidate_to_final_iou: float,
    tp_iou_threshold: float,
) -> dict:
    totals: dict[str, float] = {
        "image_count": 0,
        "candidate_count": 0,
        "unique_gt_count": 0,
    }
    per_image: list[dict] = []
    model_names = ["baseline", *models.keys()]
    all_models = {"baseline": baseline_model, **models}

    for images, targets in val_loader:
        image = images[0]
        target = targets[0]
        device_image = image.to(device)
        with torch.no_grad():
            rollout = baseline_model([device_image])[0]
            proposals = select_rpn_proposals_for_images(baseline_model, [image], args, device)[0]
        if proposals.numel() == 0:
            continue
        gt_boxes = target.get("boxes", torch.empty((0, 4))).to(device)
        gt_labels = target.get("labels", torch.empty((0,), dtype=torch.long)).to(device)
        baseline_logits, _, scaled_boxes, transformed_sizes = extract_roi_head_outputs_for_boxes(
            baseline_model,
            [device_image],
            [proposals],
        )
        scaled_gt_boxes = resize_boxes_to_image(
            gt_boxes,
            tuple(image.shape[-2:]),
            tuple(transformed_sizes[0]),
        )
        best_iou, best_labels, target_boxes = match_boxes_to_target_boxes(
            scaled_boxes[0],
            scaled_gt_boxes,
            gt_labels,
        )
        baseline_probs = matched_label_probabilities(baseline_logits, best_labels)
        positive_candidate_mask = torch.ones_like(best_labels, dtype=torch.bool, device=device)
        best_gt_indices = torch.zeros_like(best_labels, dtype=torch.long, device=device)
        if gt_boxes.numel() > 0:
            original_iou = box_iou(proposals.to(device), gt_boxes.float())
            _, best_gt_indices = original_iou.max(dim=1)
            target_cpu = {
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in target.items()
            }
            rollout_cpu = {
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in rollout.items()
            }
            from spectral_detection_posttrain.eval.rescue_oracle import unmatched_gt_candidate_mask

            positive_candidate_mask = unmatched_gt_candidate_mask(
                rollout_cpu,
                target_cpu,
                best_gt_indices.detach().cpu(),
                torch.ones((proposals.shape[0],), dtype=torch.bool),
                iou_threshold=float(tp_iou_threshold),
                score_threshold=float(score_threshold),
            ).to(device)
        candidate_mask = build_chain_rescue_candidate_mask(
            best_iou,
            best_labels,
            best_gt_indices,
            baseline_probs,
            positive_candidate_mask,
            low_conf_max=float(args.rescue_low_conf_max),
            high_iou_min=float(args.rescue_high_iou_min),
            topk_per_gt=int(args.chain_topk_per_gt),
        )
        if not candidate_mask.any():
            continue
        candidate_proposals = proposals[candidate_mask.detach().cpu()]
        candidate_labels = best_labels[candidate_mask].detach().cpu()
        candidate_gt_indices = best_gt_indices[candidate_mask].detach().cpu()
        candidate_target_boxes = target_boxes[candidate_mask].detach().cpu()
        image_record = {
            "image_id": int(target.get("image_id", torch.tensor([-1]))[0]),
            "candidate_count": int(candidate_proposals.shape[0]),
            "unique_gt_count": int(candidate_gt_indices.unique().numel()),
            "models": {},
        }
        totals["image_count"] += 1
        totals["candidate_count"] += int(candidate_proposals.shape[0])
        totals["unique_gt_count"] += int(candidate_gt_indices.unique().numel())
        model_masks = {}
        for name in model_names:
            model = all_models[name]
            with torch.no_grad():
                final_prediction = model([device_image])[0]
                outputs = model_candidate_outputs(
                    model,
                    image,
                    candidate_proposals,
                    candidate_labels,
                    candidate_target_boxes,
                    device=device,
                )
            fate = summarize_candidate_fate(
                outputs["decoded_boxes"],
                candidate_labels,
                outputs["scores"],
                target,
                final_prediction,
                score_threshold=float(score_threshold),
                candidate_to_final_iou=float(candidate_to_final_iou),
                tp_iou_threshold=float(tp_iou_threshold),
            )
            masks = candidate_fate_masks(
                outputs["decoded_boxes"],
                candidate_labels,
                outputs["scores"],
                target,
                final_prediction,
                score_threshold=float(score_threshold),
                candidate_to_final_iou=float(candidate_to_final_iou),
                tp_iou_threshold=float(tp_iou_threshold),
            )
            suppressors = classify_candidate_suppressors(
                outputs["decoded_boxes"],
                candidate_labels,
                outputs["scores"],
                candidate_gt_indices,
                target,
                final_prediction,
                score_threshold=float(score_threshold),
                candidate_to_final_iou=float(candidate_to_final_iou),
                tp_iou_threshold=float(tp_iou_threshold),
                nms_iou_threshold=0.5,
            )
            model_masks[name] = masks
            image_record["models"][name] = {
                **fate,
                **suppressors,
                "score_mean": float(outputs["scores"].mean().item()) if outputs["scores"].numel() else 0.0,
                "iou_mean": float(outputs["iou"].mean().item()) if outputs["iou"].numel() else 0.0,
            }
            add_prefixed_totals(totals, name, image_record["models"][name])

        base_masks = model_masks["baseline"]
        for name in models:
            masks = model_masks[name]
            transition_prefix = f"{name}_transition"
            rescued_to_entered = (~base_masks["entered"]) & masks["entered"]
            rescued_to_tp = (~base_masks["ap75_tp"]) & masks["ap75_tp"]
            lost_entered = base_masks["entered"] & (~masks["entered"])
            lost_tp = base_masks["ap75_tp"] & (~masks["ap75_tp"])
            totals[f"{transition_prefix}_rescued_to_entered_count"] = totals.get(
                f"{transition_prefix}_rescued_to_entered_count", 0.0
            ) + float(rescued_to_entered.sum().item())
            totals[f"{transition_prefix}_rescued_to_tp_count"] = totals.get(
                f"{transition_prefix}_rescued_to_tp_count", 0.0
            ) + float(rescued_to_tp.sum().item())
            totals[f"{transition_prefix}_lost_entered_count"] = totals.get(
                f"{transition_prefix}_lost_entered_count", 0.0
            ) + float(lost_entered.sum().item())
            totals[f"{transition_prefix}_lost_tp_count"] = totals.get(
                f"{transition_prefix}_lost_tp_count", 0.0
            ) + float(lost_tp.sum().item())
        per_image.append(image_record)

    candidate_count = max(1.0, totals["candidate_count"])
    for name in model_names:
        for key in ["entered_final_count", "ap75_tp_count", "score_ge_threshold_but_not_entered_count"]:
            totals[f"{name}_{key}_rate_global"] = totals.get(f"{name}_{key}", 0.0) / candidate_count
        for key in ["score_mean", "iou_mean"]:
            totals[f"{name}_{key}_mean"] = totals.get(f"{name}_{key}", 0.0) / max(1.0, totals["image_count"])
    for name in models:
        prefix = f"{name}_transition"
        for key in ["rescued_to_entered_count", "rescued_to_tp_count", "lost_entered_count", "lost_tp_count"]:
            totals[f"{prefix}_{key}_rate_global"] = totals.get(f"{prefix}_{key}", 0.0) / candidate_count
    return {"summary": totals, "per_image": per_image}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace shared RPN chain candidate fate across checkpoints.")
    parser.add_argument("--output", default="runs/chain_fate_trace/report.json")
    parser.add_argument("--reference-run", default="runs/round2170_rpn_chain_joint")
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--checkpoint-name", default="checkpoint_best.pth")
    parser.add_argument("--candidate-to-final-iou", type=float, default=0.9)
    parser.add_argument("--tp-iou-threshold", type=float, default=0.75)
    parser.add_argument(
        "--runs",
        nargs="+",
        default=[
            "bbox=runs/round2167_rpn_chain_bbox_only",
            "cls=runs/round2169_rpn_chain_cls_margin_bg_only",
            "joint=runs/round2170_rpn_chain_joint",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference_run = Path(args.reference_run)
    round_args = load_round_args(reference_run)
    if args.limit_val is not None:
        round_args.limit_val = int(args.limit_val)
    _, val_loader = build_loaders(round_args)
    baseline_model = build_nwpu_model(device)
    configure_detector_rollout(
        baseline_model,
        score_threshold=float(round_args.rollout_score_threshold),
        detections_per_img=int(round_args.rollout_detections_per_img),
    )
    baseline_model.eval()
    for parameter in baseline_model.parameters():
        parameter.requires_grad = False

    models = {}
    run_dirs = {}
    for spec in args.runs:
        if "=" not in spec:
            raise ValueError(f"Run spec must be name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        run_dir = Path(path)
        run_args = load_round_args(run_dir)
        models[name] = load_posttrain_model(run_dir, device, run_args, str(args.checkpoint_name))
        run_dirs[name] = str(run_dir)

    report = trace_shared_candidates(
        baseline_model=baseline_model,
        models=models,
        val_loader=val_loader,
        args=round_args,
        device=device,
        score_threshold=float(round_args.score_threshold),
        candidate_to_final_iou=float(args.candidate_to_final_iou),
        tp_iou_threshold=float(args.tp_iou_threshold),
    )
    report["metadata"] = {
        "reference_run": str(reference_run),
        "runs": run_dirs,
        "checkpoint_name": str(args.checkpoint_name),
        "limit_val": int(round_args.limit_val),
        "candidate_to_final_iou": float(args.candidate_to_final_iou),
        "tp_iou_threshold": float(args.tp_iou_threshold),
    }
    save_json(report, Path(args.output))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
