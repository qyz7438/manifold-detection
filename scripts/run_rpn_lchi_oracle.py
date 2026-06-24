from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou
from torchvision.ops.boxes import clip_boxes_to_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import round2129_nwpu_posttrain_smoke as round2129
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.eval.rescue_oracle import matched_gt_indices
from spectral_detection_posttrain.rlvr.roi_policy_loss import resize_boxes_to_image
from spectral_detection_posttrain.utils.io import save_json
from spectral_detection_posttrain.utils.seed import set_seed


@contextmanager
def _temporary_argv(argv: list[str]):
    old_argv = sys.argv
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = old_argv


def parse_args():
    parser = argparse.ArgumentParser(description="RPN pre-ROI-NMS LC-HI oracle smoke for NWPU.")
    parser.add_argument("--run-name", default="round2160_rpn_lchi_oracle")
    parser.add_argument("--limit-train", type=int, default=100000)
    parser.add_argument("--limit-val", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-rpn-proposals", type=int, default=300)
    parser.add_argument("--rollout-score-threshold", type=float, default=0.001)
    parser.add_argument("--rollout-detections-per-img", type=int, default=300)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--low-conf-max", type=float, default=0.5)
    parser.add_argument("--high-iou-min", type=float, default=0.75)
    parser.add_argument("--oracle-scores", type=float, nargs="+", default=[0.06, 0.1, 0.2, 0.5, 0.9])
    return parser.parse_args()


def make_round_args(script_args):
    with _temporary_argv(["round2129_nwpu_posttrain_smoke.py"]):
        args = round2129.parse_args()
    args.limit_train = int(script_args.limit_train)
    args.limit_val = int(script_args.limit_val)
    args.batch_size = int(script_args.batch_size)
    args.rollout_score_threshold = float(script_args.rollout_score_threshold)
    args.rollout_detections_per_img = int(script_args.rollout_detections_per_img)
    args.score_threshold = float(script_args.score_threshold)
    args.rescue_mode = True
    return args


def append_oracle_detections(prediction: dict, boxes: torch.Tensor, labels: torch.Tensor, score: float) -> dict:
    out = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in prediction.items()
    }
    if boxes.numel() == 0:
        return out
    out["boxes"] = torch.cat([out["boxes"], boxes.detach().cpu().float()], dim=0)
    out["labels"] = torch.cat([out["labels"], labels.detach().cpu().long()], dim=0)
    out["scores"] = torch.cat(
        [
            out["scores"],
            torch.full((boxes.shape[0],), float(score), dtype=out["scores"].dtype),
        ],
        dim=0,
    )
    return out


def best_candidate_per_gt(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    gt_indices: torch.Tensor,
    ious: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0:
        return boxes, labels
    chosen = []
    for gt_idx in sorted(set(int(item) for item in gt_indices.detach().cpu().tolist())):
        positions = torch.nonzero(gt_indices == int(gt_idx), as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        local = positions[torch.argmax(ious[positions])]
        chosen.append(int(local.item()))
    if not chosen:
        return boxes.new_empty((0, 4)), labels.new_empty((0,), dtype=torch.long)
    chosen_tensor = torch.tensor(chosen, dtype=torch.long, device=boxes.device)
    return boxes[chosen_tensor], labels[chosen_tensor]


@torch.no_grad()
def collect_rpn_candidates(model, image: torch.Tensor, target: dict, prediction: dict, args, device: torch.device):
    original_size = tuple(image.shape[-2:])
    transformed, _ = model.transform([image.to(device)], None)
    features = model.backbone(transformed.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    proposals, _ = model.rpn(transformed, features, None)
    proposals_scaled = proposals[0][: int(args.max_rpn_proposals)]
    if proposals_scaled.numel() == 0:
        empty_boxes = torch.empty((0, 4), dtype=torch.float32)
        empty_labels = torch.empty((0,), dtype=torch.long)
        empty_gt = torch.empty((0,), dtype=torch.long)
        empty_iou = torch.empty((0,), dtype=torch.float32)
        return empty_boxes, empty_labels, empty_gt, empty_iou, {"rpn_proposals": 0, "rpn_lchi": 0, "rpn_unmatched_lchi": 0}

    pooled = model.roi_heads.box_roi_pool(features, [proposals_scaled], transformed.image_sizes)
    box_features = model.roi_heads.box_head(pooled)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)
    decoded_scaled = model.roi_heads.box_coder.decode(box_regression, [proposals_scaled])
    num_props, num_classes = decoded_scaled.shape[:2]
    decoded_scaled = clip_boxes_to_image(decoded_scaled.reshape(-1, 4), transformed.image_sizes[0]).reshape(
        num_props,
        num_classes,
        4,
    )
    decoded_orig = resize_boxes_to_image(
        decoded_scaled.reshape(-1, 4),
        tuple(transformed.image_sizes[0]),
        original_size,
    ).reshape(num_props, num_classes, 4)

    gt_boxes = target.get("boxes", torch.empty((0, 4))).to(device).float()
    gt_labels = target.get("labels", torch.empty((0,), dtype=torch.long)).to(device).long()
    if gt_boxes.numel() == 0:
        empty_boxes = torch.empty((0, 4), dtype=torch.float32)
        empty_labels = torch.empty((0,), dtype=torch.long)
        empty_gt = torch.empty((0,), dtype=torch.long)
        empty_iou = torch.empty((0,), dtype=torch.float32)
        return empty_boxes, empty_labels, empty_gt, empty_iou, {"rpn_proposals": int(num_props), "rpn_lchi": 0, "rpn_unmatched_lchi": 0}

    best_iou = torch.zeros((num_props,), dtype=torch.float32, device=device)
    best_gt = torch.zeros((num_props,), dtype=torch.long, device=device)
    best_labels = torch.zeros((num_props,), dtype=torch.long, device=device)
    for gt_idx, gt_label in enumerate(gt_labels.tolist()):
        if int(gt_label) <= 0 or int(gt_label) >= num_classes:
            continue
        decoded_for_label = decoded_orig[:, int(gt_label), :].to(device)
        ious = box_iou(decoded_for_label, gt_boxes[gt_idx : gt_idx + 1]).squeeze(1)
        update = ious > best_iou
        best_iou[update] = ious[update]
        best_gt[update] = int(gt_idx)
        best_labels[update] = int(gt_label)

    probs = F.softmax(class_logits, dim=1)
    rows = torch.arange(num_props, device=device)
    label_probs = probs[rows, best_labels.clamp(min=0, max=num_classes - 1)]
    lchi = (
        (label_probs <= float(args.low_conf_max))
        & (best_iou >= float(args.high_iou_min))
        & (best_labels > 0)
    )
    matched_gt = matched_gt_indices(
        prediction,
        {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in target.items()},
        iou_threshold=0.75,
        score_threshold=float(args.score_threshold),
    )
    if matched_gt:
        matched_tensor = torch.tensor(sorted(matched_gt), dtype=torch.long, device=device)
        already_matched = (best_gt.unsqueeze(1) == matched_tensor.unsqueeze(0)).any(dim=1)
    else:
        already_matched = torch.zeros_like(lchi)
    unmatched_lchi = lchi & (~already_matched)
    selected_boxes = decoded_orig[rows, best_labels.clamp(min=0, max=num_classes - 1)].detach().cpu()
    selected_labels = best_labels.detach().cpu()
    return (
        selected_boxes[unmatched_lchi.detach().cpu()],
        selected_labels[unmatched_lchi.detach().cpu()],
        best_gt.detach().cpu()[unmatched_lchi.detach().cpu()],
        best_iou.detach().cpu()[unmatched_lchi.detach().cpu()],
        {
            "rpn_proposals": int(num_props),
            "rpn_lchi": int(lchi.sum().item()),
            "rpn_unmatched_lchi": int(unmatched_lchi.sum().item()),
            "rpn_unmatched_unique_gt": int(len(set(best_gt[unmatched_lchi].detach().cpu().tolist()))),
        },
    )


@torch.no_grad()
def main():
    script_args = parse_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path("runs") / script_args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    args = make_round_args(script_args)
    args.max_rpn_proposals = int(script_args.max_rpn_proposals)
    args.low_conf_max = float(script_args.low_conf_max)
    args.high_iou_min = float(script_args.high_iou_min)

    _, val_loader = round2129.build_loaders(args)
    model = round2129.build_nwpu_model(device)
    round2129.configure_detector_rollout(
        model,
        score_threshold=float(args.rollout_score_threshold),
        detections_per_img=int(args.rollout_detections_per_img),
    )
    model.eval()

    predictions = []
    targets = []
    candidate_infos = []
    totals = {"rpn_proposals": 0, "rpn_lchi": 0, "rpn_unmatched_lchi": 0, "rpn_unmatched_unique_gt": 0}
    for images, batch_targets in val_loader:
        outputs = model([image.to(device) for image in images])
        for image, target, output in zip(images, batch_targets, outputs):
            prediction = {key: value.detach().cpu() for key, value in output.items()}
            boxes, labels, gt_indices, ious, stats = collect_rpn_candidates(model, image, target, prediction, args, device)
            best_boxes, best_labels = best_candidate_per_gt(boxes, labels, gt_indices, ious)
            predictions.append(prediction)
            targets.append({key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in target.items()})
            candidate_infos.append(
                {
                    "all_boxes": boxes,
                    "all_labels": labels,
                    "best_boxes": best_boxes,
                    "best_labels": best_labels,
                }
            )
            for key, value in stats.items():
                totals[key] = totals.get(key, 0) + int(value)

    variants = {
        "baseline": evaluate_detection_predictions(
            predictions,
            targets,
            iou_threshold=0.5,
            score_threshold=float(args.score_threshold),
        )
    }
    for oracle_score in script_args.oracle_scores:
        key = str(float(oracle_score)).replace(".", "p")
        for pool_name, box_key, label_key in (
            ("all_unmatched", "all_boxes", "all_labels"),
            ("best_per_gt_unmatched", "best_boxes", "best_labels"),
        ):
            oracle_predictions = [
                append_oracle_detections(prediction, info[box_key], info[label_key], float(oracle_score))
                for prediction, info in zip(predictions, candidate_infos)
            ]
            variants[f"rpn_{pool_name}_score_{key}"] = evaluate_detection_predictions(
                oracle_predictions,
                targets,
                iou_threshold=0.5,
                score_threshold=float(args.score_threshold),
            )
    baseline_ap75 = float(variants["baseline"]["ap75"])
    report = {
        "device": str(device),
        "config": vars(script_args),
        "candidate_totals": totals,
        "variants": variants,
        "delta_ap75": {
            key: float(value["ap75"]) - baseline_ap75
            for key, value in variants.items()
            if key != "baseline"
        },
    }
    save_json(report, run_dir / "rpn_lchi_oracle_report.json")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
