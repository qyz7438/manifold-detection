from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import round2129_nwpu_posttrain_smoke as round2129
from spectral_detection_posttrain.matching.pred_gt_matcher import match_predictions_to_gt
from spectral_detection_posttrain.rlvr.confidence_rescue import match_boxes_to_targets
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
    parser = argparse.ArgumentParser(
        description="Diagnose why verifier-positive LC-HI NWPU proposals do not become final TP."
    )
    parser.add_argument("--run-name", default="diagnose_vp_lchi_flow")
    parser.add_argument("--split", choices=["train", "val", "both"], default="both")
    parser.add_argument("--limit-train", type=int, default=100000)
    parser.add_argument("--limit-val", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-proposals", type=int, default=60)
    parser.add_argument("--rollout-score-threshold", type=float, default=0.001)
    parser.add_argument("--rollout-detections-per-img", type=int, default=300)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--rescue-low-conf-max", type=float, default=0.5)
    parser.add_argument("--rescue-high-iou-min", type=float, default=0.75)
    parser.add_argument("--rescue-low-iou-max", type=float, default=0.3)
    parser.add_argument("--rescue-verifier-gate", type=float, default=0.0)
    parser.add_argument(
        "--rescue-raw-ifft-features",
        nargs="+",
        default=["fft_edge_truncation@64", "phase_edge@64", "phase_abs_high@11"],
    )
    parser.add_argument("--target-precision", type=float, default=0.8)
    parser.add_argument("--margin-std-frac", type=float, default=0.0)
    parser.add_argument("--max-examples", type=int, default=50)
    return parser.parse_args()


def make_round_args(script_args):
    with _temporary_argv(["round2129_nwpu_posttrain_smoke.py"]):
        args = round2129.parse_args()
    args.limit_train = int(script_args.limit_train)
    args.limit_val = int(script_args.limit_val)
    args.batch_size = int(script_args.batch_size)
    args.max_proposals = int(script_args.max_proposals)
    args.rollout_score_threshold = float(script_args.rollout_score_threshold)
    args.rollout_detections_per_img = int(script_args.rollout_detections_per_img)
    args.score_threshold = float(script_args.score_threshold)
    args.rescue_mode = True
    args.rescue_low_conf_max = float(script_args.rescue_low_conf_max)
    args.rescue_high_iou_min = float(script_args.rescue_high_iou_min)
    args.rescue_low_iou_max = float(script_args.rescue_low_iou_max)
    args.rescue_verifier_mode = "raw_ifft"
    args.rescue_verifier_gate = float(script_args.rescue_verifier_gate)
    args.rescue_raw_ifft_features = list(script_args.rescue_raw_ifft_features)
    args.rescue_raw_ifft_target_precision = float(script_args.target_precision)
    args.rescue_raw_ifft_margin_std_frac = float(script_args.margin_std_frac)
    args.rescue_raw_ifft_score_method = "train_effect_sum"
    return args


def safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[index])


def score_rank(probs: torch.Tensor, label: int) -> int:
    if label <= 0 or label >= probs.numel():
        return 0
    object_probs = probs[1:]
    order = torch.argsort(object_probs, descending=True)
    zero_based = int((order == (label - 1)).nonzero(as_tuple=False)[0].item())
    return zero_based + 1


def final_match_maps(prediction: dict, target: dict, score_threshold: float) -> dict[str, dict[int, dict]]:
    maps = {}
    for iou_threshold in (0.5, 0.75):
        matched = match_predictions_to_gt(
            prediction,
            target,
            iou_threshold=float(iou_threshold),
            score_threshold=float(score_threshold),
        )
        by_gt = {int(item["gt_index"]): item for item in matched["matches"]}
        by_pred = {int(item["pred_index"]): item for item in matched["matches"]}
        maps[f"gt@{iou_threshold:g}"] = by_gt
        maps[f"pred@{iou_threshold:g}"] = by_pred
    return maps


def classify_reason(
    *,
    final_score: float,
    final_label: int,
    gt_label: int,
    proposal_index: int,
    matched_gt_index: int,
    maps: dict[str, dict[int, dict]],
    score_threshold: float,
) -> str:
    gt_match_075 = maps["gt@0.75"].get(int(matched_gt_index))
    gt_match_05 = maps["gt@0.5"].get(int(matched_gt_index))
    if final_score < score_threshold:
        if gt_match_075 is not None:
            return "below_eval_score_threshold_gt_already_tp_ap75"
        if gt_match_05 is not None:
            return "below_eval_score_threshold_gt_already_tp_ap50"
        if final_label != gt_label:
            return "below_eval_score_threshold_label_mismatch_gt_unmatched"
        return "below_eval_score_threshold_gt_unmatched"
    if final_label != gt_label:
        if gt_match_075 is not None:
            return "final_label_mismatch_gt_already_tp_ap75"
        return "final_label_mismatch_gt_unmatched"
    if proposal_index in maps["pred@0.75"]:
        return "is_final_ap75_tp"
    if gt_match_075 is not None:
        if int(gt_match_075["pred_index"]) != int(proposal_index):
            return "duplicate_same_gt_taken_by_higher_score_ap75"
        return "is_final_ap75_tp"
    if gt_match_05 is not None and int(gt_match_05["pred_index"]) != int(proposal_index):
        return "duplicate_same_gt_taken_by_higher_score_ap50"
    return "correct_label_score_ok_but_unmatched_unexpected"


@torch.no_grad()
def diagnose_split(model, loader, device: torch.device, args, reference_features, reference_stats, split_name: str):
    totals = Counter()
    reason_counts = Counter()
    gt_prob_values: list[float] = []
    rollout_score_values: list[float] = []
    verifier_score_values: list[float] = []
    top_prob_values: list[float] = []
    gt_rank_values: list[float] = []
    iou_values: list[float] = []
    label_match_scores: list[float] = []
    examples = []
    per_class = defaultdict(Counter)

    model.eval()
    for images, targets in loader:
        device_images = [image.to(device) for image in images]
        outputs = model(device_images)
        for image, target, output in zip(images, targets, outputs):
            threshold = float(args.rollout_score_threshold)
            keep_indices = torch.nonzero(output["scores"].detach().cpu() >= threshold, as_tuple=False).flatten()
            keep_indices = keep_indices[: int(args.max_proposals)]
            if keep_indices.numel() == 0:
                continue

            proposals = output["boxes"].detach().cpu()[keep_indices]
            proposal_scores = output["scores"].detach().cpu()[keep_indices]
            proposal_labels = output["labels"].detach().cpu()[keep_indices]
            final_prediction = {k: v.detach().cpu() for k, v in output.items()}
            final_target = {
                k: v.detach().cpu() if torch.is_tensor(v) else v
                for k, v in target.items()
            }
            maps = final_match_maps(final_prediction, final_target, float(args.score_threshold))

            class_logits, _, box_features, _, _ = round2129.extract_roi_outputs_and_features_for_boxes(
                model,
                [image.to(device)],
                [proposals],
            )
            gt_boxes = target.get("boxes", torch.empty((0, 4))).to(device)
            gt_labels = target.get("labels", torch.empty((0,), dtype=torch.long)).to(device)
            if gt_boxes.numel() == 0:
                continue
            best_iou, best_labels = match_boxes_to_targets(proposals.to(device), gt_boxes, gt_labels)
            iou_matrix = round2129.box_iou(proposals.to(device), gt_boxes.float())
            _, best_gt_indices = iou_matrix.max(dim=1)
            gt_probs = round2129.matched_label_probabilities(class_logits, best_labels)
            verifier_scores = round2129.compute_rescue_verifier_scores(
                image=image,
                proposals=proposals,
                box_features=box_features,
                reference_features=reference_features,
                reference_stats=reference_stats,
                args=args,
                proposal_labels=best_labels,
            )
            if verifier_scores is None:
                verifier_scores = torch.zeros_like(gt_probs)

            probs = F.softmax(class_logits, dim=1)
            object_probs = probs[:, 1:]
            top_object_probs, top_object_labels = object_probs.max(dim=1)
            top_object_labels = top_object_labels + 1
            lchi = (
                (gt_probs <= float(args.rescue_low_conf_max))
                & (best_iou >= float(args.rescue_high_iou_min))
                & (best_labels > 0)
            )
            verifier_positive = lchi & (verifier_scores >= float(args.rescue_verifier_gate))

            totals["rollout_selected"] += int(proposals.shape[0])
            totals["lchi"] += int(lchi.sum().item())
            totals["verifier_positive_lchi"] += int(verifier_positive.sum().item())

            for local_idx in torch.nonzero(verifier_positive, as_tuple=False).flatten().tolist():
                gt_label = int(best_labels[local_idx].item())
                final_label = int(proposal_labels[local_idx].item())
                final_score = float(proposal_scores[local_idx].item())
                gt_prob = float(gt_probs[local_idx].item())
                verifier_score = float(verifier_scores[local_idx].item())
                top_prob = float(top_object_probs[local_idx].item())
                top_label = int(top_object_labels[local_idx].item())
                rank = score_rank(probs[local_idx].detach().cpu(), gt_label)
                matched_gt_index = int(best_gt_indices[local_idx].item())
                global_pred_index = int(keep_indices[local_idx].item())
                reason = classify_reason(
                    final_score=final_score,
                    final_label=final_label,
                    gt_label=gt_label,
                    proposal_index=global_pred_index,
                    matched_gt_index=matched_gt_index,
                    maps=maps,
                    score_threshold=float(args.score_threshold),
                )
                reason_counts[reason] += 1
                per_class[str(gt_label)][reason] += 1
                per_class[str(gt_label)]["total"] += 1
                gt_prob_values.append(gt_prob)
                rollout_score_values.append(final_score)
                verifier_score_values.append(verifier_score)
                top_prob_values.append(top_prob)
                gt_rank_values.append(float(rank))
                iou_values.append(float(best_iou[local_idx].item()))
                label_match_scores.append(1.0 if final_label == gt_label else 0.0)
                if final_score >= float(args.score_threshold):
                    totals["score_ge_eval_threshold"] += 1
                if final_label == gt_label:
                    totals["final_label_matches_gt"] += 1
                if global_pred_index in maps["pred@0.75"]:
                    totals["is_ap75_tp"] += 1
                if matched_gt_index in maps["gt@0.75"]:
                    totals["matched_gt_has_ap75_tp"] += 1

                if len(examples) < int(getattr(args, "max_examples", 50)):
                    gt_match_075 = maps["gt@0.75"].get(matched_gt_index)
                    examples.append(
                        {
                            "split": split_name,
                            "image_id": int(final_target.get("image_id", torch.tensor([-1])).flatten()[0].item())
                            if torch.is_tensor(final_target.get("image_id", None))
                            else int(final_target.get("image_id", -1)),
                            "local_proposal_index": int(local_idx),
                            "global_prediction_index": global_pred_index,
                            "matched_gt_index": matched_gt_index,
                            "best_iou": float(best_iou[local_idx].item()),
                            "gt_label": gt_label,
                            "final_label": final_label,
                            "final_rollout_score": final_score,
                            "roi_gt_label_prob": gt_prob,
                            "roi_top_label": top_label,
                            "roi_top_prob": top_prob,
                            "roi_gt_label_rank": rank,
                            "verifier_score_minus_threshold": verifier_score,
                            "reason": reason,
                            "gt_ap75_match_pred_index": None
                            if gt_match_075 is None
                            else int(gt_match_075["pred_index"]),
                            "gt_ap75_match_score": None
                            if gt_match_075 is None
                            else float(gt_match_075["score"]),
                            "gt_ap75_match_iou": None
                            if gt_match_075 is None
                            else float(gt_match_075["iou"]),
                        }
                    )

    count = max(1, int(totals["verifier_positive_lchi"]))
    return {
        "split": split_name,
        "counts": dict(totals),
        "reason_counts": dict(reason_counts),
        "per_class_reason_counts": {key: dict(value) for key, value in sorted(per_class.items())},
        "rates": {
            "score_ge_eval_threshold": float(totals["score_ge_eval_threshold"]) / count,
            "final_label_matches_gt": float(totals["final_label_matches_gt"]) / count,
            "is_ap75_tp": float(totals["is_ap75_tp"]) / count,
            "matched_gt_has_ap75_tp": float(totals["matched_gt_has_ap75_tp"]) / count,
        },
        "stats": {
            "roi_gt_label_prob_mean": safe_mean(gt_prob_values),
            "roi_gt_label_prob_p50": percentile(gt_prob_values, 0.5),
            "final_rollout_score_mean": safe_mean(rollout_score_values),
            "final_rollout_score_p50": percentile(rollout_score_values, 0.5),
            "roi_top_prob_mean": safe_mean(top_prob_values),
            "roi_gt_label_rank_mean": safe_mean(gt_rank_values),
            "best_iou_mean": safe_mean(iou_values),
            "verifier_score_mean": safe_mean(verifier_score_values),
            "final_label_match_rate": safe_mean(label_match_scores),
        },
        "examples": examples,
    }


def main():
    script_args = parse_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path("runs") / script_args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    args = make_round_args(script_args)
    train_loader, val_loader = round2129.build_loaders(args)
    model = round2129.build_nwpu_model(device)
    round2129.configure_detector_rollout(
        model,
        score_threshold=float(args.rollout_score_threshold),
        detections_per_img=int(args.rollout_detections_per_img),
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()

    reference_features, reference_stats = round2129.build_rescue_reference(model, train_loader, device, args)
    save_json(vars(script_args), run_dir / "config.json")
    save_json(reference_stats, run_dir / "rescue_reference_stats.json")

    reports = {}
    if script_args.split in {"train", "both"}:
        reports["train"] = diagnose_split(
            model,
            train_loader,
            device,
            args,
            reference_features,
            reference_stats,
            "train",
        )
    if script_args.split in {"val", "both"}:
        reports["val"] = diagnose_split(
            model,
            val_loader,
            device,
            args,
            reference_features,
            reference_stats,
            "val",
        )

    output = {
        "device": str(device),
        "round_args": {
            "score_threshold": float(args.score_threshold),
            "rollout_score_threshold": float(args.rollout_score_threshold),
            "max_proposals": int(args.max_proposals),
            "rescue_low_conf_max": float(args.rescue_low_conf_max),
            "rescue_high_iou_min": float(args.rescue_high_iou_min),
            "rescue_verifier_gate": float(args.rescue_verifier_gate),
            "rescue_raw_ifft_features": list(args.rescue_raw_ifft_features),
        },
        "reports": reports,
    }
    out_path = run_dir / "vp_lchi_flow_report.json"
    save_json(output, out_path)
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Saved report to {out_path}")


if __name__ == "__main__":
    main()
