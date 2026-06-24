from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import round2129_nwpu_posttrain_smoke as round2129
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.eval.rescue_oracle import apply_detection_score_oracle, unmatched_gt_candidate_mask
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
    parser = argparse.ArgumentParser(description="Run oracle AP75 upper-bound experiments for verifier-positive LC-HI.")
    parser.add_argument("--run-name", default="round2156_vp_lchi_oracle")
    parser.add_argument("--limit-train", type=int, default=100000)
    parser.add_argument("--limit-val", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-proposals", type=int, default=60)
    parser.add_argument("--rollout-score-threshold", type=float, default=0.001)
    parser.add_argument("--rollout-detections-per-img", type=int, default=300)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--rescue-low-conf-max", type=float, default=0.5)
    parser.add_argument("--rescue-high-iou-min", type=float, default=0.75)
    parser.add_argument("--rescue-verifier-gate", type=float, default=0.0)
    parser.add_argument(
        "--rescue-raw-ifft-features",
        nargs="+",
        default=["fft_edge_truncation@64", "phase_edge@64", "phase_abs_high@11"],
    )
    parser.add_argument("--target-precision", type=float, default=0.8)
    parser.add_argument("--margin-std-frac", type=float, default=0.0)
    parser.add_argument("--oracle-scores", type=float, nargs="+", default=[0.06, 0.1, 0.2, 0.5])
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
    args.rescue_verifier_mode = "raw_ifft"
    args.rescue_verifier_gate = float(script_args.rescue_verifier_gate)
    args.rescue_raw_ifft_features = list(script_args.rescue_raw_ifft_features)
    args.rescue_raw_ifft_target_precision = float(script_args.target_precision)
    args.rescue_raw_ifft_margin_std_frac = float(script_args.margin_std_frac)
    args.rescue_raw_ifft_score_method = "train_effect_sum"
    return args


@torch.no_grad()
def collect_predictions_and_candidates(model, loader, device, args, reference_features, reference_stats):
    predictions = []
    targets_out = []
    candidate_infos = []
    totals = {
        "rollout_selected": 0,
        "lchi": 0,
        "verifier_positive_lchi": 0,
        "vp_lchi_unmatched_ap75": 0,
        "vp_lchi_score_below_threshold": 0,
        "vp_lchi_score_ge_threshold": 0,
        "lchi_unmatched_ap75": 0,
        "lchi_score_below_threshold": 0,
        "lchi_score_ge_threshold": 0,
    }
    model.eval()
    for images, targets in loader:
        outputs = model([image.to(device) for image in images])
        for image, target, output in zip(images, targets, outputs):
            prediction = {key: value.detach().cpu() for key, value in output.items()}
            target_cpu = {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in target.items()}
            image_candidate = {
                "vp_indices": [],
                "vp_labels": [],
                "vp_unmatched_indices": [],
                "vp_unmatched_labels": [],
                "lchi_indices": [],
                "lchi_labels": [],
                "lchi_unmatched_indices": [],
                "lchi_unmatched_labels": [],
            }
            keep_indices = torch.nonzero(prediction["scores"] >= float(args.rollout_score_threshold), as_tuple=False).flatten()
            keep_indices = keep_indices[: int(args.max_proposals)]
            if keep_indices.numel() > 0:
                proposals = prediction["boxes"][keep_indices]
                proposal_scores = prediction["scores"][keep_indices]
                class_logits, _, box_features, _, _ = round2129.extract_roi_outputs_and_features_for_boxes(
                    model,
                    [image.to(device)],
                    [proposals],
                )
                gt_boxes = target.get("boxes", torch.empty((0, 4))).to(device)
                gt_labels = target.get("labels", torch.empty((0,), dtype=torch.long)).to(device)
                if gt_boxes.numel() > 0:
                    best_iou, best_labels = match_boxes_to_targets(proposals.to(device), gt_boxes, gt_labels)
                    iou_matrix = round2129.box_iou(proposals.to(device), gt_boxes.float())
                    _, best_gt_indices = iou_matrix.max(dim=1)
                    gt_probs = round2129.matched_label_probabilities(class_logits, best_labels)
                    verifier_scores = round2129.compute_rescue_verifier_scores(
                        image,
                        proposals,
                        box_features,
                        reference_features,
                        reference_stats,
                        args,
                        proposal_labels=best_labels,
                    )
                    if verifier_scores is None:
                        verifier_scores = torch.zeros_like(gt_probs)
                    lchi = (
                        (gt_probs <= float(args.rescue_low_conf_max))
                        & (best_iou >= float(args.rescue_high_iou_min))
                        & (best_labels > 0)
                    )
                    vp_lchi = lchi & (verifier_scores >= float(args.rescue_verifier_gate))
                    unmatched = unmatched_gt_candidate_mask(
                        prediction,
                        target_cpu,
                        best_gt_indices.detach().cpu(),
                        vp_lchi.detach().cpu(),
                        iou_threshold=0.75,
                        score_threshold=float(args.score_threshold),
                    )
                    lchi_unmatched = unmatched_gt_candidate_mask(
                        prediction,
                        target_cpu,
                        best_gt_indices.detach().cpu(),
                        lchi.detach().cpu(),
                        iou_threshold=0.75,
                        score_threshold=float(args.score_threshold),
                    )
                    totals["rollout_selected"] += int(proposals.shape[0])
                    totals["lchi"] += int(lchi.sum().item())
                    totals["verifier_positive_lchi"] += int(vp_lchi.sum().item())
                    totals["vp_lchi_unmatched_ap75"] += int(unmatched.sum().item())
                    totals["vp_lchi_score_below_threshold"] += int(
                        (vp_lchi.detach().cpu() & (proposal_scores < float(args.score_threshold))).sum().item()
                    )
                    totals["vp_lchi_score_ge_threshold"] += int(
                        (vp_lchi.detach().cpu() & (proposal_scores >= float(args.score_threshold))).sum().item()
                    )
                    totals["lchi_unmatched_ap75"] += int(lchi_unmatched.sum().item())
                    totals["lchi_score_below_threshold"] += int(
                        (lchi.detach().cpu() & (proposal_scores < float(args.score_threshold))).sum().item()
                    )
                    totals["lchi_score_ge_threshold"] += int(
                        (lchi.detach().cpu() & (proposal_scores >= float(args.score_threshold))).sum().item()
                    )
                    image_candidate["vp_indices"] = keep_indices[vp_lchi.detach().cpu()].long()
                    image_candidate["vp_labels"] = best_labels.detach().cpu()[vp_lchi.detach().cpu()].long()
                    image_candidate["vp_unmatched_indices"] = keep_indices[unmatched].long()
                    image_candidate["vp_unmatched_labels"] = best_labels.detach().cpu()[unmatched].long()
                    image_candidate["lchi_indices"] = keep_indices[lchi.detach().cpu()].long()
                    image_candidate["lchi_labels"] = best_labels.detach().cpu()[lchi.detach().cpu()].long()
                    image_candidate["lchi_unmatched_indices"] = keep_indices[lchi_unmatched].long()
                    image_candidate["lchi_unmatched_labels"] = best_labels.detach().cpu()[lchi_unmatched].long()
            predictions.append(prediction)
            targets_out.append(target_cpu)
            candidate_infos.append(image_candidate)
    return predictions, targets_out, candidate_infos, totals


def evaluate_oracles(predictions, targets, candidate_infos, oracle_scores, score_threshold):
    baseline = evaluate_detection_predictions(predictions, targets, iou_threshold=0.5, score_threshold=score_threshold)
    variants = {"baseline": baseline}
    pools = {
        "vp_lchi_all": ("vp_indices", "vp_labels"),
        "vp_lchi_unmatched": ("vp_unmatched_indices", "vp_unmatched_labels"),
        "lchi_all": ("lchi_indices", "lchi_labels"),
        "lchi_unmatched": ("lchi_unmatched_indices", "lchi_unmatched_labels"),
    }
    for oracle_score in oracle_scores:
        key = str(oracle_score).replace(".", "p")
        for pool_name, (indices_key, labels_key) in pools.items():
            pool_predictions = []
            for prediction, info in zip(predictions, candidate_infos):
                pool_predictions.append(
                    apply_detection_score_oracle(
                        prediction,
                        indices=torch.as_tensor(info[indices_key], dtype=torch.long),
                        labels=torch.as_tensor(info[labels_key], dtype=torch.long),
                        score=float(oracle_score),
                    )
                )
            variants[f"oracle_{pool_name}_score_{key}"] = evaluate_detection_predictions(
                pool_predictions,
                targets,
                iou_threshold=0.5,
                score_threshold=score_threshold,
            )
    return variants


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
    predictions, targets, candidate_infos, candidate_totals = collect_predictions_and_candidates(
        model,
        val_loader,
        device,
        args,
        reference_features,
        reference_stats,
    )
    variants = evaluate_oracles(
        predictions,
        targets,
        candidate_infos,
        list(script_args.oracle_scores),
        float(args.score_threshold),
    )
    baseline_ap75 = float(variants["baseline"]["ap75"])
    summary = {
        "device": str(device),
        "config": vars(script_args),
        "reference_stats": reference_stats,
        "candidate_totals": candidate_totals,
        "variants": variants,
        "delta_ap75": {
            key: float(value["ap75"]) - baseline_ap75
            for key, value in variants.items()
            if key != "baseline"
        },
    }
    save_json(summary, run_dir / "oracle_report.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
