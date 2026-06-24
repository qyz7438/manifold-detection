"""ChordEdit-style low-energy control prototype for detection RLVR.

Validates on a frozen baseline val set (no training).  Computes:
1. Per-proposal confidence and IoU from baseline predictions.
2. OT displacement field between source (current confidences) and target
   (IoU-weighted ideal confidences) via Sinkhorn algorithm.
3. Compares OT displacement variance vs PG gradient variance.
4. Compares pair-wise ranking consistency (OT displacement vs IoU).

Usage:
    python scripts/prototype_ot_control.py \
        --checkpoint runs/baseline_mid06/best_model.pth \
        --config configs/baseline_mid06.yaml \
        --run-name ot_prototype_01
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Ensure project root is on path so spectral_detection_posttrain can be imported
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.matching.pred_gt_matcher import match_predictions_to_gt
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.config import load_config
from spectral_detection_posttrain.utils.io import ensure_run_dir, save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OT-control prototype on frozen baseline.")
    parser.add_argument("--checkpoint", required=True, help="Path to frozen baseline checkpoint.")
    parser.add_argument("--config", required=True, help="Model config YAML.")
    parser.add_argument("--run-name", required=True, help="Run directory name under runs/.")
    parser.add_argument("--limit-val", type=int, default=None, help="Limit val images for quick test.")
    parser.add_argument("--score-threshold", type=float, default=0.05, help="Proposal score threshold.")
    parser.add_argument("--max-candidates", type=int, default=40, help="Max proposals per image.")
    parser.add_argument("--target-mode", default="iou", choices=["iou", "binary_top20"],
                        help="Target confidence distribution: iou=continuous, binary_top20=top20% IoU->1.")
    parser.add_argument("--sinkhorn-reg", type=float, default=0.1, help="Sinkhorn entropy regularization.")
    parser.add_argument("--sinkhorn-maxiter", type=int, default=1000, help="Sinkhorn max iterations.")
    parser.add_argument("--afm-type", default="none", choices=["none", "mplseg", "micro"],
                        help="AFM type for model architecture (default: none).")
    parser.add_argument("--afm-residual-mode", default="current", choices=["current", "delta", "norm_delta"],
                        help="AFM residual mode.")
    parser.add_argument("--strict-load", action="store_true", default=False,
                        help="Use strict=False when loading checkpoint (needed for some AFM models).")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def collect_proposals(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    score_threshold: float,
    max_candidates: int,
) -> list[dict]:
    """Collect per-image proposals with confidence, IoU, and match labels.

    Returns a list of dicts, one per image:
        {
            "confidences": np.ndarray (N,),
            "ious": np.ndarray (N,),
            "matched": np.ndarray (N, bool),
            "boxes": np.ndarray (N, 4),
            "labels": np.ndarray (N,),
            "gt_boxes": np.ndarray (M, 4),
            "gt_labels": np.ndarray (M,),
        }
    """
    model.eval()
    records = []
    for images, batch_targets in tqdm(val_loader, desc="collect"):
        outputs = model([img.to(device) for img in images])
        for out, tgt in zip(outputs, batch_targets):
            pred = {k: v.detach().cpu() for k, v in out.items()}
            tgt_cpu = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in tgt.items()}

            # threshold and cap candidates
            scores = pred.get("scores", torch.empty((0,)))
            boxes = pred.get("boxes", torch.empty((0, 4)))
            labels = pred.get("labels", torch.empty((0,), dtype=torch.long))
            keep = scores >= score_threshold
            if keep.sum() > max_candidates:
                order = torch.argsort(scores[keep], descending=True)[:max_candidates]
                keep_idx = torch.where(keep)[0][order]
                keep = torch.zeros_like(keep, dtype=torch.bool)
                keep[keep_idx] = True

            confidences = scores[keep].numpy()
            pred_boxes = boxes[keep]
            pred_labels = labels[keep]

            gt_boxes = tgt_cpu.get("boxes", torch.empty((0, 4)))
            gt_labels = tgt_cpu.get("labels", torch.empty((0,), dtype=torch.long))

            # compute per-proposal best IoU
            if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
                ious = np.zeros(len(confidences))
                matched = np.zeros(len(confidences), dtype=bool)
            else:
                iou_mat = _box_iou_matrix(pred_boxes, gt_boxes)
                best_iou, best_gt = iou_mat.max(dim=1)
                # class-aware matching
                same_class = gt_labels[best_gt] == pred_labels
                matched = (best_iou >= 0.5) & same_class
                ious = best_iou.numpy()
                matched = matched.numpy()

            records.append({
                "confidences": confidences.astype(np.float32),
                "ious": ious.astype(np.float32),
                "matched": matched,
                "boxes": pred_boxes.numpy().astype(np.float32),
                "labels": pred_labels.numpy().astype(np.int64),
                "gt_boxes": gt_boxes.numpy().astype(np.float32),
                "gt_labels": gt_labels.numpy().astype(np.int64),
            })
    return records


def _box_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Vectorized IoU matrix (N, M)."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    return inter / (area1[:, None] + area2 - inter).clamp_min(1e-6)


def build_target_distribution(confidences: np.ndarray, ious: np.ndarray, mode: str) -> np.ndarray:
    """Build target confidence distribution from IoU labels.

    Args:
        confidences: current proposal confidences (N,)
        ious: best IoU per proposal (N,)
        mode: "iou" -> target = iou; "binary_top20" -> top 20% IoU -> 1, bottom 20% -> 0
    """
    if mode == "iou":
        return ious.astype(np.float32)
    elif mode == "binary_top20":
        target = np.full_like(confidences, 0.5)
        if len(ious) > 0:
            t20 = np.percentile(ious, 80)
            b20 = np.percentile(ious, 20)
            target[ious >= t20] = 1.0
            target[ious <= b20] = 0.0
        return target
    else:
        raise ValueError(f"Unknown target_mode: {mode}")


def sinkhorn_ot_plan(source: np.ndarray, target: np.ndarray, reg: float = 0.1, max_iter: int = 1000) -> np.ndarray:
    """Compute optimal transport plan via Sinkhorn algorithm.

    Args:
        source: (N,) source samples
        target: (N,) target samples
        reg: entropy regularization
        max_iter: max iterations

    Returns:
        (N, N) transport plan matrix (row-stochastic, approximately).
    """
    import ot
    N = len(source)
    if N == 0:
        return np.zeros((0, 0), dtype=np.float32)

    # cost matrix: squared distance in 1D confidence space
    M = ot.dist(source.reshape(-1, 1), target.reshape(-1, 1), metric="sqeuclidean")
    a = np.ones(N) / N
    b = np.ones(N) / N
    plan = ot.sinkhorn(a, b, M, reg=reg, numItermax=max_iter, stopThr=1e-6, verbose=False)
    return plan.astype(np.float32)


def compute_ot_displacement(source: np.ndarray, target: np.ndarray, plan: np.ndarray) -> np.ndarray:
    """Compute per-sample displacement under OT plan.

    displacement[i] = barycentric projection of target mass assigned to i.
    """
    if len(source) == 0:
        return np.zeros(0, dtype=np.float32)
    row_sums = plan.sum(axis=1, keepdims=True).clip(min=1e-9)
    barycenter = (plan @ target.reshape(-1, 1)) / row_sums
    return barycenter.squeeze(-1) - source


def compute_pg_gradient(confidences: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute naive policy-gradient direction: (confidence - target) * noise_proxy.

    Here we treat the displacement itself as the gradient direction for comparison.
    In actual PG, the gradient is (c_i - t_i) * sampled noise.  We approximate
    the direction magnitude by the raw residual.
    """
    return confidences - target


def pairwise_ranking_consistency(displacement: np.ndarray, ious: np.ndarray) -> float:
    """Fraction of proposal pairs where displacement and IoU agree on ordering.

    For a pair (i, j): if displacement[i] > displacement[j], then ious[i] should be > ious[j].
    """
    n = len(displacement)
    if n < 2:
        return 0.0
    agree = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if (displacement[i] > displacement[j]) == (ious[i] > ious[j]):
                agree += 1
    return agree / total if total > 0 else 0.0


def fast_pairwise_ranking_consistency(displacement: np.ndarray, ious: np.ndarray) -> float:
    """Kendall tau-like consistency using argsort correlation (O(n log n))."""
    if len(displacement) < 2:
        return 0.0
    rank_disp = np.argsort(np.argsort(displacement))
    rank_iou = np.argsort(np.argsort(ious))
    # Spearman rank correlation
    d = rank_disp - rank_iou
    n = len(displacement)
    return 1.0 - (6.0 * float((d * d).sum())) / (n * (n * n - 1.0))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device({})
    run_dir = ensure_run_dir(args.run_name)

    # save args
    with open(run_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    config = load_config(args.config)
    model_cfg = dict(config)
    model_cfg["model"] = dict(config["model"])
    model_cfg["model"]["pretrained"] = False
    # Override AFM settings if requested
    if args.afm_type != "none":
        model_cfg["model"]["afm_type"] = args.afm_type
        model_cfg["model"]["afm_residual_mode"] = args.afm_residual_mode
        model_cfg["model"]["afm_channels"] = 256
    model = build_detector(model_cfg).to(device)

    # load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=not args.strict_load)
    elif isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=not args.strict_load)
    else:
        model.load_state_dict(ckpt, strict=not args.strict_load)
    model.eval()

    _, val_loader = build_penn_fudan_loaders(
        config,
        limit_train=1,
        limit_val=args.limit_val,
        batch_size=int(config.get("eval", {}).get("batch_size", 2)),
    )

    # ------------------------------------------------------------------
    # 1. Collect proposals
    # ------------------------------------------------------------------
    records = collect_proposals(
        model, val_loader, device,
        score_threshold=args.score_threshold,
        max_candidates=args.max_candidates,
    )

    # ------------------------------------------------------------------
    # 2. Per-image OT analysis and aggregation
    # ------------------------------------------------------------------
    per_image_results = []
    all_ot_displacements = []
    all_pg_gradients = []
    all_confs = []
    all_ious = []
    all_targets = []
    all_matched = []

    for idx, r in enumerate(records):
        n = len(r["confidences"])
        if n == 0:
            continue
        target = build_target_distribution(r["confidences"], r["ious"], args.target_mode)
        if n >= 2:
            plan_img = sinkhorn_ot_plan(r["confidences"], target, reg=args.sinkhorn_reg, max_iter=args.sinkhorn_maxiter)
            disp_img = compute_ot_displacement(r["confidences"], target, plan_img)
        else:
            disp_img = np.zeros(n, dtype=np.float32)
        pg_img = compute_pg_gradient(r["confidences"], target)

        all_ot_displacements.append(disp_img)
        all_pg_gradients.append(pg_img)
        all_confs.append(r["confidences"])
        all_ious.append(r["ious"])
        all_targets.append(target)
        all_matched.append(r["matched"])

        if n >= 2:
            per_image_results.append({
                "image_idx": idx,
                "num_proposals": n,
                "num_matched": int(r["matched"].sum()),
                "ot_displacement_var": float(np.var(disp_img)),
                "pg_gradient_var": float(np.var(pg_img)),
                "variance_ratio": float(np.var(pg_img) / np.var(disp_img)) if np.var(disp_img) > 1e-9 else None,
                "ranking_consistency": fast_pairwise_ranking_consistency(disp_img, r["ious"]),
            })

    if not all_confs:
        print("No proposals collected.  Abort.")
        return

    all_ot_displacement = np.concatenate(all_ot_displacements)
    all_pg_gradient = np.concatenate(all_pg_gradients)
    all_conf = np.concatenate(all_confs)
    all_iou = np.concatenate(all_ious)
    all_target = np.concatenate(all_targets)
    all_matched = np.concatenate(all_matched)

    # ------------------------------------------------------------------
    # 3. Variance comparison (per-image aggregated)
    # ------------------------------------------------------------------
    ot_var = float(np.var(all_ot_displacement))
    pg_var = float(np.var(all_pg_gradient))
    ot_std = float(np.std(all_ot_displacement))
    pg_std = float(np.std(all_pg_gradient))

    # Also report median per-image variance ratio
    per_image_variance_ratios = [
        r["variance_ratio"] for r in per_image_results
        if r["variance_ratio"] is not None
    ]
    median_per_image_variance_ratio = float(np.median(per_image_variance_ratios)) if per_image_variance_ratios else None

    # ------------------------------------------------------------------
    # 4. Pairwise ranking consistency (per-image, aggregated)
    # ------------------------------------------------------------------
    per_image_consistencies = [r["ranking_consistency"] for r in per_image_results]
    avg_per_image_consistency = float(np.mean(per_image_consistencies)) if per_image_consistencies else 0.0
    median_per_image_consistency = float(np.median(per_image_consistencies)) if per_image_consistencies else 0.0

    # ------------------------------------------------------------------
    # 5. Baseline detection metrics (for reference)
    # ------------------------------------------------------------------
    predictions = []
    targets = []
    for images, batch_targets in val_loader:
        outputs = model([img.to(device) for img in images])
        predictions.extend([{k: v.detach().cpu() for k, v in out.items()} for out in outputs])
        targets.extend([{k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in t.items()} for t in batch_targets])

    metrics = evaluate_detection_predictions(
        predictions, targets,
        iou_threshold=0.5,
        score_threshold=args.score_threshold,
    )

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    summary = {
        "num_images": len(records),
        "num_proposals": int(len(all_conf)),
        "num_matched": int(all_matched.sum()),
        "target_mode": args.target_mode,
        "sinkhorn_reg": args.sinkhorn_reg,
        "baseline_metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in metrics.items()},
        "variance": {
            "ot_displacement_var": ot_var,
            "pg_gradient_var": pg_var,
            "variance_ratio_pg_ot": pg_var / ot_var if ot_var > 1e-9 else None,
            "median_per_image_variance_ratio": median_per_image_variance_ratio,
            "ot_displacement_std": ot_std,
            "pg_gradient_std": pg_std,
        },
        "ranking_consistency": {
            "avg_per_image_ot_vs_iou": avg_per_image_consistency,
            "median_per_image_ot_vs_iou": median_per_image_consistency,
        },
        "distribution": {
            "confidence_mean": float(all_conf.mean()),
            "confidence_std": float(all_conf.std()),
            "iou_mean": float(all_iou.mean()),
            "iou_std": float(all_iou.std()),
            "target_mean": float(all_target.mean()),
            "target_std": float(all_target.std()),
            "ot_displacement_mean": float(all_ot_displacement.mean()),
            "ot_displacement_median": float(np.median(all_ot_displacement)),
            "pg_gradient_mean": float(all_pg_gradient.mean()),
            "pg_gradient_median": float(np.median(all_pg_gradient)),
        },
    }

    save_json(summary, run_dir / "summary.json")
    print(json.dumps(summary, indent=2))

    save_json(per_image_results, run_dir / "per_image.json")
    print(f"\nDone. Results saved to {run_dir}")


if __name__ == "__main__":
    main()
