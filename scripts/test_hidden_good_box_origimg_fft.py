"""Improvement 2 for real: Original-image 64x64 patch FFT → Isomap(6).

This actually extracts 64x64 patches from the original image (not ROI feature maps),
computes rfft2, extracts per-band features, then runs Isomap(6) + hidden good box eval.

Uses cross-validation to prevent overfit.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.experiments.runner_utils import (
    build_penn_fudan_loaders_320,
    decode_boxes,
)
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
set_seed(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
OUT_DIR = Path("scripts/hidden_good_box_improvements")
OUT_JSON = OUT_DIR / "results_origimg_fft.json"

# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def bm():
    return build_detector({
        "model": {
            "name": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "pretrained": True,
            "num_classes": 2,
            "min_size": 320,
            "max_size": 320,
        }
    })


# ---------------------------------------------------------------------------
# Original-image 64x64 FFT extraction
# ---------------------------------------------------------------------------
def extract_original_image_fft(proposals: torch.Tensor, image: torch.Tensor,
                                 resize_to: tuple[int, int] = (64, 64),
                                 bands: tuple[float, float] = (0.3, 0.7)) -> dict[str, torch.Tensor]:
    """Crop each proposal from original image, resize to 64x64, FFT."""
    lo_thr, hi_thr = bands
    N = proposals.shape[0]
    H, W = image.shape[1], image.shape[2]

    boxes = proposals.clone()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, W - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, H - 1)
    boxes[:, 2] = torch.maximum(boxes[:, 2], boxes[:, 0] + 1)
    boxes[:, 3] = torch.maximum(boxes[:, 3], boxes[:, 1] + 1)

    crops = []
    for i in range(N):
        x1, y1, x2, y2 = boxes[i].long()
        crop = image[:, y1:y2, x1:x2]
        crop = F.interpolate(crop.unsqueeze(0), size=resize_to, mode="bilinear", align_corners=False)
        crops.append(crop.squeeze(0))

    crops = torch.stack(crops, dim=0)  # (N, 3, 64, 64)

    fft = torch.fft.rfft2(crops, dim=(-2, -1), norm="ortho")  # (N, 3, 64, 33)
    amp = torch.abs(fft)

    freq_h = torch.fft.fftfreq(resize_to[0], device=image.device)
    freq_w = torch.fft.rfftfreq(resize_to[1], device=image.device)
    grid_y, grid_x = torch.meshgrid(freq_h, freq_w, indexing="ij")
    radius = torch.sqrt(grid_x ** 2 + grid_y ** 2)
    radius = radius / radius.max().clamp_min(1e-6)

    lo_mask = (radius <= lo_thr).float().unsqueeze(0).unsqueeze(0)
    mid_mask = ((radius > lo_thr) & (radius <= hi_thr)).float().unsqueeze(0).unsqueeze(0)
    hi_mask = (radius > hi_thr).float().unsqueeze(0).unsqueeze(0)

    return {
        "amp_lo": (amp * lo_mask).reshape(N, 3, -1).sum(dim=2),   # (N, 3)
        "amp_mid": (amp * mid_mask).reshape(N, 3, -1).sum(dim=2), # (N, 3)
        "amp_hi": (amp * hi_mask).reshape(N, 3, -1).sum(dim=2),  # (N, 3)
    }


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_proposals_with_origimg_fft(model, val_loader, device):
    """Collect per-proposal IoU, confidence, and original-image FFT."""
    model.eval()
    records = []
    sampled_props, box_head_in = {}, {}

    model.roi_heads.box_roi_pool.register_forward_pre_hook(
        lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]})
    )
    model.roi_heads.box_head.register_forward_pre_hook(
        lambda m, args: box_head_in.update({"x": args[0]})
    )

    for img_idx, (images, targets) in enumerate(tqdm(val_loader, desc="inference")):
        images_d = [img.to(device) for img in images]
        tgts_t = [{k: v.to(device) for k, v in t.items()} for t in targets]
        sampled_props.clear()
        box_head_in.clear()

        model(images_d, tgts_t)

        sp_raw = sampled_props.get("p")
        rf = box_head_in.get("x")
        if sp_raw is None or rf is None or rf.shape[0] == 0:
            continue

        bf = model.roi_heads.box_head(rf)
        cls_logits = model.roi_heads.box_predictor.cls_score(bf)
        conf = F.softmax(cls_logits, dim=-1)[:, 1].cpu()

        sp_cat = torch.cat(sp_raw, dim=0)
        N = sp_cat.shape[0]
        reg_out = model.roi_heads.box_predictor.bbox_pred(bf[:N])
        person_deltas = reg_out[:, 2:6]
        decoded = decode_boxes(sp_cat, person_deltas)

        img_tensor = images_d[0]
        bands = extract_original_image_fft(decoded, img_tensor)

        offset = 0
        for i_img, p_img in enumerate(sp_raw):
            n_p = p_img.shape[0]
            if n_p == 0:
                continue
            gt = tgts_t[i_img]["boxes"]
            gt_labels = tgts_t[i_img]["labels"]
            person_mask = gt_labels == 1
            gt_person = gt[person_mask] if person_mask.any() else gt

            iou = torch.zeros(n_p)
            gt_idx = torch.full((n_p,), -1, dtype=torch.long)
            if len(gt_person) > 0:
                iou_mat = box_iou(decoded[offset:offset + n_p], gt_person)
                iou, best_gt = iou_mat.max(dim=1)
                iou = iou.cpu()
                gt_idx = best_gt.cpu()

            img_bands = {k: v[offset:offset + n_p].cpu().numpy() for k, v in bands.items()}

            records.append({
                "iou": iou.numpy(),
                "conf": conf[offset:offset + n_p].numpy(),
                "bands": img_bands,
                "gt_idx": gt_idx.numpy(),
                "img_id": img_idx,
            })
            offset += n_p

    return records


# ---------------------------------------------------------------------------
# Build feature matrix from records
# ---------------------------------------------------------------------------
def build_features(records):
    """Build (N, D) feature matrix and labels from records."""
    all_feats = []
    all_iou = []
    all_conf = []
    all_img_id = []
    all_gt_idx = []

    for rec in records:
        n = len(rec["iou"])
        # Feature: concat all bands → (N, 9)
        feat = np.concatenate([rec["bands"][k] for k in ["amp_lo", "amp_mid", "amp_hi"]], axis=1)
        all_feats.append(feat)
        all_iou.append(rec["iou"])
        all_conf.append(rec["conf"])
        all_img_id.append(np.full(n, rec["img_id"]))
        all_gt_idx.append(rec["gt_idx"])

    X = np.concatenate(all_feats, axis=0)  # (N_total, 9)
    ious_arr = np.concatenate(all_iou)
    confs_arr = np.concatenate(all_conf)
    img_ids_arr = np.concatenate(all_img_id)
    gt_ids_arr = np.concatenate(all_gt_idx)

    return X, ious_arr, confs_arr, img_ids_arr, gt_ids_arr


# ---------------------------------------------------------------------------
# Cross-validated evaluation
# ---------------------------------------------------------------------------
def evaluate_cv(X, ious, confs, img_ids, score_fn, name: str) -> dict[str, Any]:
    """Leave-one-image-out CV for hidden good box detection."""
    M = X.shape[0]
    is_tp = ious >= 0.5
    is_fp = ious < 0.3
    uncertain_mask = (confs >= 0.1) & (confs <= 0.5)

    N_uncertain_tp = int((uncertain_mask & is_tp).sum())
    N_uncertain_fp = int((uncertain_mask & is_fp).sum())

    unique_imgs = np.unique(img_ids)
    all_pred_good = np.zeros(M, dtype=bool)
    all_uncertain = np.zeros(M, dtype=bool)

    for img_id in unique_imgs:
        img_mask = img_ids == img_id
        uncertain_img = img_mask & uncertain_mask
        if uncertain_img.sum() == 0:
            continue

        other_mask = ~img_mask
        score = score_fn(img_id, X, is_tp, other_mask, np.ones(M, dtype=bool))

        other_uncertain = other_mask & uncertain_mask
        if other_uncertain.sum() > 0:
            thresh = np.median(score[other_uncertain])
        else:
            thresh = np.median(score)

        pred_good = (score > thresh) & uncertain_img
        all_pred_good |= pred_good
        all_uncertain |= uncertain_img

    tp_found = int((all_pred_good & is_tp).sum())
    fp_found = int((all_pred_good & is_fp).sum())
    fn = int(((~all_pred_good) & all_uncertain & is_tp).sum())
    tn = int(((~all_pred_good) & all_uncertain & is_fp).sum())

    recall = tp_found / N_uncertain_tp if N_uncertain_tp > 0 else 0.0
    precision = tp_found / (tp_found + fp_found) if (tp_found + fp_found) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "name": name,
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "true_positives": tp_found,
        "false_positives": fp_found,
        "false_negatives": fn,
        "true_negatives": tn,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("IMPROVEMENT 2: ORIGINAL-IMAGE 64x64 FFT — CROSS-VALIDATED")
    print("=" * 70)

    print("\n[1/4] Loading model...")
    model = bm().to(DEV)
    ckpt = torch.load(CKPT, map_location=DEV)
    model.load_state_dict(ckpt["model"])

    print("[2/4] Loading val data...")
    _, val_loader = build_penn_fudan_loaders_320(batch_size=1)

    print("[3/4] Collecting proposals with original-image FFT...")
    records = collect_proposals_with_origimg_fft(model, val_loader, DEV)
    print(f"  -> {len(records)} images with proposals")

    print("[4/4] Building features...")
    X, ious_arr, confs_arr, img_ids_arr, gt_ids_arr = build_features(records)
    print(f"  -> {X.shape[0]} proposals, feature dim={X.shape[1]}")

    M = X.shape[0]
    is_tp = ious_arr >= 0.5
    is_fp = ious_arr < 0.3
    uncertain_mask = (confs_arr >= 0.1) & (confs_arr <= 0.5)
    print(f"  -> TP: {is_tp.sum()}, FP: {is_fp.sum()}, Uncertain: {uncertain_mask.sum()}")
    print(f"  -> Uncertain TP: {(uncertain_mask & is_tp).sum()}, Uncertain FP: {(uncertain_mask & is_fp).sum()}")

    # Preprocess: z-score + PCA
    scaler = StandardScaler()
    X_z = scaler.fit_transform(X)
    pca = PCA(n_components=min(6, X.shape[0] - 1), random_state=42)
    X_pca = pca.fit_transform(X_z)
    print(f"  -> PCA dim: {X_pca.shape[1]}, variance explained: {pca.explained_variance_ratio_.sum():.4f}")

    # -----------------------------------------------------------------------
    # Baseline on original-image features: Euclidean distance to TP centroid
    # -----------------------------------------------------------------------
    def euclidean_score_fn(img_id, X_all, is_tp_all, other_mask, all_mask):
        tp_centroid = X_all[is_tp_all].mean(axis=0)
        dist = np.linalg.norm(X_all - tp_centroid, axis=1)
        return -dist

    r_euc = evaluate_cv(X_pca, ious_arr, confs_arr, img_ids_arr, euclidean_score_fn, "OrigImg_Euclidean_CV")
    print(f"\n  {'OrigImg_Euclidean_CV':<30s}: R={r_euc['recall']*100:5.1f}% P={r_euc['precision']*100:5.1f}% F1={r_euc['f1']*100:5.1f}%")

    # -----------------------------------------------------------------------
    # Isomap(6) on original-image features
    # -----------------------------------------------------------------------
    # Precompute Isomap on all data (Isomap has no transform)
    iso = Isomap(n_neighbors=15, n_components=6)
    X_iso = iso.fit_transform(X_pca)

    def iso_score_fn(img_id, X_all, is_tp_all, other_mask, all_mask):
        return -np.linalg.norm(X_iso - X_iso[is_tp_all].mean(axis=0), axis=1)

    r_iso = evaluate_cv(X_pca, ious_arr, confs_arr, img_ids_arr, iso_score_fn, "OrigImg_Isomap6_CV")
    print(f"  {'OrigImg_Isomap6_CV':<30s}: R={r_iso['recall']*100:5.1f}% P={r_iso['precision']*100:5.1f}% F1={r_iso['f1']*100:5.1f}%")

    # -----------------------------------------------------------------------
    # Large-neighbor Isomap
    # -----------------------------------------------------------------------
    for n_nbr in [20, 30]:
        iso_n = Isomap(n_neighbors=n_nbr, n_components=6)
        X_iso_n = iso_n.fit_transform(X_pca)

        def iso_n_score_fn(img_id, X_all, is_tp_all, other_mask, all_mask):
            return -np.linalg.norm(X_iso_n - X_iso_n[is_tp_all].mean(axis=0), axis=1)

        r_iso_n = evaluate_cv(X_pca, ious_arr, confs_arr, img_ids_arr, iso_n_score_fn, f"OrigImg_Isomap_n{n_nbr}_CV")
        print(f"  {f'OrigImg_Isomap_n{n_nbr}_CV':<30s}: R={r_iso_n['recall']*100:5.1f}% P={r_iso_n['precision']*100:5.1f}% F1={r_iso_n['f1']*100:5.1f}%")

    # -----------------------------------------------------------------------
    # KDE on original-image features
    # -----------------------------------------------------------------------
    def kde_score_fn(img_id, X_all, is_tp_all, other_mask, all_mask):
        other_tp = other_mask & is_tp_all
        if other_tp.sum() < 5:
            return np.zeros(M)
        kde = KernelDensity(bandwidth="scott", kernel="gaussian")
        kde.fit(X_all[other_tp])
        return kde.score_samples(X_all)

    r_kde = evaluate_cv(X_pca, ious_arr, confs_arr, img_ids_arr, kde_score_fn, "OrigImg_KDE_CV")
    print(f"  {'OrigImg_KDE_CV':<30s}: R={r_kde['recall']*100:5.1f}% P={r_kde['precision']*100:5.1f}% F1={r_kde['f1']*100:5.1f}%")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    results = [r_euc, r_iso, r_kde]
    # Add n20, n30 if computed
    for n_nbr in [20, 30]:
        iso_n = Isomap(n_neighbors=n_nbr, n_components=6)
        X_iso_n = iso_n.fit_transform(X_pca)
        def iso_n_score_fn(img_id, X_all, is_tp_all, other_mask, all_mask):
            return -np.linalg.norm(X_iso_n - X_iso_n[is_tp_all].mean(axis=0), axis=1)
        r = evaluate_cv(X_pca, ious_arr, confs_arr, img_ids_arr, iso_n_score_fn, f"OrigImg_Isomap_n{n_nbr}_CV")
        results.append(r)

    print("\n" + "=" * 70)
    print("SUMMARY: Original-Image 64x64 FFT — Cross-Validated")
    print("=" * 70)
    print(f"{'Method':<35s} {'Recall%':>8s} {'Precision%':>10s} {'F1%':>8s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'TN':>4s}")
    print("-" * 70)

    for r in results:
        print(f"{r['name']:<35s} {r['recall']*100:>7.1f}% {r['precision']*100:>9.1f}% {r['f1']*100:>7.1f}% {r['true_positives']:>4d} {r['false_positives']:>4d} {r['false_negatives']:>4d} {r['true_negatives']:>4d}")

    best = max(results, key=lambda x: x["recall"])
    print(f"\nBEST: {best['name']} — Recall={best['recall']*100:.1f}%, Precision={best['precision']*100:.1f}%, F1={best['f1']*100:.1f}%")

    max_recall = best["recall"] * 100
    if max_recall > 70:
        print(f"\nVERDICT: CONTINUE — {best['name']} > 70% recall in CV.")
    elif max_recall > 65:
        print(f"\nVERDICT: MARGINAL — {max_recall:.1f}% between 65-70%.")
    else:
        print(f"\nVERDICT: CLOSE — {max_recall:.1f}% < 65%. Information ceiling likely.")

    print("=" * 70)

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)
    print(f"\nSaved to {OUT_JSON}")


if __name__ == "__main__":
    main()
