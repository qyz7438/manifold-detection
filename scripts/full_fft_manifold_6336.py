"""Full 6336-dim original-image FFT manifold analysis for Penn-Fudan val set.

This script preserves the COMPLETE 64x64 rfft2 amplitude structure (3 channels x 64 x 33 = 6336 dims)
instead of compressing to 3 scalar sums per band. It evaluates:

1. Pair consistency rate (all / uncertain-conf / borderline)
2. Hidden good box recall@uncertain_TP via leave-one-image-out CV
3. Comparison against 768-dim ROI FFT manifold and 9-dim scalar band sums

Pipeline:
    crop -> resize 64x64 -> rfft2 -> amplitude (N, 3, 64, 33) -> flatten (N, 6336)
    -> z-score -> PCA(50, whiten=True) -> Isomap(6, n_neighbors=15)
    -> pair consistency + hidden-good-box CV

Reference: scripts/test_hidden_good_box_origimg_fft.py (9-dim scalar version)
           scripts/validate_improved_fft_manifold.py (768-dim ROI version)
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
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
OUT_DIR = Path("scripts/full_fft_manifold_6336")
OUT_JSON = OUT_DIR / "results_6336_full.json"

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
# 1. Full 6336-dim original-image FFT extraction
# ---------------------------------------------------------------------------
def extract_full_image_fft(proposals: torch.Tensor, image: torch.Tensor,
                            resize_to: tuple[int, int] = (64, 64)) -> torch.Tensor:
    """Crop each proposal from original image, resize to 64x64, FFT, return FULL flat amplitude.

    Args:
        proposals: (N, 4) in [x1, y1, x2, y2], image coordinates.
        image: (3, H, W) original image tensor (already on device).

    Returns:
        (N, 6336) amplitude tensor, where 6336 = 3 * 64 * 33.
    """
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
    amp = torch.abs(fft)  # (N, 3, 64, 33)

    # Flatten to (N, 6336)
    amp_flat = amp.reshape(N, -1)  # (N, 6336)
    return amp_flat


# ---------------------------------------------------------------------------
# 2. Data collection
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_proposals_full_fft(model, val_loader, device):
    """Collect per-proposal IoU, confidence, and full 6336-dim original-image FFT."""
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
        full_fft = extract_full_image_fft(decoded, img_tensor)  # (N, 6336)

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

            records.append({
                "iou": iou.numpy(),
                "conf": conf[offset:offset + n_p].numpy(),
                "fft": full_fft[offset:offset + n_p].cpu().numpy(),  # (n_p, 6336)
                "gt_idx": gt_idx.numpy(),
                "img_id": img_idx,
            })
            offset += n_p

    return records


# ---------------------------------------------------------------------------
# 3. Pair building (same as validate_improved_fft_manifold.py)
# ---------------------------------------------------------------------------
def build_pairs(records, max_pairs=8000):
    """Group by GT and build all intra-group pairs."""
    pairs = []
    for img_rec in records:
        iou = img_rec["iou"]
        gt_idx = img_rec["gt_idx"]
        conf = img_rec["conf"]
        groups = defaultdict(list)
        for p in range(len(iou)):
            if gt_idx[p] >= 0:
                groups[int(gt_idx[p])].append(p)
        for gid, idxs in groups.items():
            if len(idxs) < 2:
                continue
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    a, b = idxs[i], idxs[j]
                    iou_a = float(iou[a])
                    iou_b = float(iou[b])
                    pair_type = "standard" if (iou_a > 0.5 and iou_b > 0.5) else \
                                ("relaxed" if (iou_a > 0.3 and iou_b > 0.3) else "mixed")
                    borderline = (0.3 <= iou_a <= 0.5) or (0.3 <= iou_b <= 0.5)
                    uncertain = ((conf[a] >= 0.1) and (conf[a] <= 0.5)) or \
                                ((conf[b] >= 0.1) and (conf[b] <= 0.5))
                    pairs.append({
                        "iou_a": iou_a,
                        "iou_b": iou_b,
                        "conf_a": float(conf[a]),
                        "conf_b": float(conf[b]),
                        "fft_a": img_rec["fft"][a],
                        "fft_b": img_rec["fft"][b],
                        "pair_type": pair_type,
                        "borderline": borderline,
                        "uncertain": uncertain,
                    })

    if len(pairs) > max_pairs:
        rng = np.random.RandomState(42)
        idxs = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in idxs]

    return pairs


# ---------------------------------------------------------------------------
# 4. Whitening: z-score + PCA(50, whiten=True)
#    FIT on TP subset only, transform all
# ---------------------------------------------------------------------------
def whiten_tp_fit(X: np.ndarray, tp_mask: np.ndarray, n_comp: int = 50) -> np.ndarray:
    """Z-score + PCA whitening. Fit scaler and PCA ONLY on TP samples."""
    X = np.asarray(X, dtype=np.float32)
    if X.shape[0] < 2:
        return X

    scaler = StandardScaler()
    scaler.fit(X[tp_mask])
    Xs = scaler.transform(X)

    k = min(n_comp, Xs.shape[0] - 1, Xs.shape[1])
    if k < 1:
        return Xs

    pca = PCA(n_components=k, whiten=True, random_state=42)
    pca.fit(Xs[tp_mask])
    return pca.transform(Xs)


# ---------------------------------------------------------------------------
# 5. Pair consistency metrics
# ---------------------------------------------------------------------------
def compute_pair_consistency(scores: np.ndarray, pairs) -> dict:
    """Compute pair consistency metrics for a score vector."""
    M = len(pairs)
    s_a = scores[:M]
    s_b = scores[M:]

    agree = 0; total = 0
    agree_border = 0; total_border = 0
    agree_uncertain = 0; total_uncertain = 0
    agree_standard = 0; total_standard = 0
    agree_relaxed = 0; total_relaxed = 0
    agree_mixed = 0; total_mixed = 0

    for i, p in enumerate(pairs):
        metric_order = s_a[i] > s_b[i]
        iou_order = p["iou_a"] > p["iou_b"]
        if metric_order == iou_order:
            agree += 1
        total += 1

        if p["borderline"]:
            if metric_order == iou_order:
                agree_border += 1
            total_border += 1

        if p["uncertain"]:
            if metric_order == iou_order:
                agree_uncertain += 1
            total_uncertain += 1

        if p["pair_type"] == "standard":
            if metric_order == iou_order:
                agree_standard += 1
            total_standard += 1
        elif p["pair_type"] == "relaxed":
            if metric_order == iou_order:
                agree_relaxed += 1
            total_relaxed += 1
        else:
            if metric_order == iou_order:
                agree_mixed += 1
            total_mixed += 1

    return {
        "pair_consistency": agree / total if total else 0.0,
        "n_pairs": total,
        "borderline_consistency": agree_border / total_border if total_border else 0.0,
        "n_borderline": total_border,
        "uncertain_consistency": agree_uncertain / total_uncertain if total_uncertain else 0.0,
        "n_uncertain": total_uncertain,
        "standard_consistency": agree_standard / total_standard if total_standard else 0.0,
        "n_standard": total_standard,
        "relaxed_consistency": agree_relaxed / total_relaxed if total_relaxed else 0.0,
        "n_relaxed": total_relaxed,
        "mixed_consistency": agree_mixed / total_mixed if total_mixed else 0.0,
        "n_mixed": total_mixed,
    }


# ---------------------------------------------------------------------------
# 6. Geometry methods (same as validate_improved_fft_manifold.py)
# ---------------------------------------------------------------------------
def geometry_fake_euclidean(X_w: np.ndarray, pairs) -> np.ndarray:
    """Old fake manifold: Euclidean distance to TP median in whitened space."""
    M = len(pairs)
    tp_mask = np.array([p["iou_a"] > 0.5 for p in pairs] + [p["iou_b"] > 0.5 for p in pairs])
    tp_center = X_w[tp_mask].mean(axis=0) if tp_mask.any() else X_w.mean(axis=0)
    d = np.linalg.norm(X_w - tp_center, axis=1)
    return -d


def geometry_gmm(X_w: np.ndarray, pairs) -> np.ndarray:
    """GMM log-likelihood: fit GMM on TP samples, score all."""
    from sklearn.mixture import GaussianMixture
    tp_mask = np.array([p["iou_a"] > 0.5 for p in pairs] + [p["iou_b"] > 0.5 for p in pairs])
    if tp_mask.sum() < 5:
        return geometry_fake_euclidean(X_w, pairs)
    try:
        gmm = GaussianMixture(n_components=min(5, tp_mask.sum()), random_state=42,
                              covariance_type="full", max_iter=100)
        gmm.fit(X_w[tp_mask])
        return gmm.score_samples(X_w)
    except Exception as e:
        print(f"    [GMM failed: {e}, falling back to fake_euclidean]")
        return geometry_fake_euclidean(X_w, pairs)


def geometry_kde(X_w: np.ndarray, pairs) -> np.ndarray:
    """KDE log-density: fit KDE on TP samples, score all."""
    tp_mask = np.array([p["iou_a"] > 0.5 for p in pairs] + [p["iou_b"] > 0.5 for p in pairs])
    if tp_mask.sum() < 5:
        return geometry_fake_euclidean(X_w, pairs)
    try:
        n_tp = tp_mask.sum()
        d = X_w.shape[1]
        bandwidth = max(0.5, n_tp ** (-1.0 / (d + 4)))
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth, rtol=1e-4)
        kde.fit(X_w[tp_mask])
        return kde.score_samples(X_w)
    except Exception as e:
        print(f"    [KDE failed: {e}, falling back to fake_euclidean]")
        return geometry_fake_euclidean(X_w, pairs)


def geometry_isomap(X_w: np.ndarray, pairs, n_neighbors: int = 15, n_components: int = 6) -> np.ndarray:
    """Isomap(6) on whitened features, then Euclidean distance to TP centroid in Isomap space."""
    tp_mask = np.array([p["iou_a"] > 0.5 for p in pairs] + [p["iou_b"] > 0.5 for p in pairs])
    try:
        iso = Isomap(n_neighbors=n_neighbors, n_components=n_components)
        X_iso = iso.fit_transform(X_w)
        tp_center = X_iso[tp_mask].mean(axis=0) if tp_mask.any() else X_iso.mean(axis=0)
        d = np.linalg.norm(X_iso - tp_center, axis=1)
        return -d
    except Exception as e:
        print(f"    [Isomap failed: {e}, falling back to fake_euclidean]")
        return geometry_fake_euclidean(X_w, pairs)


# ---------------------------------------------------------------------------
# 7. Hidden good box evaluation (leave-one-image-out CV)
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
# 8. Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("FULL 6336-DIM ORIGINAL-IMAGE FFT MANIFOLD ANALYSIS")
    print("=" * 80)

    print("\n[1/6] Loading model...")
    model = bm().to(DEV)
    ckpt = torch.load(CKPT, map_location=DEV)
    model.load_state_dict(ckpt["model"])

    print("[2/6] Loading val data...")
    _, val_loader = build_penn_fudan_loaders_320(batch_size=1)

    print("[3/6] Collecting proposals with full 6336-dim FFT...")
    records = collect_proposals_full_fft(model, val_loader, DEV)
    print(f"  -> {len(records)} images with proposals")

    # Build global arrays for CV
    all_fft = np.concatenate([r["fft"] for r in records], axis=0).astype(np.float32)
    all_iou = np.concatenate([r["iou"] for r in records])
    all_conf = np.concatenate([r["conf"] for r in records])
    all_img_id = np.concatenate([np.full(len(r["iou"]), r["img_id"]) for r in records])
    print(f"  -> Total proposals: {all_fft.shape[0]}, feature dim: {all_fft.shape[1]}")

    is_tp = all_iou >= 0.5
    is_fp = all_iou < 0.3
    uncertain_mask = (all_conf >= 0.1) & (all_conf <= 0.5)
    print(f"  -> TP: {is_tp.sum()}, FP: {is_fp.sum()}, Uncertain: {uncertain_mask.sum()}")
    print(f"  -> Uncertain TP: {(uncertain_mask & is_tp).sum()}, Uncertain FP: {(uncertain_mask & is_fp).sum()}")

    print("\n[4/6] Building pairs...")
    pairs = build_pairs(records, max_pairs=8000)
    print(f"  -> {len(pairs)} pairs")
    if len(pairs) == 0:
        print("No pairs found. Exiting.")
        return

    # Build pair feature matrix (2M, 6336)
    M = len(pairs)
    all_a = np.stack([p["fft_a"] for p in pairs], axis=0)
    all_b = np.stack([p["fft_b"] for p in pairs], axis=0)
    X_pairs = np.concatenate([all_a, all_b], axis=0).astype(np.float32)
    print(f"  -> Pair feature matrix: {X_pairs.shape}")

    # TP mask for pair matrix
    tp_mask_pairs = np.array([p["iou_a"] > 0.5 for p in pairs] + [p["iou_b"] > 0.5 for p in pairs])

    print("\n[5/6] Whitening (z-score + PCA(50, whiten=True), fit on TP only)...")
    X_w_pairs = whiten_tp_fit(X_pairs, tp_mask_pairs, n_comp=50)
    print(f"  -> Whitened dim: {X_w_pairs.shape[1]}")

    # Also whiten global proposal features for CV
    X_w_global = whiten_tp_fit(all_fft, is_tp, n_comp=50)
    print(f"  -> Global whitened dim: {X_w_global.shape[1]}")

    # -----------------------------------------------------------------------
    # Pair consistency evaluation
    # -----------------------------------------------------------------------
    print("\n[6/6] Pair consistency evaluation...")
    pair_results = {}

    for geo_name, geo_fn in [
        ("fake_euclidean", geometry_fake_euclidean),
        ("gmm", geometry_gmm),
        ("kde", geometry_kde),
        ("isomap_n15", lambda X, p: geometry_isomap(X, p, n_neighbors=15, n_components=6)),
        ("isomap_n20", lambda X, p: geometry_isomap(X, p, n_neighbors=20, n_components=6)),
        ("isomap_n30", lambda X, p: geometry_isomap(X, p, n_neighbors=30, n_components=6)),
    ]:
        r = compute_pair_consistency(geo_fn(X_w_pairs, pairs), pairs)
        pair_results[geo_name] = r
        print(f"  {geo_name:<20s}: all={r['pair_consistency']*100:5.1f}%  "
              f"border={r['borderline_consistency']*100:5.1f}%  "
              f"uncertain={r['uncertain_consistency']*100:5.1f}%  "
              f"std={r['standard_consistency']*100:5.1f}%  "
              f"relaxed={r['relaxed_consistency']*100:5.1f}%")

    # -----------------------------------------------------------------------
    # Hidden good box CV evaluation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("HIDDEN GOOD BOX DETECTION (Leave-One-Image-Out CV)")
    print("=" * 80)

    cv_results = []

    # Precompute Isomap embeddings for CV (fit on all data - no transform)
    iso_n15 = Isomap(n_neighbors=15, n_components=6)
    X_iso_n15 = iso_n15.fit_transform(X_w_global)

    iso_n20 = Isomap(n_neighbors=20, n_components=6)
    X_iso_n20 = iso_n20.fit_transform(X_w_global)

    # Euclidean on whitened space
    def euc_score_fn(img_id, X_all, is_tp_all, other_mask, all_mask):
        tp_centroid = X_all[is_tp_all].mean(axis=0)
        return -np.linalg.norm(X_all - tp_centroid, axis=1)

    r_euc = evaluate_cv(X_w_global, all_iou, all_conf, all_img_id, euc_score_fn, "6336_Euclidean_CV")
    print(f"  {'6336_Euclidean_CV':<30s}: R={r_euc['recall']*100:5.1f}% P={r_euc['precision']*100:5.1f}% F1={r_euc['f1']*100:5.1f}%")
    cv_results.append(r_euc)

    # Isomap n15
    def iso15_score_fn(img_id, X_all, is_tp_all, other_mask, all_mask):
        return -np.linalg.norm(X_iso_n15 - X_iso_n15[is_tp_all].mean(axis=0), axis=1)

    r_iso15 = evaluate_cv(X_w_global, all_iou, all_conf, all_img_id, iso15_score_fn, "6336_Isomap_n15_CV")
    print(f"  {'6336_Isomap_n15_CV':<30s}: R={r_iso15['recall']*100:5.1f}% P={r_iso15['precision']*100:5.1f}% F1={r_iso15['f1']*100:5.1f}%")
    cv_results.append(r_iso15)

    # Isomap n20
    def iso20_score_fn(img_id, X_all, is_tp_all, other_mask, all_mask):
        return -np.linalg.norm(X_iso_n20 - X_iso_n20[is_tp_all].mean(axis=0), axis=1)

    r_iso20 = evaluate_cv(X_w_global, all_iou, all_conf, all_img_id, iso20_score_fn, "6336_Isomap_n20_CV")
    print(f"  {'6336_Isomap_n20_CV':<30s}: R={r_iso20['recall']*100:5.1f}% P={r_iso20['precision']*100:5.1f}% F1={r_iso20['f1']*100:5.1f}%")
    cv_results.append(r_iso20)

    # KDE
    def kde_score_fn(img_id, X_all, is_tp_all, other_mask, all_mask):
        other_tp = other_mask & is_tp_all
        if other_tp.sum() < 5:
            return np.zeros(all_fft.shape[0])
        kde = KernelDensity(bandwidth="scott", kernel="gaussian")
        kde.fit(X_all[other_tp])
        return kde.score_samples(X_all)

    r_kde = evaluate_cv(X_w_global, all_iou, all_conf, all_img_id, kde_score_fn, "6336_KDE_CV")
    print(f"  {'6336_KDE_CV':<30s}: R={r_kde['recall']*100:5.1f}% P={r_kde['precision']*100:5.1f}% F1={r_kde['f1']*100:5.1f}%")
    cv_results.append(r_kde)

    # GMM
    def gmm_score_fn(img_id, X_all, is_tp_all, other_mask, all_mask):
        other_tp = other_mask & is_tp_all
        if other_tp.sum() < 5:
            return np.zeros(all_fft.shape[0])
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=min(5, other_tp.sum()), random_state=42,
                              covariance_type="full", max_iter=100, reg_covar=1e-2)
        gmm.fit(X_all[other_tp])
        return gmm.score_samples(X_all)

    r_gmm = evaluate_cv(X_w_global, all_iou, all_conf, all_img_id, gmm_score_fn, "6336_GMM_CV")
    print(f"  {'6336_GMM_CV':<30s}: R={r_gmm['recall']*100:5.1f}% P={r_gmm['precision']*100:5.1f}% F1={r_gmm['f1']*100:5.1f}%")
    cv_results.append(r_gmm)

    # -----------------------------------------------------------------------
    # Summary tables
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("SUMMARY TABLE 1: Pair Consistency (6336-dim Full FFT)")
    print("=" * 100)
    header = f"{'Geometry':<20} {'Pairs':>7} {'All%':>8} {'Border%':>8} {'Uncertain%':>10} {'Std%':>8} {'Relaxed%':>8}"
    print(header)
    print("-" * 100)
    for geo_name, r in pair_results.items():
        print(f"{geo_name:<20} {r['n_pairs']:>7} {r['pair_consistency']*100:>7.1f}% "
              f"{r['borderline_consistency']*100:>7.1f}% {r['uncertain_consistency']*100:>9.1f}% "
              f"{r['standard_consistency']*100:>7.1f}% {r['relaxed_consistency']*100:>7.1f}%")
    print("-" * 100)
    print("Target: >70% consistency for DPO viability")
    print("=" * 100)

    print("\n" + "=" * 100)
    print("SUMMARY TABLE 2: Hidden Good Box Recall @ Uncertain (CV)")
    print("=" * 100)
    header = f"{'Method':<30} {'Recall%':>8} {'Precision%':>10} {'F1%':>8} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}"
    print(header)
    print("-" * 100)
    for r in cv_results:
        print(f"{r['name']:<30} {r['recall']*100:>7.1f}% {r['precision']*100:>9.1f}% {r['f1']*100:>7.1f}% "
              f"{r['true_positives']:>4d} {r['false_positives']:>4d} {r['false_negatives']:>4d} {r['true_negatives']:>4d}")
    print("-" * 100)

    best_cv = max(cv_results, key=lambda x: x["recall"])
    print(f"\nBEST CV RECALL: {best_cv['name']} — Recall={best_cv['recall']*100:.1f}%, Precision={best_cv['precision']*100:.1f}%, F1={best_cv['f1']*100:.1f}%")

    max_recall = best_cv["recall"] * 100
    if max_recall > 70:
        print(f"\nVERDICT: CONTINUE — {best_cv['name']} > 70% recall in CV.")
    elif max_recall > 65:
        print(f"\nVERDICT: MARGINAL — {max_recall:.1f}% between 65-70%.")
    else:
        print(f"\nVERDICT: CLOSE — {max_recall:.1f}% < 65%. Information ceiling likely.")

    # -----------------------------------------------------------------------
    # Comparison with prior baselines
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("COMPARISON WITH PRIOR BASELINES")
    print("=" * 100)
    print("""
Prior baselines (from earlier scripts):
  - 9-dim scalar band sums (lo/mid/hi per channel): ~55.8% pair consistency
  - 768-dim ROI FFT manifold (7x7 ROI features):   ~58-62% pair consistency
  - 7168-dim full ROI flatten:                      ~60-65% pair consistency

Current 6336-dim full original-image FFT:
  - Best pair consistency: {:.1f}% ({})
  - Best CV recall:        {:.1f}% ({})
""".format(
        max(pair_results.values(), key=lambda x: x["pair_consistency"])["pair_consistency"] * 100,
        max(pair_results, key=lambda k: pair_results[k]["pair_consistency"]),
        max_recall,
        best_cv["name"]
    ))

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "pair_results": {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                            for kk, vv in v.items()} for k, v in pair_results.items()},
        "cv_results": cv_results,
        "n_images": len(records),
        "n_pairs": len(pairs),
        "n_proposals": int(all_fft.shape[0]),
        "feature_dim": int(all_fft.shape[1]),
        "whitened_dim": int(X_w_pairs.shape[1]),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)
    print(f"\nSaved to {OUT_JSON}")


if __name__ == "__main__":
    main()
