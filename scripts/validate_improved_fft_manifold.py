"""Validate two improvements on Penn-Fudan val set (no training).

Improvement 1: FFT on original image proposal crops (not ROI feature maps).
    - Crop proposal from original image -> resize to 64x64 -> rfft2 -> amplitude bands
    - This gives physically meaningful frequencies (not 7x7 artifact).

Improvement 2: Real manifold geometry (GMM / KDE / geodesic) instead of fake
    Euclidean-distance-to-median.

For each GT group, we build proposal pairs and measure:
    - Pair consistency rate: does the metric rank the pair the same as IoU?
    - AUC of (confidence + geometric_score) vs confidence-only
    - Comparison against old "fake manifold" (Euclidean to TP median)

Groups:
    - Standard TP   : IoU > 0.5
    - Relaxed TP    : IoU > 0.3
    - Borderline    : IoU in [0.3, 0.5)

Target: pair consistency > 70% (old fake manifold baseline).
"""
from __future__ import annotations

import sys
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity, NearestNeighbors
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from scipy import stats

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.experiments.runner_utils import (
    build_penn_fudan_loaders_320,
    decode_boxes,
)
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed

warnings.filterwarnings("ignore", category=UserWarning)

set_seed(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
OUT_DIR = Path("scripts/validate_improved_fft_manifold")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
# 1. Original-image FFT extraction
# ---------------------------------------------------------------------------

def extract_original_image_fft(proposals: torch.Tensor, image: torch.Tensor,
                                 resize_to: tuple[int, int] = (64, 64),
                                 bands: tuple[float, float] = (0.3, 0.7)) -> dict[str, torch.Tensor]:
    """Crop each proposal from the original image, resize, then FFT.

    Args:
        proposals: (N, 4) in [x1, y1, x2, y2], image coordinates.
        image: (3, H, W) original image tensor (already on device).

    Returns:
        Dict with per-band amplitude tensors, each (N, 3, Hf, Wf).
        3 channels = RGB; Hf, Wf from rfft2 of resize_to.
    """
    lo_thr, hi_thr = bands
    N = proposals.shape[0]
    H, W = image.shape[1], image.shape[2]

    # Clamp boxes to image bounds and ensure valid
    boxes = proposals.clone()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, W - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, H - 1)
    # Ensure x2 > x1, y2 > y1
    boxes[:, 2] = torch.maximum(boxes[:, 2], boxes[:, 0] + 1)
    boxes[:, 3] = torch.maximum(boxes[:, 3], boxes[:, 1] + 1)

    crops = []
    for i in range(N):
        x1, y1, x2, y2 = boxes[i].long()
        crop = image[:, y1:y2, x1:x2]  # (3, h, w)
        # Resize to target
        crop = F.interpolate(crop.unsqueeze(0), size=resize_to, mode="bilinear", align_corners=False)
        crops.append(crop.squeeze(0))  # (3, 64, 64)

    crops = torch.stack(crops, dim=0)  # (N, 3, 64, 64)

    # FFT per channel
    fft = torch.fft.rfft2(crops, dim=(-2, -1), norm="ortho")  # (N, 3, 64, 33)
    amp = torch.abs(fft)

    # Frequency masks
    freq_h = torch.fft.fftfreq(resize_to[0], device=image.device)
    freq_w = torch.fft.rfftfreq(resize_to[1], device=image.device)
    grid_y, grid_x = torch.meshgrid(freq_h, freq_w, indexing="ij")
    radius = torch.sqrt(grid_x ** 2 + grid_y ** 2)
    radius = radius / radius.max().clamp_min(1e-6)

    lo_mask = (radius <= lo_thr).float().unsqueeze(0).unsqueeze(0)      # (1, 1, 64, 33)
    mid_mask = ((radius > lo_thr) & (radius <= hi_thr)).float().unsqueeze(0).unsqueeze(0)
    hi_mask = (radius > hi_thr).float().unsqueeze(0).unsqueeze(0)

    return {
        "amp_lo": amp * lo_mask,
        "amp_mid": amp * mid_mask,
        "amp_hi": amp * hi_mask,
    }


def feat_band_stats(band: torch.Tensor) -> np.ndarray:
    """Per-channel mean/std/max over freq grid.
    Input: (N, C, Hf, Wf) -> (N, 3*C).
    Returns: (N, D) numpy.
    """
    if band.dim() == 3:
        band = band.unsqueeze(0)
    N, C, Hf, Wf = band.shape
    flat = band.reshape(N, C, -1)  # (N, C, Hf*Wf)
    mu = flat.mean(dim=-1)         # (N, C)
    sg = flat.std(dim=-1)          # (N, C)
    mx = flat.max(dim=-1).values   # (N, C)
    return torch.cat([mu, sg, mx], dim=1).cpu().numpy()


def feat_band_topk(band: torch.Tensor, k: int = 8) -> np.ndarray:
    """Per-channel top-k frequency values. (N, C, Hf, Wf) -> (N, C*k)."""
    if band.dim() == 3:
        band = band.unsqueeze(0)
    N, C, Hf, Wf = band.shape
    flat = band.reshape(N, C, -1)
    topk_vals, _ = torch.topk(flat, k=min(k, flat.shape[-1]), dim=-1)
    return topk_vals.reshape(N, -1).cpu().numpy()


def feat_band_flatten(band: torch.Tensor) -> np.ndarray:
    """Direct flatten. (N, C, Hf, Wf) -> (N, C*Hf*Wf)."""
    if band.dim() == 3:
        band = band.unsqueeze(0)
    return band.flatten(1).cpu().numpy()


EXTRACTORS = {
    "stats": feat_band_stats,
    "topk8": lambda b: feat_band_topk(b, k=8),
    "topk16": lambda b: feat_band_topk(b, k=16),
    "flatten": feat_band_flatten,
}


# ---------------------------------------------------------------------------
# 2. Data collection: proposals + original-image FFT
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_proposals_with_image_fft(model, val_loader, device):
    """Collect per-proposal IoU, confidence, and original-image FFT bands."""
    model.eval()
    records = []
    sampled_props, box_head_in = {}, {}

    model.roi_heads.box_roi_pool.register_forward_pre_hook(
        lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]})
    )
    model.roi_heads.box_head.register_forward_pre_hook(
        lambda m, args: box_head_in.update({"x": args[0]})
    )

    for images, targets in tqdm(val_loader, desc="inference"):
        images_d = [img.to(device) for img in images]
        tgts_t = [{k: v.to(device) for k, v in t.items()} for t in targets]
        sampled_props.clear()
        box_head_in.clear()

        model(images_d, tgts_t)

        sp_raw = sampled_props.get("p")
        rf = box_head_in.get("x")
        if sp_raw is None or rf is None or rf.shape[0] == 0:
            continue

        # classifier confidence
        bf = model.roi_heads.box_head(rf)
        cls_logits = model.roi_heads.box_predictor.cls_score(bf)
        conf = F.softmax(cls_logits, dim=-1)[:, 1].cpu()

        sp_cat = torch.cat(sp_raw, dim=0)
        N = sp_cat.shape[0]
        reg_out = model.roi_heads.box_predictor.bbox_pred(bf[:N])
        person_deltas = reg_out[:, 2:6]
        decoded = decode_boxes(sp_cat, person_deltas)

        # Original-image FFT on decoded boxes
        img_tensor = images_d[0]  # (3, H, W)
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

            img_bands = {k: v[offset:offset + n_p].cpu() for k, v in bands.items()}

            records.append({
                "iou": iou,
                "conf": conf[offset:offset + n_p],
                "bands": img_bands,
                "gt_idx": gt_idx,
                "boxes": decoded[offset:offset + n_p].cpu(),
                "image": img_tensor.cpu(),
            })
            offset += n_p

    return records


# ---------------------------------------------------------------------------
# 3. Pair building
# ---------------------------------------------------------------------------

def build_pairs(records, max_pairs=8000):
    """Group by GT and build all intra-group pairs."""
    pairs = []
    for img_rec in records:
        iou = img_rec["iou"]
        bands = img_rec["bands"]
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
                    pairs.append({
                        "iou_a": iou_a,
                        "iou_b": iou_b,
                        "conf_a": float(conf[a]),
                        "conf_b": float(conf[b]),
                        "bands_a": {k: v[a].numpy() for k, v in bands.items()},
                        "bands_b": {k: v[b].numpy() for k, v in bands.items()},
                        "pair_type": pair_type,
                        "borderline": borderline,
                    })

    # Subsample to manageable size
    if len(pairs) > max_pairs:
        rng = np.random.RandomState(42)
        idxs = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in idxs]

    return pairs


# ---------------------------------------------------------------------------
# 4. Feature extraction + whitening
# ---------------------------------------------------------------------------

def whiten(X: np.ndarray, n_comp: int = 50) -> np.ndarray:
    """Z-score + PCA whitening."""
    X = np.asarray(X, dtype=np.float32)
    if X.shape[0] < 2:
        return X
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    k = min(n_comp, Xs.shape[0] - 1, Xs.shape[1])
    if k < 1:
        return Xs
    pca = PCA(n_components=k, whiten=True, random_state=42)
    return pca.fit_transform(Xs)


def build_feature_matrix(pairs, band_names, extractor_fn):
    """Build (2M, D) feature matrix from pairs."""
    all_vecs = []
    for p in pairs:
        for side in ["bands_a", "bands_b"]:
            feat = np.concatenate([extractor_fn(torch.from_numpy(p[side][bn])).flatten()
                                      for bn in band_names], axis=0).astype(np.float32)
            all_vecs.append(feat)
    return np.stack(all_vecs, axis=0)


# ---------------------------------------------------------------------------
# 5. Geometry methods
# ---------------------------------------------------------------------------

def geometry_fake_euclidean(X_w: np.ndarray, pairs) -> np.ndarray:
    """Old fake manifold: Euclidean distance to TP median in whitened space."""
    M = len(pairs)
    tp_mask = np.array([p["iou_a"] > 0.5 for p in pairs] + [p["iou_b"] > 0.5 for p in pairs])
    tp_center = X_w[tp_mask].mean(axis=0) if tp_mask.any() else X_w.mean(axis=0)
    d = np.linalg.norm(X_w - tp_center, axis=1)
    return -d  # higher = closer to TP = better


def geometry_gmm(X_w: np.ndarray, pairs) -> np.ndarray:
    """GMM log-likelihood: fit GMM on TP samples, score all. Higher = more like TP."""
    tp_mask = np.array([p["iou_a"] > 0.5 for p in pairs] + [p["iou_b"] > 0.5 for p in pairs])
    if tp_mask.sum() < 5:
        return geometry_fake_euclidean(X_w, pairs)

    try:
        gmm = GaussianMixture(n_components=min(5, tp_mask.sum()), random_state=42,
                              covariance_type="full", max_iter=100)
        gmm.fit(X_w[tp_mask])
        scores = gmm.score_samples(X_w)  # log-likelihood
        return scores
    except Exception as e:
        print(f"    [GMM failed: {e}, falling back to fake_euclidean]")
        return geometry_fake_euclidean(X_w, pairs)


def geometry_kde(X_w: np.ndarray, pairs) -> np.ndarray:
    """KDE log-density: fit KDE on TP samples, score all. Higher = more like TP."""
    tp_mask = np.array([p["iou_a"] > 0.5 for p in pairs] + [p["iou_b"] > 0.5 for p in pairs])
    if tp_mask.sum() < 5:
        return geometry_fake_euclidean(X_w, pairs)

    try:
        # Bandwidth heuristic: Scott's rule ~ n^(-1/(d+4))
        n_tp = tp_mask.sum()
        d = X_w.shape[1]
        bandwidth = max(0.5, n_tp ** (-1.0 / (d + 4)))
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth, rtol=1e-4)
        kde.fit(X_w[tp_mask])
        scores = kde.score_samples(X_w)
        return scores
    except Exception as e:
        print(f"    [KDE failed: {e}, falling back to fake_euclidean]")
        return geometry_fake_euclidean(X_w, pairs)


def geometry_geodesic(X_w: np.ndarray, pairs, k_nn: int = 15) -> np.ndarray:
    """Geodesic distance on k-NN graph: Dijkstra from TP center."""
    n_total = X_w.shape[0]

    # k-NN graph
    nbrs = NearestNeighbors(n_neighbors=min(k_nn + 1, n_total), algorithm="auto",
                             metric="euclidean", n_jobs=-1)
    nbrs.fit(X_w)
    distances, indices = nbrs.kneighbors(X_w)
    indices = indices[:, 1:]   # remove self
    distances = distances[:, 1:]

    row_idx = np.repeat(np.arange(n_total), indices.shape[1])
    col_idx = indices.flatten()
    data = distances.flatten()
    adj = csr_matrix((data, (row_idx, col_idx)), shape=(n_total, n_total))
    adj = adj.maximum(adj.T)

    # TP center = median of 5 closest to TP centroid
    tp_mask = np.array([p["iou_a"] > 0.5 for p in pairs] + [p["iou_b"] > 0.5 for p in pairs])
    if tp_mask.sum() < 5:
        return geometry_fake_euclidean(X_w, pairs)

    tp_centroid = X_w[tp_mask].mean(axis=0)
    tp_indices = np.where(tp_mask)[0]
    tp_to_centroid = np.linalg.norm(X_w[tp_indices] - tp_centroid, axis=1)
    closest5 = tp_indices[np.argsort(tp_to_centroid)[:5]]
    tp_center_idx = int(np.median(closest5))

    geodesic_dist = shortest_path(adj, method="D", directed=False, unweighted=False)
    geo_to_tp = geodesic_dist[:, tp_center_idx]

    # Handle disconnected
    n_inf = np.isinf(geo_to_tp).sum()
    if n_inf > 0:
        max_geo = geo_to_tp[np.isfinite(geo_to_tp)].max()
        geo_to_tp = np.where(np.isinf(geo_to_tp), max_geo * 2, geo_to_tp)

    return -geo_to_tp  # higher = closer to TP = better


# ---------------------------------------------------------------------------
# 6. Metrics
# ---------------------------------------------------------------------------

def compute_pair_consistency(scores: np.ndarray, pairs) -> dict:
    """Compute pair consistency metrics for a score vector."""
    M = len(pairs)
    s_a = scores[:M]
    s_b = scores[M:]

    agree = 0; total = 0
    agree_border = 0; total_border = 0
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
        "standard_consistency": agree_standard / total_standard if total_standard else 0.0,
        "n_standard": total_standard,
        "relaxed_consistency": agree_relaxed / total_relaxed if total_relaxed else 0.0,
        "n_relaxed": total_relaxed,
        "mixed_consistency": agree_mixed / total_mixed if total_mixed else 0.0,
        "n_mixed": total_mixed,
    }


def compute_auc(conf_scores: np.ndarray, geo_scores: np.ndarray, ious: np.ndarray) -> dict:
    """Compute AUC for (conf + geo) vs conf-only in binary classification (TP vs FP)."""
    from sklearn.metrics import roc_auc_score

    labels = (ious > 0.5).astype(int)
    if labels.sum() == 0 or (1 - labels).sum() == 0:
        return {"auc_conf_only": 0.5, "auc_conf_geo": 0.5, "delta": 0.0}

    # Normalize geo_scores to [0, 1] for combination
    geo_min, geo_max = geo_scores.min(), geo_scores.max()
    geo_norm = (geo_scores - geo_min) / (geo_max - geo_min + 1e-8)

    # Conf-only
    auc_conf = roc_auc_score(labels, conf_scores)
    # Conf + geo (equal weight)
    combined = conf_scores + geo_norm
    auc_combined = roc_auc_score(labels, combined)

    return {
        "auc_conf_only": float(auc_conf),
        "auc_conf_geo": float(auc_combined),
        "delta": float(auc_combined - auc_conf),
    }


def evaluate_geometry(X_w: np.ndarray, pairs, geometry_fn, name: str) -> dict:
    """Run a geometry method and compute all metrics."""
    scores = geometry_fn(X_w, pairs)
    M = len(pairs)

    # Pair consistency
    consistency = compute_pair_consistency(scores, pairs)

    # AUC (need per-proposal scores, not pair scores)
    all_conf = np.array([p["conf_a"] for p in pairs] + [p["conf_b"] for p in pairs])
    all_iou = np.array([p["iou_a"] for p in pairs] + [p["iou_b"] for p in pairs])

    auc = compute_auc(all_conf, scores, all_iou)

    return {
        "name": name,
        **consistency,
        **auc,
    }


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("Validation: Original-Image FFT + Real Manifold Geometry")
    print("=" * 80)

    print("\n[1/6] Loading model...")
    model = bm().to(DEV)
    ckpt = torch.load(CKPT, map_location=DEV)
    model.load_state_dict(ckpt["model"])

    print("[2/6] Loading val data...")
    _, val_loader = build_penn_fudan_loaders_320(batch_size=1)

    print("[3/6] Collecting proposals with original-image FFT...")
    records = collect_proposals_with_image_fft(model, val_loader, DEV)
    print(f"  -> {len(records)} images with proposals")

    print("[4/6] Building pairs within same GT...")
    pairs = build_pairs(records, max_pairs=8000)
    print(f"  -> {len(pairs)} pairs")
    if len(pairs) == 0:
        print("No pairs found. Exiting.")
        return

    # -----------------------------------------------------------------------
    # Try multiple band + extractor combinations
    # -----------------------------------------------------------------------
    best_results = {}

    for band_combo_name, band_names in [
        ("all_bands", ["amp_lo", "amp_mid", "amp_hi"]),
        ("lo_mid", ["amp_lo", "amp_mid"]),
        ("mid_hi", ["amp_mid", "amp_hi"]),
    ]:
        for ext_name in ["stats", "topk8"]:
            ext_fn = EXTRACTORS[ext_name]
            config_key = f"{band_combo_name}_{ext_name}"
            print(f"\n[5/6] Config: {config_key}")

            # Build feature matrix
            X = build_feature_matrix(pairs, band_names, ext_fn)
            print(f"  Feature dim: {X.shape[1]}")

            # Whiten
            X_w = whiten(X, n_comp=50)
            print(f"  Whitened dim: {X_w.shape[1]}")

            # Evaluate geometry methods (skip geodesic - too expensive)
            results = {}
            for geo_name, geo_fn in [
                ("fake_euclidean", geometry_fake_euclidean),
                ("gmm", geometry_gmm),
                ("kde", geometry_kde),
            ]:
                r = evaluate_geometry(X_w, pairs, geo_fn, geo_name)
                results[geo_name] = r
                print(f"    {geo_name:15s}: pair_consist={r['pair_consistency']*100:5.1f}%  "
                      f"border={r['borderline_consistency']*100:5.1f}%  "
                      f"standard={r['standard_consistency']*100:5.1f}%  "
                      f"AUC delta={r['delta']:+.4f}")

            best_results[config_key] = results

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("SUMMARY: Best pair consistency per config")
    print("=" * 100)
    header = f"{'Config':<25} {'Geometry':<15} {'Pairs':>7} {'Consist%':>10} {'Border%':>10} {'Std%':>10} {'Relaxed%':>10} {'AUC+d':>8}"
    print(header)
    print("-" * 100)

    overall_best = None
    overall_best_key = None
    overall_best_geo = None

    for config_key in sorted(best_results.keys()):
        for geo_name in ["fake_euclidean", "gmm", "kde"]:
            r = best_results[config_key][geo_name]
            print(f"{config_key:<25} {geo_name:<15} {r['n_pairs']:>7} "
                  f"{r['pair_consistency']*100:>9.1f}% {r['borderline_consistency']*100:>9.1f}% "
                  f"{r['standard_consistency']*100:>9.1f}% {r['relaxed_consistency']*100:>9.1f}% "
                  f"{r['delta']:>+7.4f}")

            if overall_best is None or r["pair_consistency"] > overall_best["pair_consistency"]:
                overall_best = r
                overall_best_key = config_key
                overall_best_geo = geo_name

    print("-" * 100)
    print(f"\nOVERALL BEST: {overall_best_key} + {overall_best_geo}")
    print(f"  Pair consistency      = {overall_best['pair_consistency']*100:.1f}%")
    print(f"  Borderline consistency  = {overall_best['borderline_consistency']*100:.1f}%")
    print(f"  Standard consistency    = {overall_best['standard_consistency']*100:.1f}%")
    print(f"  Relaxed consistency     = {overall_best['relaxed_consistency']*100:.1f}%")
    print(f"  AUC improvement       = {overall_best['delta']:+.4f}")

    # Compare to old fake baseline
    fake_baseline = None
    for config_key in best_results:
        if "fake_euclidean" in best_results[config_key]:
            if fake_baseline is None or best_results[config_key]["fake_euclidean"]["pair_consistency"] > fake_baseline["pair_consistency"]:
                fake_baseline = best_results[config_key]["fake_euclidean"]

    if fake_baseline:
        print(f"\nvs OLD FAKE BASELINE (best config):")
        print(f"  Old pair consistency = {fake_baseline['pair_consistency']*100:.1f}%")
        print(f"  New - Old            = {(overall_best['pair_consistency'] - fake_baseline['pair_consistency'])*100:+.1f} pp")
        if overall_best['pair_consistency'] > fake_baseline['pair_consistency']:
            print("  => IMPROVED")
        else:
            print("  => NO IMPROVEMENT")

    # Target check
    if overall_best['pair_consistency'] > 0.70:
        print(f"\nTARGET MET: pair consistency > 70%")
    else:
        print(f"\nTARGET NOT MET: pair consistency = {overall_best['pair_consistency']*100:.1f}% (need > 70%)")

    # Save
    out = {
        "best_config": overall_best_key,
        "best_geometry": overall_best_geo,
        "best_result": {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in overall_best.items()},
        "all_results": {
            ck: {
                gk: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                     for kk, vv in gv.items()}
                for gk, gv in cv.items()
            }
            for ck, cv in best_results.items()
        },
    }
    out_path = OUT_DIR / "validation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
