"""Complete manifold geometry diagnostic for FFT features.

Before using manifold structure as a reward signal, verify whether the data
actually forms a usable manifold.  This script runs five diagnostic tests:

1. Clustering quality (silhouette, Davies-Bouldin)
2. Permutation test for pair-consistency
3. Intrinsic dimension (skdim TLE)
4. k-NN graph connectivity
5. Local curvature estimate

Data: PennFudan val set (34 images), per-proposal FFT amplitude features.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import connected_components
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torchvision.ops import box_iou
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.experiments.runner_utils import (
    build_penn_fudan_loaders_320,
    decode_boxes,
)
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed

warnings.filterwarnings("ignore")
set_seed(42)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
OUT_DIR = Path("scripts/manifold_geometry_diagnostic")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load model & data
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


print("Loading model...")
model = bm().to(DEV)
ckpt = torch.load(CKPT, map_location=DEV)
model.load_state_dict(ckpt["model"])
model.eval()

# Hooks
box_head_in: dict = {}
roi_crops: dict = {}
sampled_props: dict = {}

model.roi_heads.box_head.register_forward_pre_hook(
    lambda m, args: box_head_in.update({"x": args[0]})
)
model.roi_heads.box_roi_pool.register_forward_pre_hook(
    lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]})
)
model.roi_heads.box_roi_pool.register_forward_hook(
    lambda m, i, o: roi_crops.update({"c": o.clone()})
)

_, vl = build_penn_fudan_loaders_320(batch_size=1)

# ---------------------------------------------------------------------------
# Helper: FFT band extraction (copied from bandwise_fft_manifold.py)
# ---------------------------------------------------------------------------


def extract_fft_bands(x: torch.Tensor, bands: tuple[float, float] = (0.3, 0.7)) -> dict[str, torch.Tensor]:
    lo_thr, hi_thr = bands
    fft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
    amp = torch.abs(fft)

    freq_h = torch.fft.fftfreq(x.shape[-2], device=x.device)
    freq_w = torch.fft.rfftfreq(x.shape[-1], device=x.device)
    grid_y, grid_x = torch.meshgrid(freq_h, freq_w, indexing="ij")
    radius = torch.sqrt(grid_x ** 2 + grid_y ** 2)
    radius = radius / radius.max().clamp_min(1e-6)

    lo_mask = (radius <= lo_thr).float()
    mid_mask = ((radius > lo_thr) & (radius <= hi_thr)).float()
    hi_mask = (radius > hi_thr).float()

    lo_m = lo_mask.unsqueeze(0).unsqueeze(0)
    mid_m = mid_mask.unsqueeze(0).unsqueeze(0)
    hi_m = hi_mask.unsqueeze(0).unsqueeze(0)

    return {
        "amp_lo": amp * lo_m,
        "amp_mid": amp * mid_m,
        "amp_hi": amp * hi_m,
    }


def feat_per_channel_stats(band: torch.Tensor) -> np.ndarray:
    """Per-channel mean/std/max over freq bins. Returns (N, 3*C)."""
    if band.dim() == 3:
        band = band.unsqueeze(0)
    flat = band.reshape(band.shape[0], band.shape[1], -1)
    mu = flat.mean(dim=-1)
    sg = flat.std(dim=-1)
    mx = flat.max(dim=-1).values
    return torch.cat([mu, sg, mx], dim=1).cpu().numpy()


# ---------------------------------------------------------------------------
# 2. Extract proposals + FFT features
# ---------------------------------------------------------------------------

all_fft_amp: list[np.ndarray] = []
all_iou: list[float] = []
all_conf: list[float] = []
all_gt_id: list[int] = []
all_img_id: list[int] = []

print("Extracting proposals and FFT features from val set...")
for img_idx, (img, tgt) in enumerate(tqdm(vl, desc="Val images")):
    img_d = [img[0].to(DEV)]
    tgt_d = [{k: v.to(DEV) for k, v in tgt[0].items()}]
    box_head_in.clear()
    roi_crops.clear()
    sampled_props.clear()

    with torch.no_grad():
        model(img_d, tgt_d)

    rf = box_head_in.get("x")
    crops = roi_crops.get("c")
    sp_raw = sampled_props.get("p")
    if rf is None or crops is None or sp_raw is None or rf.shape[0] == 0:
        continue

    sp_cat = torch.cat(sp_raw, dim=0)
    bf = model.roi_heads.box_head(rf)
    reg_out = model.roi_heads.box_predictor.bbox_pred(bf)
    person_deltas = reg_out[:, 2:6]
    decoded = decode_boxes(sp_cat, person_deltas)
    gt_boxes = tgt_d[0]["boxes"]

    N = sp_cat.shape[0]

    # --- Feature extraction: two variants ---
    # Variant A: amp_lo per-channel stats (768-dim) from ROI features
    bands = extract_fft_bands(rf[:N].cpu())
    amp_lo = bands["amp_lo"]  # (N, C, Hf, Wf)
    feat_A = feat_per_channel_stats(amp_lo)  # (N, 3*C) -> flattened by helper

    # Variant B: raw 64x64 crop FFT (6336-dim)
    f = torch.fft.rfft2(crops, dim=(-2, -1), norm="ortho")  # (N, C, 7, 4)
    amp_raw = torch.abs(f)
    feat_B = amp_raw.view(N, -1).cpu().numpy()  # (N, 7168)

    conf = F.softmax(model.roi_heads.box_predictor.cls_score(bf), dim=-1)[:, 1]

    if len(gt_boxes) == 0:
        for i in range(N):
            all_fft_amp.append(feat_A[i])
            all_iou.append(0.0)
            all_conf.append(conf[i].item())
            all_gt_id.append(-1)
            all_img_id.append(img_idx)
        continue

    ious = box_iou(decoded, gt_boxes)
    best_iou, best_gt = ious.max(dim=1)

    for i in range(N):
        all_fft_amp.append(feat_A[i])
        all_iou.append(best_iou[i].item())
        all_conf.append(conf[i].item())
        all_gt_id.append(int(best_gt[i].item()))
        all_img_id.append(img_idx)

X_raw = np.stack(all_fft_amp, axis=0)  # (N_total, D)
ious_arr = np.array(all_iou, dtype=np.float64)
confs_arr = np.array(all_conf, dtype=np.float64)
gt_ids_arr = np.array(all_gt_id, dtype=np.int32)
img_ids_arr = np.array(all_img_id, dtype=np.int32)

N_total, D = X_raw.shape
print(f"\nTotal proposals: {N_total}, Feature dim: {D}")
print(f"TP (IoU>=0.5): {(ious_arr >= 0.5).sum()}, FP (IoU<0.5): {(ious_arr < 0.5).sum()}")

# Labels for TP vs FP
tp_fp_labels = (ious_arr >= 0.5).astype(int)

# ---------------------------------------------------------------------------
# Preprocessing: z-score + PCA whitening
# ---------------------------------------------------------------------------

print("\n[Preprocessing] Z-score + PCA whitening...")
scaler = StandardScaler()
X_z = scaler.fit_transform(X_raw)

pca = PCA(n_components=min(50, N_total - 1, D), whiten=True, random_state=42)
X_w = pca.fit_transform(X_z)
print(f"  Whitened shape: {X_w.shape}, explained variance: {pca.explained_variance_ratio_.sum():.4f}")

# Split TP / FP
tp_mask = ious_arr >= 0.5
fp_mask = ~tp_mask
X_w_tp = X_w[tp_mask]
X_w_fp = X_w[fp_mask]

# ---------------------------------------------------------------------------
# Diagnostic 1: Clustering Quality
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("DIAGNOSTIC 1: Clustering Quality")
print("=" * 70)

# 1a. TP vs FP clustering
if len(np.unique(tp_fp_labels)) > 1 and X_w.shape[0] > 2:
    try:
        sil_all = silhouette_score(X_w, tp_fp_labels)
        db_all = davies_bouldin_score(X_w, tp_fp_labels)
        print(f"  TP vs FP silhouette: {sil_all:.4f}")
        print(f"  TP vs FP Davies-Bouldin: {db_all:.4f}")
        print(f"  -> {'Has cluster structure' if sil_all > 0.3 else 'Weak clustering' if sil_all > 0.1 else 'No cluster structure (uniform)'}")
    except Exception as e:
        print(f"  Error computing clustering metrics: {e}")
        sil_all = db_all = None
else:
    print("  Skipped: insufficient labels")
    sil_all = db_all = None

# 1b. GMM-like clustering within TP (by GT assignment)
tp_gt_ids = gt_ids_arr[tp_mask]
unique_gts = np.unique(tp_gt_ids[tp_gt_ids >= 0])
if len(unique_gts) > 1 and X_w_tp.shape[0] > 2:
    # Create labels: each GT is a cluster
    tp_cluster_labels = np.full(X_w_tp.shape[0], -1, dtype=int)
    for idx, gid in enumerate(unique_gts):
        tp_cluster_labels[tp_gt_ids == gid] = idx
    valid = tp_cluster_labels >= 0
    if valid.sum() > 2 and len(np.unique(tp_cluster_labels[valid])) > 1:
        sil_tp = silhouette_score(X_w_tp[valid], tp_cluster_labels[valid])
        print(f"  TP-by-GT silhouette: {sil_tp:.4f}")
        print(f"  -> {'TPs cluster by GT' if sil_tp > 0.3 else 'Weak TP clustering' if sil_tp > 0.1 else 'TPs do not cluster by GT'}")
    else:
        sil_tp = None
        print("  Skipped: insufficient TP clusters")
else:
    sil_tp = None
    print("  Skipped: only one GT or no TP")

# ---------------------------------------------------------------------------
# Diagnostic 2: Permutation Test for Pair-Consistency
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("DIAGNOSTIC 2: Permutation Test for Pair-Consistency")
print("=" * 70)


def pair_agreement(distances: np.ndarray, ious: np.ndarray, gt_ids: np.ndarray) -> float:
    """Compute pair agreement rate: fraction of intra-GT pairs where distance order matches IoU order.

    distances: (N, N) distance matrix or (N,) distance-to-center vector.
    """
    agree = 0
    total = 0
    for gid in np.unique(gt_ids):
        if gid < 0:
            continue
        gmask = gt_ids == gid
        n = gmask.sum()
        if n < 2:
            continue
        gidx = np.where(gmask)[0]
        giou = ious[gmask]
        for ii in range(n):
            for jj in range(ii + 1, n):
                total += 1
                i, j = gidx[ii], gidx[jj]
                if distances.ndim == 1:
                    dist_i = distances[i]
                    dist_j = distances[j]
                else:
                    dist_i = distances[i, j]
                    dist_j = distances[j, i]
                dist_order = dist_i < dist_j
                iou_order = giou[ii] > giou[jj]
                if bool(dist_order) == bool(iou_order):
                    agree += 1
    return agree / total if total > 0 else 0.0


# Compute observed consistency using Euclidean distance in whitened space
euclid_dist = np.linalg.norm(X_w[:, None] - X_w[None, :], axis=2)
observed_consistency = pair_agreement(euclid_dist, ious_arr, gt_ids_arr)
print(f"  Observed pair-consistency: {observed_consistency:.4f}")

# Permutation test: shuffle GT labels 100 times
print("  Running permutation test (100 shuffles)...")
permuted_consistencies = []
for perm_idx in tqdm(range(100), desc="Permutation"):
    shuffled_gt = gt_ids_arr.copy()
    # Shuffle within each image independently
    for img_id in np.unique(img_ids_arr):
        img_mask = img_ids_arr == img_id
        gt_in_img = shuffled_gt[img_mask]
        np.random.shuffle(gt_in_img)
        shuffled_gt[img_mask] = gt_in_img
    perm_cons = pair_agreement(euclid_dist, ious_arr, shuffled_gt)
    permuted_consistencies.append(perm_cons)

permuted_consistencies = np.array(permuted_consistencies)
p_value = (np.sum(permuted_consistencies >= observed_consistency) + 1) / 101
print(f"  Permuted consistency: mean={permuted_consistencies.mean():.4f}, std={permuted_consistencies.std():.4f}")
print(f"  p-value: {p_value:.4f}")
print(f"  -> {'Significant (p<0.05)' if p_value < 0.05 else 'NOT significant (p>=0.05)'}: observed consistency is {'higher than' if p_value < 0.05 else 'not higher than'} random")

# ---------------------------------------------------------------------------
# Diagnostic 3: Intrinsic Dimension (skdim TLE)
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("DIAGNOSTIC 3: Intrinsic Dimension (skdim TLE)")
print("=" * 70)

try:
    import skdim
    has_skdim = True
except ImportError:
    has_skdim = False
    print("  skdim not installed. Run: pip install scikit-dimension")

if has_skdim:
    if X_w_tp.shape[0] > 20:
        id_tp = skdim.id.TLE().fit(X_w_tp).dimension_
        print(f"  TP intrinsic dimension (TLE): {id_tp:.2f}")
    else:
        id_tp = None
        print(f"  Skipped TP ID: only {X_w_tp.shape[0]} TP samples")

    if X_w_fp.shape[0] > 20:
        id_fp = skdim.id.TLE().fit(X_w_fp).dimension_
        print(f"  FP intrinsic dimension (TLE): {id_fp:.2f}")
    else:
        id_fp = None
        print(f"  Skipped FP ID: only {X_w_fp.shape[0]} FP samples")

    env_dim = X_w.shape[1]
    print(f"  Embedding dimension: {env_dim}")

    if id_tp is not None and id_fp is not None:
        ratio_tp = id_tp / env_dim
        ratio_fp = id_fp / env_dim
        print(f"  ID/Env ratio: TP={ratio_tp:.2%}, FP={ratio_fp:.2%}")
        if ratio_tp > 0.8 and ratio_fp > 0.8:
            print("  -> Data fills the embedding space (no low-D manifold)")
        elif ratio_tp < 0.5 or ratio_fp < 0.5:
            print("  -> Evidence of low-dimensional manifold structure")
        else:
            print("  -> Ambiguous: moderate dimensionality reduction")
else:
    id_tp = id_fp = None

# ---------------------------------------------------------------------------
# Diagnostic 4: k-NN Graph Connectivity
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("DIAGNOSTIC 4: k-NN Graph Connectivity")
print("=" * 70)

K_NN = 15
nbrs = NearestNeighbors(n_neighbors=K_NN + 1, algorithm="auto", metric="euclidean", n_jobs=-1)
nbrs.fit(X_w)
distances_knn, indices_knn = nbrs.kneighbors(X_w)
indices_knn = indices_knn[:, 1:]  # exclude self
distances_knn = distances_knn[:, 1:]

from scipy.sparse import csr_matrix
row_idx = np.repeat(np.arange(N_total), K_NN)
col_idx = indices_knn.flatten()
data = np.ones_like(col_idx)  # unweighted for connectivity
knn_graph = csr_matrix((data, (row_idx, col_idx)), shape=(N_total, N_total))

n_components, labels = connected_components(knn_graph, directed=False, return_labels=True)
component_sizes = np.bincount(labels)
largest_size = component_sizes.max()
isolated_count = (component_sizes == 1).sum()

print(f"  k-NN graph (k={K_NN}):")
print(f"    Connected components: {n_components}")
print(f"    Largest component: {largest_size} / {N_total} ({largest_size / N_total:.1%})")
print(f"    Isolated points: {isolated_count} / {N_total} ({isolated_count / N_total:.1%})")

if isolated_count / N_total > 0.05:
    print("  -> WARNING: >5% isolated points — graph construction problematic")
else:
    print("  -> Graph connectivity acceptable")

if n_components > 1 and largest_size / N_total < 0.95:
    print("  -> WARNING: Graph is fragmented — manifold assumption weak")
else:
    print("  -> Graph is well-connected")

# ---------------------------------------------------------------------------
# Diagnostic 5: Local Curvature Estimate
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("DIAGNOSTIC 5: Local Curvature Estimate")
print("=" * 70)

print(f"  Computing local PCA curvature (k={K_NN})...")
curvatures = np.zeros(N_total)
for i in tqdm(range(N_total), desc="Local curvature"):
    neighbors = X_w[indices_knn[i]]  # (K, d)
    centered = neighbors - neighbors.mean(axis=0)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    ev = (s ** 2) / (K_NN - 1)
    ev_total = ev.sum() + 1e-12
    # Curvature = residual variance / total variance = 1 - (sum top-k / total)
    # Use 95% variance capture threshold
    ev_cumsum = np.cumsum(ev)
    n_comp_95 = np.searchsorted(ev_cumsum / ev_total, 0.95) + 1
    # Curvature proxy: fraction of variance in "neglected" dimensions
    curvature_proxy = 1.0 - ev_cumsum[min(n_comp_95 - 1, len(ev) - 1)] / ev_total
    curvatures[i] = curvature_proxy

print(f"  Curvature proxy (residual variance after 95% capture):")
print(f"    Mean:   {curvatures.mean():.4f}")
print(f"    Median: {np.median(curvatures):.4f}")
print(f"    Max:    {curvatures.max():.4f}")
print(f"    Min:    {curvatures.min():.4f}")
print(f"    Std:    {curvatures.std():.4f}")

# Compare TP vs FP curvature
tp_curv = curvatures[tp_mask]
fp_curv = curvatures[fp_mask]
if len(tp_curv) > 0 and len(fp_curv) > 0:
    print(f"\n  TP curvature: mean={tp_curv.mean():.4f}, median={np.median(tp_curv):.4f}")
    print(f"  FP curvature: mean={fp_curv.mean():.4f}, median={np.median(fp_curv):.4f}")
    # t-test
    t_stat, t_p = stats.ttest_ind(tp_curv, fp_curv)
    print(f"  t-test: t={t_stat:.3f}, p={t_p:.4f}")

if curvatures.mean() < 0.01:
    print("  -> Very flat: Euclidean distance is a good approximation of geodesic")
elif curvatures.mean() < 0.05:
    print("  -> Moderate curvature: geodesic may help slightly over Euclidean")
else:
    print("  -> High curvature: Euclidean distance is unreliable; geodesic strongly preferred")

# ---------------------------------------------------------------------------
# Summary Report
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL SUMMARY: Is this data a usable manifold for reward?")
print("=" * 70)

report: dict[str, Any] = {
    "n_proposals": int(N_total),
    "feature_dim": int(D),
    "embedding_dim": int(X_w.shape[1]),
    "tp_count": int(tp_mask.sum()),
    "fp_count": int(fp_mask.sum()),
}

# Decision criteria
verdicts = []

# 1. Clustering
if sil_all is not None:
    report["clustering"] = {
        "tp_vs_fp_silhouette": float(sil_all),
        "tp_vs_fp_davies_bouldin": float(db_all),
    }
    if sil_all > 0.3:
        verdicts.append(("Clustering", "PASS", "Clear cluster structure (sil>0.3)"))
    elif sil_all > 0.1:
        verdicts.append(("Clustering", "WEAK", "Weak cluster structure (sil 0.1-0.3)"))
    else:
        verdicts.append(("Clustering", "FAIL", "No cluster structure (sil<0.1)"))
else:
    verdicts.append(("Clustering", "SKIP", "Could not compute"))

if sil_tp is not None:
    report["clustering"]["tp_by_gt_silhouette"] = float(sil_tp)

# 2. Permutation
report["permutation_test"] = {
    "observed_consistency": float(observed_consistency),
    "permuted_mean": float(permuted_consistencies.mean()),
    "permuted_std": float(permuted_consistencies.std()),
    "p_value": float(p_value),
}
if p_value < 0.05:
    verdicts.append(("Permutation", "PASS", f"Significant pair-consistency (p={p_value:.3f})"))
else:
    verdicts.append(("Permutation", "FAIL", f"Not significant (p={p_value:.3f}) — no structure beyond random"))

# 3. Intrinsic dimension
if id_tp is not None and id_fp is not None:
    report["intrinsic_dimension"] = {
        "tp_tle": float(id_tp),
        "fp_tle": float(id_fp),
        "embedding_dim": int(env_dim),
        "tp_ratio": float(id_tp / env_dim),
        "fp_ratio": float(id_fp / env_dim),
    }
    if id_tp / env_dim < 0.5 and id_fp / env_dim < 0.5:
        verdicts.append(("Intrinsic Dim", "PASS", f"Low-D manifold (TP={id_tp:.1f}, FP={id_fp:.1f} vs env={env_dim})"))
    elif id_tp / env_dim > 0.8 and id_fp / env_dim > 0.8:
        verdicts.append(("Intrinsic Dim", "FAIL", f"Fills embedding space (TP={id_tp:.1f}, FP={id_fp:.1f})"))
    else:
        verdicts.append(("Intrinsic Dim", "WEAK", f"Moderate dimensionality (TP={id_tp:.1f}, FP={id_fp:.1f})"))
else:
    verdicts.append(("Intrinsic Dim", "SKIP", "skdim not available or insufficient samples"))

# 4. Connectivity
report["connectivity"] = {
    "k_nn": K_NN,
    "n_components": int(n_components),
    "largest_component_ratio": float(largest_size / N_total),
    "isolated_ratio": float(isolated_count / N_total),
}
if isolated_count / N_total > 0.05:
    verdicts.append(("Connectivity", "FAIL", f"{isolated_count/N_total:.1%} isolated points"))
else:
    verdicts.append(("Connectivity", "PASS", f"Well-connected ({isolated_count/N_total:.1%} isolated)"))

# 5. Curvature
report["curvature"] = {
    "mean": float(curvatures.mean()),
    "median": float(np.median(curvatures)),
    "std": float(curvatures.std()),
    "tp_mean": float(tp_curv.mean()) if len(tp_curv) > 0 else None,
    "fp_mean": float(fp_curv.mean()) if len(fp_curv) > 0 else None,
}
if curvatures.mean() < 0.01:
    verdicts.append(("Curvature", "PASS", "Flat manifold — Euclidean is fine"))
elif curvatures.mean() < 0.05:
    verdicts.append(("Curvature", "WEAK", "Moderate curvature — geodesic may help"))
else:
    verdicts.append(("Curvature", "FAIL", "High curvature — Euclidean unreliable"))

# Print verdict table
print(f"\n{'Test':<20} {'Verdict':<8} {'Detail'}")
print("-" * 70)
for name, verdict, detail in verdicts:
    print(f"{name:<20} {verdict:<8} {detail}")

# Overall decision
passes = sum(1 for _, v, _ in verdicts if v == "PASS")
fails = sum(1 for _, v, _ in verdicts if v == "FAIL")
weak = sum(1 for _, v, _ in verdicts if v == "WEAK")
skips = sum(1 for _, v, _ in verdicts if v == "SKIP")

print("\n" + "=" * 70)
if fails >= 2:
    overall = "REJECT"
    overall_msg = "Data does NOT form a usable manifold. Do not use for reward."
elif passes >= 3 and fails == 0:
    overall = "ACCEPT"
    overall_msg = "Data shows manifold structure. Viable for reward design."
else:
    overall = "AMBIGUOUS"
    overall_msg = "Mixed signals. Consider more data or alternative features."

print(f"OVERALL VERDICT: {overall}")
print(f"  Pass: {passes}, Weak: {weak}, Fail: {fails}, Skip: {skips}")
print(f"  {overall_msg}")
print("=" * 70)

report["verdicts"] = [{"test": n, "verdict": v, "detail": d} for n, v, d in verdicts]
report["overall"] = overall
report["overall_message"] = overall_msg

# Save
import json
with open(OUT_DIR / "manifold_geometry_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False, default=float)

print(f"\n[Done] Report saved to {OUT_DIR / 'manifold_geometry_report.json'}")
