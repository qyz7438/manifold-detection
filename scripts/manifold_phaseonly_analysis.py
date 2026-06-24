"""Phase-only reconstruction manifold analysis for RLIimage proposals.

Pipeline:
    proposal ROI feature区域 (64x64x256)
        -> rFFT2 -> amplitude归一化 -> 只保留相位
        -> iRFFT2
    phase-only 重建图 (64x64x256) = 1,048,576 维
        -> z-score -> PCA(50, whiten=True, fit on TP) -> Isomap(6)
        -> 测地距离 -> pair 一致率 vs IoU

对比:
    - 原图直出 1,048,576 维 (不加 FFT)
    - Phase-only 重建 1,048,576 维
    - 之前 best: Isomap(6) ROI-FFT = 60.6%

内存策略: 使用 numpy memmap 将特征存到磁盘，避免同时加载两个完整 float32 数组到内存。
"""
from __future__ import annotations

import sys
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import shortest_path
from scipy import stats

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.experiments.runner_utils import (
    build_penn_fudan_loaders_320,
    decode_boxes,
)
from scripts.round2102_runner import bm
from spectral_detection_posttrain.utils.seed import set_seed

# ---------------------------------------------------------------------------
# 1. Data collection: extract phase-only reconstructions + raw crops
# ---------------------------------------------------------------------------

set_seed(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
OUT_DIR = Path("scripts/manifold_phaseonly_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CROP_SIZE = 64
# ROI crops from box_roi_pool are 256-channel feature maps, not 3-channel RGB
# Flattened dim = CROP_SIZE * CROP_SIZE * 256 = 1,048,576

print("Loading model...")
model = bm().to(DEV)
ckpt = torch.load(CKPT, map_location=DEV)
model.load_state_dict(ckpt["model"])
model.eval()

# Hooks to capture internal tensors
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

all_iou: list[float] = []
all_conf: list[float] = []
all_gt_id: list[int] = []
all_img_id: list[int] = []
all_is_best: list[bool] = []


def phase_only_reconstruct(crop: torch.Tensor) -> torch.Tensor:
    """Apply phase-only filter: set |FFT| = 1, preserve phase, iRFFT2 back.

    Args:
        crop: (N, C, H, W) tensor.

    Returns:
        (N, C, H, W) phase-only reconstruction.
    """
    fft = torch.fft.rfft2(crop, dim=(-2, -1), norm="ortho")
    phase = torch.angle(fft)
    fft_phase_only = torch.exp(1j * phase)
    recon = torch.fft.irfft2(fft_phase_only, s=crop.shape[-2:], dim=(-2, -1), norm="ortho")
    return recon


# First pass: count total proposals to pre-allocate memmap
print("Pass 1: Counting proposals...")
N_total = 0
for img_idx, (img, tgt) in enumerate(tqdm(vl, desc="Counting")):
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
    N_total += rf.shape[0]

print(f"Total proposals to extract: {N_total}")
D = CROP_SIZE * CROP_SIZE * 256  # 1,048,576

# Pre-allocate memmap files (float32)
X_phase_mmap = np.memmap(OUT_DIR / "X_phase.dat", dtype=np.float32, mode="w+", shape=(N_total, D))
X_raw_mmap = np.memmap(OUT_DIR / "X_raw.dat", dtype=np.float32, mode="w+", shape=(N_total, D))

# Second pass: write features to memmap
print("Pass 2: Extracting phase-only reconstructions and raw crops...")
offset = 0
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

    # Resize crops to 64x64 for phase-only analysis
    crops_resized = F.interpolate(crops, size=(CROP_SIZE, CROP_SIZE), mode="bilinear", align_corners=False)

    # Phase-only reconstruction
    phase_recon = phase_only_reconstruct(crops_resized)  # (N, C, 64, 64)
    phase_flat = phase_recon.reshape(N, -1).cpu().numpy()  # (N, 1048576)

    # Raw crop (for comparison)
    raw_flat = crops_resized.reshape(N, -1).cpu().numpy()  # (N, 1048576)

    conf = F.softmax(model.roi_heads.box_predictor.cls_score(bf), dim=-1)[:, 1]

    # Write to memmap
    X_phase_mmap[offset:offset+N] = phase_flat
    X_raw_mmap[offset:offset+N] = raw_flat

    if len(gt_boxes) == 0:
        for i in range(N):
            all_iou.append(0.0)
            all_conf.append(conf[i].item())
            all_gt_id.append(-1)
            all_img_id.append(img_idx)
            all_is_best.append(False)
        offset += N
        continue

    ious = box_iou(decoded, gt_boxes)  # (N, G)
    best_iou, best_gt = ious.max(dim=1)  # (N,)

    for i in range(N):
        matched_gt = best_gt[i].item()
        matched_iou = best_iou[i].item()
        is_best = True
        for j in range(N):
            if j != i and best_gt[j].item() == matched_gt and best_iou[j].item() > matched_iou:
                is_best = False
                break

        all_iou.append(matched_iou)
        all_conf.append(conf[i].item())
        all_gt_id.append(int(matched_gt))
        all_img_id.append(img_idx)
        all_is_best.append(is_best)

    offset += N

# Flush memmap to disk
X_phase_mmap.flush()
X_raw_mmap.flush()

ious_arr = np.array(all_iou, dtype=np.float64)
confs_arr = np.array(all_conf, dtype=np.float64)
gt_ids_arr = np.array(all_gt_id, dtype=np.int32)
img_ids_arr = np.array(all_img_id, dtype=np.int32)
is_best_arr = np.array(all_is_best, dtype=bool)

print(f"\nTotal proposals: {N_total}, feature dim: {D}")
print(f"TP (IoU>=0.5): {(ious_arr >= 0.5).sum()}, FP: {(ious_arr < 0.5).sum()}")
print(f"Best-for-GT: {is_best_arr.sum()}")

# Save labels for reproducibility
np.savez(
    OUT_DIR / "labels.npz",
    ious=ious_arr,
    confs=confs_arr,
    gt_ids=gt_ids_arr,
    img_ids=img_ids_arr,
    is_best=is_best_arr,
)

# ---------------------------------------------------------------------------
# 2. Preprocessing: z-score normalization (fit on TP subset)
#    Use incremental (batch) fitting to avoid loading full array into memory
# ---------------------------------------------------------------------------

tp_mask = ious_arr >= 0.5
print(f"\n[Step 1] Per-dim z-score normalization (fit on TP={tp_mask.sum()})...")

# Incremental z-score: first compute mean/std on TP subset via batches
BATCH = 256

# Phase-only: compute mean and std on TP subset
tp_indices = np.where(tp_mask)[0]
mean_phase = np.zeros(D, dtype=np.float64)
for start in tqdm(range(0, len(tp_indices), BATCH), desc="Phase mean"):
    end = min(start + BATCH, len(tp_indices))
    idx = tp_indices[start:end]
    mean_phase += X_phase_mmap[idx].sum(axis=0)
mean_phase /= len(tp_indices)

var_phase = np.zeros(D, dtype=np.float64)
for start in tqdm(range(0, len(tp_indices), BATCH), desc="Phase var"):
    end = min(start + BATCH, len(tp_indices))
    idx = tp_indices[start:end]
    diff = X_phase_mmap[idx] - mean_phase
    var_phase += (diff ** 2).sum(axis=0)
std_phase = np.sqrt(var_phase / len(tp_indices)) + 1e-8

# Raw: compute mean and std on TP subset
mean_raw = np.zeros(D, dtype=np.float64)
for start in tqdm(range(0, len(tp_indices), BATCH), desc="Raw mean"):
    end = min(start + BATCH, len(tp_indices))
    idx = tp_indices[start:end]
    mean_raw += X_raw_mmap[idx].sum(axis=0)
mean_raw /= len(tp_indices)

var_raw = np.zeros(D, dtype=np.float64)
for start in tqdm(range(0, len(tp_indices), BATCH), desc="Raw var"):
    end = min(start + BATCH, len(tp_indices))
    idx = tp_indices[start:end]
    diff = X_raw_mmap[idx] - mean_raw
    var_raw += (diff ** 2).sum(axis=0)
std_raw = np.sqrt(var_raw / len(tp_indices)) + 1e-8

# Create normalized memmaps
X_phase_z_mmap = np.memmap(OUT_DIR / "X_phase_z.dat", dtype=np.float32, mode="w+", shape=(N_total, D))
X_raw_z_mmap = np.memmap(OUT_DIR / "X_raw_z.dat", dtype=np.float32, mode="w+", shape=(N_total, D))

for start in tqdm(range(0, N_total, BATCH), desc="Normalize phase"):
    end = min(start + BATCH, N_total)
    X_phase_z_mmap[start:end] = (X_phase_mmap[start:end] - mean_phase) / std_phase

for start in tqdm(range(0, N_total, BATCH), desc="Normalize raw"):
    end = min(start + BATCH, N_total)
    X_raw_z_mmap[start:end] = (X_raw_mmap[start:end] - mean_raw) / std_raw

X_phase_z_mmap.flush()
X_raw_z_mmap.flush()

# ---------------------------------------------------------------------------
# 3. Evaluation functions
# ---------------------------------------------------------------------------

K_NN = 15


def evaluate_manifold(X_w: np.ndarray, label: str, pca_dim: int | None, isomap_dim: int | None) -> dict:
    """Run k-NN graph + geodesic distances + pair agreement for a given embedding."""
    print(f"\n{'='*60}")
    print(f"[Config] {label}")
    print(f"{'='*60}")
    print(f"  Embedding shape: {X_w.shape}")

    # k-NN + local PCA for intrinsic dimension
    print(f"  [Step A] k-NN (k={K_NN}) + local PCA for intrinsic dimension...")
    nbrs = NearestNeighbors(n_neighbors=K_NN + 1, algorithm="auto", metric="euclidean", n_jobs=-1)
    nbrs.fit(X_w)
    distances, indices = nbrs.kneighbors(X_w)  # includes self
    indices = indices[:, 1:]
    distances = distances[:, 1:]

    local_dims = np.zeros(N_total, dtype=np.int32)
    for i in tqdm(range(N_total), desc="Local PCA"):
        neighbors = X_w[indices[i]]  # (K, d)
        centered = neighbors - neighbors.mean(axis=0)
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        ev = (s ** 2) / (K_NN - 1)
        ev_cumsum = np.cumsum(ev)
        ev_total = ev_cumsum[-1] + 1e-12
        n_comp = np.searchsorted(ev_cumsum / ev_total, 0.95) + 1
        local_dims[i] = int(n_comp)

    print(f"    Local intrinsic dimension: mean={local_dims.mean():.2f}, median={np.median(local_dims):.1f}, "
          f"min={local_dims.min()}, max={local_dims.max()}")

    # Geodesic distance via k-NN graph shortest path
    print("  [Step B] Building k-NN graph and computing geodesic distances...")
    from scipy.sparse import csr_matrix
    row_idx = np.repeat(np.arange(N_total), K_NN)
    col_idx = indices.flatten()
    data = distances.flatten()
    adj = csr_matrix((data, (row_idx, col_idx)), shape=(N_total, N_total))
    adj = adj.maximum(adj.T)

    print("    Running shortest_path on k-NN graph (Dijkstra)...")
    t0 = time.time()
    geodesic_dist = shortest_path(adj, method="D", directed=False, unweighted=False)
    print(f"    Done in {time.time() - t0:.1f}s")

    # Euclidean distance matrix (for comparison)
    print("    Computing Euclidean distance matrix...")
    from scipy.spatial.distance import cdist
    euclid_dist = cdist(X_w, X_w, metric="euclidean")

    # TP cluster center
    print("  [Step C] Finding TP cluster center...")
    print(f"    TP proposals: {tp_mask.sum()}")
    if tp_mask.sum() == 0:
        print("    ERROR: No TP proposals found! Aborting.")
        sys.exit(1)

    tp_centroid = X_w[tp_mask].mean(axis=0)
    tp_indices = np.where(tp_mask)[0]
    tp_to_centroid = np.linalg.norm(X_w[tp_indices] - tp_centroid, axis=1)
    closest5 = tp_indices[np.argsort(tp_to_centroid)[:5]]
    tp_center_idx = int(np.median(closest5))
    print(f"    TP center index (median of 5 closest to centroid): {tp_center_idx}, IoU={ious_arr[tp_center_idx]:.3f}")

    # Distance to TP center
    print("  [Step D] Computing distances to TP center...")
    geodesic_to_tp = geodesic_dist[:, tp_center_idx]
    euclid_to_tp = euclid_dist[:, tp_center_idx]

    n_inf = np.isinf(geodesic_to_tp).sum()
    if n_inf > 0:
        print(f"    Warning: {n_inf} proposals disconnected from TP center in k-NN graph. "
              "Falling back to Euclidean for those.")
        max_geo = geodesic_to_tp[np.isfinite(geodesic_to_tp)].max()
        geodesic_to_tp = np.where(np.isinf(geodesic_to_tp), max_geo * 2, geodesic_to_tp)

    # Pair agreement + rank correlation
    print("  [Step E] Evaluating DPO pair agreement rates...")

    def pair_agreement(distances: np.ndarray, ious: np.ndarray, gt_ids: np.ndarray, confs: np.ndarray) -> dict:
        total_pairs = 0
        agree_pairs = 0
        agree_pairs_conf_controlled = 0
        total_pairs_conf_controlled = 0

        for gid in np.unique(gt_ids):
            if gid < 0:
                continue
            gmask = gt_ids == gid
            n = gmask.sum()
            if n < 2:
                continue
            gdist = distances[gmask]
            giou = ious[gmask]
            gconf = confs[gmask]

            for i in range(n):
                for j in range(i + 1, n):
                    total_pairs += 1
                    if (gdist[i] < gdist[j]) == (giou[i] > giou[j]):
                        agree_pairs += 1

            for i in range(n):
                for j in range(i + 1, n):
                    if abs(gconf[i] - gconf[j]) < 0.1:
                        total_pairs_conf_controlled += 1
                        if (gdist[i] < gdist[j]) == (giou[i] > giou[j]):
                            agree_pairs_conf_controlled += 1

        return {
            "total_pairs": total_pairs,
            "agree_pairs": agree_pairs,
            "agreement_rate": agree_pairs / total_pairs if total_pairs > 0 else 0.0,
            "total_pairs_conf_controlled": total_pairs_conf_controlled,
            "agree_pairs_conf_controlled": agree_pairs_conf_controlled,
            "agreement_rate_conf_controlled": (
                agree_pairs_conf_controlled / total_pairs_conf_controlled if total_pairs_conf_controlled > 0 else 0.0
            ),
        }

    def rank_correlation(distances: np.ndarray, ious: np.ndarray, gt_ids: np.ndarray) -> dict:
        cors = []
        for gid in np.unique(gt_ids):
            if gid < 0:
                continue
            gmask = gt_ids == gid
            n = gmask.sum()
            if n < 3:
                continue
            gdist = distances[gmask]
            giou = ious[gmask]
            r, _ = stats.spearmanr(-gdist, giou)
            if not np.isnan(r):
                cors.append(r)
        return {
            "mean_spearman": np.mean(cors) if cors else 0.0,
            "median_spearman": np.median(cors) if cors else 0.0,
            "n_groups": len(cors),
        }

    geo_agree = pair_agreement(geodesic_to_tp, ious_arr, gt_ids_arr, confs_arr)
    euc_agree = pair_agreement(euclid_to_tp, ious_arr, gt_ids_arr, confs_arr)
    geo_rank = rank_correlation(geodesic_to_tp, ious_arr, gt_ids_arr)
    euc_rank = rank_correlation(euclid_to_tp, ious_arr, gt_ids_arr)

    print(f"\n    --- Pair Agreement (all pairs) ---")
    print(f"    Geodesic:  {geo_agree['agree_pairs']}/{geo_agree['total_pairs']} = {geo_agree['agreement_rate']:.4f}")
    print(f"    Euclidean: {euc_agree['agree_pairs']}/{euc_agree['total_pairs']} = {euc_agree['agreement_rate']:.4f}")
    print(f"    Delta:     {geo_agree['agreement_rate'] - euc_agree['agreement_rate']:+.4f}")

    print(f"\n    --- Pair Agreement (conf-controlled, |dconf|<0.1) ---")
    print(f"    Geodesic:  {geo_agree['agree_pairs_conf_controlled']}/{geo_agree['total_pairs_conf_controlled']} = {geo_agree['agreement_rate_conf_controlled']:.4f}")
    print(f"    Euclidean: {euc_agree['agree_pairs_conf_controlled']}/{euc_agree['total_pairs_conf_controlled']} = {euc_agree['agreement_rate_conf_controlled']:.4f}")
    print(f"    Delta:     {geo_agree['agreement_rate_conf_controlled'] - euc_agree['agreement_rate_conf_controlled']:+.4f}")

    print(f"\n    --- Per-GT Spearman (-dist vs IoU) ---")
    print(f"    Geodesic:  mean={geo_rank['mean_spearman']:.4f}, median={geo_rank['median_spearman']:.4f}, n_groups={geo_rank['n_groups']}")
    print(f"    Euclidean: mean={euc_rank['mean_spearman']:.4f}, median={euc_rank['median_spearman']:.4f}, n_groups={euc_rank['n_groups']}")

    return {
        "label": label,
        "pca_dim": pca_dim,
        "isomap_dim": isomap_dim,
        "embedding_dim": int(X_w.shape[1]),
        "k_nn": K_NN,
        "local_intrinsic_dim_mean": float(local_dims.mean()),
        "local_intrinsic_dim_median": float(np.median(local_dims)),
        "tp_count": int(tp_mask.sum()),
        "tp_center_idx": int(tp_center_idx),
        "tp_center_iou": float(ious_arr[tp_center_idx]),
        "geodesic": {
            "pair_agreement": geo_agree,
            "rank_correlation": geo_rank,
        },
        "euclidean": {
            "pair_agreement": euc_agree,
            "rank_correlation": euc_rank,
        },
    }


# ---------------------------------------------------------------------------
# 4. Run configurations
# ---------------------------------------------------------------------------

results = []

# --- Phase-only configurations ---

# 4a. Phase-only: z-score only (1M-dim)
print("\n[Phase-only Config 1] z-score only (1M-dim)...")
results.append(evaluate_manifold(np.array(X_phase_z_mmap), "Phase-only z-score (1M)", None, None))

# 4b. Phase-only: PCA(50) on z-score (fit on TP)
print("\n[Phase-only Config 2] PCA(50, whiten=True) on z-score...")
pca50 = PCA(n_components=50, whiten=True, random_state=42)
pca50.fit(X_phase_z_mmap[tp_mask])
X_phase_pca50 = pca50.transform(X_phase_z_mmap)
print(f"  Explained variance (50 comp): {pca50.explained_variance_ratio_.sum():.4f}")
results.append(evaluate_manifold(X_phase_pca50, "Phase-only PCA=50", 50, None))

# 4c. Phase-only: Isomap(6) on z-score
print("\n[Phase-only Config 3] Isomap(6) on z-score...")
isomap6 = Isomap(n_neighbors=K_NN, n_components=6, path_method="auto", n_jobs=-1)
X_phase_iso6 = isomap6.fit_transform(np.array(X_phase_z_mmap))
results.append(evaluate_manifold(X_phase_iso6, "Phase-only Isomap=6", None, 6))

# 4d. Phase-only: PCA(50) -> Isomap(6)
print("\n[Phase-only Config 4] PCA(50) -> Isomap(6)...")
isomap6_pca = Isomap(n_neighbors=K_NN, n_components=6, path_method="auto", n_jobs=-1)
X_phase_pca50_iso6 = isomap6_pca.fit_transform(X_phase_pca50)
results.append(evaluate_manifold(X_phase_pca50_iso6, "Phase-only PCA=50 -> Isomap=6", 50, 6))

# --- Raw crop configurations (for comparison) ---

# 4e. Raw: z-score only (1M-dim)
print("\n[Raw Config 1] z-score only (1M-dim)...")
results.append(evaluate_manifold(np.array(X_raw_z_mmap), "Raw z-score (1M)", None, None))

# 4f. Raw: PCA(50) on z-score
print("\n[Raw Config 2] PCA(50, whiten=True) on z-score...")
pca50_raw = PCA(n_components=50, whiten=True, random_state=42)
pca50_raw.fit(X_raw_z_mmap[tp_mask])
X_raw_pca50 = pca50_raw.transform(X_raw_z_mmap)
print(f"  Explained variance (50 comp): {pca50_raw.explained_variance_ratio_.sum():.4f}")
results.append(evaluate_manifold(X_raw_pca50, "Raw PCA=50", 50, None))

# 4g. Raw: Isomap(6) on z-score
print("\n[Raw Config 3] Isomap(6) on z-score...")
isomap6_raw = Isomap(n_neighbors=K_NN, n_components=6, path_method="auto", n_jobs=-1)
X_raw_iso6 = isomap6_raw.fit_transform(np.array(X_raw_z_mmap))
results.append(evaluate_manifold(X_raw_iso6, "Raw Isomap=6", None, 6))

# 4h. Raw: PCA(50) -> Isomap(6)
print("\n[Raw Config 4] PCA(50) -> Isomap(6)...")
isomap6_raw_pca = Isomap(n_neighbors=K_NN, n_components=6, path_method="auto", n_jobs=-1)
X_raw_pca50_iso6 = isomap6_raw_pca.fit_transform(X_raw_pca50)
results.append(evaluate_manifold(X_raw_pca50_iso6, "Raw PCA=50 -> Isomap=6", 50, 6))

# ---------------------------------------------------------------------------
# 5. Summary comparison table
# ---------------------------------------------------------------------------

print(f"\n{'='*80}")
print("SUMMARY COMPARISON TABLE — ALL PAIRS")
print(f"{'='*80}")
print(f"{'Config':<35s} {'Pair Agree':>12s} {'Geo-Euc Delta':>14s} {'Spearman':>10s}")
print(f"{'-'*80}")
for r in results:
    geo_agree = r["geodesic"]["pair_agreement"]["agreement_rate"]
    euc_agree = r["euclidean"]["pair_agreement"]["agreement_rate"]
    delta = geo_agree - euc_agree
    spear = r["geodesic"]["rank_correlation"]["mean_spearman"]
    print(f"{r['label']:<35s} {geo_agree:12.4f} {delta:14.4f} {spear:10.4f}")

print(f"\n{'='*80}")
print("CONFIDENCE-CONTROLLED PAIRS (|dconf|<0.1)")
print(f"{'='*80}")
print(f"{'Config':<35s} {'Pair Agree':>12s} {'Geo-Euc Delta':>14s} {'Spearman':>10s}")
print(f"{'-'*80}")
for r in results:
    geo_agree = r["geodesic"]["pair_agreement"]["agreement_rate_conf_controlled"]
    euc_agree = r["euclidean"]["pair_agreement"]["agreement_rate_conf_controlled"]
    delta = geo_agree - euc_agree
    spear = r["geodesic"]["rank_correlation"]["mean_spearman"]
    print(f"{r['label']:<35s} {geo_agree:12.4f} {delta:14.4f} {spear:10.4f}")

# ---------------------------------------------------------------------------
# 6. Uncertain confidence zone analysis
# ---------------------------------------------------------------------------

print(f"\n{'='*80}")
print("UNCERTAIN CONFIDENCE ZONE ANALYSIS")
print(f"{'='*80}")

# Define uncertain zone: confidence in [0.3, 0.7]
uncertain_mask = (confs_arr >= 0.3) & (confs_arr <= 0.7)
print(f"Uncertain zone proposals: {uncertain_mask.sum()} / {N_total}")

print(f"\n{'Config':<35s} {'Uncertain Pairs':>15s} {'Agree Rate':>12s}")
print(f"{'-'*80}")

for embed, label in [(X_phase_iso6, "Phase-only Isomap=6"), (X_raw_iso6, "Raw Isomap=6")]:
    nbrs = NearestNeighbors(n_neighbors=K_NN + 1, algorithm="auto", metric="euclidean", n_jobs=-1)
    nbrs.fit(embed)
    distances, indices = nbrs.kneighbors(embed)
    indices = indices[:, 1:]
    distances = distances[:, 1:]

    from scipy.sparse import csr_matrix
    row_idx = np.repeat(np.arange(N_total), K_NN)
    col_idx = indices.flatten()
    data = distances.flatten()
    adj = csr_matrix((data, (row_idx, col_idx)), shape=(N_total, N_total))
    adj = adj.maximum(adj.T)
    geodesic_dist = shortest_path(adj, method="D", directed=False, unweighted=False)

    tp_centroid = embed[tp_mask].mean(axis=0)
    tp_indices = np.where(tp_mask)[0]
    tp_to_centroid = np.linalg.norm(embed[tp_indices] - tp_centroid, axis=1)
    closest5 = tp_indices[np.argsort(tp_to_centroid)[:5]]
    tp_center_idx = int(np.median(closest5))
    geodesic_to_tp = geodesic_dist[:, tp_center_idx]
    n_inf = np.isinf(geodesic_to_tp).sum()
    if n_inf > 0:
        max_geo = geodesic_to_tp[np.isfinite(geodesic_to_tp)].max()
        geodesic_to_tp = np.where(np.isinf(geodesic_to_tp), max_geo * 2, geodesic_to_tp)

    def pair_agreement_masked(distances: np.ndarray, ious: np.ndarray, gt_ids: np.ndarray,
                              confs: np.ndarray, mask: np.ndarray) -> dict:
        total_pairs = 0
        agree_pairs = 0
        for gid in np.unique(gt_ids):
            if gid < 0:
                continue
            gmask = (gt_ids == gid) & mask
            n = gmask.sum()
            if n < 2:
                continue
            gdist = distances[gmask]
            giou = ious[gmask]
            for i in range(n):
                for j in range(i + 1, n):
                    total_pairs += 1
                    if (gdist[i] < gdist[j]) == (giou[i] > giou[j]):
                        agree_pairs += 1
        return {
            "total_pairs": total_pairs,
            "agree_pairs": agree_pairs,
            "agreement_rate": agree_pairs / total_pairs if total_pairs > 0 else 0.0,
        }

    ua = pair_agreement_masked(geodesic_to_tp, ious_arr, gt_ids_arr, confs_arr, uncertain_mask)
    print(f"{label:<35s} {ua['total_pairs']:>15d} {ua['agreement_rate']:>12.4f}")

# ---------------------------------------------------------------------------
# 7. Save JSON
# ---------------------------------------------------------------------------

summary = {
    "N_total": int(N_total),
    "feature_dim": int(D),
    "crop_size": CROP_SIZE,
    "k_nn": K_NN,
    "configs": results,
}

with open(OUT_DIR / "summary_phaseonly.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=float)

print(f"\n[Done] Results saved to {OUT_DIR}/")
print(f"  - X_phase.dat, X_raw.dat (memmap)")
print(f"  - labels.npz")
print(f"  - summary_phaseonly.json")
