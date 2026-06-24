"""Non-linear manifold embedding comparison for baseline FFT spectral features.

Pipeline:
    7168-dim FFT amplitude -> z-score -> skip PCA
    -> Isomap / LLE / Diffusion Maps -> 6-dim embedding
    -> TP cluster center distance -> DPO pair agreement rate

Compares non-linear embeddings against PCA(6) + geodesic distance baseline.
"""
from __future__ import annotations

import sys
import time
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap, LocallyLinearEmbedding, SpectralEmbedding
from scipy import stats
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.experiments.runner_utils import (
    build_penn_fudan_loaders_320,
    decode_boxes,
    extract_perchan_fft,
)
from scripts.round2102_runner import bm
from spectral_detection_posttrain.utils.seed import set_seed

# ---------------------------------------------------------------------------
# 1. Data collection (identical to manifold_fft_analysis.py)
# ---------------------------------------------------------------------------

set_seed(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
OUT_DIR = Path("scripts/manifold_nonlinear_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

all_fft_amp: list[np.ndarray] = []
all_iou: list[float] = []
all_conf: list[float] = []
all_gt_id: list[int] = []
all_img_id: list[int] = []
all_is_best: list[bool] = []

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
    fft = extract_perchan_fft(crops)  # (N, 6*C)
    ch = fft.shape[1] // 6
    # Amplitude features only: [:, 0*ch : 3*ch] -> (N, 3*C)
    amp_features = fft[:, 0 * ch : 3 * ch]  # (N, 3*C)
    amp_vec = amp_features.cpu().numpy()  # (N, 3*C)

    conf = F.softmax(model.roi_heads.box_predictor.cls_score(bf), dim=-1)[:, 1]

    if len(gt_boxes) == 0:
        for i in range(N):
            all_fft_amp.append(amp_vec[i])
            all_iou.append(0.0)
            all_conf.append(conf[i].item())
            all_gt_id.append(-1)
            all_img_id.append(img_idx)
            all_is_best.append(False)
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

        all_fft_amp.append(amp_vec[i])
        all_iou.append(matched_iou)
        all_conf.append(conf[i].item())
        all_gt_id.append(int(matched_gt))
        all_img_id.append(img_idx)
        all_is_best.append(is_best)

X_raw = np.stack(all_fft_amp, axis=0)  # (N_total, D)
ious_arr = np.array(all_iou, dtype=np.float64)
confs_arr = np.array(all_conf, dtype=np.float64)
gt_ids_arr = np.array(all_gt_id, dtype=np.int32)
img_ids_arr = np.array(all_img_id, dtype=np.int32)
is_best_arr = np.array(all_is_best, dtype=bool)

N_total, D = X_raw.shape
print(f"\nTotal proposals: {N_total}, FFT amp feature dim: {D}")
print(f"TP (IoU>=0.5): {(ious_arr >= 0.5).sum()}, FP: {(ious_arr < 0.5).sum()}")
print(f"Best-for-GT: {is_best_arr.sum()}")

# Save raw features for reproducibility
np.savez(
    OUT_DIR / "raw_features.npz",
    X_raw=X_raw,
    ious=ious_arr,
    confs=confs_arr,
    gt_ids=gt_ids_arr,
    img_ids=img_ids_arr,
    is_best=is_best_arr,
)

# ---------------------------------------------------------------------------
# 2. Preprocessing: z-score only (skip PCA)
# ---------------------------------------------------------------------------

print("\n[Step 1] Per-dim z-score normalization...")
scaler = StandardScaler()
X_z = scaler.fit_transform(X_raw)

# ---------------------------------------------------------------------------
# 3. Non-linear embeddings -> 6-dim
# ---------------------------------------------------------------------------

N_COMPONENTS = 6
N_NEIGHBORS = 15

embeddings: dict[str, np.ndarray] = {}

# 3a. Isomap
print(f"\n[Step 2a] Isomap (n_components={N_COMPONENTS}, n_neighbors={N_NEIGHBORS})...")
t0 = time.time()
isomap = Isomap(n_components=N_COMPONENTS, n_neighbors=N_NEIGHBORS, n_jobs=-1)
X_isomap = isomap.fit_transform(X_z)
embeddings["Isomap"] = X_isomap
print(f"  Done in {time.time() - t0:.1f}s  shape={X_isomap.shape}")

# 3b. LLE
print(f"\n[Step 2b] LLE (n_components={N_COMPONENTS}, n_neighbors={N_NEIGHBORS})...")
t0 = time.time()
lle = LocallyLinearEmbedding(n_components=N_COMPONENTS, n_neighbors=N_NEIGHBORS, method="standard", random_state=42, n_jobs=-1)
X_lle = lle.fit_transform(X_z)
embeddings["LLE"] = X_lle
print(f"  Done in {time.time() - t0:.1f}s  shape={X_lle.shape}")

# 3c. Diffusion Maps (SpectralEmbedding approximation)
print(f"\n[Step 2c] Diffusion Maps via SpectralEmbedding (n_components={N_COMPONENTS}, n_neighbors={N_NEIGHBORS}, gamma=1.0)...")
t0 = time.time()
# SpectralEmbedding with affinity='nearest_neighbors' approximates diffusion maps
# gamma parameter is not directly exposed; we use the default affinity with nearest neighbors
spectral = SpectralEmbedding(n_components=N_COMPONENTS, n_neighbors=N_NEIGHBORS, random_state=42, affinity="nearest_neighbors")
X_diffusion = spectral.fit_transform(X_z)
embeddings["DiffusionMaps"] = X_diffusion
print(f"  Done in {time.time() - t0:.1f}s  shape={X_diffusion.shape}")

# 3d. PCA(6) baseline (for comparison, no whitening)
print(f"\n[Step 2d] PCA(6) baseline (no whitening)...")
from sklearn.decomposition import PCA
pca6 = PCA(n_components=N_COMPONENTS, random_state=42)
X_pca6 = pca6.fit_transform(X_z)
embeddings["PCA6"] = X_pca6
print(f"  Explained variance: {pca6.explained_variance_ratio_.sum():.4f}  shape={X_pca6.shape}")

# ---------------------------------------------------------------------------
# 4. TP cluster center in each embedding space
# ---------------------------------------------------------------------------

tp_mask = ious_arr >= 0.5
print(f"\n[Step 3] TP proposals: {tp_mask.sum()}")
if tp_mask.sum() == 0:
    print("  ERROR: No TP proposals found! Aborting.")
    sys.exit(1)

tp_indices = np.where(tp_mask)[0]

tp_centers: dict[str, int] = {}
for name, X_emb in embeddings.items():
    tp_centroid = X_emb[tp_mask].mean(axis=0)
    tp_to_centroid = np.linalg.norm(X_emb[tp_indices] - tp_centroid, axis=1)
    closest5 = tp_indices[np.argsort(tp_to_centroid)[:5]]
    tp_center_idx = int(np.median(closest5))
    tp_centers[name] = tp_center_idx
    print(f"  {name}: TP center idx={tp_center_idx}, IoU={ious_arr[tp_center_idx]:.3f}")

# ---------------------------------------------------------------------------
# 5. Evaluation functions
# ---------------------------------------------------------------------------


def pair_agreement(distances: np.ndarray, ious: np.ndarray, gt_ids: np.ndarray, confs: np.ndarray) -> dict:
    """For each GT group with >=2 proposals, count how many pairs are ranked
    consistently by distance vs IoU (controlling for confidence).
    """
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

        # All pairs
        for i in range(n):
            for j in range(i + 1, n):
                total_pairs += 1
                if (gdist[i] < gdist[j]) == (giou[i] > giou[j]):
                    agree_pairs += 1

        # Confidence-controlled pairs: only pairs with |conf diff| < 0.1
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
    """Spearman correlation per-GT-group, averaged."""
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
        # Distance should be anti-correlated with IoU
        r, _ = stats.spearmanr(-gdist, giou)
        if not np.isnan(r):
            cors.append(r)
    return {
        "mean_spearman": np.mean(cors) if cors else 0.0,
        "median_spearman": np.median(cors) if cors else 0.0,
        "n_groups": len(cors),
    }


# ---------------------------------------------------------------------------
# 6. Evaluate each embedding
# ---------------------------------------------------------------------------

print("\n[Step 4] Evaluating all embeddings...\n")

results: dict[str, dict] = {}

for name, X_emb in embeddings.items():
    tp_center_idx = tp_centers[name]
    dist_to_tp = np.linalg.norm(X_emb - X_emb[tp_center_idx], axis=1)

    agree = pair_agreement(dist_to_tp, ious_arr, gt_ids_arr, confs_arr)
    rank = rank_correlation(dist_to_tp, ious_arr, gt_ids_arr)

    results[name] = {
        "pair_agreement": agree,
        "rank_correlation": rank,
        "tp_center_idx": tp_center_idx,
        "tp_center_iou": float(ious_arr[tp_center_idx]),
    }

    print(f"--- {name} ---")
    print(f"  Pair agreement (all):       {agree['agree_pairs']}/{agree['total_pairs']} = {agree['agreement_rate']:.4f}")
    print(f"  Pair agreement (conf<0.1):  {agree['agree_pairs_conf_controlled']}/{agree['total_pairs_conf_controlled']} = {agree['agreement_rate_conf_controlled']:.4f}")
    print(f"  Spearman mean={rank['mean_spearman']:.4f}, median={rank['median_spearman']:.4f}, n_groups={rank['n_groups']}")
    print()

# ---------------------------------------------------------------------------
# 7. Summary table
# ---------------------------------------------------------------------------

print("=" * 70)
print(f"{'Method':<18s} {'Pair Agree':>12s} {'Pair Agree*':>12s} {'Spearman':>10s} {'Notes':>15s}")
print("-" * 70)
for name in ["PCA6", "Isomap", "LLE", "DiffusionMaps"]:
    r = results[name]
    pa = r["pair_agreement"]["agreement_rate"]
    pac = r["pair_agreement"]["agreement_rate_conf_controlled"]
    sp = r["rank_correlation"]["mean_spearman"]
    note = ""
    if pa > 0.65:
        note = "WORTH PURSUING"
    print(f"{name:<18s} {pa:12.4f} {pac:12.4f} {sp:10.4f} {note:>15s}")
print("=" * 70)

# ---------------------------------------------------------------------------
# 8. Save results
# ---------------------------------------------------------------------------

summary = {
    "N_total": int(N_total),
    "feature_dim": int(D),
    "n_components": N_COMPONENTS,
    "n_neighbors": N_NEIGHBORS,
    "tp_count": int(tp_mask.sum()),
    "methods": results,
}

with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=float)

# Save embeddings for potential downstream use
np.savez(
    OUT_DIR / "embeddings.npz",
    **{name: emb for name, emb in embeddings.items()},
    ious=ious_arr,
    confs=confs_arr,
    gt_ids=gt_ids_arr,
    is_best=is_best_arr,
)

print(f"\n[Done] Results saved to {OUT_DIR}/")
print(f"  - raw_features.npz")
print(f"  - embeddings.npz")
print(f"  - summary.json")
